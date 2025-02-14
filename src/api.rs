use std::path;
use std::time;

use actix_web::web;
use futures::StreamExt;
use lazy_static::lazy_static;
use llama_cpp_2::model;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{ai, embedding, knowledge, EMBEDDING_TIME, REQUEST_RESPONSE_TIME};
use prometheus::Encoder;

lazy_static! {
    static ref MODEL: model::LlamaModel = ai::load_model(
        &std::env::var("MODEL_PATH").unwrap_or("model/model.gguf".to_string()),
        ai::init_backend()
    );
    static ref EMBEDDING_MODEL: model::LlamaModel = embedding::get_embedding_model(
        &std::env::var("EMBEDDING_MODEL").unwrap_or("model/embed.gguf".to_string()),
        ai::init_backend()
    );
}

/// Loads knowledge from a set of text files into an in-memory KV-store.
pub async fn load_knowledge(path: &path::Path, db: &mut knowledge::KnowledgeBase) {
    let backend = ai::init_backend();

    let entries = match std::fs::read_dir(path) {
        Ok(v) => v,
        Err(_) => {
            log::error!(
                "Could find directory: {:?} - will NOT load any data.",
                &path
            );
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(v) => v,
            Err(_) => {
                log::error!("Could deal with entry {:?}, skipping!", &entry);
                continue;
            }
        };
        let path = entry.path();

        // check if it ends with ".txt"
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let contents = std::fs::read_to_string(path);
            // Use a match statement to handle the result
            match contents {
                Ok(data) => {
                    let tkn_context = embedding::embed(&data, &EMBEDDING_MODEL, backend);
                    knowledge::add_context(&data, tkn_context, db);
                }
                Err(err) => {
                    log::error!("Could not read file {}.", err);
                    continue;
                }
            }
        }
    }
}

/// holds some general information for the service.
#[derive(Clone)]
pub struct AppState {
    pub threads: i32,
    pub max_token: i32,
    pub prompt: String,
}

/// Represents a message in a chat.
#[derive(serde::Deserialize)]
struct Message {
    role: String,
    content: String,
}

/// Represents a JSON style request.
#[derive(serde::Deserialize)]
struct Request {
    stream: bool,
    model: String,
    messages: Vec<Message>,
}

/// Handles prometheus style queries for observability.
#[actix_web::get("/metrics")]
async fn metrics() -> impl actix_web::Responder {
    let encoder = prometheus::TextEncoder::new();
    let mut buffer = vec![];
    let mf = prometheus::gather();
    encoder.encode(&mf, &mut buffer).unwrap();

    actix_web::HttpResponse::Ok().body(buffer)
}

/// List the available models.
#[actix_web::get("/v1/models")]
async fn list_models() -> actix_web::HttpResponse {
    let response_data = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "rusty_llm",
                "object": "model",
                "created": 1733007600,
                "owned_by": "foo"
            }
        ]
    });

    actix_web::HttpResponse::Ok()
        .content_type("application/json")
        .json(response_data)
}

/// Adds the chat history to the context.
fn add_chat_history(messages: &[Message]) -> Vec<String> {
    let mut output = vec![
        "Use the following context as your chat history, inside <chat></chat> XML tags.\n<chat>"
            .to_string(),
    ];
    for message in &messages[..messages.len() - 1] {
        output.push(format!("{}: {}", message.role, message.content));
    }
    output.push("</chat>".to_string());
    output
}

#[actix_web::post("/v1/chat/completions")]
async fn stream_response(
    state: web::Data<AppState>,
    db: web::Data<knowledge::KnowledgeBase>,
    req_body: web::Json<Request>,
) -> impl actix_web::Responder {
    if req_body.messages.is_empty()
        || req_body.messages.last().unwrap().content.is_empty()
        || !req_body.stream
        || req_body.model != "rusty_llm"
    {
        return actix_web::HttpResponse::BadRequest()
            .body("Only support streaming mode with the model rusty_llm, requires at least 1 message in the request - not triggering LLM model.".to_string());
    }
    let overall_start_time = time::Instant::now();
    let backend = ai::init_backend();

    // To make this work with many models we take the last message as main query. The previous chat messages go in as context...
    let query = req_body.messages.last().unwrap().content.clone();

    let query_ = query.clone();
    let context = match actix_web::web::block(move || {
        // Timing the embedding step
        let embedding_start_time = time::Instant::now();
        let tkn_query = embedding::embed(&query_, &EMBEDDING_MODEL, backend);
        let mut context = knowledge::get_context(tkn_query, db.get_ref());
        let embedding_duration = embedding_start_time.elapsed();
        EMBEDDING_TIME
            .with_label_values(&[])
            .observe(embedding_duration.as_secs_f64());

        // ...now add the old chat stuff (if any)...
        if req_body.messages.len() >= 3 {
            context.extend(add_chat_history(&req_body.messages));
        }
        context
    })
    .await
    {
        Ok(context) => context,
        Err(e) => {
            return actix_web::HttpResponse::InternalServerError().body(e.to_string());
        }
    };

    // Initialize the AI query context
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let ai_worker = actix_web::rt::task::spawn_blocking(move || {
        let mut ai_context = ai::AiQueryContext::new(
            &query,
            context,
            state.threads,
            state.max_token,
            &state.prompt,
            &MODEL,
            backend,
        );
        while let Some(token) = ai_context.next_token() {
            if tx.send(token).is_err() {
                break;
            }
        }
    });

    // Create a token stream
    let token_stream = UnboundedReceiverStream::new(rx).map(|token| {
        let json_tmp = format!(
            r#"data: {{"id":"foo","object":"chat.completion.chunk","created":1733007600,"model":"{}", "system_fingerprint": "fp0", "choices":[{{"index":0,"delta":{{"content": "{}"}},"logprobs":null,"finish_reason":null}}]}}"#,
            "rusty_llm", token
        );
        Ok::<_, actix_web::Error>(web::Bytes::from(json_tmp + "\n\n"))
    });

    // Measure the overall request-response time
    let overall_duration = overall_start_time.elapsed();
    REQUEST_RESPONSE_TIME
        .with_label_values(&[])
        .observe(overall_duration.as_secs_f64());

    actix_web::HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(token_stream)
}

#[cfg(test)]
mod tests {
    use actix_web::body::MessageBody;

    use super::*;

    #[actix_web::test]
    async fn test_load_knowledge_for_success() {
        let mut db = knowledge::get_db();
        load_knowledge(path::Path::new("data"), &mut db).await;
    }

    #[actix_web::test]
    async fn test_metrics_for_success() {
        let app = actix_web::test::init_service(actix_web::App::new().service(metrics)).await;
        let req = actix_web::test::TestRequest::default()
            .uri("/metrics")
            .insert_header(actix_web::http::header::ContentType::plaintext())
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_list_models_for_success() {
        let app = actix_web::test::init_service(actix_web::App::new().service(list_models)).await;
        let req = actix_web::test::TestRequest::default()
            .uri("/v1/models")
            .insert_header(actix_web::http::header::ContentType::plaintext())
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[test]
    fn test_add_chat_history() {
        let input = vec![
            Message {
                role: "user".to_string(),
                content: "Who was Albert Einstein?".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "A physicist".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Tell me more!".to_string(),
            },
        ];

        let result = add_chat_history(&input);
        let expected = vec![
            "Use the following context as your chat history, inside <chat></chat> XML tags.\n<chat>".to_string(),
            "user: Who was Albert Einstein?".to_string(),
            "assistant: A physicist".to_string(),
            "</chat>".to_string(),
        ];

        assert_eq!(result, expected);
    }

    #[actix_web::test]
    async fn test_stream_response_for_success() {
        let kv_store = knowledge::get_db();
        let prompt =
            "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>"
                .to_string();
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(AppState {
                    max_token: 32,
                    threads: 4,
                    prompt,
                }))
                .app_data(web::Data::new(kv_store))
                .service(stream_response),
        )
        .await;
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/chat/completions")
            .insert_header(actix_web::http::header::ContentType::json())
            .set_payload(
                "{\"stream\": true, \"model\": \"rusty_llm\", \"messages\": [{\"role\": \"user\", \"content\": \"Who was Albert Einstein?\"}]}"
                    .try_into_bytes()
                    .unwrap(),
            )
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_stream_response_for_failure() {
        let kv_store = knowledge::get_db();
        let prompt =
            "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>"
                .to_string();
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(AppState {
                    max_token: 5,
                    threads: 4,
                    prompt,
                }))
                .app_data(web::Data::new(kv_store))
                .service(stream_response),
        )
        .await;
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/chat/completions")
            .insert_header(actix_web::http::header::ContentType::json())
            .set_payload("Garbage".try_into_bytes().unwrap())
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 400);
    }
}
