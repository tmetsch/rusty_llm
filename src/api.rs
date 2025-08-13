use crate::{
    ai, cache, embedding, knowledge, EMBEDDING_TIME, INSTANCE_LABEL, REQUEST_RESPONSE_TIME,
};
use actix_web::web;
use futures::StreamExt;
use lazy_static::lazy_static;
use llama_cpp_2::{llama_backend, model, token};
use prometheus::Encoder;
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};
use std::{collections, path};
use tokio::sync::RwLock;
use tokio_stream::wrappers::UnboundedReceiverStream;

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
                    log::error!("Could not read file {err}.");
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
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Message {
    pub(crate) role: String,
    pub(crate) content: String,
}

/// Represents a JSON style request.
#[derive(serde::Serialize, serde::Deserialize)]
struct Request {
    stream: bool,
    model: String,
    messages: Vec<Message>,
}

/// Handles prometheus style queries for observability.
#[actix_web::get("/metrics")]
async fn metrics() -> impl actix_web::Responder {
    let encoder = prometheus::TextEncoder::new();
    let mut buffer = Vec::new();
    let mf = prometheus::gather();

    if let Err(e) = encoder.encode(&mf, &mut buffer) {
        eprintln!("Failed to encode Prometheus metrics: {e}");
        return actix_web::HttpResponse::InternalServerError().finish();
    }

    actix_web::HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4") // Correct Content-Type
        .body(buffer)
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
    // No previous history if only the query is present
    if messages.len() <= 1 {
        return vec![];
    }
    let mut output = vec![
        "Use the following context as your chat history, inside <chat></chat> XML tags.\n<chat>"
            .to_string(),
    ];
    // All but the last message are "history"
    for message in &messages[..messages.len() - 1] {
        output.push(format!("{}: {}", message.role, message.content));
    }
    output.push("</chat>".to_string());
    output
}

fn tokenize(
    chat: &[Message],
    prompt_template: &str,
    db: &knowledge::KnowledgeBase,
    backend: &llama_backend::LlamaBackend,
) -> Vec<token::LlamaToken> {
    assert!(
        !chat.is_empty(),
        "Chat history must contain at least one message (the query)."
    );
    // Last message is always the query
    let query_message = chat.last().unwrap();
    let query = &query_message.content;
    // Embed query & fetch relevant knowledge
    let embedding_start_time = Instant::now();
    let tkn_query = embedding::embed(query, &EMBEDDING_MODEL, backend);
    let mut context = knowledge::get_context(tkn_query, db);
    let embedding_duration = embedding_start_time.elapsed();
    EMBEDDING_TIME
        .with_label_values(&[&INSTANCE_LABEL])
        .observe(embedding_duration.as_secs_f64());

    // Add previous messages to context (if any)
    context.extend(add_chat_history(chat));

    // Fill prompt template
    let mut vars = collections::HashMap::new();
    vars.insert("context".to_string(), context.join(","));
    vars.insert("query".to_string(), query.to_string());
    let prompt = strfmt::strfmt(prompt_template, &vars).unwrap();
    log::debug!("Prompt is: {prompt}");

    // Tokenize
    let token_list = &MODEL
        .str_to_token(&prompt, model::AddBos::Always)
        .expect("Failed to tokenize prompt...");
    token_list.clone()
}

#[actix_web::post("/v1/chat/completions")]
async fn stream_response(
    state: web::Data<AppState>,
    db: web::Data<knowledge::KnowledgeBase>,
    cache_guard: web::Data<Arc<RwLock<cache::TokenCache>>>,
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
    let overall_start_time = Instant::now();
    let backend = ai::init_backend();

    let mut cache = cache_guard.write().await;

    let fp = cache::fingerprint(&req_body.messages);
    let tokens = match &fp {
        Some(fp) => {
            // Case 0 - we have a fingerprint...
            match cache.get(fp) {
                Some(data) => {
                    // ... and we already have tokens.
                    let mut tokens = (*data).clone();
                    // trick: we add the last system message as well; we could cache it but the
                    // overhead for doing so is high; this is simple. Should not bail as we can
                    // fingerprint we know we have enough messages...

                    let msg_len = req_body.messages.len();
                    let start = msg_len.saturating_sub(2);
                    let last_two = &req_body.messages[start..];

                    let additional_tokens = tokenize(last_two, &state.prompt, &db, backend);
                    tokens.extend(additional_tokens.clone());
                    cache.extend(&fp.clone(), additional_tokens);
                    Arc::new(tokens)
                }
                None => {
                    // ... we do not have tokens.
                    let tokens =
                        Arc::new(tokenize(&req_body.messages, &state.prompt, &db, backend));
                    cache.insert(fp.clone(), Arc::clone(&tokens));
                    tokens
                }
            }
        }
        None => Arc::new(tokenize(&req_body.messages, &state.prompt, &db, backend)),
    };

    // Instant would be more correct, but we can't get the timestamp in seconds
    let timestamp = UNIX_EPOCH
        .elapsed()
        .expect("System time is before epoch")
        .as_secs();

    // Initialize the AI query context
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tokens_clone = Arc::clone(&tokens);
    actix_web::rt::task::spawn_blocking(move || {
        let mut ai_context = ai::AiQueryContext::new(
            &tokens_clone,
            state.threads,
            state.max_token,
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
    let token_stream = UnboundedReceiverStream::new(rx).map(move |token| {
        let json_tmp = format!(
            r#"data: {{"id":"foo","object":"chat.completion.chunk","created":{timestamp},"model":"{}", "system_fingerprint": "fp0", "choices":[{{"index":0,"delta":{{"content": {}}},"logprobs":null,"finish_reason":null}}]}}"#,
            "rusty_llm", serde_json::to_string(&token).unwrap()
        );
        Ok::<_, actix_web::Error>(web::Bytes::from(json_tmp + "\n\n"))
    });

    // Measure the overall request-response time
    let overall_duration = overall_start_time.elapsed();
    REQUEST_RESPONSE_TIME
        .with_label_values(&[&INSTANCE_LABEL])
        .observe(overall_duration.as_secs_f64());

    actix_web::HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(token_stream)
}

#[cfg(test)]
mod tests {
    use actix_web::body::MessageBody;
    use std::{thread, time};

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
        let cache = Arc::new(RwLock::new(cache::TokenCache::new(
            time::Duration::from_secs(60),
        )));
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(cache))
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
        let cache = Arc::new(RwLock::new(cache::TokenCache::new(
            time::Duration::from_secs(60),
        )));
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(cache))
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

    #[actix_web::test]
    async fn test_caching_for_success() {
        let kv_store = knowledge::get_db();
        let prompt =
            "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>"
                .to_string();
        let cache = Arc::new(RwLock::new(cache::TokenCache::new(
            time::Duration::from_secs(5),
        )));
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(cache.clone()))
                .app_data(web::Data::new(AppState {
                    max_token: 10,
                    threads: 4,
                    prompt,
                }))
                .app_data(web::Data::new(kv_store))
                .service(stream_response),
        )
        .await;
        // request.
        let non_fp_request = Request {
            stream: true,
            model: "rusty_llm".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Who was Albert Einstein?".to_string(),
            }],
        };
        let fp_request = Request {
            stream: true,
            model: "rusty_llm".to_string(),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: "Who was Albert Einstein?".to_string(),
                },
                Message {
                    role: "system".to_string(),
                    content: "A physicist".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "Who was Madame Curie?".to_string(),
                },
            ],
        };
        let fp = cache::fingerprint(&fp_request.messages).unwrap();
        // This should do nothing to the cache...
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/chat/completions")
            .insert_header(actix_web::http::header::ContentType::json())
            .set_json(non_fp_request)
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        // First call will add a cache entry...
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/chat/completions")
            .insert_header(actix_web::http::header::ContentType::json())
            .set_json(&fp_request)
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        {
            let tmp = cache.read().await;
            assert!(tmp.has_key(&fp));
        }
        // Second call will enable caching...
        let req = actix_web::test::TestRequest::post()
            .uri("/v1/chat/completions")
            .insert_header(actix_web::http::header::ContentType::json())
            .set_json(fp_request)
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        {
            let tmp = cache.read().await;
            assert!(tmp.has_key(&fp));
        }
        thread::sleep(time::Duration::from_secs(5));
        {
            let mut tmp = cache.write().await;
            assert!(tmp.get(&fp).is_none());
        }
    }
}
