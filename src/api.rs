use std::path;
use std::time;

use actix_web::web;
use lazy_static::lazy_static;
use prometheus::Encoder;

use crate::{ai, embedding, knowledge, EMBEDDING_TIME, REQUEST_RESPONSE_TIME};

lazy_static! {
    static ref MODEL: llama_cpp::LlamaModel =
        ai::load_model(&std::env::var("MODEL_PATH").unwrap_or("model/model.gguf".to_string()),);
    static ref EMBEDDING_MODEL: llama_cpp::LlamaModel = embedding::get_embedding_model(
        &std::env::var("EMBEDDING_MODEL").unwrap_or("model/embed.gguf".to_string())
    );
}

/// Loads knowledge from a set of text files into an in-memory KV-store.
pub async fn load_knowledge(path: &path::Path, db: &mut knowledge::KnowledgeBase) {
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
                    let tkn_context = embedding::embed(&data, &EMBEDDING_MODEL);
                    knowledge::add_context(&data, tkn_context, db).await;
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
    pub batch_size: u32,
    pub threads: u32,
    pub max_token: usize,
}

/// Represents a JSON style request.
#[derive(serde::Deserialize)]
struct QueryRequest {
    query: String,
}

/// Represents a JSON style response.
#[derive(serde::Serialize)]
struct QueryResponse {
    response: String,
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

/// Handles an HTTP post & triggers a prompt on the LLM.
#[actix_web::post("/query")]
async fn query(
    state: web::Data<AppState>,
    db: web::Data<knowledge::KnowledgeBase>,
    req_body: web::Json<QueryRequest>,
) -> impl actix_web::Responder {
    if req_body.query.is_empty() {
        return actix_web::HttpResponse::Ok().json(QueryResponse {
            response: "Empty query - not triggering LLM model.".to_string(),
        });
    }

    let s_0 = time::Instant::now();
    let tkn_query = embedding::embed(&req_body.query, &EMBEDDING_MODEL);
    let context = knowledge::get_context(tkn_query, db.get_ref()).await;
    let s_1 = time::Instant::now();
    let duration = s_1.duration_since(s_0);
    EMBEDDING_TIME
        .with_label_values(&[])
        .observe(duration.as_secs_f64());

    let res = ai::query_ai(
        &req_body.query,
        context,
        state.threads,
        state.batch_size,
        state.max_token,
        &MODEL,
    )
    .await;
    let s_2 = time::Instant::now();
    let duration = s_2.duration_since(s_0);
    REQUEST_RESPONSE_TIME
        .with_label_values(&[])
        .observe(duration.as_secs_f64());

    // TODO: make this a streaming response.
    actix_web::HttpResponse::Ok().json(QueryResponse { response: res.1 })
}

#[cfg(test)]
mod tests {
    use actix_web::body::MessageBody;

    use super::*;

    #[actix_web::test]
    async fn test_load_knowledge_for_success() {
        let mut db = crate::knowledge::get_db().await;
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
    async fn test_query_for_success() {
        let kv_store = crate::knowledge::get_db().await;
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(AppState {
                    batch_size: 8,
                    max_token: 5,
                    threads: 4,
                }))
                .app_data(web::Data::new(kv_store))
                .service(query),
        )
        .await;
        let req = actix_web::test::TestRequest::post()
            .uri("/query")
            .insert_header(actix_web::http::header::ContentType::json())
            .set_payload(
                "{\"query\": \"Who was Albert Einstein?\"}"
                    .try_into_bytes()
                    .unwrap(),
            )
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_query_for_failure() {
        let kv_store = crate::knowledge::get_db().await;
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(AppState {
                    max_token: 5,
                    batch_size: 8,
                    threads: 4,
                }))
                .app_data(web::Data::new(kv_store))
                .service(query),
        )
        .await;
        let req = actix_web::test::TestRequest::post()
            .uri("/query")
            .insert_header(actix_web::http::header::ContentType::json())
            .set_payload("Garbage".try_into_bytes().unwrap())
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), 400);
    }
}
