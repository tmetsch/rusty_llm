use std::path;

use actix_web::web;
use rusty_llm::api;

/// Helper function to get an environment variable or return a default value.
fn get_env_or_default<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

/// The main code.
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();

    // set up the in-memory KV-store & load some knowledge.
    let mut kv_store = rusty_llm::knowledge::get_db().await;
    api::load_knowledge(
        path::Path::new(&std::env::var("DATA_PATH").unwrap_or("data".to_string())),
        &mut kv_store,
    )
    .await;

    // parse the environment variables.
    let prom_addr = get_env_or_default("PROMETHEUS_HTTP_ADDRESS", "127.0.0.1:8081".to_string());
    let addr = get_env_or_default("HTTP_ADDRESS", "127.0.0.1:8080".to_string());
    let workers = get_env_or_default("HTTP_WORKERS", 1);
    let threads = get_env_or_default("MODEL_THREADS", 6);
    let max_token = get_env_or_default("MODEL_MAX_TOKEN", 128);
    let prompt = get_env_or_default(
        "MODEL_PROMPT_TEMPLATE",
        "<s>[INST]Using this information: {context}. Answer the Question: {query}[/INST]</s>"
            .to_string(),
    );

    // check prompt template.
    if !prompt.contains("{context}") || !prompt.contains("{query}") {
        panic!("The prompt template should contain context and query fields.")
    }

    // run the web servers.
    let prom = tokio::spawn(async {
        actix_web::HttpServer::new(|| actix_web::App::new().service(api::metrics))
            .workers(1)
            .bind(prom_addr)
            .expect("Could not bing to given address!")
            .run()
            .await
    });

    let state = api::AppState {
        threads,
        max_token,
        prompt,
    };
    let server = tokio::spawn(async move {
        actix_web::HttpServer::new(move || {
            actix_web::App::new()
                .app_data(web::Data::new(state.clone()))
                .app_data(web::Data::new(kv_store.clone()))
                .service(api::stream_response)
                .service(api::list_models)
        })
        .workers(workers)
        .bind(addr)
        .expect("Could not bind to given address!")
        .run()
        .await
    });

    let _ = prom.await;
    let _ = server.await;

    Ok(())
}
