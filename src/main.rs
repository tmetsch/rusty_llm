use std::path;

use actix_web::web;

use rusty_llm::api;

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
    let prom_addr: String = match std::env::var("PROMETHEUS_HTTP_ADDRESS") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse the prometheus bind address environment variable!"),
        Err(_) => "127.0.0.1:8081".parse().unwrap(),
    };
    let addr: String = match std::env::var("HTTP_ADDRESS") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse bind address environment variable!"),
        Err(_) => "127.0.0.1:8080".parse().unwrap(),
    };
    let workers: usize = match std::env::var("HTTP_WORKERS") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse number of workers variable!"),
        Err(_) => 1,
    };
    let threads: u32 = match std::env::var("MODEL_THREADS") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse model thread environment variable!"),
        Err(_) => 6,
    };
    let batch_size: u32 = match std::env::var("MODEL_BATCH_SIZE") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse model batch size environment variable!"),
        Err(_) => 8,
    };
    let max_token: usize = match std::env::var("MODEL_MAX_TOKEN") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse model max token length environment variable!"),
        Err(_) => 128,
    };
    let prompt: String = match std::env::var("MODEL_PROMPT_TEMPLATE") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse prompt template environment variable!"),
        Err(_) => {
            "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>"
                .parse()
                .unwrap()
        }
    };

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
        batch_size,
        threads,
        max_token,
        prompt,
    };
    let server = tokio::spawn(async move {
        actix_web::HttpServer::new(move || {
            actix_web::App::new()
                .app_data(web::Data::new(state.clone()))
                .app_data(web::Data::new(kv_store.clone()))
                .service(api::query)
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
