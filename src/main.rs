use std::path;

use actix_web::web;

use rusty_llm::api;

/// The main code.
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();

    // set up the in-memory KV-store & load some knowledge.
    let kv_store = rusty_llm::knowledge::get_db().await;
    api::load_knowledge(
        path::Path::new(&std::env::var("DATA_PATH").unwrap_or("data".to_string())),
        &kv_store,
    )
        .await;

    // parse the environment variables.
    let addr: String = match std::env::var("HTTP_ADDRESS") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse bind address environment variable!"),
        Err(_) => "127.0.0.1:8080".parse().unwrap(),
    };
    let workers: usize = match std::env::var("HTTP_WORKERS") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse port environment variable!"),
        Err(_) => 2,
    };
    let threads: usize = match std::env::var("MODEL_THREADS") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse model thread environment variable!"),
        Err(_) => 6,
    };
    let max_token: usize = match std::env::var("MODEL_MAX_TOKEN") {
        Ok(val) => val
            .parse()
            .expect("Unable to parse model context length environment variable!"),
        Err(_) => 128,
    };

    // run the web server.
    let state = api::AppState { threads, max_token };
    actix_web::HttpServer::new(move || {
        actix_web::App::new()
            .app_data(web::Data::new(state.clone()))
            .app_data(web::Data::new(kv_store.clone()))
            .service(api::query)
            .service(api::metrics)
    })
        .workers(workers)
        .bind(addr)
        .expect("Could not bind to vien address!")
        .run()
        .await
}
