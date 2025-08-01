extern crate rusty_llm;

#[cfg(test)]
mod tests {
    use actix_web::body::MessageBody;
    use actix_web::web;

    macro_rules! test_query_request {
        ($name:ident, $($prompt:expr, $expected_status:expr, $expected_len:expr),+) => {
            #[actix_web::test]
            async fn $name() {
                $(
                    let mut kv_store = rusty_llm::knowledge::get_db();

        let prompt = "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>".to_string();
                    rusty_llm::api::load_knowledge(std::path::Path::new("data/"), &mut kv_store).await;
                    let app = actix_web::test::init_service(
                        actix_web::App::new()
                            .app_data(web::Data::new(rusty_llm::api::AppState {
                                max_token: 1024,
                                threads: 4,
                                prompt,
                            }))
                            .app_data(web::Data::new(kv_store.clone()))
                            .service(rusty_llm::api::stream_response)
                    )
                    .await;
                    let req = actix_web::test::TestRequest::post()
                        .uri("/v1/chat/completions")
                        .insert_header(actix_web::http::header::ContentType::json())
                        .set_payload(
                            format!(r#"{{"stream": true, "model": "rusty_llm", "messages": [{{"role": "user", "content": "{}"}}]}}"#, $prompt)
                                .try_into_bytes()
                                .unwrap(),
                        )
                        .to_request();
                    let resp = actix_web::test::call_service(&app, req).await;
                    assert_eq!(resp.status().as_u16(), $expected_status);
                    // TODO: capture all chunks and do this:
                    // assert!(body_bytes.len() >= $expected_len);
                )*
            }
        }
    }

    // run a simple test - will not trigger adding knowledge...
    test_query_request!(simple_query_test, "Who was Albert Einstein?", 200, 40);
    // this will trigger adding context.
    test_query_request!(simple_rag_query_test, "Who was Thom Rhubarb?", 200, 40);
    // make sure this doesn't break anything.
    test_query_request!(empty_request, "", 400, 40);

    #[actix_web::test]
    async fn test_if_observability_works_example() {
        let kv_store = rusty_llm::knowledge::get_db();

        let prompt =
            "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>"
                .to_string();
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(rusty_llm::api::AppState {
                    max_token: 32,
                    threads: 4,
                    prompt,
                }))
                .app_data(web::Data::new(kv_store.clone()))
                .service(rusty_llm::api::stream_response)
                .service(rusty_llm::api::metrics),
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
        let _resp = actix_web::test::call_service(&app, req).await;

        // check what metrics report...
        let req = actix_web::test::TestRequest::default()
            .uri("/metrics")
            .insert_header(actix_web::http::header::ContentType::plaintext())
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert!(!resp.into_body().try_into_bytes().unwrap().is_empty());
    }
}
