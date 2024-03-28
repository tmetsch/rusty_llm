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
                    let kv_store = rusty_llm::knowledge::get_db().await;
                    rusty_llm::api::load_knowledge(std::path::Path::new("data/"), &kv_store).await;
                    let app = actix_web::test::init_service(
                        actix_web::App::new()
                            .app_data(web::Data::new(rusty_llm::api::AppState {
                                batch_size: 8,
                                max_token: 10,
                                threads: 4,
                            }))
                            .app_data(web::Data::new(kv_store.clone()))
                            .service(rusty_llm::api::query)
                    )
                    .await;
                    let req = actix_web::test::TestRequest::post()
                        .uri("/query")
                        .insert_header(actix_web::http::header::ContentType::json())
                        .set_payload(
                            format!("{{\"query\": \"{}\" }}", $prompt)
                                .try_into_bytes()
                                .unwrap(),
                        )
                        .to_request();
                    let resp = actix_web::test::call_service(&app, req).await;
                    assert_eq!(resp.status().as_u16(), $expected_status);
                    assert!(resp.into_body().try_into_bytes().unwrap().len() >= $expected_len);
                )*
            }
        }
    }

    // tune a simple test - will not trigger adding knowledge.
    test_query_request!(simple_query_test, "Who was Albert Einstein?", 200, 120);
    // this will trigger adding context, hence prompt should be longer.
    test_query_request!(simple_rag_query_test, "Who was Thom Rhubarb?", 200, 700);
    // make sure this doesn't break anything.
    test_query_request!(empty_request, "", 200, 100);

    #[actix_web::test]
    async fn test_if_observability_works_example() {
        let kv_store = rusty_llm::knowledge::get_db().await;
        let app = actix_web::test::init_service(
            actix_web::App::new()
                .app_data(web::Data::new(rusty_llm::api::AppState {
                    batch_size: 8,
                    max_token: 10,
                    threads: 4,
                }))
                .app_data(web::Data::new(kv_store.clone()))
                .service(rusty_llm::api::query)
                .service(rusty_llm::api::metrics),
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
        let _resp = actix_web::test::call_service(&app, req).await;

        // check what metrics report...
        let req = actix_web::test::TestRequest::default()
            .uri("/metrics")
            .insert_header(actix_web::http::header::ContentType::plaintext())
            .to_request();
        let resp = actix_web::test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert!(resp.into_body().try_into_bytes().unwrap().len() > 0);
    }
}
