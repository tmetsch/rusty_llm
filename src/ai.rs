use crate::TOKEN_RESPONSE_TIME;
use std::time;

/// Load a LLM model & Tokenizer from a file.
pub(crate) fn load_model(gguf_path: &str) -> llama_cpp::LlamaModel {
    let mut gpu_layers: u32 = 0;
    if let Ok(v) = std::env::var("MODEL_GPU_LAYERS") {
        gpu_layers = v.parse().expect("Could not parse number og GPU layers");
    }

    let model_options = llama_cpp::LlamaParams {
        n_gpu_layers: gpu_layers,
        main_gpu: 1,
        ..Default::default()
    };
    llama_cpp::LlamaModel::load_from_file(gguf_path, model_options).expect("Failed to load model!")
}

/// Creates a prompt based on the given query and retrieved context and tries to predict some text.
pub(crate) async fn query_ai(
    query: &str,
    references: Vec<String>,
    threads: u32,
    batch_size: u32,
    max_token: usize,
    model: &llama_cpp::LlamaModel,
) -> (usize, String) {
    let mut context = Vec::new();
    for reference in references.clone() {
        context.push(serde_json::json!({"content": reference}))
    }
    let context = serde_json::json!(context).to_string();

    // Mistral AI optimized prompt...
    let prompt = format!(
        "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>",
        context = context,
        query = query
    );

    // initialize a session...
    let inference_session_config = llama_cpp::SessionParams {
        n_threads: threads,
        n_batch: batch_size,
        ..Default::default()
    };

    // ...and push the prompt.
    let mut ctx = model
        .create_session(inference_session_config)
        .expect("Failed to create session");
    let mut s_0 = time::Instant::now();
    ctx.advance_context(prompt)
        .expect("Failed to insert the prompt");

    let mut completions = ctx
        .start_completing_with(
            llama_cpp::standard_sampler::StandardSampler::default(),
            max_token,
        )
        .expect("Failed to start the completion.");

    let mut tokens = Vec::new();
    while let Some(token) = completions.next_token_async().await {
        tokens.push(token);
        let s_1 = time::Instant::now();
        let duration = s_1.duration_since(s_0);
        TOKEN_RESPONSE_TIME
            .with_label_values(&[])
            .observe(duration.as_secs_f64());
        log::info!("Token took: {}s.", duration.as_secs_f64());
        s_0 = s_1;
    }

    // return the result.
    (tokens.len(), model.decode_tokens(tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model_for_success() {
        load_model("model/model.gguf");
    }

    #[test]
    #[should_panic(expected = "Failed to load model!")]
    fn test_load_model_for_failure() {
        load_model("foo/bar.gguf");
    }

    #[test]
    fn test_load_model_for_sanity() {
        std::env::set_var("MODEL_GPU_LAYERS", "1");
        load_model("model/model.gguf");
        std::env::remove_var("MODEL_GPU_LAYERS");
    }

    #[actix_web::test]
    async fn test_query_ai_for_sanity() {
        let model = load_model("model/model.gguf");
        let res = query_ai("Who was Albert Einstein", vec![], 4, 8, 10, &model).await;
        assert_eq!(res.0, 11); // not quite sure why this is 11 :-(
    }
}
