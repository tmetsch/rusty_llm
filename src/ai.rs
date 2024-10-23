use crate::TOKEN_RESPONSE_TIME;
use llama_cpp_2::llama_backend;
use llama_cpp_2::model;
use llama_cpp_2::token;
use std::collections;
use std::num;
use std::time;

/// Initialize the LLama backend.
pub(crate) fn init_backend() -> llama_backend::LlamaBackend {
    llama_backend::LlamaBackend::init_numa(llama_backend::NumaStrategy::ISOLATE)
        .expect("Failed to initialize LlamaBackend.")
}

/// Load a LLM model & Tokenizer from a file.
pub(crate) fn load_model(
    gguf_path: &str,
    backend: &llama_backend::LlamaBackend,
) -> model::LlamaModel {
    let gpu_layers = std::env::var("MODEL_GPU_LAYERS")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .unwrap_or(0);
    let main_gpu = std::env::var("MAIN_GPU")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .unwrap_or(0);

    let model_options = model::params::LlamaModelParams::default()
        .with_main_gpu(main_gpu)
        .with_n_gpu_layers(gpu_layers);

    model::LlamaModel::load_from_file(backend, gguf_path, &model_options)
        .expect("Failed to load model!")
}

/// Creates a prompt based on the given query and retrieved context and tries to predict some text.
pub(crate) async fn query_ai(
    query: &str,
    references: Vec<String>,
    threads: i32,
    max_token: i32,
    prompt_template: &str,
    model: &model::LlamaModel,
    backend: &llama_backend::LlamaBackend,
) -> (usize, String) {
    // Format the prompt
    let mut vars = collections::HashMap::new();
    vars.insert("context".to_string(), references.join(","));
    vars.insert("query".to_string(), query.to_string());

    let prompt = strfmt::strfmt(prompt_template, &vars).unwrap();

    // Initialize a session
    let inference_context = llama_cpp_2::context::params::LlamaContextParams::default()
        .with_n_ctx(Option::from(num::NonZeroU32::new(2048).unwrap()))
        .with_n_threads(threads);

    let mut ctx = model
        .new_context(backend, inference_context)
        .expect("Could not create context!");

    let token_list = model
        .str_to_token(&prompt, model::AddBos::Always)
        .expect("Failed to tokenize prompt...");

    let last_index = (token_list.len() - 1) as i32;
    let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);

    for (i, token) in token_list.into_iter().enumerate() {
        batch
            .add(token, i as i32, &[0], i as i32 == last_index)
            .expect("Failed to add token...");
    }

    ctx.decode(&mut batch).expect("Failed to decode batch!");

    let mut n_cur = batch.n_tokens();
    let mut tokens = Vec::with_capacity(max_token as usize);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut s_0 = time::Instant::now();

    // Main loop
    while n_cur <= max_token {
        let candidates = ctx.candidates();
        let candidates_p = token::data_array::LlamaTokenDataArray::from_iter(candidates, false);
        let new_token_id = ctx.sample_token_greedy(candidates_p);

        if model.is_eog_token(new_token_id) {
            break;
        }

        let output_bytes = model
            .token_to_bytes(new_token_id, model::Special::Tokenize)
            .expect("Failed to convert token to bytes");

        let mut output_string = String::with_capacity(32);
        let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
        tokens.push(output_string);

        let s_1 = time::Instant::now();
        let duration = s_1.duration_since(s_0);
        TOKEN_RESPONSE_TIME
            .with_label_values(&[])
            .observe(duration.as_secs_f64());
        log::info!("Token took: {}s.", duration.as_secs_f64());
        s_0 = s_1;

        batch.clear();
        batch
            .add(new_token_id, n_cur, &[0], true)
            .expect("Failed to add token...");

        ctx.decode(&mut batch).expect("Failed to decode batch!");
        n_cur += 1;
    }

    (tokens.len(), tokens.concat())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model_for_success() {
        let backend = init_backend();
        load_model("model/model.gguf", &backend);
        drop(backend);
    }

    #[test]
    #[should_panic(expected = "\"foo/bar.gguf\" does not exist")]
    fn test_load_model_for_failure() {
        let backend = init_backend();
        load_model("foo/bar.gguf", &backend);
        drop(backend);
    }

    #[test]
    fn test_load_model_for_sanity() {
        let backend = init_backend();
        std::env::set_var("MODEL_GPU_LAYERS", "1");
        load_model("model/model.gguf", &backend);
        std::env::remove_var("MODEL_GPU_LAYERS");
        drop(backend);
    }

    #[actix_web::test]
    async fn test_query_ai_for_sanity() {
        let backend = init_backend();
        let model = load_model("model/model.gguf", &backend);
        let prompt =
            "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>";
        let res = query_ai(
            "Who was Albert Einstein",
            vec![],
            4,
            30,
            prompt,
            &model,
            &backend,
        )
        .await;
        assert_eq!(res.0, 8); // not quite sure why this is 8 :-(

        drop(backend);
    }
}
