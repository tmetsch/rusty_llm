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

pub struct AiQueryContext<'a> {
    ctx: llama_cpp_2::context::LlamaContext<'a>,
    decoder: encoding_rs::Decoder,
    n_cur: i32,
    max_token: i32,
    model: &'a model::LlamaModel,
    s_0: time::Instant,
}

impl<'a> AiQueryContext<'a> {
    /// Creates a new QueryAiContext for token generation.
    pub fn new(
        query: &str,
        references: Vec<String>,
        threads: i32,
        max_token: i32,
        prompt_template: &str,
        model: &'a model::LlamaModel,
        backend: &'a llama_backend::LlamaBackend,
    ) -> Self {
        let mut vars = collections::HashMap::new();
        vars.insert("context".to_string(), references.join(","));
        vars.insert("query".to_string(), query.to_string());

        let prompt = strfmt::strfmt(prompt_template, &vars).unwrap();

        let inference_context = llama_cpp_2::context::params::LlamaContextParams::default()
            .with_n_ctx(Some(num::NonZeroU32::new(2048).unwrap()))
            .with_n_threads(threads);

        let mut ctx = model
            .new_context(backend, inference_context)
            .expect("Could not create context!");

        let token_list = model
            .str_to_token(&prompt, model::AddBos::Always)
            .expect("Failed to tokenize prompt...");

        if token_list.len() >= max_token as usize {
            panic!("Maximum token length is smaller than the prompt...");
        }

        let last_index = (token_list.len() - 1) as i32;
        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);

        for (i, token) in token_list.into_iter().enumerate() {
            batch
                .add(token, i as i32, &[0], i as i32 == last_index)
                .expect("Failed to add token...");
        }

        ctx.decode(&mut batch).expect("Failed to decode batch!");

        AiQueryContext {
            ctx,
            decoder: encoding_rs::UTF_8.new_decoder(),
            n_cur: batch.n_tokens(),
            max_token,
            model,
            s_0: time::Instant::now(),
        }
    }

    pub async fn next_token(&mut self) -> Option<String> {
        if self.n_cur >= self.max_token {
            return None;
        }

        let candidates = self.ctx.candidates();
        let candidates_p = token::data_array::LlamaTokenDataArray::from_iter(candidates, false);
        let new_token_id = self.ctx.sample_token_greedy(candidates_p);

        if self.model.is_eog_token(new_token_id) {
            return None;
        }

        let output_bytes = self
            .model
            .token_to_bytes(new_token_id, model::Special::Tokenize)
            .expect("Failed to convert token to bytes");

        let mut output_string = String::with_capacity(32);
        let _ = self
            .decoder
            .decode_to_string(&output_bytes, &mut output_string, false);
        self.n_cur += 1;

        // Timing the token generation
        let token_duration = self.s_0.elapsed();
        TOKEN_RESPONSE_TIME
            .with_label_values(&[])
            .observe(token_duration.as_secs_f64());
        println!("Token took: {:?}.", token_duration);
        self.s_0 = time::Instant::now();

        // Prepare for the next token
        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);
        batch
            .add(new_token_id, self.n_cur, &[0], true)
            .expect("Failed to add token...");
        self.ctx
            .decode(&mut batch)
            .expect("Failed to decode batch!");

        Some(output_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn process_tokens(mut query_context: AiQueryContext<'_>) -> usize {
        let mut count = 0;
        while let Some(_token) = query_context.next_token().await {
            count += 1;
        }
        count
    }

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
    async fn test_ai_query_context_for_sanity() {
        let backend = init_backend();
        let model = load_model("model/model.gguf", &backend);
        let prompt =
            "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>";

        let query_context = AiQueryContext::new(
            "Who was Albert Einstein",
            vec![],
            4,
            30,
            prompt,
            &model,
            &backend,
        );
        let i = process_tokens(query_context).await;
        assert_eq!(i, 7); // max tokens is 30; subtract prompt and we get 7.

        drop(backend);
    }
}
