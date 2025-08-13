use crate::{INSTANCE_LABEL, TOKEN_RESPONSE_TIME};
use llama_cpp_2::{llama_backend, model, sampling, token};
use std::num;
use std::sync::LazyLock;
use std::time;

const CONTEXT_LEN: usize = 8196; // TODO: pick up from env variable.

/// Initialize the LLama backend.
pub(crate) fn init_backend() -> &'static llama_backend::LlamaBackend {
    static LLAMA_BACKEND: LazyLock<llama_backend::LlamaBackend> = LazyLock::new(|| {
        llama_backend::LlamaBackend::init_numa(llama_backend::NumaStrategy::ISOLATE)
            .expect("Failed to initialize LlamaBackend")
    });
    &LLAMA_BACKEND
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
    batch: llama_cpp_2::llama_batch::LlamaBatch,
    ctx: llama_cpp_2::context::LlamaContext<'a>,
    decoder: encoding_rs::Decoder,
    generated_len: usize,
    max_token: i32,
    model: &'a model::LlamaModel,
    n_cur: i32,
    s_0: time::Instant,
    sampler: sampling::LlamaSampler,
}

impl<'a> AiQueryContext<'a> {
    pub fn new(
        tokens: &[token::LlamaToken],
        threads: i32,
        max_token: i32,
        model: &'a model::LlamaModel,
        backend: &'a llama_backend::LlamaBackend,
    ) -> Self {
        let prompt_len = tokens.len();

        if prompt_len > CONTEXT_LEN {
            panic!("Prompt length ({prompt_len}) exceeds model context length ({CONTEXT_LEN})");
        }

        let adjusted_max_token = if prompt_len + max_token as usize > CONTEXT_LEN {
            CONTEXT_LEN - prompt_len
        } else {
            max_token as usize
        };

        let n_batch_needed = (prompt_len + adjusted_max_token).min(CONTEXT_LEN);

        let inference_context = llama_cpp_2::context::params::LlamaContextParams::default()
            .with_n_ctx(Some(num::NonZeroU32::new(CONTEXT_LEN as u32).unwrap()))
            .with_n_batch(n_batch_needed.try_into().unwrap())
            .with_n_threads(threads);

        let mut ctx = model
            .new_context(backend, inference_context)
            .expect("Could not create context!");

        let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(n_batch_needed, 1);

        let last_index = (tokens.len() - 1) as i32;

        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i as i32 == last_index)
                .expect("Failed to add token...");
        }

        ctx.decode(&mut batch).expect("Failed to decode batch!");

        let sampler = sampling::LlamaSampler::chain_simple([
            sampling::LlamaSampler::dist(42),
            sampling::LlamaSampler::greedy(),
        ]);

        AiQueryContext {
            batch,
            ctx,
            decoder: encoding_rs::UTF_8.new_decoder(),
            generated_len: 0,
            max_token: adjusted_max_token as i32,
            model,
            n_cur: prompt_len as i32,
            s_0: time::Instant::now(),
            sampler,
        }
    }

    // rest of the impl...

    pub fn next_token(&mut self) -> Option<String> {
        if self.generated_len >= self.max_token as usize {
            return None;
        }

        let token = self.sampler.sample(&self.ctx, self.batch.n_tokens() - 1);
        self.sampler.accept(token);

        if self.model.is_eog_token(token) {
            return None;
        }

        let output_bytes = self
            .model
            .token_to_bytes(token, model::Special::Tokenize)
            .expect("Failed to convert token to bytes");
        let mut output_string = String::with_capacity(32);
        let _ = self
            .decoder
            .decode_to_string(&output_bytes, &mut output_string, false);

        // Timing the token generation
        let token_duration = self.s_0.elapsed();
        TOKEN_RESPONSE_TIME
            .with_label_values(&[&INSTANCE_LABEL])
            .observe(token_duration.as_secs_f64());
        log::info!("Token took: {token_duration:?}.");
        self.s_0 = time::Instant::now();

        // Prepare for the next token
        self.batch.clear();
        self.batch
            .add(token, self.n_cur, &[0], true)
            .expect("Failed to add token...");

        self.n_cur += 1;
        self.generated_len += 1;

        self.ctx
            .decode(&mut self.batch)
            .expect("Failed to decode batch!");

        Some(output_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn process_tokens(mut query_context: AiQueryContext<'_>) -> usize {
        let mut count = 0;
        while let Some(_token) = query_context.next_token() {
            count += 1;
        }
        count
    }

    #[test]
    fn test_load_model_for_success() {
        let backend = init_backend();
        load_model("model/model.gguf", backend);
    }

    #[test]
    #[should_panic(expected = "\"foo/bar.gguf\" does not exist")]
    fn test_load_model_for_failure() {
        let backend = init_backend();
        load_model("foo/bar.gguf", backend);
    }

    #[test]
    fn test_load_model_for_sanity() {
        let backend = init_backend();
        std::env::set_var("MODEL_GPU_LAYERS", "1");
        load_model("model/model.gguf", backend);
        std::env::remove_var("MODEL_GPU_LAYERS");
    }

    #[actix_web::test]
    async fn test_ai_query_context_for_sanity() {
        let backend = init_backend();
        let model = load_model("model/model.gguf", backend);
        let prompt =
            "<s>[INST]Using this information: [] answer the Question: Who was Albert Einstein?[/INST]</s>";

        let tokens = model
            .str_to_token(prompt, model::AddBos::Always)
            .expect("Failed to tokenize prompt...");

        let query_context = AiQueryContext::new(&tokens, 4, 30, &model, backend);
        let i = process_tokens(query_context).await;
        assert_eq!(i, 30);
    }
}
