use std::path;
use std::time;

use crate::knowledge::ContextEntry;
use crate::TOKEN_RESPONSE_TIME;

/// Default context size will be set to 2048 for now.
const CONTEXT_SIZE: usize = 2048;

/// Load a LLM model & Tokenizer from a file.
pub(crate) fn load_model(gguf_path: &path::Path, tok_path: &path::Path) -> Box<dyn llm::Model> {
    let tokenizer = llm::TokenizerSource::HuggingFaceTokenizerFile(path::PathBuf::from(tok_path));

    let mut use_gpu = false;
    let mut gpu_layers: usize = 0;
    if let Ok(v) = std::env::var("MODEL_GPU_LAYERS") {
        use_gpu = true;
        gpu_layers = v.parse().expect("Could not parse number og GPU layers");
    }
    let model_params = llm::ModelParameters {
        prefer_mmap: true,
        context_size: CONTEXT_SIZE,
        lora_adapters: None,
        use_gpu,
        gpu_layers: Some(gpu_layers),
        rope_overrides: None,
    };
    (llm::load(
        gguf_path,
        tokenizer,
        model_params,
        llm::load_progress_callback_stdout,
    )
        .expect("Failed to load model!")) as _
}

/// Creates a prompt based on the given query and retrieved context and tries to predict some text.
pub(crate) async fn query_ai(
    query: &str,
    references: Vec<ContextEntry>,
    threads: usize,
    context_len: usize,
    model: &dyn llm::Model,
) -> Vec<String> {
    let mut context = Vec::new();
    for reference in references.clone() {
        context.push(serde_json::json!({"content": reference.content}))
    }
    let context = serde_json::json!(context).to_string();

    // Mistral AI optimized prompt...
    let prompt = format!(
        "<s>[INST]Using this information: {context} answer the Question: {query}[/INST]</s>",
        context = context,
        query = query
    );

    // initialize a session to talk with the model...
    let inference_session_config = llm::InferenceSessionConfig {
        memory_k_type: llm::ModelKVMemoryType::Float16,
        memory_v_type: llm::ModelKVMemoryType::Float16,
        n_batch: 8,
        n_threads: threads,
    };
    let mut tokens: Vec<String> = vec![];
    let mut session = model.start_session(inference_session_config);
    let mut s_0 = time::Instant::now();
    let _res = session.infer::<std::convert::Infallible>(
        model,
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: llm::Prompt::from(&prompt),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: Some(context_len),
        },
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                tokens.push(t);
                let s_1 = time::Instant::now();
                let duration = s_1.duration_since(s_0);
                TOKEN_RESPONSE_TIME
                    .with_label_values(&[])
                    .observe(duration.as_secs_f64());
                log::info!("Token took: {}", duration.as_secs_f64());
                s_0 = s_1;
                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model_for_success() {
        load_model(
            path::Path::new("model/model.gguf"),
            path::Path::new("model/tokenizer.json"),
        );
    }

    #[test]
    #[should_panic(expected = "Failed to load model!")]
    fn test_load_model_for_failure() {
        load_model(
            path::Path::new("foo/bar.gguf"),
            path::Path::new("model/tokenizer.json"),
        );
    }

    #[test]
    fn test_load_model_for_sanity() {
        std::env::set_var("MODEL_GPU_LAYERS", "1");
        load_model(
            path::Path::new("model/model.gguf"),
            path::Path::new("model/tokenizer.json"),
        );
        std::env::remove_var("MODEL_GPU_LAYERS");
    }

    #[actix_web::test]
    async fn test_query_ai_for_sanity() {
        let model = load_model(
            path::Path::new("model/model.gguf"),
            path::Path::new("model/tokenizer.json"),
        );
        let res = query_ai("Who was Albert Einstein", vec![], 4, 10, model.as_ref()).await;
        assert_eq!(res.len(), 33); // 33 = 10 for result & 23 for prompt.
    }
}
