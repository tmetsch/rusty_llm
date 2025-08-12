use llama_cpp_2::context;
use llama_cpp_2::llama_backend;
use llama_cpp_2::llama_batch;
use llama_cpp_2::model;

/// Loads a model.
pub(crate) fn get_embedding_model(
    gguf_path: &str,
    backend: &llama_backend::LlamaBackend,
) -> model::LlamaModel {
    let model_params = model::params::LlamaModelParams::default();

    model::LlamaModel::load_from_file(backend, gguf_path, &model_params)
        .expect("Failed to load embedding model!")
}

/// Given a string will return a vector representing the (average) embedding.
pub(crate) fn embed(
    content: &str,
    model: &model::LlamaModel,
    backend: &llama_backend::LlamaBackend,
) -> Vec<f32> {
    let ctx_params = context::params::LlamaContextParams::default().with_embeddings(true);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .expect("Failed to construct new LlamaContext.");

    // For now let's not go line by line but do it all in one goal...might make this configurable later.
    let tokens_list = model
        .str_to_token(content, model::AddBos::Always)
        .expect("Failed to convert content to token list.");
    let mut output = Vec::with_capacity(tokens_list.len());

    let mut batch = llama_batch::LlamaBatch::new(ctx.n_ctx() as usize, 1);
    batch
        .add_sequence(&tokens_list, 0, false)
        .expect("Failed to add sequence");
    batch_decode(&mut ctx, &mut batch, 1, &mut output);

    output[0].clone()
}

fn batch_decode(
    ctx: &mut context::LlamaContext,
    batch: &mut llama_batch::LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
) {
    ctx.clear_kv_cache();
    ctx.decode(batch).expect("Failed to decode");
    for i in 0..s_batch {
        let embedding = ctx
            .embeddings_seq_ith(i)
            .expect("Failed to get embeddings.");
        output.push(normalize(embedding));
    }

    batch.clear();
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input.iter().map(|&val| val * val).sum::<f32>().sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}

// /// Computes the average embedding.
// fn compute_average_embedding(embeddings: Vec<Vec<f32>>, embedding_size: usize) -> Vec<f32> {
//     let mut average_embedding = vec![0.0; embedding_size];
//     let num_embeddings = embeddings.len() as f32;
//
//     for embedding in embeddings {
//         for (i, value) in embedding.iter().enumerate() {
//             average_embedding[i] += value;
//         }
//     }
//
//     for value in &mut average_embedding {
//         *value /= num_embeddings;
//     }
//
//     average_embedding
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai;

    #[test]
    fn test_get_embedding_model_for_success() {
        let backend = ai::init_backend();
        get_embedding_model("model/embed.gguf", backend);
    }

    #[test]
    fn test_embed_for_success() {
        let backend = ai::init_backend();
        let model = get_embedding_model("model/embed.gguf", backend);
        embed("Lorem ipsum dolor sit amet.", &model, backend);
    }

    #[test]
    #[should_panic(expected = "\"foo/bar42\" does not exist")]
    fn test_get_embedding_model_for_failure() {
        let backend = ai::init_backend();
        get_embedding_model("foo/bar42", backend);
    }

    #[test]
    fn test_embed_for_sanity() {
        let backend = ai::init_backend();
        let model = get_embedding_model("model/embed.gguf", backend);
        let result = embed("hello", &model, backend);
        assert_eq!(
            (result.iter().sum::<f32>() * 1000.0).round() / 1000.0,
            -0.526
        );

        let result = embed("💩", &model, backend);
        assert_eq!(
            (result.iter().sum::<f32>() * 1000.0).round() / 1000.0,
            -0.460
        );
    }
}
