// chunk size for splitting longer content.
const CHUNK_SIZE: usize = 384;

/// Loads a model.
pub(crate) fn get_embedding_model(gguf_path: &str) -> llama_cpp::LlamaModel {
    let model_params = llama_cpp::LlamaParams::default();
    llama_cpp::LlamaModel::load_from_file(gguf_path, model_params)
        .expect("Failed to load embedding model!")
}

/// Given a string will return a vector representing the (average) embedding.
pub(crate) fn embed(content: &str, model: &llama_cpp::LlamaModel) -> Vec<f32> {
    // Chunk the content.
    let tokens = content.split_whitespace();
    let binding = tokens.collect::<Vec<&str>>();
    let chunks = binding.chunks(CHUNK_SIZE);

    // Get embeddings
    let embeddings: Vec<Vec<f32>> = chunks
        .map(|chunk| {
            let chunk_str = chunk.join(" ");
            let params = llama_cpp::EmbeddingsParams::default();
            model
                .embeddings(&[chunk_str], params)
                .expect("Failed to get embeddings.")
                .into_iter()
                .next()
                .expect("Could not unpack embedding.")
        })
        .collect();
    if embeddings.is_empty() {
        return Vec::new();
    }

    // Calculate an average embedding
    let mut average_embedding = vec![0.0; embeddings[0].len()];
    for embedding in &embeddings {
        for (i, value) in embedding.iter().enumerate() {
            average_embedding[i] += value;
        }
    }
    let num_embeddings = embeddings.len() as f32;
    for value in &mut average_embedding {
        *value /= num_embeddings;
    }

    average_embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_embedding_model_for_success() {
        get_embedding_model("model/embed.gguf");
    }

    #[test]
    fn test_embed_for_success() {
        let model = get_embedding_model("model/embed.gguf");
        embed("Lorem ipsum dolor sit amet.", &model);
    }

    #[test]
    #[should_panic(expected = "Failed to load embedding model!")]
    fn test_get_embedding_model_for_failure() {
        get_embedding_model("foo/bar42");
    }

    #[test]
    fn test_embed_for_sanity() {
        let model = get_embedding_model("model/embed.gguf");
        let result = embed("hello", &model);
        assert_eq!(result.iter().sum::<f32>(), -0.52504015);

        let result = embed("💩", &model);
        assert_eq!(result.iter().sum::<f32>(), -0.46041253);
    }
}
