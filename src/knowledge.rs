use std::cmp;
use std::collections;

/// Defines the minimum threshold for the cosine similarity search.
const THRESHOLD: f32 = 0.5;

/// Limits the number of entries we return from the knowledge base.
const MAX_ENTRIES: usize = 3;

/// Represents an entry in the priority queue we use when searching for relevant information.
#[derive(PartialEq)]
struct SimilarityContext<'a> {
    similarity: f32,
    content: &'a str,
}

impl Eq for SimilarityContext<'_> {}

impl Ord for SimilarityContext<'_> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // Reverse the order so the BinaryHeap becomes a min-heap
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(cmp::Ordering::Equal)
    }
}

impl PartialOrd for SimilarityContext<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Holds the actual knowledge.
#[derive(Clone)]
pub struct KnowledgeBase {
    data: Vec<(String, Vec<f32>)>,
}

/// Returns a knowledge base.
pub async fn get_db() -> KnowledgeBase {
    KnowledgeBase { data: Vec::new() }
}

/// Calculates the cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let magnitude_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b).max(1e-10)
}

/// Adds context to the knowledge base.
pub(crate) async fn add_context(content: &str, embedding: Vec<f32>, db: &mut KnowledgeBase) {
    db.data.push((content.to_string(), embedding));
}

/// Retrieves context relevant to the given embedding.
pub(crate) async fn get_context(embedding: Vec<f32>, db: &KnowledgeBase) -> Vec<String> {
    if db.data.is_empty() {
        return Vec::new();
    }

    // Calculate similarities between input embedding and all stored embeddings
    let mut heap = collections::BinaryHeap::with_capacity(MAX_ENTRIES + 1);

    for (content, existing_embedding) in &db.data {
        let similarity = cosine_similarity(&embedding, existing_embedding);
        if similarity >= THRESHOLD {
            heap.push(SimilarityContext {
                similarity,
                content,
            });
            if heap.len() > MAX_ENTRIES {
                heap.pop(); // Maintain the heap size to MAX_ENTRIES
            }
        }
    }

    // Return the heap as a sorted vector.
    heap.into_sorted_vec()
        .into_iter()
        .map(|sc| sc.content.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai;
    use crate::embedding::{embed, get_embedding_model};

    #[actix_web::test]
    async fn test_add_content_for_success() {
        let text = "hello";
        let tokens = Vec::new();
        let mut db = get_db().await;
        add_context(text, tokens, &mut db).await;
    }

    #[actix_web::test]
    async fn test_get_content_for_success() {
        let text = "hello";
        let backend = ai::init_backend();
        let model = get_embedding_model("model/embed.gguf", &backend);
        let tokens = embed(text, &model, &backend);
        let mut db = get_db().await;
        add_context(text, tokens.clone(), &mut db).await;
        let tmp = get_context(tokens, &db).await;
        assert_eq!(tmp.len(), 1)
    }

    #[actix_web::test]
    async fn test_get_content_for_sanity() {
        let albert = "Albert Einstein was a theoretical physicist and knew Thomas Jefferson.";
        let thomas = "Thomas Jefferson was the third president of the United States.";
        let johan = "Johan van Oldenbarnevelt founded the Dutch East India Company.";

        let mut db = get_db().await;
        let backend = ai::init_backend();
        let model = get_embedding_model("model/embed.gguf", &backend);

        for val in &[albert, thomas, johan] {
            let tokens = embed(val, &model, &backend);
            add_context(val, tokens, &mut db).await;
        }

        // test single return...
        let query = "Who was Johan Oldenbarneveld?";
        let query_tokens = embed(query, &model, &backend);
        let tmp = get_context(query_tokens, &db).await;
        assert_eq!(tmp.len(), 1);
        assert_eq!(tmp[0], johan);

        // test double return...
        let query = "Who was Thomas Jefferson?";
        let query_tokens = embed(query, &model, &backend);
        let tmp = get_context(query_tokens, &db).await;
        assert_eq!(tmp.len(), 2);
        for item in &tmp {
            assert_ne!(item, johan)
        }

        // empty result...
        let query = "Who was the painter Frans Hals?";
        let query_tokens = embed(query, &model, &backend);
        let tmp = get_context(query_tokens, &db).await;
        assert_eq!(tmp.len(), 0);
    }
}
