use surrealdb::engine::local;
use surrealdb::sql;

/// Bert model has a dimension of 384.
const LENGTH: usize = 384;

/// Represent an entry in the knowledge base.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(crate) struct ContextEntry {
    pub id: sql::Thing,
    pub content: String,
    pub vector: Vec<f32>,
    pub created_at: sql::Datetime,
}

/// Returns a handle to the in memory database (will panic if it doesn't work; no db --> no chance of all of this to work).
pub async fn get_db() -> surrealdb::Surreal<local::Db> {
    let db = surrealdb::Surreal::new::<local::Mem>(())
        .await
        .expect("Could not initialize the surreal db...");
    db.use_ns("rag")
        .use_db("content")
        .await
        .expect("Could not set the ns and db.");
    db
}

/// Adds content to the knowledge base.
pub(crate) async fn add_context(
    content: &str,
    tensor_content: &candle_core::Tensor,
    db: &surrealdb::Surreal<local::Db>,
) {
    let id = sql::Uuid::new_v4().0.to_string().replace('-', "");
    let id = match sql::thing(format!("vector_index:{}", id).as_str()) {
        Ok(id) => id,
        Err(err) => {
            log::error!("Could not create unique id: {}.", err);
            return;
        }
    };

    let vector = match tensor_content.reshape((LENGTH, )) {
        Ok(val) => val.to_vec1().unwrap_or_default(),
        Err(err) => {
            log::error!("Could create vector: {}.", err);
            return;
        }
    };

    let _vector_index: Option<ContextEntry> = db
        .create(("vector_index", id.clone()))
        .content(ContextEntry {
            id: id.clone(),
            content: content.to_string(),
            vector,
            created_at: sql::Datetime::default(),
        })
        .await
        .expect("Unable to insert new content!");
}

/// Retrieves a limit number of entries from the knowledge base if the cosine similarity score is above a threshold.
pub(crate) async fn get_context(
    query: &candle_core::Tensor,
    db: &surrealdb::Surreal<local::Db>,
) -> Vec<ContextEntry> {
    let embeddings: Vec<f32> = match query.reshape((LENGTH, )) {
        Ok(val) => val.to_vec1().unwrap_or_default(),
        Err(err) => {
            log::error!("Could reshape: {}.", err);
            return vec![];
        }
    };

    let mut result = match db
        .query(
            "SELECT *, vector::similarity::cosine(vector, $query) AS score \
        FROM vector_index \
        WHERE vector::similarity::cosine(vector, $query) > 0.5 \
        ORDER BY score DESC \
        LIMIT 3",
        ) //
        .bind(("query", embeddings))
        .await
    {
        Ok(res) => res,
        Err(err) => {
            log::error!("Could not run the query: {};", err);
            return vec![];
        }
    };
    let vector_indexes: Vec<ContextEntry> = result.take(0).expect("This should not fail!");
    vector_indexes
}

#[cfg(test)]
mod tests {
    use crate::embedding::{get_embedding_model, tokenize};

    use super::*;

    #[actix_web::test]
    async fn test_add_content_for_success() {
        let text = "hello";
        let (model, tokenizer) = get_embedding_model("BAAI/bge-small-en-v1.5");
        let tokens = tokenize(text, &model, &tokenizer);
        let db = get_db().await;
        add_context(text, &tokens, &db).await;
    }

    #[actix_web::test]
    async fn test_get_content_for_success() {
        let text = "hello";
        let (model, tokenizer) = get_embedding_model("BAAI/bge-small-en-v1.5");
        let tokens = tokenize(text, &model, &tokenizer);
        let db = get_db().await;
        add_context(text, &tokens, &db).await;
        let tmp = get_context(&tokens, &db).await;
        assert_eq!(tmp.len(), 1)
    }

    #[actix_web::test]
    async fn test_get_content_for_sanity() {
        let albert = "Albert Einstein was a theoretical physicist and knew Thomas Jefferson.";
        let thomas = "Thomas Jefferson was the third president of the United States.";
        let johan = "Johan van Oldenbarnevelt founded the Dutch East India Company.";

        let db = get_db().await;
        let (model, tokenizer) = get_embedding_model("BAAI/bge-small-en-v1.5");

        for val in vec![albert, thomas, johan] {
            let tokens = tokenize(val, &model, &tokenizer);
            add_context(val, &tokens, &db).await;
        }

        // test single return...
        let query = "Who was Johan Oldenbarneveld?";
        let query_tokens = tokenize(query, &model, &tokenizer);
        let tmp = get_context(&query_tokens, &db).await;
        assert_eq!(tmp.len(), 1);
        assert_eq!(tmp[0].content, johan);

        // test double return...
        let query = "Who was Thomas Jefferson?";
        let query_tokens = tokenize(query, &model, &tokenizer);
        let tmp = get_context(&query_tokens, &db).await;
        assert_eq!(tmp.len(), 2);
        for item in &tmp {
            assert_ne!(item.content, johan)
        }

        // empty result...
        let query = "Who was Frans Hals?";
        let query_tokens = tokenize(query, &model, &tokenizer);
        let tmp = get_context(&query_tokens, &db).await;
        assert_eq!(tmp.len(), 0);
    }
}
