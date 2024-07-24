use candle_transformers::models::bert;

/// This currently uses a bge-small-en-v1.5" model that transforms a text into a vector space. Eventually we'll also want to replace candle here.
pub(crate) fn get_embedding_model(repo: &str) -> (bert::BertModel, tokenizers::Tokenizer) {
    let api = hf_hub::api::sync::Api::new().expect("Could not instantiate HF api - aborting!");
    let api_repo = api.repo(hf_hub::Repo::model(repo.to_string()));

    // get model from HF.
    let weights = api_repo
        .get("pytorch_model.bin")
        .expect("Could not find or download weights file - aborting!");

    // read the files.
    let cfg_file = api_repo
        .get("config.json")
        .expect("Could not find or download the config file - aborting!");
    let config = std::fs::read_to_string(cfg_file).expect("Could not read config.");
    let config: bert::Config = serde_json::from_str(&config).expect("Could not parse config.");

    let vb = candle_nn::VarBuilder::from_pth(&weights, bert::DTYPE, &candle_core::Device::Cpu)
        .expect("Could not create var builder.");
    let model = bert::BertModel::load(vb, &config).expect("Could not load model!");

    // set up the tokenizer.
    let tkn_file = api_repo
        .get("tokenizer.json")
        .expect("Could not find or download the tokenizer file - aborting!");
    let tokenizer = tokenizers::Tokenizer::from_file(tkn_file).expect("Could not load tokenizer.");

    (model, tokenizer)
}

/// Tokenizes a string.
pub(crate) fn embed(
    content: &str,
    model: &bert::BertModel,
    tokenizer: &tokenizers::Tokenizer,
) -> candle_core::Tensor {
    let tokens = tokenizer
        .encode_batch(vec![content], true)
        .expect("Could not encode!");

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            candle_core::Tensor::new(tokens.as_slice(), &candle_core::Device::Cpu)
                .expect("Token to slice didn't work.")
        })
        .collect::<Vec<_>>();

    let token_ids = candle_core::Tensor::stack(&token_ids, 0).expect("Unable to stack token ids.");
    let token_type_ids = token_ids
        .zeros_like()
        .expect("Unable to get token type ids.");

    let embeddings = model
        .forward(&token_ids, &token_type_ids)
        .expect("Unable to get embeddings.");
    let (_n_sentence, n_tokens, _hidden_size) = embeddings
        .dims3()
        .expect("Unable to get embeddings dimensions.");
    let embeddings = (embeddings.sum(1).expect("could not sum.") / (n_tokens as f64))
        .expect("Unable to get embeddings sum.");
    embeddings
        .broadcast_div(
            &embeddings
                .sqr()
                .expect("sqr failed.")
                .sum_keepdim(1)
                .expect("sum failed.")
                .sqrt()
                .expect("sqrt failed."),
        )
        .expect("Unable to get embeddings broadcast_div.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_embedding_model_for_success() {
        get_embedding_model("BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn test_embed_for_success() {
        let (model, tokenizer) = get_embedding_model("BAAI/bge-small-en-v1.5");
        embed("hello", &model, &tokenizer);
    }

    #[test]
    #[should_panic(expected = "Could not find or download weights file - aborting!")]
    fn test_get_embedding_model_for_failure() {
        get_embedding_model("foo/bar42");
    }

    #[test]
    fn test_embed_for_sanity() {
        let (model, tokenizer) = get_embedding_model("BAAI/bge-small-en-v1.5");
        let result = embed("hello", &model, &tokenizer);
        assert_eq!(
            result.sum_all().unwrap().to_scalar::<f32>().unwrap(),
            0.10454147
        );

        let result = embed("💩", &model, &tokenizer);
        assert_eq!(
            result.sum_all().unwrap().to_scalar::<f32>().unwrap(),
            0.12179252
        );
    }
}
