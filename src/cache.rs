use crate::api;
use llama_cpp_2::token;
use std::hash::{Hash, Hasher};
use std::{collections, time};

pub(crate) struct CacheEntry {
    tokens: Vec<token::LlamaToken>,
    expires_at: time::Instant,
}

pub(crate) struct TokenCache {
    inner: collections::HashMap<String, CacheEntry>,
    ttl: time::Duration,
}

impl TokenCache {
    pub fn new(ttl: time::Duration) -> Self {
        Self {
            inner: collections::HashMap::new(),
            ttl,
        }
    }
}

/// Fingerprint generation from first two user messages - good for chat context caching.
pub fn fingerprint(messages: &[api::Message]) -> Option<String> {
    let user_msgs: Vec<&api::Message> = messages.iter().filter(|m| m.role == "user").collect();
    if user_msgs.len() < 2 {
        return None;
    }
    let combined = format!("{}|{}", user_msgs[0].content, user_msgs[1].content);
    let mut hasher = collections::hash_map::DefaultHasher::new();
    combined.hash(&mut hasher);
    let hash_value = hasher.finish();
    Some(format!("{:016x}", hash_value))
}
