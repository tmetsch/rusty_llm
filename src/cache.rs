use crate::api;
use llama_cpp_2::token;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::{collections, time};

// TODO: check for max size of cache.

#[derive(Clone)]
pub(crate) struct CacheEntry {
    tokens: Arc<Vec<token::LlamaToken>>,
    expires_at: time::Instant,
}

#[derive(Clone)]
pub struct TokenCache {
    inner: collections::HashMap<String, CacheEntry>,
    ttl: time::Duration,
}

/// A simple in-memory token cache for Llama tokens.
///
/// This cache stores token sequences keyed by a string and expires entries after a configurable
/// TTL. Normally, one might cache the full KV store for efficiency, but managing `LlamaContext` in
/// async code is tricky due to raw pointers. This implementation is simple and safe for most use
/// cases, but it probably won't outperform more advanced solutions like the ones in `vllm`.
///
/// Future improvement: could explore using channels or other mechanisms to safely cache
/// `LlamaContext` across async tasks.
impl TokenCache {
    pub fn new(ttl: time::Duration) -> Self {
        Self {
            inner: collections::HashMap::new(),
            ttl,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<Arc<Vec<token::LlamaToken>>> {
        if let Some(entry) = self.inner.get(key) {
            if entry.expires_at > time::Instant::now() {
                return Some(Arc::clone(&entry.tokens));
            }
            self.inner.remove(key);
        }
        None
    }

    pub fn insert(&mut self, key: String, tokens: Arc<Vec<token::LlamaToken>>) {
        let entry = CacheEntry {
            tokens,
            expires_at: time::Instant::now() + self.ttl,
        };
        self.inner.insert(key, entry);
    }

    pub fn has_key(&self, key: &str) -> bool {
        self.inner
            .get(key)
            .map(|entry| entry.expires_at > time::Instant::now())
            .unwrap_or(false)
    }

    pub fn extend(&mut self, key: &str, new_tokens: Vec<token::LlamaToken>) {
        if let Some(entry) = self.inner.get_mut(key) {
            let mut combined_tokens = Vec::with_capacity(entry.tokens.len() + new_tokens.len());
            combined_tokens.extend(entry.tokens.iter().cloned());
            combined_tokens.extend(new_tokens);
            entry.tokens = Arc::new(combined_tokens);
            entry.expires_at = time::Instant::now() + self.ttl;
        }
    }
}

/// Fingerprint generation from first two user messages - good for chat context caching.
pub(crate) fn fingerprint(messages: &[api::Message]) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_for_sanity() {
        let mut message = vec![api::Message {
            role: "user".to_string(),
            content: "foo".to_string(),
        }];
        let res = fingerprint(&message);
        assert!(res.is_none()); // not enough messages...
        message.push(api::Message {
            role: "system".to_string(),
            content: "bar".to_string(),
        });
        let res = fingerprint(&message);
        assert!(res.is_none()); // ...still not enough user messages...
        message.push(api::Message {
            role: "user".to_string(),
            content: "baz".to_string(),
        });
        let res = fingerprint(&message);
        assert!(res.is_some()); // ...now it should work.
    }

    #[test]
    fn test_token_cache_for_sanity() {
        let mut cache = TokenCache::new(time::Duration::from_secs(1));

        // Create dummy tokens
        let tokens = Arc::new(vec![token::LlamaToken(1)]);

        // Insert tokens into cache
        cache.insert("key1".to_string(), Arc::clone(&tokens));
        assert!(cache.has_key("key1"));
        assert_eq!(cache.get("key1").unwrap()[0].0, 1);

        // Extend the cache entry
        let extra_tokens = vec![token::LlamaToken(2)];
        cache.extend("key1", extra_tokens);
        let cached_tokens = cache.get("key1").unwrap();
        assert_eq!(cached_tokens.len(), 2);
        assert_eq!(cached_tokens[0].0, 1);
        assert_eq!(cached_tokens[1].0, 2);

        // Wait for the cache to expire
        std::thread::sleep(time::Duration::from_secs(2));
        assert!(cache.get("key1").is_none());
        assert!(!cache.has_key("key1"));
    }
}
