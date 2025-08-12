//! This is documentation for the `rusty_llm` crate.

#![doc = include_str!("../README.md")]

use lazy_static::lazy_static;

pub(crate) mod ai;
pub mod api;
pub(crate) mod cache;
pub(crate) mod embedding;
pub mod knowledge;

// A set of buckets.
const BUCKETS: &[f64] = &[
    0.001,
    0.01,
    0.05,
    0.1,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    2.0,
    5.0,
    10.0,
    f64::INFINITY,
];

lazy_static!(
    static ref INSTANCE_LABEL: String = std::env::var("INSTANCE_LABEL").unwrap_or_else(|_| "default".to_string());

    /// Histogram for collecting token creation time.
    static ref TOKEN_RESPONSE_TIME: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "token_creation_duration",
            "Histogram of token generation times in seconds.",
            &["instance_label"],
            BUCKETS.to_vec()
        )
        .unwrap();
    /// Histogram for capturing the embedding time.
    static ref EMBEDDING_TIME: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "embedding_duration",
            "Histogram of embedding time in seconds.",
            &["instance_label"],
            BUCKETS.to_vec()
        )
        .unwrap();
    /// Histogram for capturing the overall request time.
    static ref REQUEST_RESPONSE_TIME: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "inference_response_duration",
            "Histogram of response generation times in seconds.",
            &["instance_label"],
            BUCKETS.to_vec()
        )
        .unwrap();
);
