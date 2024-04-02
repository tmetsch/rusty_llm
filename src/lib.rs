//! This is documentation for the `rusty_llm` crate.

#![doc = include_str!("../README.md")]

use lazy_static::lazy_static;

pub(crate) mod ai;
pub mod api;
pub(crate) mod embedding;
pub mod knowledge;

lazy_static!(
    /// Histogram for collecting token creation time.
    static ref TOKEN_RESPONSE_TIME: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "token_creation_duration",
            "Histogram of token generation times in seconds.",
            &[],
            vec![0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        )
        .unwrap();
    /// Histogram for capturing the embedding time.
    static ref EMBEDDING_TIME: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "embedding_duration",
            "Histogram of embedding time in seconds.",
            &[],
            vec![0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        )
        .unwrap();
    /// Histogram for capturing the overall request time.
    static ref REQUEST_RESPONSE_TIME: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "inference_response_duration",
            "Histogram of response generation times in seconds.",
            &[],
            vec![0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        )
        .unwrap();
);
