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
            vec![0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
        )
        .unwrap();
    /// Histogram for capturing the overall request time.
    static ref REQUEST_RESPONSE_TIME: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "inference_response_duration",
            "Histogram of response generation times in seconds.",
            &[],
            vec![0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
        )
        .unwrap();
);
