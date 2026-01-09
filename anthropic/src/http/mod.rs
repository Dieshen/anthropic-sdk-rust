//! HTTP infrastructure for the Anthropic SDK.
//!
//! This module provides the core HTTP functionality including request building,
//! response handling, retry logic with exponential backoff, and middleware support.
//!
//! # Architecture
//!
//! The HTTP module is organized into several submodules:
//!
//! - [`request`]: Request building with Anthropic-specific headers
//! - [`response`]: Response parsing and error handling
//! - [`retry`]: Retry policies with exponential backoff and jitter
//! - [`middleware`]: Composable middleware chain for request/response processing
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::http::{RequestBuilder, RetryConfig, MiddlewareStack};
//!
//! // Build a request with Anthropic headers
//! let request = RequestBuilder::new(client, "POST", "/v1/messages")
//!     .json_body(&params)?
//!     .add_anthropic_headers("api-key", "2023-06-01")
//!     .build()?;
//!
//! // Configure retry behavior
//! let retry_config = RetryConfig::default();
//!
//! // Set up middleware stack
//! let stack = MiddlewareStack::new()
//!     .push(LoggingMiddleware::new())
//!     .push(RetryMiddleware::new(retry_config));
//! ```

mod middleware;
mod request;
mod response;
mod retry;

pub use middleware::{
    HeaderMiddleware, LoggingMiddleware, Middleware, MiddlewareStack, Next, RetryMiddleware,
    TimingMiddleware,
};
pub use request::{
    RequestBuilder, CONTENT_TYPE_JSON, HEADER_API_KEY, HEADER_API_VERSION, HEADER_BETA,
};
pub use response::{
    extract_request_id, parse_retry_after, read_body, read_body_string, ApiResponse, ResponseExt,
    HEADER_REQUEST_ID, HEADER_RETRY_AFTER, HEADER_RETRY_AFTER_MS,
};
pub use retry::{
    ConstantBackoff, ExponentialBackoff, NoRetry, RetryConfig, RetryConfigBuilder, RetryPolicy,
    DEFAULT_INITIAL_DELAY_MS, DEFAULT_JITTER_FACTOR, DEFAULT_MAX_DELAY_MS, DEFAULT_MAX_RETRIES,
    DEFAULT_MULTIPLIER,
};
