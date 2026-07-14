//! Middleware chain for HTTP request/response processing.
//!
//! This module provides a composable middleware system for processing HTTP
//! requests and responses. Middleware can be used for logging, retry logic,
//! header injection, and more.
//!
//! # Architecture
//!
//! The middleware system uses a chain-of-responsibility pattern where each
//! middleware can:
//! - Modify the request before passing it to the next middleware
//! - Decide whether to continue the chain or short-circuit
//! - Modify the response after receiving it from the next middleware
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::http::{MiddlewareStack, LoggingMiddleware, RetryMiddleware, RetryConfig};
//!
//! let stack = MiddlewareStack::new()
//!     .push(LoggingMiddleware::new())
//!     .push(RetryMiddleware::new(RetryConfig::default()));
//!
//! let response = stack.execute(client, request).await?;
//! ```

use crate::error::{Error, Result};
use crate::http::retry::{ExponentialBackoff, RetryConfig, RetryPolicy};
use futures::future::BoxFuture;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Request, Response};
use std::sync::Arc;
use std::time::Instant;

/// A middleware that processes HTTP requests and responses.
///
/// Implement this trait to create custom middleware for the HTTP pipeline.
pub trait Middleware: Send + Sync {
    /// Processes a request and returns a response.
    ///
    /// # Arguments
    ///
    /// * `request` - The HTTP request to process
    /// * `client` - The HTTP client for making requests
    /// * `next` - The next middleware in the chain
    ///
    /// # Returns
    ///
    /// The HTTP response or an error.
    fn handle<'a>(
        &'a self,
        request: Request,
        client: &'a Client,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<Response>>;

    /// Returns the name of this middleware for logging purposes.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Represents the next middleware or final handler in the chain.
///
/// This type is used to pass control to the next middleware in the chain.
pub struct Next<'a> {
    /// The remaining middlewares in the chain.
    middlewares: &'a [Arc<dyn Middleware>],
    /// The HTTP client.
    client: &'a Client,
}

impl std::fmt::Debug for Next<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Next")
            .field("remaining_middlewares", &self.middlewares.len())
            .finish_non_exhaustive()
    }
}

impl<'a> Next<'a> {
    /// Creates a new Next with the given middlewares.
    pub(crate) fn new(middlewares: &'a [Arc<dyn Middleware>], client: &'a Client) -> Self {
        Self {
            middlewares,
            client,
        }
    }

    /// Calls the next middleware in the chain or executes the request.
    pub fn run(self, request: Request) -> BoxFuture<'a, Result<Response>> {
        Box::pin(async move {
            if let Some((first, rest)) = self.middlewares.split_first() {
                let next = Next::new(rest, self.client);
                first.handle(request, self.client, next).await
            } else {
                // No more middleware, execute the request
                Ok(self.client.execute(request).await?)
            }
        })
    }
}

/// A stack of middleware that processes requests in order.
///
/// Middleware is executed in the order it was added (FIFO).
/// The first middleware added will be the first to process the request.
#[derive(Clone, Default)]
pub struct MiddlewareStack {
    middlewares: Vec<Arc<dyn Middleware>>,
}

impl std::fmt::Debug for MiddlewareStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareStack")
            .field(
                "middlewares",
                &self
                    .middlewares
                    .iter()
                    .map(|m| m.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl MiddlewareStack {
    /// Creates a new empty middleware stack.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a middleware to the end of the stack.
    ///
    /// Middleware is executed in the order it was added.
    #[must_use]
    pub fn push<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    /// Adds a middleware to the beginning of the stack.
    #[must_use]
    pub fn unshift<M: Middleware + 'static>(mut self, middleware: M) -> Self {
        self.middlewares.insert(0, Arc::new(middleware));
        self
    }

    /// Executes a request through the middleware stack.
    pub async fn execute(&self, client: &Client, request: Request) -> Result<Response> {
        let next = Next::new(&self.middlewares, client);
        next.run(request).await
    }

    /// Returns the number of middleware in the stack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Returns true if the stack is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Returns the middleware names in execution order.
    #[must_use]
    pub fn middleware_names(&self) -> Vec<&'static str> {
        self.middlewares.iter().map(|m| m.name()).collect()
    }
}

/// Middleware that logs requests and responses using the `tracing` crate.
///
/// This middleware logs:
/// - Request method and URL at the start
/// - Response status and duration on completion
/// - Errors if the request fails
#[derive(Debug, Clone, Default)]
pub struct LoggingMiddleware {
    /// Whether to log request headers (may contain sensitive data).
    log_headers: bool,
    /// Whether to log request/response bodies (may be large).
    log_bodies: bool,
}

impl LoggingMiddleware {
    /// Creates a new logging middleware with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables logging of request headers.
    ///
    /// **Warning**: Headers may contain sensitive data like API keys.
    #[must_use]
    pub fn with_headers(mut self) -> Self {
        self.log_headers = true;
        self
    }

    /// Enables logging of request/response bodies.
    ///
    /// **Warning**: Bodies may be large and impact performance.
    #[must_use]
    pub fn with_bodies(mut self) -> Self {
        self.log_bodies = true;
        self
    }
}

impl Middleware for LoggingMiddleware {
    fn handle<'a>(
        &'a self,
        request: Request,
        _client: &'a Client,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<Response>> {
        Box::pin(async move {
            let method = request.method().clone();
            let url = request.url().clone();
            let start = Instant::now();

            // Log request start
            if self.log_headers {
                tracing::debug!(
                    method = %method,
                    url = %url,
                    headers = ?request.headers(),
                    "Starting HTTP request"
                );
            } else {
                tracing::debug!(
                    method = %method,
                    url = %url,
                    "Starting HTTP request"
                );
            }

            // Execute the request
            let result: Result<Response> = next.run(request).await;
            let duration = start.elapsed();

            match &result {
                Ok(response) => {
                    tracing::debug!(
                        method = %method,
                        url = %url,
                        status = %response.status(),
                        duration_ms = duration.as_millis(),
                        "HTTP request completed"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        method = %method,
                        url = %url,
                        error = %error,
                        duration_ms = duration.as_millis(),
                        "HTTP request failed"
                    );
                }
            }

            result
        })
    }

    fn name(&self) -> &'static str {
        "LoggingMiddleware"
    }
}

/// Middleware that applies retry logic with exponential backoff.
///
/// This middleware automatically retries requests that fail with retryable
/// errors (rate limits, server errors, network issues).
#[derive(Debug, Clone)]
pub struct RetryMiddleware<P: RetryPolicy = ExponentialBackoff> {
    policy: P,
}

impl RetryMiddleware<ExponentialBackoff> {
    /// Creates a new retry middleware with the given configuration.
    #[must_use]
    pub fn new(config: RetryConfig) -> Self {
        Self {
            policy: ExponentialBackoff::new(config),
        }
    }
}

impl<P: RetryPolicy> RetryMiddleware<P> {
    /// Creates a new retry middleware with a custom policy.
    #[must_use]
    pub fn with_policy(policy: P) -> Self {
        Self { policy }
    }
}

impl Default for RetryMiddleware<ExponentialBackoff> {
    fn default() -> Self {
        Self::new(RetryConfig::default())
    }
}

impl<P: RetryPolicy + Clone + 'static> Middleware for RetryMiddleware<P> {
    fn handle<'a>(
        &'a self,
        request: Request,
        client: &'a Client,
        _next: Next<'a>,
    ) -> BoxFuture<'a, Result<Response>> {
        Box::pin(async move {
            let max_retries = self.policy.max_retries();

            // We need to clone the request for retries
            // This is necessary because Request doesn't implement Clone
            let method = request.method().clone();
            let url = request.url().clone();
            let headers = request.headers().clone();
            let body = request
                .body()
                .and_then(|b| b.as_bytes())
                .map(|b| b.to_vec());

            let mut attempt = 0u32;

            loop {
                attempt += 1;

                // Rebuild the request for each attempt
                let mut builder = client
                    .request(method.clone(), url.clone())
                    .headers(headers.clone());
                if let Some(ref body_bytes) = body {
                    builder = builder.body(body_bytes.clone());
                }

                let request = builder.build().map_err(Error::from)?;

                // Create a fresh next chain for each attempt
                // Since we're at the retry middleware, we call the client directly
                let result = client.execute(request).await;

                match result {
                    Ok(response) => {
                        // Check if response indicates an error that should be retried
                        let status = response.status();
                        if status.is_success() || attempt > max_retries {
                            return Ok(response);
                        }

                        // For error responses, we need to check if they're retryable
                        if status.is_server_error()
                            || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        {
                            // Get retry delay from response headers
                            let retry_after =
                                crate::http::response::parse_retry_after(response.headers());

                            // Create an error to check retry policy
                            let error = Error::RateLimited {
                                retry_after,
                                error: Box::new(crate::error::ApiError::new(
                                    status,
                                    crate::error::ApiErrorType::RateLimitError,
                                    "Rate limited",
                                )),
                            };

                            if let Some(delay) = self.policy.should_retry(&error, attempt) {
                                tracing::debug!(
                                    attempt = attempt,
                                    max_retries = max_retries,
                                    delay_ms = delay.as_millis(),
                                    status = %status,
                                    "Retrying request after server error"
                                );

                                #[cfg(feature = "tokio-runtime")]
                                tokio::time::sleep(delay).await;
                                #[cfg(not(feature = "tokio-runtime"))]
                                futures::executor::block_on(async {
                                    // Fallback for non-tokio runtimes
                                    std::thread::sleep(delay);
                                });

                                continue;
                            }
                        }

                        return Ok(response);
                    }
                    Err(e) => {
                        let error = Error::from(e);

                        if let Some(delay) = self.policy.should_retry(&error, attempt) {
                            tracing::debug!(
                                attempt = attempt,
                                max_retries = max_retries,
                                delay_ms = delay.as_millis(),
                                error = %error,
                                "Retrying request after error"
                            );

                            // Store error in case we need it for debugging later
                            drop(error);

                            #[cfg(feature = "tokio-runtime")]
                            tokio::time::sleep(delay).await;
                            #[cfg(not(feature = "tokio-runtime"))]
                            futures::executor::block_on(async {
                                std::thread::sleep(delay);
                            });

                            continue;
                        }

                        return Err(error);
                    }
                }
            }
        })
    }

    fn name(&self) -> &'static str {
        "RetryMiddleware"
    }
}

/// Middleware that adds default headers to all requests.
///
/// This is useful for adding headers like `User-Agent`, authentication
/// headers, or other headers that should be present on every request.
#[derive(Debug, Clone, Default)]
pub struct HeaderMiddleware {
    headers: HeaderMap,
}

impl HeaderMiddleware {
    /// Creates a new header middleware.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a header middleware with the given headers.
    #[must_use]
    pub fn with_headers(headers: HeaderMap) -> Self {
        Self { headers }
    }

    /// Adds a header to be included in all requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the header name or value is invalid.
    pub fn add_header(mut self, name: &str, value: &str) -> Result<Self> {
        let header_name = HeaderName::try_from(name)?;
        let header_value = HeaderValue::try_from(value)?;
        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Adds a header with a static value.
    #[must_use]
    pub fn add_header_static(mut self, name: HeaderName, value: &'static str) -> Self {
        self.headers.insert(name, HeaderValue::from_static(value));
        self
    }

    /// Sets the User-Agent header.
    #[must_use]
    pub fn user_agent(self, user_agent: &'static str) -> Self {
        self.add_header_static(reqwest::header::USER_AGENT, user_agent)
    }
}

impl Middleware for HeaderMiddleware {
    fn handle<'a>(
        &'a self,
        mut request: Request,
        _client: &'a Client,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<Response>> {
        Box::pin(async move {
            // Add headers to the request
            let request_headers = request.headers_mut();
            for (name, value) in &self.headers {
                // Only add if not already present
                if !request_headers.contains_key(name) {
                    request_headers.insert(name.clone(), value.clone());
                }
            }

            next.run(request).await
        })
    }

    fn name(&self) -> &'static str {
        "HeaderMiddleware"
    }
}

/// Middleware that measures and records request timing.
///
/// This middleware can be used to collect metrics about request latency.
#[derive(Debug, Clone, Default)]
pub struct TimingMiddleware {
    /// Optional callback to receive timing information.
    /// Using Option<Arc<dyn Fn>> instead of a direct callback for Clone support.
    _marker: std::marker::PhantomData<()>,
}

impl TimingMiddleware {
    /// Creates a new timing middleware.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Middleware for TimingMiddleware {
    fn handle<'a>(
        &'a self,
        request: Request,
        _client: &'a Client,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<Response>> {
        Box::pin(async move {
            let start = Instant::now();
            let method = request.method().to_string();
            let path = request.url().path().to_string();

            let result: Result<Response> = next.run(request).await;
            let duration = start.elapsed();

            let status = match &result {
                Ok(response) => response.status().as_u16().to_string(),
                Err(_) => "error".to_string(),
            };

            tracing::info!(
                method = method,
                path = path,
                status = status,
                duration_ms = duration.as_millis(),
                "Request timing"
            );

            result
        })
    }

    fn name(&self) -> &'static str {
        "TimingMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_stack_creation() {
        let stack = MiddlewareStack::new()
            .push(LoggingMiddleware::new())
            .push(HeaderMiddleware::new());

        assert_eq!(stack.len(), 2);
        assert!(!stack.is_empty());
    }

    #[test]
    fn test_middleware_stack_names() {
        let stack = MiddlewareStack::new()
            .push(LoggingMiddleware::new())
            .push(HeaderMiddleware::new());

        let names = stack.middleware_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"LoggingMiddleware"));
        assert!(names.contains(&"HeaderMiddleware"));
    }

    #[test]
    fn test_header_middleware() {
        let middleware = HeaderMiddleware::new()
            .add_header("X-Custom", "value")
            .unwrap()
            .user_agent("test-agent/1.0");

        assert!(middleware.headers.contains_key("X-Custom"));
        assert!(middleware.headers.contains_key(reqwest::header::USER_AGENT));
    }

    #[test]
    fn test_logging_middleware_configuration() {
        let middleware = LoggingMiddleware::new().with_headers().with_bodies();

        assert!(middleware.log_headers);
        assert!(middleware.log_bodies);
    }

    #[test]
    fn test_retry_middleware_default() {
        let middleware = RetryMiddleware::default();
        assert_eq!(middleware.policy.config().max_retries, 2);
    }

    #[test]
    fn test_retry_middleware_custom_config() {
        let config = RetryConfig::builder().max_retries(5).build();
        let middleware = RetryMiddleware::new(config);
        assert_eq!(middleware.policy.config().max_retries, 5);
    }

    #[test]
    fn test_middleware_stack_unshift() {
        let stack = MiddlewareStack::new()
            .push(LoggingMiddleware::new())
            .unshift(HeaderMiddleware::new());

        let names = stack.middleware_names();
        // HeaderMiddleware should be first since it was unshifted
        assert_eq!(names[0], "HeaderMiddleware");
        assert_eq!(names[1], "LoggingMiddleware");
    }
}
