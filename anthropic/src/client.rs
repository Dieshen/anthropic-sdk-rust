//! Anthropic API client.
//!
//! This module provides the main entry point for interacting with the Anthropic API.
//! The [`Anthropic`] client handles authentication, request building, retry logic,
//! and response parsing.
//!
//! # Example
//!
//! ```rust,no_run
//! use anthropic::Anthropic;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic::Error> {
//!     // Create client from environment (reads ANTHROPIC_API_KEY)
//!     let client = Anthropic::new()?;
//!
//!     // Or with explicit configuration
//!     let client = Anthropic::builder()
//!         .api_key("sk-ant-...")
//!         .max_retries(3)
//!         .build()?;
//!
//!     Ok(())
//! }
//! ```

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Method, StatusCode};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, trace};
use url::Url;

use crate::config::{ClientConfig, ClientConfigBuilder};
use crate::error::{ApiError, ApiErrorType, Error, RawApiErrorResponse, Result};
use crate::http::{
    parse_retry_after, ExponentialBackoff, RetryConfig, RetryPolicy, HEADER_REQUEST_ID,
};

// =============================================================================
// Constants
// =============================================================================

/// Current Anthropic API version.
pub const API_VERSION: &str = "2023-06-01";

/// Header name for Anthropic API version.
pub const HEADER_ANTHROPIC_VERSION: &str = "anthropic-version";

/// Header name for API key authentication.
pub const HEADER_API_KEY: &str = "x-api-key";

/// Default user agent for SDK requests.
const USER_AGENT: &str = concat!("anthropic-rust/", env!("CARGO_PKG_VERSION"));

// =============================================================================
// Anthropic Client
// =============================================================================

/// The main Anthropic API client.
///
/// This is the primary entry point for making API calls. It handles
/// authentication, request building, retry logic, and response parsing.
///
/// # Thread Safety
///
/// `Anthropic` is `Clone` and can be shared across threads safely.
/// Internally, it uses `Arc` for shared state.
///
/// # Example
///
/// ```rust,no_run
/// use anthropic::Anthropic;
///
/// #[tokio::main]
/// async fn main() -> Result<(), anthropic::Error> {
///     let client = Anthropic::new()?;
///     // Use the client...
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Anthropic {
    /// Internal state shared across clones.
    inner: Arc<AnthropicInner>,
}

/// Internal state for the Anthropic client.
#[derive(Debug)]
struct AnthropicInner {
    /// HTTP client for making requests.
    http_client: reqwest::Client,
    /// Client configuration.
    config: ClientConfig,
    /// Retry policy for failed requests.
    retry_policy: ExponentialBackoff,
}

impl Anthropic {
    /// Creates a new client from environment variables.
    ///
    /// This reads the following environment variables:
    /// - `ANTHROPIC_API_KEY`: API key for authentication (required unless auth token is set)
    /// - `ANTHROPIC_AUTH_TOKEN`: OAuth/JWT token (alternative to API key)
    /// - `ANTHROPIC_BASE_URL`: Base URL override (optional)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Neither `ANTHROPIC_API_KEY` nor `ANTHROPIC_AUTH_TOKEN` is set
    /// - `ANTHROPIC_BASE_URL` contains an invalid URL
    /// - The HTTP client fails to initialize
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use anthropic::Anthropic;
    ///
    /// let client = Anthropic::new()?;
    /// # Ok::<(), anthropic::Error>(())
    /// ```
    pub fn new() -> Result<Self> {
        let config = ClientConfig::from_env()?;
        Self::with_config(config)
    }

    /// Creates a new client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Client configuration including API key, base URL, and retry settings
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client fails to initialize.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use anthropic::{Anthropic, ClientConfig};
    ///
    /// let config = ClientConfig::builder()
    ///     .api_key("sk-ant-...")
    ///     .build()?;
    ///
    /// let client = Anthropic::with_config(config)?;
    /// # Ok::<(), anthropic::Error>(())
    /// ```
    pub fn with_config(config: ClientConfig) -> Result<Self> {
        // Build default headers
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static(HEADER_ANTHROPIC_VERSION),
            HeaderValue::from_static(API_VERSION),
        );

        // Add authentication header
        if let Some((name, value)) = config.auth_header() {
            headers.insert(HeaderName::try_from(name)?, HeaderValue::try_from(value)?);
        }

        // Merge with custom headers from config
        for (name, value) in config.default_headers() {
            headers.insert(name.clone(), value.clone());
        }

        // Build HTTP client
        let http_client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .default_headers(headers)
            .timeout(config.timeout())
            .connect_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(Error::Http)?;

        // Create retry policy from config
        let retry_config = RetryConfig::builder()
            .max_retries(config.max_retries())
            .build();
        let retry_policy = ExponentialBackoff::new(retry_config);

        Ok(Self {
            inner: Arc::new(AnthropicInner {
                http_client,
                config,
                retry_policy,
            }),
        })
    }

    /// Creates a new client builder.
    ///
    /// This provides a fluent interface for configuring the client.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use anthropic::Anthropic;
    /// use std::time::Duration;
    ///
    /// let client = Anthropic::builder()
    ///     .api_key("sk-ant-...")
    ///     .timeout(Duration::from_secs(60))
    ///     .max_retries(5)
    ///     .build()?;
    /// # Ok::<(), anthropic::Error>(())
    /// ```
    #[must_use]
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::new()
    }

    /// Returns the base URL for API requests.
    #[must_use]
    pub fn base_url(&self) -> &Url {
        self.inner.config.base_url()
    }

    /// Returns the configured timeout.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.inner.config.timeout()
    }

    /// Returns the maximum number of retries.
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.inner.config.max_retries()
    }

    /// Returns a clone of the underlying HTTP client.
    ///
    /// This is primarily intended for internal use by resource modules
    /// that need direct access to the HTTP client.
    #[must_use]
    pub(crate) fn http_client(&self) -> reqwest::Client {
        self.inner.http_client.clone()
    }

    // =========================================================================
    // Internal request methods
    // =========================================================================

    /// Makes a GET request to the specified path.
    pub(crate) async fn get<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.request(Method::GET, path, Option::<&()>::None).await
    }

    /// Makes a POST request with a JSON body.
    // `B` is only bound by `Serialize`, not `Sync`, so the `&B` reference held
    // across the `.await` points inside `request` (retry loop) makes this
    // future `!Send`. Adding a `Sync` bound would be a breaking API change for
    // this crate's request helpers, so we accept the `!Send` future instead.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn post<T, B>(&self, path: &str, body: &B) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {
        self.request(Method::POST, path, Some(body)).await
    }

    /// Makes a POST request with a JSON body and additional per-request headers.
    ///
    /// Used by the beta APIs, which must send `anthropic-beta` describing the
    /// features the caller opted into.
    // See `post` above for why this future is `!Send`.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn post_with_headers<T, B>(
        &self,
        path: &str,
        body: &B,
        extra_headers: &[(&str, String)],
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {
        self.request_with_headers(Method::POST, path, Some(body), extra_headers)
            .await
    }

    /// Makes a GET request with query parameters.
    // `Q` is only bound by `Serialize`, not `Sync`; see `post` above for why
    // this makes the returned future `!Send`.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn get_with_query<T, Q>(&self, path: &str, query: &Q) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        Q: serde::Serialize,
    {
        self.request_with_query(Method::GET, path, query, Option::<&()>::None)
            .await
    }

    /// Makes a POST request and returns the raw response for streaming.
    ///
    /// This is used internally for streaming requests where we need access
    /// to the raw response body.
    // `B` is only bound by `Serialize`, not `Sync`; see `post` above for why
    // this makes the returned future `!Send`.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn post_raw<B>(&self, path: &str, body: &B) -> Result<reqwest::Response>
    where
        B: serde::Serialize,
    {
        let url = self.build_url(path)?;

        let response = self
            .inner
            .http_client
            .request(Method::POST, url)
            .json(body)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response)
        } else {
            // Parse error response
            let status = response.status();
            let request_id = response
                .headers()
                .get(HEADER_REQUEST_ID)
                .and_then(|v| v.to_str().ok())
                .map(String::from);
            let retry_after = parse_retry_after(response.headers());

            let error = self
                .parse_error_response(status, request_id, retry_after, response)
                .await?;

            Err(error.into())
        }
    }

    /// Makes an HTTP request with query parameters and retry logic.
    // `Q`/`B` are only bound by `Serialize`, not `Sync`, and `query`/`body`
    // are held across `.await` points across retry-loop iterations; see
    // `post` above for why this makes the returned future `!Send`.
    #[allow(clippy::future_not_send)]
    async fn request_with_query<T, Q, B>(
        &self,
        method: Method,
        path: &str,
        query: &Q,
        body: Option<&B>,
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        Q: serde::Serialize,
        B: serde::Serialize,
    {
        let mut url = self.build_url(path)?;

        // Serialize query parameters and add to URL
        let query_string = serde_urlencoded::to_string(query).map_err(|e| {
            Error::RequestBuild(format!("Failed to serialize query parameters: {e}"))
        })?;
        if !query_string.is_empty() {
            url.set_query(Some(&query_string));
        }

        let mut attempt = 0;

        loop {
            attempt += 1;
            trace!(method = %method, url = %url, attempt, "Making request with query");

            // Build the request
            let mut request_builder = self.inner.http_client.request(method.clone(), url.clone());

            // Add body if present
            if let Some(body) = body {
                request_builder = request_builder.json(body);
            }

            // Add retry count header
            if attempt > 1 {
                request_builder = request_builder.header("x-stainless-retry-count", attempt - 1);
            }

            // Send the request
            let result = request_builder.send().await;

            match result {
                Ok(response) => {
                    let status = response.status();
                    let request_id = response
                        .headers()
                        .get(HEADER_REQUEST_ID)
                        .and_then(|v| v.to_str().ok())
                        .map(String::from);
                    let retry_after = parse_retry_after(response.headers());

                    debug!(
                        status = %status,
                        request_id = ?request_id,
                        "Received response"
                    );

                    if status.is_success() {
                        // Parse successful response
                        let data = response.json::<T>().await?;
                        return Ok(data);
                    }

                    // Parse error response
                    let error = self
                        .parse_error_response(status, request_id, retry_after, response)
                        .await?;

                    // Check if we should retry
                    if error.is_retryable() {
                        if let Some(delay) = self
                            .inner
                            .retry_policy
                            .should_retry(&error.clone().into(), attempt)
                        {
                            debug!(delay = ?delay, attempt, "Retrying after error");
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                    }

                    return Err(error.into());
                }
                Err(err) => {
                    let error = Error::Http(err);

                    // Check if we should retry
                    if error.is_retryable() {
                        if let Some(delay) = self.inner.retry_policy.should_retry(&error, attempt) {
                            debug!(delay = ?delay, attempt, "Retrying after error");
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                    }

                    return Err(error);
                }
            }
        }
    }

    /// Makes an HTTP request with retry logic.
    // `B` is only bound by `Serialize`, not `Sync`, and `body` is held across
    // `.await` points across retry-loop iterations; see `post` above for why
    // this makes the returned future `!Send`.
    #[allow(clippy::future_not_send)]
    async fn request<T, B>(&self, method: Method, path: &str, body: Option<&B>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {
        self.request_with_headers(method, path, body, &[]).await
    }

    /// Core request path. `extra_headers` are applied to every attempt, including
    /// retries — a retried beta request must still carry its `anthropic-beta` header.
    // `B` is only bound by `Serialize`, not `Sync`; the `&B` held across the
    // `.await` points in the retry loop makes this future `!Send`. See `post`.
    #[allow(clippy::future_not_send)]
    async fn request_with_headers<T, B>(
        &self,
        method: Method,
        path: &str,
        body: Option<&B>,
        extra_headers: &[(&str, String)],
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {
        let url = self.build_url(path)?;
        let mut attempt = 0;

        loop {
            attempt += 1;
            trace!(method = %method, url = %url, attempt, "Making request");

            // Build the request
            let mut request_builder = self.inner.http_client.request(method.clone(), url.clone());

            // Add body if present
            if let Some(body) = body {
                request_builder = request_builder.json(body);
            }

            for (name, value) in extra_headers {
                request_builder = request_builder.header(*name, value);
            }

            // Add retry count header
            if attempt > 1 {
                request_builder = request_builder.header("x-stainless-retry-count", attempt - 1);
            }

            // Send the request
            let result = request_builder.send().await;

            match result {
                Ok(response) => {
                    let status = response.status();
                    let request_id = response
                        .headers()
                        .get(HEADER_REQUEST_ID)
                        .and_then(|v| v.to_str().ok())
                        .map(String::from);
                    let retry_after = parse_retry_after(response.headers());

                    debug!(
                        status = %status,
                        request_id = ?request_id,
                        "Received response"
                    );

                    if status.is_success() {
                        // Parse successful response
                        let data = response.json::<T>().await?;
                        return Ok(data);
                    }

                    // Parse error response
                    let error = self
                        .parse_error_response(status, request_id, retry_after, response)
                        .await?;

                    // Check if we should retry
                    if error.is_retryable() {
                        if let Some(delay) = self
                            .inner
                            .retry_policy
                            .should_retry(&error.clone().into(), attempt)
                        {
                            debug!(delay = ?delay, attempt, "Retrying after error");
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                    }

                    return Err(error.into());
                }
                Err(err) => {
                    let error = Error::Http(err);

                    // Check if we should retry
                    if error.is_retryable() {
                        if let Some(delay) = self.inner.retry_policy.should_retry(&error, attempt) {
                            debug!(delay = ?delay, attempt, "Retrying after error");
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                    }

                    return Err(error);
                }
            }
        }
    }

    /// Builds a full URL from a path.
    fn build_url(&self, path: &str) -> Result<Url> {
        let base = self.inner.config.base_url();
        let path = path.trim_start_matches('/');
        base.join(path).map_err(Error::Url)
    }

    /// Parses an error response from the API.
    async fn parse_error_response(
        &self,
        status: StatusCode,
        request_id: Option<String>,
        retry_after: Option<Duration>,
        response: reqwest::Response,
    ) -> Result<ApiError> {
        // Try to parse the error body
        let error_body = response.text().await.unwrap_or_default();

        let (error_type, message) =
            if let Ok(raw_error) = serde_json::from_str::<RawApiErrorResponse>(&error_body) {
                (raw_error.error.error_type, raw_error.error.message)
            } else {
                // Fallback: infer error type from status code
                let error_type = match status.as_u16() {
                    400 => ApiErrorType::InvalidRequestError,
                    401 => ApiErrorType::AuthenticationError,
                    403 => ApiErrorType::PermissionError,
                    404 => ApiErrorType::NotFoundError,
                    429 => ApiErrorType::RateLimitError,
                    529 => ApiErrorType::OverloadedError,
                    _ if status.is_server_error() => ApiErrorType::ApiError,
                    _ => ApiErrorType::Unknown,
                };
                let message = if error_body.is_empty() {
                    status
                        .canonical_reason()
                        .unwrap_or("Unknown error")
                        .to_string()
                } else {
                    error_body
                };
                (error_type, message)
            };

        let mut error = ApiError::new(status, error_type, message);

        if let Some(id) = request_id {
            error = error.with_request_id(id);
        }

        if let Some(duration) = retry_after {
            error = error.with_retry_after(duration);
        }

        Ok(error)
    }
}

// =============================================================================
// Anthropic Builder
// =============================================================================

/// Builder for creating an [`Anthropic`] client.
///
/// This provides a fluent interface for configuring all aspects of the client.
///
/// # Example
///
/// ```rust,no_run
/// use anthropic::Anthropic;
/// use std::time::Duration;
///
/// let client = Anthropic::builder()
///     .api_key("sk-ant-...")
///     .base_url("https://custom-api.example.com")
///     .timeout(Duration::from_secs(30))
///     .max_retries(5)
///     .header("X-Custom-Header", "value")
///     .build()?;
/// # Ok::<(), anthropic::Error>(())
/// ```
#[derive(Default)]
pub struct AnthropicBuilder {
    config_builder: ClientConfigBuilder,
}

impl std::fmt::Debug for AnthropicBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicBuilder")
            .field("config_builder", &"ClientConfigBuilder { ... }")
            .finish()
    }
}

impl AnthropicBuilder {
    /// Creates a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key for authentication.
    ///
    /// This is the primary authentication method. The key should start with `sk-ant-`.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config_builder.api_key(key);
        self
    }

    /// Sets the OAuth/JWT auth token for authentication.
    ///
    /// Alternative to API key authentication for specific use cases.
    #[must_use]
    pub fn auth_token(mut self, token: impl Into<String>) -> Self {
        self.config_builder.auth_token(token);
        self
    }

    /// Sets the base URL for API requests.
    ///
    /// Defaults to `https://api.anthropic.com`. Override for testing or proxies.
    #[must_use]
    pub fn base_url(mut self, url: impl AsRef<str>) -> Self {
        if let Ok(parsed) = Url::parse(url.as_ref()) {
            self.config_builder.base_url(parsed);
        }
        self
    }

    /// Sets the maximum number of retry attempts.
    ///
    /// Defaults to 2. Set to 0 to disable retries.
    #[must_use]
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config_builder.max_retries(retries);
        self
    }

    /// Sets the request timeout.
    ///
    /// Defaults to 120 seconds.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config_builder.timeout(timeout);
        self
    }

    /// Adds a custom header to all requests.
    #[must_use]
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        self.config_builder.header(name, value);
        self
    }

    /// Loads configuration from environment variables.
    ///
    /// This reads `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_BASE_URL`.
    /// Values set explicitly via builder methods take precedence.
    #[must_use]
    pub fn from_env(mut self) -> Self {
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            if !key.is_empty() {
                self.config_builder.api_key(key);
            }
        }
        if let Ok(token) = std::env::var("ANTHROPIC_AUTH_TOKEN") {
            if !token.is_empty() {
                self.config_builder.auth_token(token);
            }
        }
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            if let Ok(parsed) = Url::parse(&url) {
                self.config_builder.base_url(parsed);
            }
        }
        self
    }

    /// Builds the Anthropic client.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Neither API key nor auth token is configured
    /// - The HTTP client fails to initialize
    pub fn build(self) -> Result<Anthropic> {
        let config = self.config_builder.build()?;
        Anthropic::with_config(config)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_requires_auth() {
        let result = Anthropic::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_api_key() {
        let result = Anthropic::builder().api_key("sk-ant-test-key").build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_auth_token() {
        let result = Anthropic::builder().auth_token("test-token").build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_custom_config() {
        let client = Anthropic::builder()
            .api_key("sk-ant-test-key")
            .base_url("https://custom.example.com")
            .max_retries(5)
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build client");

        assert_eq!(client.base_url().as_str(), "https://custom.example.com/");
        assert_eq!(client.max_retries(), 5);
        assert_eq!(client.timeout(), Duration::from_secs(30));
    }

    #[test]
    fn test_build_url() {
        let client = Anthropic::builder()
            .api_key("test-key")
            .build()
            .expect("Failed to build client");

        let url = client
            .build_url("/v1/messages")
            .expect("Failed to build URL");
        assert_eq!(url.as_str(), "https://api.anthropic.com/v1/messages");

        let url = client
            .build_url("v1/messages")
            .expect("Failed to build URL");
        assert_eq!(url.as_str(), "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn test_client_is_clone() {
        let client = Anthropic::builder()
            .api_key("test-key")
            .build()
            .expect("Failed to build client");

        let cloned = client.clone();
        assert_eq!(client.base_url(), cloned.base_url());
    }

    #[test]
    fn test_constants() {
        assert_eq!(API_VERSION, "2023-06-01");
        assert_eq!(HEADER_ANTHROPIC_VERSION, "anthropic-version");
        assert_eq!(HEADER_API_KEY, "x-api-key");
        assert!(USER_AGENT.starts_with("anthropic-rust/"));
    }
}
