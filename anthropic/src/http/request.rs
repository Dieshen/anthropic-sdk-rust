//! Request building infrastructure for the Anthropic API.
//!
//! This module provides a [`RequestBuilder`] that wraps `reqwest::RequestBuilder`
//! and adds Anthropic-specific functionality like automatic header injection.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::http::RequestBuilder;
//!
//! let request = RequestBuilder::new(client, "POST", "https://api.anthropic.com/v1/messages")
//!     .json_body(&message_params)?
//!     .add_anthropic_headers("sk-ant-...", "2023-06-01")
//!     .header("X-Custom-Header", "value")
//!     .build()?;
//! ```

use crate::error::{Error, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use reqwest::{Client, Method, Url};
use serde::Serialize;
use std::str::FromStr;

/// Header name for the API key.
pub const HEADER_API_KEY: &str = "x-api-key";
/// Header name for the API version.
pub const HEADER_API_VERSION: &str = "anthropic-version";
/// Header name for beta features.
pub const HEADER_BETA: &str = "anthropic-beta";
/// Default content type for JSON requests.
pub const CONTENT_TYPE_JSON: &str = "application/json";

/// Builder for constructing HTTP requests with Anthropic-specific configuration.
///
/// This builder wraps `reqwest::RequestBuilder` and provides a fluent API for
/// constructing requests with proper headers, query parameters, and body content.
#[derive(Debug)]
pub struct RequestBuilder {
    client: Client,
    method: Method,
    url: Url,
    headers: HeaderMap,
    query: Vec<(String, String)>,
    body: Option<Vec<u8>>,
}

impl RequestBuilder {
    /// Creates a new request builder.
    ///
    /// # Arguments
    ///
    /// * `client` - The reqwest client to use for building requests
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `url` - The full URL for the request
    ///
    /// # Errors
    ///
    /// Returns an error if the method string is invalid.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = RequestBuilder::new(client, "POST", "https://api.anthropic.com/v1/messages")?;
    /// ```
    pub fn new(client: Client, method: &str, url: Url) -> Result<Self> {
        let method = Method::from_str(method)
            .map_err(|e| Error::request_build(format!("Invalid HTTP method: {e}")))?;

        Ok(Self {
            client,
            method,
            url,
            headers: HeaderMap::new(),
            query: Vec::new(),
            body: None,
        })
    }

    /// Creates a new request builder with a parsed method.
    ///
    /// This is a convenience constructor when you already have a `Method` instance.
    #[must_use]
    pub fn with_method(client: Client, method: Method, url: Url) -> Self {
        Self {
            client,
            method,
            url,
            headers: HeaderMap::new(),
            query: Vec::new(),
            body: None,
        }
    }

    /// Sets the HTTP method.
    #[must_use]
    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Appends a path segment to the URL.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // URL: https://api.anthropic.com/v1
    /// builder.path("messages")
    /// // Result: https://api.anthropic.com/v1/messages
    /// ```
    #[must_use]
    pub fn path(mut self, path: &str) -> Self {
        // Ensure we don't double-slash
        let base = self.url.as_str().trim_end_matches('/');
        let path = path.trim_start_matches('/');
        if let Ok(new_url) = Url::parse(&format!("{base}/{path}")) {
            self.url = new_url;
        }
        self
    }

    /// Adds a query parameter to the request.
    ///
    /// Multiple calls will append additional parameters.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder
    ///     .query("limit", "10")
    ///     .query("offset", "0")
    /// ```
    #[must_use]
    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into()));
        self
    }

    /// Adds multiple query parameters from an iterator.
    #[must_use]
    pub fn query_pairs<I, K, V>(mut self, pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (key, value) in pairs {
            self.query.push((key.into(), value.into()));
        }
        self
    }

    /// Sets the request body as JSON.
    ///
    /// This method serializes the provided value to JSON and sets the
    /// appropriate Content-Type header.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// #[derive(Serialize)]
    /// struct MessageParams {
    ///     model: String,
    ///     max_tokens: u32,
    /// }
    ///
    /// let params = MessageParams {
    ///     model: "claude-sonnet-4-20250514".to_string(),
    ///     max_tokens: 1024,
    /// };
    ///
    /// builder.json_body(&params)?
    /// ```
    pub fn json_body<T: Serialize>(mut self, body: &T) -> Result<Self> {
        let json = serde_json::to_vec(body)?;
        self.body = Some(json);
        self.headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static(CONTENT_TYPE_JSON),
        );
        Ok(self)
    }

    /// Sets the request body as raw bytes.
    #[must_use]
    pub fn body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Adds a single header to the request.
    ///
    /// # Errors
    ///
    /// Returns an error if the header name or value is invalid.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.header("X-Custom-Header", "custom-value")?
    /// ```
    pub fn header(mut self, name: &str, value: &str) -> Result<Self> {
        let header_name = HeaderName::from_str(name)?;
        let header_value = HeaderValue::from_str(value)?;
        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Adds a header with a static string value.
    ///
    /// This is more efficient than `header()` when the value is a compile-time constant.
    #[must_use]
    pub fn header_static(mut self, name: HeaderName, value: &'static str) -> Self {
        self.headers.insert(name, HeaderValue::from_static(value));
        self
    }

    /// Adds multiple headers from a `HeaderMap`.
    #[must_use]
    pub fn headers(mut self, headers: HeaderMap) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Adds multiple headers from an iterator of tuples.
    ///
    /// # Errors
    ///
    /// Returns an error if any header name or value is invalid.
    pub fn headers_iter<I>(mut self, headers: I) -> Result<Self>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        for (name, value) in headers {
            let header_name = HeaderName::from_str(&name)?;
            let header_value = HeaderValue::from_str(&value)?;
            self.headers.insert(header_name, header_value);
        }
        Ok(self)
    }

    /// Adds the standard Anthropic API headers.
    ///
    /// This method adds:
    /// - `x-api-key`: The API key for authentication
    /// - `anthropic-version`: The API version being used
    /// - `Content-Type`: application/json (if not already set)
    ///
    /// # Errors
    ///
    /// Returns an error if the API key contains invalid characters.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.add_anthropic_headers("sk-ant-api-key", "2023-06-01")?
    /// ```
    pub fn add_anthropic_headers(mut self, api_key: &str, version: &str) -> Result<Self> {
        let api_key_value = HeaderValue::from_str(api_key)?;
        let version_value = HeaderValue::from_str(version)?;

        self.headers.insert(
            HeaderName::from_static(HEADER_API_KEY),
            api_key_value,
        );
        self.headers.insert(
            HeaderName::from_static(HEADER_API_VERSION),
            version_value,
        );

        // Set content type if not already set
        if !self.headers.contains_key(CONTENT_TYPE) {
            self.headers.insert(
                CONTENT_TYPE,
                HeaderValue::from_static(CONTENT_TYPE_JSON),
            );
        }

        Ok(self)
    }

    /// Adds beta feature headers.
    ///
    /// Beta features are enabled by passing their feature names.
    /// Multiple features can be enabled by calling this method multiple times
    /// or by passing a comma-separated list.
    ///
    /// # Errors
    ///
    /// Returns an error if the beta feature name contains invalid characters.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.add_beta_header("prompt-caching-2024-07-31")?
    /// ```
    pub fn add_beta_header(mut self, beta_features: &str) -> Result<Self> {
        let header_name = HeaderName::from_static(HEADER_BETA);

        if let Some(existing) = self.headers.get(&header_name) {
            // Append to existing beta features
            let existing_str = existing.to_str().unwrap_or("");
            let combined = format!("{existing_str},{beta_features}");
            let header_value = HeaderValue::from_str(&combined)?;
            self.headers.insert(header_name, header_value);
        } else {
            let header_value = HeaderValue::from_str(beta_features)?;
            self.headers.insert(header_name, header_value);
        }

        Ok(self)
    }

    /// Builds the final `reqwest::Request`.
    ///
    /// # Errors
    ///
    /// Returns an error if the request cannot be built (e.g., invalid URL with query params).
    pub fn build(self) -> Result<reqwest::Request> {
        let mut url = self.url;

        // Add query parameters
        if !self.query.is_empty() {
            let mut query_pairs = url.query_pairs_mut();
            for (key, value) in &self.query {
                query_pairs.append_pair(key, value);
            }
            drop(query_pairs);
        }

        // Build the request
        let mut builder = self.client.request(self.method, url).headers(self.headers);

        if let Some(body) = self.body {
            builder = builder.body(body);
        }

        builder
            .build()
            .map_err(|e| Error::request_build(format!("Failed to build request: {e}")))
    }

    /// Returns a reference to the current URL.
    #[must_use]
    pub fn url(&self) -> &Url {
        &self.url
    }

    /// Returns a reference to the current method.
    #[must_use]
    pub fn get_method(&self) -> &Method {
        &self.method
    }

    /// Returns a reference to the current headers.
    #[must_use]
    pub fn get_headers(&self) -> &HeaderMap {
        &self.headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client() -> Client {
        Client::new()
    }

    fn test_url() -> Url {
        Url::parse("https://api.anthropic.com/v1").unwrap()
    }

    #[test]
    fn test_request_builder_basic() {
        let builder = RequestBuilder::new(test_client(), "POST", test_url()).unwrap();
        assert_eq!(builder.get_method(), Method::POST);
        assert_eq!(builder.url().as_str(), "https://api.anthropic.com/v1");
    }

    #[test]
    fn test_request_builder_path() {
        let builder = RequestBuilder::new(test_client(), "POST", test_url())
            .unwrap()
            .path("messages");
        assert_eq!(builder.url().as_str(), "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn test_request_builder_query() {
        let builder = RequestBuilder::new(test_client(), "GET", test_url())
            .unwrap()
            .query("limit", "10")
            .query("offset", "0");

        let request = builder.build().unwrap();
        let url = request.url();
        assert!(url.query().unwrap().contains("limit=10"));
        assert!(url.query().unwrap().contains("offset=0"));
    }

    #[test]
    fn test_request_builder_headers() {
        let builder = RequestBuilder::new(test_client(), "POST", test_url())
            .unwrap()
            .header("X-Custom", "value")
            .unwrap();

        let request = builder.build().unwrap();
        assert_eq!(
            request.headers().get("X-Custom").unwrap().to_str().unwrap(),
            "value"
        );
    }

    #[test]
    fn test_anthropic_headers() {
        let builder = RequestBuilder::new(test_client(), "POST", test_url())
            .unwrap()
            .add_anthropic_headers("test-api-key", "2023-06-01")
            .unwrap();

        let request = builder.build().unwrap();
        assert_eq!(
            request.headers().get(HEADER_API_KEY).unwrap().to_str().unwrap(),
            "test-api-key"
        );
        assert_eq!(
            request.headers().get(HEADER_API_VERSION).unwrap().to_str().unwrap(),
            "2023-06-01"
        );
    }

    #[test]
    fn test_beta_headers() {
        let builder = RequestBuilder::new(test_client(), "POST", test_url())
            .unwrap()
            .add_beta_header("feature1")
            .unwrap()
            .add_beta_header("feature2")
            .unwrap();

        let request = builder.build().unwrap();
        let beta_header = request.headers().get(HEADER_BETA).unwrap().to_str().unwrap();
        assert!(beta_header.contains("feature1"));
        assert!(beta_header.contains("feature2"));
    }

    #[test]
    fn test_json_body() {
        #[derive(serde::Serialize)]
        struct TestBody {
            field: String,
        }

        let body = TestBody {
            field: "value".to_string(),
        };

        let builder = RequestBuilder::new(test_client(), "POST", test_url())
            .unwrap()
            .json_body(&body)
            .unwrap();

        let request = builder.build().unwrap();
        assert_eq!(
            request.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap(),
            CONTENT_TYPE_JSON
        );
    }

    #[test]
    fn test_invalid_method() {
        // HTTP methods must be valid tokens - spaces are not allowed
        let result = RequestBuilder::new(test_client(), "INVALID METHOD", test_url());
        assert!(result.is_err());

        // Empty string should also fail
        let result = RequestBuilder::new(test_client(), "", test_url());
        assert!(result.is_err());
    }
}
