//! Client configuration for the Anthropic SDK.
//!
//! This module provides configuration options for the Anthropic API client,
//! including authentication, base URL, retry behavior, and timeouts.
//!
//! # Example
//!
//! ```rust,no_run
//! use anthropic::config::{ClientConfig, ClientConfigBuilder};
//! use std::time::Duration;
//!
//! // Build configuration with explicit values
//! let config = ClientConfigBuilder::default()
//!     .api_key("sk-ant-...")
//!     .timeout(Duration::from_secs(60))
//!     .max_retries(3u32)
//!     .build()
//!     .expect("Failed to build config");
//!
//! // Or load from environment variables
//! let config = ClientConfig::from_env()
//!     .expect("Failed to load config from environment");
//! ```

use derive_builder::Builder;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use std::time::Duration;
use url::Url;

use crate::{Error, Result};

// =============================================================================
// Constants
// =============================================================================

/// Default base URL for the Anthropic API.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Default maximum number of retry attempts for failed requests.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// Default request timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Environment variable name for the API key.
pub const ENV_API_KEY: &str = "ANTHROPIC_API_KEY";

/// Environment variable name for the auth token (OAuth/JWT).
pub const ENV_AUTH_TOKEN: &str = "ANTHROPIC_AUTH_TOKEN";

/// Environment variable name for the base URL override.
pub const ENV_BASE_URL: &str = "ANTHROPIC_BASE_URL";

// =============================================================================
// ClientConfig
// =============================================================================

/// Configuration for the Anthropic API client.
///
/// This struct holds all configuration options needed to communicate with
/// the Anthropic API. It supports both API key authentication and OAuth/JWT
/// token authentication.
///
/// # Authentication
///
/// The client supports two authentication methods:
///
/// 1. **API Key**: Set via `api_key` field or `ANTHROPIC_API_KEY` environment variable.
///    This is the most common authentication method for direct API access.
///
/// 2. **Auth Token**: Set via `auth_token` field or `ANTHROPIC_AUTH_TOKEN` environment variable.
///    Used for OAuth/JWT-based authentication flows.
///
/// At least one authentication method must be configured.
///
/// # Example
///
/// ```rust,no_run
/// use anthropic::config::ClientConfig;
///
/// // Load from environment (recommended for production)
/// let config = ClientConfig::from_env()?;
///
/// // Or use the builder for explicit configuration
/// use anthropic::config::ClientConfigBuilder;
///
/// let config = ClientConfigBuilder::default()
///     .api_key("sk-ant-api03-...")
///     .build()?;
/// # Ok::<(), anthropic::Error>(())
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(
    name = "ClientConfigBuilder",
    setter(into, strip_option),
    build_fn(validate = "Self::validate", error = "crate::Error")
)]
pub struct ClientConfig {
    /// API key for authentication.
    ///
    /// This is the primary authentication method for the Anthropic API.
    /// Keys typically start with `sk-ant-`.
    #[builder(default, setter(custom))]
    api_key: Option<SecretString>,

    /// OAuth/JWT auth token for authentication.
    ///
    /// Alternative authentication method using OAuth or JWT tokens.
    /// Used in specific authentication flows where API keys are not appropriate.
    #[builder(default, setter(custom))]
    auth_token: Option<SecretString>,

    /// Base URL for API requests.
    ///
    /// Defaults to `https://api.anthropic.com`. Override this for testing
    /// or when using a proxy.
    #[builder(default = "Self::default_base_url()")]
    base_url: Url,

    /// Maximum number of retry attempts for failed requests.
    ///
    /// Defaults to 2. Set to 0 to disable retries.
    /// Retries use exponential backoff with jitter.
    #[builder(default = "DEFAULT_MAX_RETRIES")]
    max_retries: u32,

    /// Request timeout duration.
    ///
    /// Defaults to 120 seconds. This is the total time allowed for a request,
    /// including connection, sending, and receiving.
    #[builder(default = "Duration::from_secs(DEFAULT_TIMEOUT_SECS)")]
    timeout: Duration,

    /// Additional headers to include in all requests.
    ///
    /// These headers are merged with the default headers (authentication,
    /// content-type, etc.). Custom headers take precedence over defaults.
    #[builder(default, setter(custom))]
    default_headers: HeaderMap,
}

impl ClientConfigBuilder {
    /// Sets the API key for authentication.
    ///
    /// The key will be stored securely using `SecretString` to prevent
    /// accidental exposure in logs or debug output.
    ///
    /// # Arguments
    ///
    /// * `key` - The API key, typically starting with `sk-ant-`
    ///
    /// # Example
    ///
    /// ```rust
    /// use anthropic::config::ClientConfigBuilder;
    ///
    /// let builder = ClientConfigBuilder::default()
    ///     .api_key("sk-ant-api03-...");
    /// ```
    pub fn api_key(&mut self, key: impl Into<String>) -> &mut Self {
        self.api_key = Some(Some(SecretString::from(key.into())));
        self
    }

    /// Sets the OAuth/JWT auth token for authentication.
    ///
    /// The token will be stored securely using `SecretString` to prevent
    /// accidental exposure in logs or debug output.
    ///
    /// # Arguments
    ///
    /// * `token` - The authentication token
    ///
    /// # Example
    ///
    /// ```rust
    /// use anthropic::config::ClientConfigBuilder;
    ///
    /// let builder = ClientConfigBuilder::default()
    ///     .auth_token("eyJhbGciOiJSUzI1NiIs...");
    /// ```
    pub fn auth_token(&mut self, token: impl Into<String>) -> &mut Self {
        self.auth_token = Some(Some(SecretString::from(token.into())));
        self
    }

    /// Adds a custom header to include in all requests.
    ///
    /// Headers added here will be included in every API request.
    /// If the same header is added multiple times, the last value wins.
    ///
    /// # Arguments
    ///
    /// * `name` - The header name
    /// * `value` - The header value
    ///
    /// # Example
    ///
    /// ```rust
    /// use anthropic::config::ClientConfigBuilder;
    ///
    /// let builder = ClientConfigBuilder::default()
    ///     .api_key("sk-ant-...")
    ///     .header("X-Custom-Header", "custom-value");
    /// ```
    pub fn header(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> &mut Self {
        let headers = self.default_headers.get_or_insert_with(HeaderMap::new);
        if let (Ok(name), Ok(value)) = (
            HeaderName::try_from(name.as_ref()),
            HeaderValue::try_from(value.as_ref()),
        ) {
            headers.insert(name, value);
        }
        self
    }

    /// Returns the default base URL as a `Url`.
    fn default_base_url() -> Url {
        // This should never fail for a hardcoded valid URL
        Url::parse(DEFAULT_BASE_URL).expect("Invalid default base URL")
    }

    /// Validates the configuration before building.
    ///
    /// Ensures that at least one authentication method is configured.
    fn validate(&self) -> std::result::Result<(), Error> {
        let has_api_key = self
            .api_key
            .as_ref()
            .is_some_and(std::option::Option::is_some);

        let has_auth_token = self
            .auth_token
            .as_ref()
            .is_some_and(std::option::Option::is_some);

        if !has_api_key && !has_auth_token {
            return Err(Error::config(
                "Either api_key or auth_token must be provided",
            ));
        }

        Ok(())
    }
}

impl ClientConfig {
    /// Creates a new `ClientConfigBuilder`.
    ///
    /// This is a convenience method equivalent to `ClientConfigBuilder::default()`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anthropic::config::ClientConfig;
    ///
    /// let config = ClientConfig::builder()
    ///     .api_key("sk-ant-...")
    ///     .build()?;
    /// # Ok::<(), anthropic::Error>(())
    /// ```
    #[must_use]
    pub fn builder() -> ClientConfigBuilder {
        ClientConfigBuilder::default()
    }

    /// Creates a configuration by reading from environment variables.
    ///
    /// This method reads the following environment variables:
    ///
    /// - `ANTHROPIC_API_KEY`: API key for authentication
    /// - `ANTHROPIC_AUTH_TOKEN`: OAuth/JWT token for authentication
    /// - `ANTHROPIC_BASE_URL`: Base URL override (optional)
    ///
    /// At least one of `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` must be set.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Neither `ANTHROPIC_API_KEY` nor `ANTHROPIC_AUTH_TOKEN` is set
    /// - `ANTHROPIC_BASE_URL` is set but contains an invalid URL
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use anthropic::config::ClientConfig;
    ///
    /// // Ensure ANTHROPIC_API_KEY is set in your environment
    /// let config = ClientConfig::from_env()?;
    /// # Ok::<(), anthropic::Error>(())
    /// ```
    pub fn from_env() -> Result<Self> {
        let mut builder = ClientConfigBuilder::default();

        // Read API key from environment
        if let Ok(key) = std::env::var(ENV_API_KEY) {
            if !key.is_empty() {
                builder.api_key(key);
            }
        }

        // Read auth token from environment
        if let Ok(token) = std::env::var(ENV_AUTH_TOKEN) {
            if !token.is_empty() {
                builder.auth_token(token);
            }
        }

        // Read base URL from environment
        if let Ok(url_str) = std::env::var(ENV_BASE_URL) {
            if !url_str.is_empty() {
                let url = Url::parse(&url_str)
                    .map_err(|e| Error::config(format!("Invalid base URL '{url_str}': {e}")))?;
                builder.base_url(url);
            }
        }

        builder.build()
    }

    /// Returns the API key as a string slice, if configured.
    ///
    /// This method exposes the secret for use in HTTP headers.
    /// Use with caution and avoid logging the returned value.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use anthropic::config::ClientConfig;
    ///
    /// let config = ClientConfig::from_env()?;
    /// if let Some(key) = config.api_key() {
    ///     println!("API key is configured (length: {})", key.len());
    /// }
    /// # Ok::<(), anthropic::Error>(())
    /// ```
    #[must_use]
    pub fn api_key(&self) -> Option<&str> {
        self.api_key
            .as_ref()
            .map(secrecy::ExposeSecret::expose_secret)
    }

    /// Returns the auth token as a string slice, if configured.
    ///
    /// This method exposes the secret for use in HTTP headers.
    /// Use with caution and avoid logging the returned value.
    #[must_use]
    pub fn auth_token(&self) -> Option<&str> {
        self.auth_token
            .as_ref()
            .map(secrecy::ExposeSecret::expose_secret)
    }

    /// Returns the authentication header name and value.
    ///
    /// This method returns the appropriate header for authentication:
    /// - If an API key is configured: `("x-api-key", "<api_key>")`
    /// - If an auth token is configured: `("Authorization", "Bearer <token>")`
    /// - API key takes precedence if both are configured
    ///
    /// # Returns
    ///
    /// A tuple of (`header_name`, `header_value`) if authentication is configured,
    /// or `None` if no authentication is available.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use anthropic::config::ClientConfig;
    ///
    /// let config = ClientConfig::from_env()?;
    /// if let Some((name, value)) = config.auth_header() {
    ///     println!("Using {} header for authentication", name);
    /// }
    /// # Ok::<(), anthropic::Error>(())
    /// ```
    #[must_use]
    pub fn auth_header(&self) -> Option<(&'static str, String)> {
        // API key takes precedence
        if let Some(ref api_key) = self.api_key {
            return Some(("x-api-key", api_key.expose_secret().to_string()));
        }

        // Fall back to auth token (Bearer token)
        if let Some(ref auth_token) = self.auth_token {
            return Some((
                "Authorization",
                format!("Bearer {}", auth_token.expose_secret()),
            ));
        }

        None
    }

    /// Returns the base URL for API requests.
    #[must_use]
    pub const fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Returns the maximum number of retry attempts.
    #[must_use]
    pub const fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Returns the request timeout duration.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the default headers to include in all requests.
    #[must_use]
    pub const fn default_headers(&self) -> &HeaderMap {
        &self.default_headers
    }
}

impl Default for ClientConfig {
    /// Creates a default configuration by reading from environment variables.
    ///
    /// # Panics
    ///
    /// Panics if environment configuration is invalid. For fallible construction,
    /// use [`ClientConfig::from_env()`] instead.
    fn default() -> Self {
        Self::from_env().expect(
            "Failed to create ClientConfig from environment. \
             Ensure ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is set.",
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_BASE_URL, "https://api.anthropic.com");
        assert_eq!(DEFAULT_MAX_RETRIES, 2);
        assert_eq!(DEFAULT_TIMEOUT_SECS, 120);
        assert_eq!(ENV_API_KEY, "ANTHROPIC_API_KEY");
        assert_eq!(ENV_AUTH_TOKEN, "ANTHROPIC_AUTH_TOKEN");
        assert_eq!(ENV_BASE_URL, "ANTHROPIC_BASE_URL");
    }

    #[test]
    fn test_builder_with_api_key() {
        let config = ClientConfigBuilder::default()
            .api_key("sk-ant-test-key")
            .build()
            .expect("Failed to build config");

        assert_eq!(config.api_key(), Some("sk-ant-test-key"));
        assert_eq!(config.auth_token(), None);
        assert_eq!(config.base_url().as_str(), "https://api.anthropic.com/");
        assert_eq!(config.max_retries(), 2);
        assert_eq!(config.timeout(), Duration::from_secs(120));
    }

    #[test]
    fn test_builder_with_auth_token() {
        let config = ClientConfigBuilder::default()
            .auth_token("eyJhbGciOiJSUzI1NiIs")
            .build()
            .expect("Failed to build config");

        assert_eq!(config.api_key(), None);
        assert_eq!(config.auth_token(), Some("eyJhbGciOiJSUzI1NiIs"));
    }

    #[test]
    fn test_builder_requires_authentication() {
        let result = ClientConfigBuilder::default().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_custom_values() {
        let config = ClientConfigBuilder::default()
            .api_key("test-key")
            .base_url(Url::parse("https://custom.example.com").unwrap())
            .max_retries(5u32)
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build config");

        assert_eq!(config.base_url().as_str(), "https://custom.example.com/");
        assert_eq!(config.max_retries(), 5);
        assert_eq!(config.timeout(), Duration::from_secs(30));
    }

    #[test]
    fn test_builder_with_custom_header() {
        let config = ClientConfigBuilder::default()
            .api_key("test-key")
            .header("X-Custom-Header", "custom-value")
            .header("X-Another-Header", "another-value")
            .build()
            .expect("Failed to build config");

        let headers = config.default_headers();
        assert_eq!(
            headers.get("X-Custom-Header").map(|v| v.to_str().unwrap()),
            Some("custom-value")
        );
        assert_eq!(
            headers.get("X-Another-Header").map(|v| v.to_str().unwrap()),
            Some("another-value")
        );
    }

    #[test]
    fn test_auth_header_with_api_key() {
        let config = ClientConfigBuilder::default()
            .api_key("sk-ant-test")
            .build()
            .expect("Failed to build config");

        let (name, value) = config.auth_header().expect("Should have auth header");
        assert_eq!(name, "x-api-key");
        assert_eq!(value, "sk-ant-test");
    }

    #[test]
    fn test_auth_header_with_auth_token() {
        let config = ClientConfigBuilder::default()
            .auth_token("my-token")
            .build()
            .expect("Failed to build config");

        let (name, value) = config.auth_header().expect("Should have auth header");
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my-token");
    }

    #[test]
    fn test_auth_header_api_key_takes_precedence() {
        let config = ClientConfigBuilder::default()
            .api_key("api-key")
            .auth_token("auth-token")
            .build()
            .expect("Failed to build config");

        let (name, value) = config.auth_header().expect("Should have auth header");
        assert_eq!(name, "x-api-key");
        assert_eq!(value, "api-key");
    }

    #[test]
    fn test_client_config_builder_method() {
        let config = ClientConfig::builder()
            .api_key("test-key")
            .build()
            .expect("Failed to build config");

        assert_eq!(config.api_key(), Some("test-key"));
    }

    #[test]
    fn test_config_debug_does_not_expose_secrets() {
        let config = ClientConfigBuilder::default()
            .api_key("super-secret-key")
            .build()
            .expect("Failed to build config");

        let debug_output = format!("{config:?}");

        // The debug output should not contain the actual secret
        assert!(!debug_output.contains("super-secret-key"));
    }
}
