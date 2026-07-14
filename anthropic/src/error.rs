//! Error types for the Anthropic SDK.
//!
//! This module provides comprehensive error handling for all SDK operations,
//! including API errors, network errors, and validation errors.

use reqwest::StatusCode;
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Result type alias for SDK operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The primary error type for all SDK operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// API error returned by the Anthropic API.
    #[error("API error: {0}")]
    Api(#[from] ApiError),

    /// HTTP/network error from the underlying HTTP client.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// URL parsing error.
    #[error("URL error: {0}")]
    Url(#[from] url::ParseError),

    /// Request building error.
    #[error("Request build error: {0}")]
    RequestBuild(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Authentication error (missing or invalid API key).
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Timeout error.
    #[error("Request timed out after {0:?}")]
    Timeout(Duration),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded, retry after: {retry_after:?}")]
    RateLimited {
        /// Duration to wait before retrying.
        retry_after: Option<Duration>,
        /// The underlying API error.
        error: Box<ApiError>,
    },

    /// Streaming error.
    #[error("Streaming error: {0}")]
    Streaming(String),

    /// Invalid header value.
    #[error("Invalid header value: {0}")]
    InvalidHeader(#[from] reqwest::header::InvalidHeaderValue),

    /// Invalid header name.
    #[error("Invalid header name: {0}")]
    InvalidHeaderName(#[from] reqwest::header::InvalidHeaderName),
}

impl Error {
    /// Creates a new configuration error.
    #[must_use]
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    /// Creates a new authentication error.
    #[must_use]
    pub fn auth(message: impl Into<String>) -> Self {
        Self::Authentication(message.into())
    }

    /// Creates a new request build error.
    #[must_use]
    pub fn request_build(message: impl Into<String>) -> Self {
        Self::RequestBuild(message.into())
    }

    /// Creates a new streaming error.
    #[must_use]
    pub fn streaming(message: impl Into<String>) -> Self {
        Self::Streaming(message.into())
    }

    /// Returns `true` if this error is retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Api(api_err) => api_err.is_retryable(),
            Self::Http(http_err) => {
                // Retry on connection errors and timeouts
                http_err.is_connect() || http_err.is_timeout()
            }
            Self::RateLimited { .. } => true,
            Self::Timeout(_) => true,
            _ => false,
        }
    }

    /// Returns the HTTP status code if this is an API error.
    #[must_use]
    pub fn status_code(&self) -> Option<StatusCode> {
        match self {
            Self::Api(api_err) => Some(api_err.status),
            Self::RateLimited { error, .. } => Some(error.status),
            Self::Http(http_err) => http_err.status(),
            _ => None,
        }
    }

    /// Returns the request ID if available.
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Api(api_err) => api_err.request_id.as_deref(),
            Self::RateLimited { error, .. } => error.request_id.as_deref(),
            _ => None,
        }
    }

    /// Returns the retry-after duration if this is a rate limit error.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after, .. } => *retry_after,
            Self::Api(api_err) => api_err.retry_after,
            _ => None,
        }
    }
}

/// Error response from the Anthropic API.
#[derive(Debug, Clone)]
pub struct ApiError {
    /// HTTP status code.
    pub status: StatusCode,
    /// Error type from the API.
    pub error_type: ApiErrorType,
    /// Human-readable error message.
    pub message: String,
    /// Request ID for debugging.
    pub request_id: Option<String>,
    /// Retry-after duration if rate limited.
    pub retry_after: Option<Duration>,
    /// Additional error details.
    pub details: Option<HashMap<String, serde_json::Value>>,
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (status: {}, type: {})",
            self.message, self.status, self.error_type
        )?;
        if let Some(ref request_id) = self.request_id {
            write!(f, " [request_id: {request_id}]")?;
        }
        Ok(())
    }
}

impl std::error::Error for ApiError {}

impl ApiError {
    /// Creates a new API error.
    #[must_use]
    pub fn new(status: StatusCode, error_type: ApiErrorType, message: impl Into<String>) -> Self {
        Self {
            status,
            error_type,
            message: message.into(),
            request_id: None,
            retry_after: None,
            details: None,
        }
    }

    /// Sets the request ID.
    #[must_use]
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Sets the retry-after duration.
    #[must_use]
    pub fn with_retry_after(mut self, retry_after: Duration) -> Self {
        self.retry_after = Some(retry_after);
        self
    }

    /// Sets additional error details.
    #[must_use]
    pub fn with_details(mut self, details: HashMap<String, serde_json::Value>) -> Self {
        self.details = Some(details);
        self
    }

    /// Returns `true` if this error is retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.error_type,
            ApiErrorType::RateLimitError | ApiErrorType::OverloadedError
        ) || self.status.is_server_error()
    }

    /// Returns `true` if this is a rate limit error.
    #[must_use]
    pub fn is_rate_limited(&self) -> bool {
        self.error_type == ApiErrorType::RateLimitError
            || self.status == StatusCode::TOO_MANY_REQUESTS
    }

    /// Returns `true` if this is an overloaded error.
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        self.error_type == ApiErrorType::OverloadedError
    }

    /// Returns `true` if this is an authentication error.
    #[must_use]
    pub fn is_auth_error(&self) -> bool {
        self.error_type == ApiErrorType::AuthenticationError
            || self.status == StatusCode::UNAUTHORIZED
    }

    /// Returns `true` if this is a permission error.
    #[must_use]
    pub fn is_permission_error(&self) -> bool {
        self.error_type == ApiErrorType::PermissionError || self.status == StatusCode::FORBIDDEN
    }
}

/// Error types returned by the Anthropic API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorType {
    /// Invalid request parameters.
    InvalidRequestError,
    /// Authentication failure.
    AuthenticationError,
    /// Permission denied.
    PermissionError,
    /// Resource not found.
    NotFoundError,
    /// Rate limit exceeded.
    RateLimitError,
    /// API is overloaded.
    OverloadedError,
    /// Internal server error.
    ApiError,
    /// Unknown error type.
    #[serde(other)]
    Unknown,
}

impl fmt::Display for ApiErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::InvalidRequestError => "invalid_request_error",
            Self::AuthenticationError => "authentication_error",
            Self::PermissionError => "permission_error",
            Self::NotFoundError => "not_found_error",
            Self::RateLimitError => "rate_limit_error",
            Self::OverloadedError => "overloaded_error",
            Self::ApiError => "api_error",
            Self::Unknown => "unknown",
        };
        write!(f, "{s}")
    }
}

impl Default for ApiErrorType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Raw error response structure from the API.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct RawApiErrorResponse {
    /// The response type (usually "error").
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub response_type: Option<String>,
    /// The actual error details.
    pub error: RawApiError,
}

/// Raw error details from the API.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct RawApiError {
    #[serde(rename = "type")]
    pub error_type: ApiErrorType,
    pub message: String,
    #[serde(flatten)]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_display() {
        let error = ApiError::new(
            StatusCode::BAD_REQUEST,
            ApiErrorType::InvalidRequestError,
            "Invalid model specified",
        )
        .with_request_id("req_123");

        let display = error.to_string();
        assert!(display.contains("Invalid model specified"));
        assert!(display.contains("400"));
        assert!(display.contains("req_123"));
    }

    #[test]
    fn test_error_is_retryable() {
        let rate_limit = ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            ApiErrorType::RateLimitError,
            "Rate limited",
        );
        assert!(rate_limit.is_retryable());

        let invalid_request = ApiError::new(
            StatusCode::BAD_REQUEST,
            ApiErrorType::InvalidRequestError,
            "Bad request",
        );
        assert!(!invalid_request.is_retryable());

        let server_error = ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiErrorType::ApiError,
            "Server error",
        );
        assert!(server_error.is_retryable());
    }

    #[test]
    fn test_api_error_type_deserialize() {
        let json = r#""rate_limit_error""#;
        let error_type: ApiErrorType = serde_json::from_str(json).unwrap();
        assert_eq!(error_type, ApiErrorType::RateLimitError);

        let json = r#""some_unknown_type""#;
        let error_type: ApiErrorType = serde_json::from_str(json).unwrap();
        assert_eq!(error_type, ApiErrorType::Unknown);
    }
}
