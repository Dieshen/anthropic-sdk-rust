//! Error types for the Bedrock integration.
//!
//! This module provides error handling for all Bedrock operations,
//! including configuration errors, AWS signing errors, and API errors.

use reqwest::StatusCode;
use std::fmt;

/// Result type alias for Bedrock operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The primary error type for Bedrock operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration error (missing credentials, invalid region, etc.).
    #[error("Configuration error: {0}")]
    Config(String),

    /// AWS Signature V4 signing error.
    #[error("Signing error: {0}")]
    Signing(String),

    /// API error from Bedrock.
    #[error("Bedrock API error: {0}")]
    Api(BedrockApiError),

    /// HTTP/network error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// URL parsing error.
    #[error("URL error: {0}")]
    Url(#[from] url::ParseError),
}

impl Error {
    /// Creates a new configuration error.
    #[must_use]
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    /// Creates a new signing error.
    #[must_use]
    pub fn signing(message: impl Into<String>) -> Self {
        Self::Signing(message.into())
    }

    /// Returns `true` if this error is retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Api(api_err) => api_err.is_retryable(),
            Self::Http(http_err) => http_err.is_connect() || http_err.is_timeout(),
            _ => false,
        }
    }

    /// Returns the HTTP status code if this is an API error.
    #[must_use]
    pub fn status_code(&self) -> Option<StatusCode> {
        match self {
            Self::Api(api_err) => Some(api_err.status),
            Self::Http(http_err) => http_err.status(),
            _ => None,
        }
    }
}

/// Error response from the Bedrock API.
#[derive(Debug, Clone)]
pub struct BedrockApiError {
    /// HTTP status code.
    pub status: StatusCode,
    /// Error type from AWS/Bedrock.
    pub error_type: String,
    /// Human-readable error message.
    pub message: String,
}

impl fmt::Display for BedrockApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (status: {}, type: {})",
            self.message, self.status, self.error_type
        )
    }
}

impl std::error::Error for BedrockApiError {}

impl BedrockApiError {
    /// Creates a new Bedrock API error.
    #[must_use]
    pub fn new(
        status: StatusCode,
        error_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            error_type: error_type.into(),
            message: message.into(),
        }
    }

    /// Returns `true` if this error is retryable.
    ///
    /// Retryable errors include:
    /// - Rate limit errors (429)
    /// - Service unavailable (503)
    /// - Internal server errors (500, 502, 504)
    /// - Throttling errors
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        // Retry on rate limits
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            return true;
        }

        // Retry on server errors
        if self.status.is_server_error() {
            return true;
        }

        // Check for throttling error types
        let error_type_lower = self.error_type.to_lowercase();
        if error_type_lower.contains("throttl") || error_type_lower.contains("rate") {
            return true;
        }

        false
    }

    /// Returns `true` if this is a throttling/rate limit error.
    #[must_use]
    pub fn is_throttled(&self) -> bool {
        self.status == StatusCode::TOO_MANY_REQUESTS
            || self.error_type.to_lowercase().contains("throttl")
    }

    /// Returns `true` if this is an authentication error.
    #[must_use]
    pub fn is_auth_error(&self) -> bool {
        self.status == StatusCode::UNAUTHORIZED
            || self.status == StatusCode::FORBIDDEN
            || self.error_type.to_lowercase().contains("access")
            || self.error_type.to_lowercase().contains("credential")
    }

    /// Returns `true` if this is a validation error.
    #[must_use]
    pub fn is_validation_error(&self) -> bool {
        self.status == StatusCode::BAD_REQUEST
            || self.error_type.to_lowercase().contains("validation")
    }

    /// Returns `true` if the model was not found.
    #[must_use]
    pub fn is_model_not_found(&self) -> bool {
        self.status == StatusCode::NOT_FOUND || self.error_type.to_lowercase().contains("model")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_config() {
        let err = Error::config("Missing AWS credentials");
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains("Missing AWS credentials"));
    }

    #[test]
    fn test_error_signing() {
        let err = Error::signing("Invalid signature");
        assert!(matches!(err, Error::Signing(_)));
        assert!(err.to_string().contains("Invalid signature"));
    }

    #[test]
    fn test_bedrock_api_error_display() {
        let err = BedrockApiError::new(
            StatusCode::BAD_REQUEST,
            "ValidationException",
            "Invalid model ID",
        );

        let display = err.to_string();
        assert!(display.contains("Invalid model ID"));
        assert!(display.contains("400"));
        assert!(display.contains("ValidationException"));
    }

    #[test]
    fn test_bedrock_api_error_is_retryable() {
        // Rate limit is retryable
        let err = BedrockApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "ThrottlingException",
            "Rate exceeded",
        );
        assert!(err.is_retryable());

        // Server error is retryable
        let err = BedrockApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "InternalError",
            "Server error",
        );
        assert!(err.is_retryable());

        // Validation error is not retryable
        let err = BedrockApiError::new(
            StatusCode::BAD_REQUEST,
            "ValidationException",
            "Invalid input",
        );
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_bedrock_api_error_is_throttled() {
        let err = BedrockApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "ThrottlingException",
            "Rate exceeded",
        );
        assert!(err.is_throttled());

        let err = BedrockApiError::new(
            StatusCode::BAD_REQUEST,
            "ValidationException",
            "Invalid input",
        );
        assert!(!err.is_throttled());
    }

    #[test]
    fn test_bedrock_api_error_is_auth_error() {
        let err = BedrockApiError::new(
            StatusCode::FORBIDDEN,
            "AccessDeniedException",
            "Access denied",
        );
        assert!(err.is_auth_error());

        let err = BedrockApiError::new(
            StatusCode::UNAUTHORIZED,
            "UnauthorizedException",
            "Invalid credentials",
        );
        assert!(err.is_auth_error());
    }

    #[test]
    fn test_bedrock_api_error_is_validation_error() {
        let err = BedrockApiError::new(
            StatusCode::BAD_REQUEST,
            "ValidationException",
            "Invalid parameter",
        );
        assert!(err.is_validation_error());
    }

    #[test]
    fn test_error_status_code() {
        let err = Error::Api(BedrockApiError::new(
            StatusCode::BAD_REQUEST,
            "ValidationException",
            "Invalid",
        ));
        assert_eq!(err.status_code(), Some(StatusCode::BAD_REQUEST));

        let err = Error::config("Missing credentials");
        assert_eq!(err.status_code(), None);
    }
}
