//! Response handling for the Anthropic API.
//!
//! This module provides utilities for parsing API responses, extracting
//! metadata like request IDs, and converting error responses into proper
//! error types.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::http::ApiResponse;
//!
//! // Parse a successful response
//! let response: ApiResponse<MessageResponse> = ApiResponse::from_response(response).await?;
//! println!("Request ID: {:?}", response.request_id);
//! println!("Status: {}", response.status);
//! let message = response.data;
//! ```

use crate::error::{ApiError, ApiErrorType, Error, RawApiErrorResponse, Result};
use bytes::Bytes;
use reqwest::header::HeaderMap;
use reqwest::{Response, StatusCode};
use serde::de::DeserializeOwned;
use std::time::Duration;

/// Header name for request ID.
pub const HEADER_REQUEST_ID: &str = "request-id";
/// Header name for retry-after (seconds).
pub const HEADER_RETRY_AFTER: &str = "retry-after";
/// Header name for retry-after in milliseconds.
pub const HEADER_RETRY_AFTER_MS: &str = "retry-after-ms";

/// A parsed API response containing the deserialized data and metadata.
///
/// This struct wraps the response data along with HTTP metadata like
/// status code, headers, and the request ID for debugging.
#[derive(Debug, Clone)]
pub struct ApiResponse<T> {
    /// The deserialized response data.
    pub data: T,
    /// HTTP status code.
    pub status: StatusCode,
    /// Response headers.
    pub headers: HeaderMap,
    /// Request ID from the `request-id` header.
    pub request_id: Option<String>,
}

impl<T> ApiResponse<T> {
    /// Creates a new API response.
    #[must_use]
    pub fn new(data: T, status: StatusCode, headers: HeaderMap) -> Self {
        let request_id = extract_request_id(&headers);
        Self {
            data,
            status,
            headers,
            request_id,
        }
    }

    /// Maps the response data to a different type.
    pub fn map<U, F>(self, f: F) -> ApiResponse<U>
    where
        F: FnOnce(T) -> U,
    {
        ApiResponse {
            data: f(self.data),
            status: self.status,
            headers: self.headers,
            request_id: self.request_id,
        }
    }

    /// Unwraps the response, returning only the data.
    #[must_use]
    pub fn into_data(self) -> T {
        self.data
    }
}

impl<T: DeserializeOwned> ApiResponse<T> {
    /// Parses an API response from a reqwest response.
    ///
    /// This method handles both successful responses (2xx) and error responses,
    /// parsing them into the appropriate types.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The response status indicates an error (4xx, 5xx)
    /// - The response body cannot be read
    /// - The response body cannot be deserialized
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = client.get("https://api.anthropic.com/v1/messages").send().await?;
    /// let api_response: ApiResponse<MessageResponse> = ApiResponse::from_response(response).await?;
    /// ```
    pub async fn from_response(response: Response) -> Result<Self> {
        let status = response.status();
        let headers = response.headers().clone();
        let request_id = extract_request_id(&headers);

        if status.is_success() {
            let bytes = response.bytes().await?;
            let data: T = serde_json::from_slice(&bytes)?;
            Ok(Self {
                data,
                status,
                headers,
                request_id,
            })
        } else {
            // Handle error response
            let bytes = response.bytes().await?;
            let api_error = parse_error_response(status, &headers, &bytes, request_id)?;
            Err(Error::Api(api_error))
        }
    }
}

/// Extension trait for working with reqwest responses.
pub trait ResponseExt {
    /// Extracts the request ID from the response headers.
    fn request_id(&self) -> Option<String>;

    /// Extracts the retry-after duration from the response headers.
    fn retry_after(&self) -> Option<Duration>;
}

impl ResponseExt for Response {
    fn request_id(&self) -> Option<String> {
        extract_request_id(self.headers())
    }

    fn retry_after(&self) -> Option<Duration> {
        parse_retry_after(self.headers())
    }
}

impl ResponseExt for HeaderMap {
    fn request_id(&self) -> Option<String> {
        extract_request_id(self)
    }

    fn retry_after(&self) -> Option<Duration> {
        parse_retry_after(self)
    }
}

/// Extracts the request ID from response headers.
///
/// The request ID is useful for debugging and support requests.
#[must_use]
pub fn extract_request_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get(HEADER_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

/// Parses the retry-after duration from response headers.
///
/// Supports multiple formats:
/// - `retry-after-ms`: Milliseconds (preferred)
/// - `retry-after`: Seconds or HTTP-date format
///
/// # Example Values
///
/// - `retry-after-ms: 5000` -> 5 seconds
/// - `retry-after: 30` -> 30 seconds
/// - `retry-after: Wed, 21 Oct 2024 07:28:00 GMT` -> Duration until that time
#[must_use]
pub fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    // Try milliseconds header first (most precise)
    if let Some(ms_value) = headers.get(HEADER_RETRY_AFTER_MS) {
        if let Ok(ms_str) = ms_value.to_str() {
            if let Ok(ms) = ms_str.parse::<u64>() {
                return Some(Duration::from_millis(ms));
            }
        }
    }

    // Try seconds header
    if let Some(value) = headers.get(HEADER_RETRY_AFTER) {
        if let Ok(value_str) = value.to_str() {
            // Try parsing as seconds (integer)
            if let Ok(seconds) = value_str.parse::<u64>() {
                return Some(Duration::from_secs(seconds));
            }

            // Try parsing as HTTP-date
            if let Some(duration) = parse_http_date_to_duration(value_str) {
                return Some(duration);
            }
        }
    }

    None
}

/// Parses an HTTP-date string and returns the duration until that time.
///
/// Supports the preferred format: `Wed, 21 Oct 2024 07:28:00 GMT`
fn parse_http_date_to_duration(date_str: &str) -> Option<Duration> {
    // Parse HTTP-date format using chrono
    use chrono::{DateTime, Utc};

    // Try RFC 2822 format (common HTTP-date format)
    if let Ok(parsed) = DateTime::parse_from_rfc2822(date_str) {
        let target = parsed.with_timezone(&Utc);
        let now = Utc::now();
        if target > now {
            let diff = target - now;
            return diff.to_std().ok();
        }
    }

    // Try RFC 3339 format as fallback
    if let Ok(parsed) = DateTime::parse_from_rfc3339(date_str) {
        let target = parsed.with_timezone(&Utc);
        let now = Utc::now();
        if target > now {
            let diff = target - now;
            return diff.to_std().ok();
        }
    }

    None
}

/// Parses an error response body into an `ApiError`.
///
/// This function handles the standard Anthropic error response format:
///
/// ```json
/// {
///     "type": "error",
///     "error": {
///         "type": "invalid_request_error",
///         "message": "Error message here"
///     }
/// }
/// ```
fn parse_error_response(
    status: StatusCode,
    headers: &HeaderMap,
    body: &Bytes,
    request_id: Option<String>,
) -> Result<ApiError> {
    let retry_after = parse_retry_after(headers);

    // Try to parse as structured error response
    if let Ok(raw_error) = serde_json::from_slice::<RawApiErrorResponse>(body) {
        let mut api_error = ApiError::new(status, raw_error.error.error_type, raw_error.error.message);

        if let Some(req_id) = request_id {
            api_error = api_error.with_request_id(req_id);
        }

        if let Some(retry) = retry_after {
            api_error = api_error.with_retry_after(retry);
        }

        if let Some(extra) = raw_error.error.extra {
            api_error = api_error.with_details(extra);
        }

        return Ok(api_error);
    }

    // Fallback: Create error from status code and body
    let message = String::from_utf8_lossy(body);
    let error_type = error_type_from_status(status);

    let mut api_error = ApiError::new(status, error_type, message.to_string());

    if let Some(req_id) = request_id {
        api_error = api_error.with_request_id(req_id);
    }

    if let Some(retry) = retry_after {
        api_error = api_error.with_retry_after(retry);
    }

    Ok(api_error)
}

/// Maps HTTP status codes to error types.
#[must_use]
fn error_type_from_status(status: StatusCode) -> ApiErrorType {
    match status.as_u16() {
        400 => ApiErrorType::InvalidRequestError,
        401 => ApiErrorType::AuthenticationError,
        403 => ApiErrorType::PermissionError,
        404 => ApiErrorType::NotFoundError,
        429 => ApiErrorType::RateLimitError,
        529 => ApiErrorType::OverloadedError,
        500..=599 => ApiErrorType::ApiError,
        _ => ApiErrorType::Unknown,
    }
}

/// Reads the response body as bytes.
///
/// This is a convenience function for reading the entire response body.
pub async fn read_body(response: Response) -> Result<Bytes> {
    Ok(response.bytes().await?)
}

/// Reads the response body as a string.
///
/// This is a convenience function for reading the response body as UTF-8 text.
pub async fn read_body_string(response: Response) -> Result<String> {
    Ok(response.text().await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn test_extract_request_id() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_REQUEST_ID,
            HeaderValue::from_static("req_abc123"),
        );

        let request_id = extract_request_id(&headers);
        assert_eq!(request_id, Some("req_abc123".to_string()));
    }

    #[test]
    fn test_extract_request_id_missing() {
        let headers = HeaderMap::new();
        let request_id = extract_request_id(&headers);
        assert_eq!(request_id, None);
    }

    #[test]
    fn test_parse_retry_after_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_RETRY_AFTER, HeaderValue::from_static("30"));

        let retry_after = parse_retry_after(&headers);
        assert_eq!(retry_after, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_parse_retry_after_milliseconds() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_RETRY_AFTER_MS, HeaderValue::from_static("5000"));

        let retry_after = parse_retry_after(&headers);
        assert_eq!(retry_after, Some(Duration::from_millis(5000)));
    }

    #[test]
    fn test_parse_retry_after_milliseconds_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_RETRY_AFTER, HeaderValue::from_static("60"));
        headers.insert(HEADER_RETRY_AFTER_MS, HeaderValue::from_static("5000"));

        let retry_after = parse_retry_after(&headers);
        // Milliseconds header should take precedence
        assert_eq!(retry_after, Some(Duration::from_millis(5000)));
    }

    #[test]
    fn test_error_type_from_status() {
        assert_eq!(
            error_type_from_status(StatusCode::BAD_REQUEST),
            ApiErrorType::InvalidRequestError
        );
        assert_eq!(
            error_type_from_status(StatusCode::UNAUTHORIZED),
            ApiErrorType::AuthenticationError
        );
        assert_eq!(
            error_type_from_status(StatusCode::FORBIDDEN),
            ApiErrorType::PermissionError
        );
        assert_eq!(
            error_type_from_status(StatusCode::NOT_FOUND),
            ApiErrorType::NotFoundError
        );
        assert_eq!(
            error_type_from_status(StatusCode::TOO_MANY_REQUESTS),
            ApiErrorType::RateLimitError
        );
        assert_eq!(
            error_type_from_status(StatusCode::INTERNAL_SERVER_ERROR),
            ApiErrorType::ApiError
        );
    }

    #[test]
    fn test_api_response_map() {
        let response = ApiResponse {
            data: 42i32,
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            request_id: Some("req_123".to_string()),
        };

        let mapped = response.map(|n| n.to_string());
        assert_eq!(mapped.data, "42");
        assert_eq!(mapped.request_id, Some("req_123".to_string()));
    }

    #[test]
    fn test_parse_error_response() {
        let body = br#"{
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid model specified"
            }
        }"#;

        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_REQUEST_ID,
            HeaderValue::from_static("req_test123"),
        );

        let error = parse_error_response(
            StatusCode::BAD_REQUEST,
            &headers,
            &Bytes::from_static(body),
            Some("req_test123".to_string()),
        )
        .unwrap();

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.error_type, ApiErrorType::InvalidRequestError);
        assert_eq!(error.message, "Invalid model specified");
        assert_eq!(error.request_id, Some("req_test123".to_string()));
    }

    #[test]
    fn test_parse_error_response_fallback() {
        let body = b"Internal Server Error";
        let headers = HeaderMap::new();

        let error = parse_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &headers,
            &Bytes::from_static(body),
            None,
        )
        .unwrap();

        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.error_type, ApiErrorType::ApiError);
        assert_eq!(error.message, "Internal Server Error");
    }
}
