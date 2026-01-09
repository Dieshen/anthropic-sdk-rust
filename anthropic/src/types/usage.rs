//! Usage and token counting types.
//!
//! This module provides types for tracking token usage in API requests
//! and responses, including cache-related metrics.

use serde::{Deserialize, Serialize};

/// Token usage statistics from an API response.
///
/// This tracks the number of tokens consumed by a request, including
/// both input and output tokens, as well as cache-related metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens (prompt tokens).
    pub input_tokens: i64,

    /// Number of output tokens (completion tokens).
    pub output_tokens: i64,

    /// Number of tokens used for cache creation.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_creation_input_tokens: i64,

    /// Number of tokens read from cache.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_read_input_tokens: i64,

    /// Cache creation details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<CacheCreation>,

    /// Server-side tool usage statistics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUsage>,

    /// The service tier used for this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
}

/// Cache creation statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheCreation {
    /// Tokens used for 5-minute ephemeral cache.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ephemeral_5m_input_tokens: i64,

    /// Tokens used for 1-hour ephemeral cache.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ephemeral_1h_input_tokens: i64,
}

/// Server-side tool usage statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerToolUsage {
    /// Number of web search requests made.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub web_search_requests: i64,
}

/// Service tier for request processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    /// Standard processing tier.
    Standard,
    /// Priority processing tier.
    Priority,
    /// Batch processing tier.
    Batch,
}

impl Default for ServiceTier {
    fn default() -> Self {
        Self::Standard
    }
}

/// Response from the token counting endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageTokensCount {
    /// The number of input tokens in the request.
    pub input_tokens: i64,
}

impl Usage {
    /// Creates a new usage with the given input and output tokens.
    #[must_use]
    pub fn new(input_tokens: i64, output_tokens: i64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            ..Default::default()
        }
    }

    /// Returns the total number of tokens (input + output).
    #[must_use]
    pub fn total_tokens(&self) -> i64 {
        self.input_tokens + self.output_tokens
    }

    /// Returns the total input tokens including cache.
    #[must_use]
    pub fn total_input_tokens(&self) -> i64 {
        self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
    }
}

/// Helper function for serde skip_serializing_if
fn is_zero(val: &i64) -> bool {
    *val == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_new() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens(), 150);
    }

    #[test]
    fn test_usage_serialize() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 25,
            ..Default::default()
        };

        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("\"input_tokens\":100"));
        assert!(json.contains("\"output_tokens\":50"));
        assert!(json.contains("\"cache_read_input_tokens\":25"));
        // cache_creation_input_tokens should be skipped (is_zero)
        assert!(!json.contains("cache_creation_input_tokens"));
    }

    #[test]
    fn test_usage_deserialize() {
        let json = r#"{
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 25
        }"#;

        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, 10);
        assert_eq!(usage.cache_read_input_tokens, 25);
    }

    #[test]
    fn test_service_tier_serialize() {
        assert_eq!(
            serde_json::to_string(&ServiceTier::Standard).unwrap(),
            "\"standard\""
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Priority).unwrap(),
            "\"priority\""
        );
        assert_eq!(
            serde_json::to_string(&ServiceTier::Batch).unwrap(),
            "\"batch\""
        );
    }

    #[test]
    fn test_message_tokens_count() {
        let json = r#"{"input_tokens": 150}"#;
        let count: MessageTokensCount = serde_json::from_str(json).unwrap();
        assert_eq!(count.input_tokens, 150);
    }
}
