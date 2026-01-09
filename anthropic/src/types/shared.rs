//! Shared types used across the Anthropic SDK.
//!
//! This module contains common types that are used by multiple
//! other type modules, such as cache control and metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Cache Control
// =============================================================================

/// Cache control configuration for request content.
///
/// Enables ephemeral caching of content blocks to reduce token costs
/// for repeated content in subsequent requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    /// The type of cache control (always "ephemeral").
    #[serde(rename = "type")]
    pub control_type: CacheControlType,

    /// Time-to-live for the cached content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheTtl>,
}

/// Cache control type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheControlType {
    /// Ephemeral cache that expires after TTL.
    Ephemeral,
}

/// Time-to-live duration for cached content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTtl {
    /// 5 minute cache duration.
    #[serde(rename = "5m")]
    FiveMinutes,
    /// 1 hour cache duration.
    #[serde(rename = "1h")]
    OneHour,
}

impl Default for CacheTtl {
    fn default() -> Self {
        Self::FiveMinutes
    }
}

impl CacheControl {
    /// Creates a new ephemeral cache control with default TTL (5 minutes).
    #[must_use]
    pub fn ephemeral() -> Self {
        Self {
            control_type: CacheControlType::Ephemeral,
            ttl: None,
        }
    }

    /// Creates a new ephemeral cache control with 5 minute TTL.
    #[must_use]
    pub fn ephemeral_5m() -> Self {
        Self {
            control_type: CacheControlType::Ephemeral,
            ttl: Some(CacheTtl::FiveMinutes),
        }
    }

    /// Creates a new ephemeral cache control with 1 hour TTL.
    #[must_use]
    pub fn ephemeral_1h() -> Self {
        Self {
            control_type: CacheControlType::Ephemeral,
            ttl: Some(CacheTtl::OneHour),
        }
    }
}

// =============================================================================
// Metadata
// =============================================================================

/// Metadata for tracking requests.
///
/// This can be used to attach a user identifier to requests for
/// tracking and abuse prevention purposes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metadata {
    /// An external identifier for the user making the request.
    ///
    /// This should be a UUID, hash, or other opaque identifier.
    /// Do not include PII (names, emails, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl Metadata {
    /// Creates new metadata with the given user ID.
    #[must_use]
    pub fn with_user_id(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
        }
    }
}

// =============================================================================
// Role
// =============================================================================

/// Message role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User message.
    User,
    /// Assistant (Claude) message.
    Assistant,
}

impl Default for Role {
    fn default() -> Self {
        Self::User
    }
}

// =============================================================================
// Stop Reason
// =============================================================================

/// Reason why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural stopping point reached.
    EndTurn,
    /// Maximum token limit reached.
    MaxTokens,
    /// Custom stop sequence was generated.
    StopSequence,
    /// Model invoked one or more tools.
    ToolUse,
    /// Long-running turn was paused.
    PauseTurn,
    /// Content was refused due to policy.
    Refusal,
}

// =============================================================================
// Source Types (for Content)
// =============================================================================

/// Source for base64-encoded image data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Base64Source {
    /// The type of source (always "base64").
    #[serde(rename = "type")]
    pub source_type: String,

    /// The media type (e.g., "image/png", "image/jpeg").
    pub media_type: String,

    /// Base64-encoded image data.
    pub data: String,
}

impl Base64Source {
    /// Creates a new base64 source.
    #[must_use]
    pub fn new(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source_type: "base64".to_string(),
            media_type: media_type.into(),
            data: data.into(),
        }
    }
}

/// Source for URL-based content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlSource {
    /// The type of source (always "url").
    #[serde(rename = "type")]
    pub source_type: String,

    /// The URL of the content.
    pub url: String,
}

impl UrlSource {
    /// Creates a new URL source.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            source_type: "url".to_string(),
            url: url.into(),
        }
    }
}

/// Source for file content (by file ID).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSource {
    /// The type of source (always "file").
    #[serde(rename = "type")]
    pub source_type: String,

    /// The file ID.
    pub file_id: String,
}

impl FileSource {
    /// Creates a new file source.
    #[must_use]
    pub fn new(file_id: impl Into<String>) -> Self {
        Self {
            source_type: "file".to_string(),
            file_id: file_id.into(),
        }
    }
}

// =============================================================================
// Image Media Types
// =============================================================================

/// Supported image media types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageMediaType {
    /// JPEG image.
    Jpeg,
    /// PNG image.
    Png,
    /// GIF image.
    Gif,
    /// WebP image.
    Webp,
}

impl ImageMediaType {
    /// Returns the MIME type string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
        }
    }
}

impl std::fmt::Display for ImageMediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Extra/Extension fields
// =============================================================================

/// A map of additional fields not covered by the schema.
pub type ExtraFields = HashMap<String, serde_json::Value>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_control_serialize() {
        let cc = CacheControl::ephemeral();
        let json = serde_json::to_string(&cc).unwrap();
        assert!(json.contains("\"type\":\"ephemeral\""));
    }

    #[test]
    fn test_cache_control_with_ttl() {
        let cc = CacheControl::ephemeral_1h();
        let json = serde_json::to_string(&cc).unwrap();
        assert!(json.contains("\"ttl\":\"1h\""));
    }

    #[test]
    fn test_metadata_serialize() {
        let meta = Metadata::with_user_id("user-123");
        let json = serde_json::to_string(&meta).unwrap();
        assert_eq!(json, r#"{"user_id":"user-123"}"#);
    }

    #[test]
    fn test_role_serialize() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    #[test]
    fn test_stop_reason_deserialize() {
        let json = "\"end_turn\"";
        let reason: StopReason = serde_json::from_str(json).unwrap();
        assert_eq!(reason, StopReason::EndTurn);

        let json = "\"tool_use\"";
        let reason: StopReason = serde_json::from_str(json).unwrap();
        assert_eq!(reason, StopReason::ToolUse);
    }

    #[test]
    fn test_base64_source() {
        let source = Base64Source::new("image/png", "iVBORw0KGgo...");
        assert_eq!(source.source_type, "base64");
        assert_eq!(source.media_type, "image/png");
    }

    #[test]
    fn test_url_source() {
        let source = UrlSource::new("https://example.com/image.png");
        assert_eq!(source.source_type, "url");
        assert_eq!(source.url, "https://example.com/image.png");
    }
}
