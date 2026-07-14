//! Model constants for the Anthropic API.
//!
//! This module provides type-safe model identifiers and constants for
//! all available Claude models.
//!
//! # Example
//!
//! ```rust
//! use anthropic::types::Model;
//!
//! // Use predefined constants
//! let model = Model::claude_sonnet_4_5_latest();
//!
//! // Or create a custom model string
//! let model = Model::new("claude-sonnet-4-5-20250929");
//! ```

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;

/// A Claude model identifier.
///
/// This can be one of the predefined model constants or a custom model string.
/// Model identifiers are case-sensitive.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Model(Cow<'static, str>);

impl Model {
    // =========================================================================
    // Claude Opus 4.5 Models
    // =========================================================================

    /// Claude Opus 4.5 - Latest version (2025-11-01)
    #[must_use]
    pub const fn claude_opus_4_5_20251101() -> Self {
        Self(Cow::Borrowed("claude-opus-4-5-20251101"))
    }

    /// Claude Opus 4.5 - Latest alias
    #[must_use]
    pub const fn claude_opus_4_5_latest() -> Self {
        Self(Cow::Borrowed("claude-opus-4-5-latest"))
    }

    /// Claude Opus 4.5 - Short alias
    #[must_use]
    pub const fn claude_opus_4_5() -> Self {
        Self(Cow::Borrowed("claude-opus-4-5"))
    }

    // =========================================================================
    // Claude Sonnet 4.5 Models
    // =========================================================================

    /// Claude Sonnet 4.5 - Latest version (2025-09-29)
    #[must_use]
    pub const fn claude_sonnet_4_5_20250929() -> Self {
        Self(Cow::Borrowed("claude-sonnet-4-5-20250929"))
    }

    /// Claude Sonnet 4.5 - Latest alias
    #[must_use]
    pub const fn claude_sonnet_4_5_latest() -> Self {
        Self(Cow::Borrowed("claude-sonnet-4-5-latest"))
    }

    /// Claude Sonnet 4.5 - Short alias
    #[must_use]
    pub const fn claude_sonnet_4_5() -> Self {
        Self(Cow::Borrowed("claude-sonnet-4-5"))
    }

    // =========================================================================
    // Claude Sonnet 4 Models
    // =========================================================================

    /// Claude Sonnet 4 - Version 2025-05-14
    #[must_use]
    pub const fn claude_sonnet_4_20250514() -> Self {
        Self(Cow::Borrowed("claude-sonnet-4-20250514"))
    }

    /// Claude Sonnet 4 - Latest alias
    #[must_use]
    pub const fn claude_sonnet_4_latest() -> Self {
        Self(Cow::Borrowed("claude-sonnet-4-latest"))
    }

    /// Claude Sonnet 4 - Short alias
    #[must_use]
    pub const fn claude_sonnet_4() -> Self {
        Self(Cow::Borrowed("claude-sonnet-4-0"))
    }

    // =========================================================================
    // Claude Opus 4 Models
    // =========================================================================

    /// Claude Opus 4 - Version 2025-05-14
    #[must_use]
    pub const fn claude_opus_4_20250514() -> Self {
        Self(Cow::Borrowed("claude-opus-4-20250514"))
    }

    /// Claude Opus 4 - Latest alias
    #[must_use]
    pub const fn claude_opus_4_latest() -> Self {
        Self(Cow::Borrowed("claude-opus-4-latest"))
    }

    /// Claude Opus 4 - Short alias
    #[must_use]
    pub const fn claude_opus_4() -> Self {
        Self(Cow::Borrowed("claude-opus-4-0"))
    }

    // =========================================================================
    // Claude Haiku 4.5 Models
    // =========================================================================

    /// Claude Haiku 4.5 - Version 2025-10-01
    #[must_use]
    pub const fn claude_haiku_4_5_20251001() -> Self {
        Self(Cow::Borrowed("claude-haiku-4-5-20251001"))
    }

    /// Claude Haiku 4.5 - Latest alias
    #[must_use]
    pub const fn claude_haiku_4_5_latest() -> Self {
        Self(Cow::Borrowed("claude-haiku-4-5-latest"))
    }

    /// Claude Haiku 4.5 - Short alias
    #[must_use]
    pub const fn claude_haiku_4_5() -> Self {
        Self(Cow::Borrowed("claude-haiku-4-5"))
    }

    // =========================================================================
    // Claude 3.5 Models (Legacy)
    // =========================================================================

    /// Claude 3.5 Sonnet - Version 2024-10-22
    #[must_use]
    pub const fn claude_3_5_sonnet_20241022() -> Self {
        Self(Cow::Borrowed("claude-3-5-sonnet-20241022"))
    }

    /// Claude 3.5 Sonnet - Version 2024-06-20
    #[must_use]
    pub const fn claude_3_5_sonnet_20240620() -> Self {
        Self(Cow::Borrowed("claude-3-5-sonnet-20240620"))
    }

    /// Claude 3.5 Sonnet - Latest alias
    #[must_use]
    pub const fn claude_3_5_sonnet_latest() -> Self {
        Self(Cow::Borrowed("claude-3-5-sonnet-latest"))
    }

    /// Claude 3.5 Haiku - Version 2024-10-22
    #[must_use]
    pub const fn claude_3_5_haiku_20241022() -> Self {
        Self(Cow::Borrowed("claude-3-5-haiku-20241022"))
    }

    /// Claude 3.5 Haiku - Latest alias
    #[must_use]
    pub const fn claude_3_5_haiku_latest() -> Self {
        Self(Cow::Borrowed("claude-3-5-haiku-latest"))
    }

    // =========================================================================
    // Claude 3 Models (Legacy)
    // =========================================================================

    /// Claude 3 Opus - Version 2024-02-29
    #[must_use]
    pub const fn claude_3_opus_20240229() -> Self {
        Self(Cow::Borrowed("claude-3-opus-20240229"))
    }

    /// Claude 3 Opus - Latest alias
    #[must_use]
    pub const fn claude_3_opus_latest() -> Self {
        Self(Cow::Borrowed("claude-3-opus-latest"))
    }

    /// Claude 3 Sonnet - Version 2024-02-29
    #[must_use]
    pub const fn claude_3_sonnet_20240229() -> Self {
        Self(Cow::Borrowed("claude-3-sonnet-20240229"))
    }

    /// Claude 3 Haiku - Version 2024-03-07
    #[must_use]
    pub const fn claude_3_haiku_20240307() -> Self {
        Self(Cow::Borrowed("claude-3-haiku-20240307"))
    }

    // =========================================================================
    // Associated Constants (string values only)
    // =========================================================================

    /// Model string for Claude Sonnet 4.5 Latest
    pub const CLAUDE_SONNET_4_5_LATEST: &'static str = "claude-sonnet-4-5-latest";

    /// Model string for Claude Opus 4.5 Latest
    pub const CLAUDE_OPUS_4_5_LATEST: &'static str = "claude-opus-4-5-latest";

    /// Model string for Claude Haiku 4.5 Latest
    pub const CLAUDE_HAIKU_4_5_LATEST: &'static str = "claude-haiku-4-5-latest";

    // =========================================================================
    // Constructors
    // =========================================================================

    /// Creates a new model from a string.
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(Cow::Owned(s.into()))
    }

    /// Creates a model from a static string.
    #[must_use]
    pub const fn from_static(s: &'static str) -> Self {
        Self(Cow::Borrowed(s))
    }

    /// Returns the model identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the model and returns the inner string.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0.into_owned()
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Model {
    fn from(s: &str) -> Self {
        Self(Cow::Owned(s.to_string()))
    }
}

impl From<String> for Model {
    fn from(s: String) -> Self {
        Self(Cow::Owned(s))
    }
}

impl AsRef<str> for Model {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::claude_sonnet_4_5_latest()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_display() {
        assert_eq!(
            Model::claude_sonnet_4_5_latest().to_string(),
            "claude-sonnet-4-5-latest"
        );
    }

    #[test]
    fn test_model_from_str() {
        let model: Model = "claude-sonnet-4-5-20250929".into();
        assert_eq!(model.as_str(), "claude-sonnet-4-5-20250929");
    }

    #[test]
    fn test_model_serialize() {
        let model = Model::claude_opus_4_5_latest();
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(json, "\"claude-opus-4-5-latest\"");
    }

    #[test]
    fn test_model_deserialize() {
        let json = "\"claude-3-5-haiku-20241022\"";
        let model: Model = serde_json::from_str(json).unwrap();
        assert_eq!(model.as_str(), "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_model_equality() {
        let m1 = Model::new("claude-sonnet-4-5");
        let m2 = Model::claude_sonnet_4_5();
        assert_eq!(m1, m2);
        assert_eq!(m1.as_str(), "claude-sonnet-4-5");
    }

    #[test]
    fn test_model_from_static() {
        let model = Model::from_static("claude-custom-model");
        assert_eq!(model.as_str(), "claude-custom-model");
    }

    #[test]
    fn test_model_const_string() {
        let model = Model::from_static(Model::CLAUDE_SONNET_4_5_LATEST);
        assert_eq!(model.as_str(), "claude-sonnet-4-5-latest");
    }
}
