//! Beta API features for the Anthropic SDK.
//!
//! This module provides access to beta features that are not yet part of the
//! stable API. All types and functions in this module are subject to change
//! without notice and should be used with caution in production environments.
//!
//! # Feature Flag
//!
//! All beta features are gated behind the `beta` feature flag:
//!
//! ```toml
//! [dependencies]
//! anthropic = { version = "0.1", features = ["beta"] }
//! ```
//!
//! # Beta Features
//!
//! ## Extended Thinking
//!
//! Extended thinking allows Claude to show its reasoning process before
//! providing a final answer. This can help with complex problems where
//! step-by-step reasoning is beneficial.
//!
//! ```rust,ignore
//! use anthropic::beta::{BetaMessageCreateParams, ThinkingConfig};
//!
//! let params = BetaMessageCreateParams::new(
//!     Model::claude_sonnet_4_5_latest(),
//!     vec![MessageParam::user("What is 15 * 23?")],
//!     16384,
//! )
//! .with_thinking(ThinkingConfig::enabled(10000));
//! ```
//!
//! ## Server Tools
//!
//! Beta server tools include:
//! - **Web Search**: Search the web for information
//! - **Computer Use Tools**: Bash, Text Editor, and Computer control
//!
//! ```rust,ignore
//! use anthropic::beta::{BetaTool, WebSearchTool20250305};
//!
//! let web_search = BetaTool::WebSearch(WebSearchTool20250305::new());
//! ```
//!
//! # Beta Headers
//!
//! When using beta features, the SDK automatically adds the appropriate
//! `anthropic-beta` header to requests based on the features being used.

pub mod files;
pub mod messages;
pub mod skills;
pub mod tools;

// Re-export commonly used types
pub use messages::{
    BetaContentBlock, BetaMessage, BetaMessageCreateParams, BetaMessageCreateParamsBuilder,
    BetaMessageService, BetaStopReason, ThinkingConfig, ThinkingConfigParam,
};

pub use tools::{
    // Computer Use
    BashTool20250124,
    BetaTool,
    BetaToolUnion,
    CodeExecutionOutputBlock,
    CodeExecutionResultBlock,
    // Code Execution
    CodeExecutionTool,
    ComputerTool20250124,
    TextEditorTool20250124,
    // Web Search
    WebSearchTool20250305,
    WebSearchToolResultBlock,
    WebSearchToolResultContent,
    WebSearchToolResultError,
    WebSearchToolResultErrorCode,
    WebSearchUserLocation,
};

pub use files::{
    DeletedFile, FileMetadata, FileUploadParams, Files, FilesListParams, FilesListResponse,
};

pub use skills::{
    DeletedSkill, DeletedSkillVersion, Skill, SkillCreateParams, SkillType, SkillVersion, Skills,
    SkillsListParams, SkillsListResponse, VersionCreateParams, VersionsListParams,
    VersionsListResponse,
};

// Re-export Beta resource (will be defined below)
// pub use Beta; - exported via the struct definition

/// Beta feature identifiers for the anthropic-beta header.
///
/// These are the valid values that can be passed in the `betas` field
/// of beta API requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BetaFeature {
    /// Extended thinking / extended output
    MaxTokens35Outputs20250131,
    /// Computer use tools
    ComputerUse20250124,
    /// Web search tool
    WebSearch20250305,
    /// Token counting
    TokenCounting20241101,
    /// Message batches
    MessageBatches20241112,
    /// PDF files support
    PdfFiles20241218,
    /// Interleaved thinking
    InterleavedThinking20250122,
    /// Code execution
    CodeExecution20250522,
    /// Files support
    Files20250114,
    /// MCP client
    McpClient20250124,
    /// Skills API
    Skills20251002,
}

impl BetaFeature {
    /// Returns the string value for the beta header.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::MaxTokens35Outputs20250131 => "max-tokens-3-5-sonnet-2024-07-15",
            Self::ComputerUse20250124 => "computer-use-2025-01-24",
            Self::WebSearch20250305 => "web-search-2025-03-05",
            Self::TokenCounting20241101 => "token-counting-2024-11-01",
            Self::MessageBatches20241112 => "message-batches-2024-11-12",
            Self::PdfFiles20241218 => "pdf-files-2024-12-18",
            Self::InterleavedThinking20250122 => "interleaved-thinking-2025-01-22",
            Self::CodeExecution20250522 => "code-execution-2025-05-22",
            Self::Files20250114 => "files-2025-01-14",
            Self::McpClient20250124 => "mcp-client-2025-01-24",
            Self::Skills20251002 => "skills-2025-10-02",
        }
    }
}

impl std::fmt::Display for BetaFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Header name for beta feature flags.
pub const HEADER_ANTHROPIC_BETA: &str = "anthropic-beta";

// =============================================================================
// Beta Resource
// =============================================================================

use crate::Anthropic;
use std::sync::Arc;

/// Access point for all beta API features.
///
/// Provides methods to access beta resources like Files and Skills.
///
/// # Example
///
/// ```rust,ignore
/// use anthropic::Anthropic;
///
/// let client = Anthropic::new()?;
///
/// // Access beta files API
/// let files = client.beta().files();
/// let all_files = files.list_all().await?;
///
/// // Access beta skills API
/// let skills = client.beta().skills();
/// let all_skills = skills.list_all().await?;
/// ```
#[derive(Debug, Clone)]
pub struct Beta {
    client: Arc<Anthropic>,
}

impl Beta {
    /// Creates a new Beta resource.
    pub(crate) const fn new(client: Arc<Anthropic>) -> Self {
        Self { client }
    }

    /// Returns the Files resource for managing server-side files.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let files = client.beta().files();
    ///
    /// // Upload a file
    /// let file = files.upload(FileUploadParams::text(
    ///     "notes.txt",
    ///     b"Hello, world!".to_vec(),
    /// )).await?;
    ///
    /// // List all files
    /// let all_files = files.list_all().await?;
    /// ```
    #[must_use]
    pub fn files(&self) -> Files {
        Files::new(self.client.http_client(), self.client.base_url().clone())
    }

    /// Returns the Skills resource for managing reusable functionality packages.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let skills = client.beta().skills();
    ///
    /// // List all skills
    /// let all_skills = skills.list_all().await?;
    ///
    /// // Get a specific skill
    /// let skill = skills.get("skill_123").await?;
    /// ```
    #[must_use]
    pub fn skills(&self) -> Skills {
        Skills::new(self.client.http_client(), self.client.base_url().clone())
    }

    /// Returns the beta message service for extended thinking and server tools.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let beta_messages = client.beta().messages();
    ///
    /// let response = beta_messages.create(
    ///     BetaMessageCreateParams::new(
    ///         Model::claude_sonnet_4_5_latest(),
    ///         vec![MessageParam::user("Think step by step...")],
    ///         16384,
    ///     )
    ///     .with_thinking(ThinkingConfig::enabled(10000))
    /// ).await?;
    /// ```
    #[must_use]
    pub fn messages(&self) -> BetaMessageService {
        BetaMessageService::new(self.client.clone())
    }
}

/// Extension trait to add `beta()` method to Anthropic client.
impl Anthropic {
    /// Returns the Beta resource for accessing beta API features.
    ///
    /// Beta features include:
    /// - **Files API**: Server-side file storage and management
    /// - **Skills API**: Reusable functionality packages
    /// - **Extended Thinking**: Show Claude's reasoning process
    /// - **Server Tools**: Web search, code execution, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic::Anthropic;
    ///
    /// let client = Anthropic::new()?;
    /// let beta = client.beta();
    ///
    /// // Upload a file
    /// let file = beta.files().upload(params).await?;
    ///
    /// // Use extended thinking
    /// let response = beta.messages().create(params).await?;
    /// ```
    #[must_use]
    pub fn beta(&self) -> Beta {
        Beta::new(Arc::new(self.clone()))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_feature_strings() {
        assert_eq!(
            BetaFeature::WebSearch20250305.as_str(),
            "web-search-2025-03-05"
        );
        assert_eq!(
            BetaFeature::ComputerUse20250124.as_str(),
            "computer-use-2025-01-24"
        );
        assert_eq!(
            BetaFeature::TokenCounting20241101.as_str(),
            "token-counting-2024-11-01"
        );
    }

    #[test]
    fn test_beta_feature_display() {
        let feature = BetaFeature::WebSearch20250305;
        assert_eq!(format!("{feature}"), "web-search-2025-03-05");
    }
}
