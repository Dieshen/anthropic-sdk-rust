//! Beta Messages API for the Anthropic SDK.
//!
//! This module provides the beta messages API which includes:
//! - Extended thinking support with thinking blocks and budget tokens
//! - Server tools (web search, computer use)
//! - Beta header injection for feature flags
//!
//! # Extended Thinking
//!
//! Extended thinking allows Claude to show its reasoning process:
//!
//! ```rust,ignore
//! use anthropic::beta::{BetaMessageCreateParams, ThinkingConfig};
//!
//! let params = BetaMessageCreateParams::new(
//!     Model::claude_sonnet_4_5_latest(),
//!     vec![BetaMessageParam::user("Solve this complex math problem...")],
//!     16384,
//! )
//! .with_thinking(ThinkingConfig::enabled(10000))
//! .with_betas(vec![BetaFeature::InterleavedThinking20250122]);
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::client::Anthropic;
use crate::error::Result;
use crate::types::content::{
    ContentBlockParam, RedactedThinkingBlock, ServerToolUseBlock, TextBlock, ThinkingBlock,
    ToolUseBlock, WebSearchToolResultBlock,
};
use crate::types::model::Model;
use crate::types::shared::{CacheControl, Metadata, Role};

use super::tools::BetaToolUnion;
use super::BetaFeature;

// =============================================================================
// Beta Message Service
// =============================================================================

/// Service for beta message API operations.
///
/// Provides access to beta features like extended thinking and server tools.
#[derive(Debug, Clone)]
pub struct BetaMessageService {
    client: Arc<Anthropic>,
}

impl BetaMessageService {
    /// Creates a new beta message service.
    pub(crate) const fn new(client: Arc<Anthropic>) -> Self {
        Self { client }
    }

    /// Creates a new beta message.
    ///
    /// This endpoint supports extended thinking and server tools.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters for the message creation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API request fails
    /// - The response cannot be parsed
    /// - Authentication fails
    /// - Rate limits are exceeded
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic::beta::{BetaMessageCreateParams, ThinkingConfig};
    ///
    /// let params = BetaMessageCreateParams::new(
    ///     Model::claude_sonnet_4_5_latest(),
    ///     vec![BetaMessageParam::user("Hello!")],
    ///     1024,
    /// );
    ///
    /// let message = client.beta().messages().create(params).await?;
    /// ```
    pub async fn create(&self, params: BetaMessageCreateParams) -> Result<BetaMessage> {
        // Build beta headers from features
        let beta_header = params
            .betas
            .iter()
            .map(super::BetaFeature::as_str)
            .collect::<Vec<_>>()
            .join(",");

        // Make the request with beta headers
        // Note: In a full implementation, we would add the header to the request
        // For now, we use the standard endpoint
        // TODO: inject beta_header into the request (not currently wired up).
        let _ = beta_header;
        self.client.post("v1/messages?beta=true", &params).await
    }

    /// Counts tokens in a beta message without creating it.
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters for token counting
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API request fails
    /// - The response cannot be parsed
    /// - Authentication fails
    pub async fn count_tokens(
        &self,
        params: BetaMessageCountTokensParams,
    ) -> Result<BetaMessageTokensCount> {
        let beta_header = params
            .betas
            .iter()
            .map(super::BetaFeature::as_str)
            .collect::<Vec<_>>()
            .join(",");

        // TODO: inject beta_header into the request (not currently wired up).
        let _ = beta_header;
        self.client
            .post("v1/messages/count_tokens?beta=true", &params)
            .await
    }
}

// =============================================================================
// Thinking Configuration
// =============================================================================

/// Configuration for extended thinking.
///
/// When enabled, Claude will show its reasoning process in thinking blocks
/// before providing the final answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// Thinking is enabled with a token budget.
    Enabled {
        /// Maximum tokens for thinking.
        /// Minimum is 1024, and thinking tokens count towards `max_tokens`.
        budget_tokens: i64,
    },

    /// Thinking is disabled.
    Disabled,
}

impl ThinkingConfig {
    /// Creates an enabled thinking config with the specified budget.
    ///
    /// # Arguments
    ///
    /// * `budget_tokens` - Maximum tokens for thinking (minimum 1024)
    ///
    /// # Panics
    ///
    /// Panics if `budget_tokens` is less than 1024.
    #[must_use]
    pub fn enabled(budget_tokens: i64) -> Self {
        assert!(
            budget_tokens >= 1024,
            "budget_tokens must be at least 1024, got {budget_tokens}"
        );
        Self::Enabled { budget_tokens }
    }

    /// Creates a disabled thinking config.
    #[must_use]
    pub const fn disabled() -> Self {
        Self::Disabled
    }

    /// Returns true if thinking is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }

    /// Returns the budget tokens if enabled.
    #[must_use]
    pub const fn budget_tokens(&self) -> Option<i64> {
        match self {
            Self::Enabled { budget_tokens } => Some(*budget_tokens),
            Self::Disabled => None,
        }
    }
}

/// Thinking configuration parameter for requests.
///
/// This is an alias for `ThinkingConfig` used in request parameters.
pub type ThinkingConfigParam = ThinkingConfig;

// =============================================================================
// Beta Message Response
// =============================================================================

/// A beta message response from the API.
///
/// This extends the standard Message with additional beta-specific fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaMessage {
    /// Unique identifier for this message.
    pub id: String,

    /// Object type (always "message").
    #[serde(rename = "type")]
    pub message_type: String,

    /// The role of this message (always "assistant").
    pub role: Role,

    /// Content blocks in the response.
    /// May include thinking blocks when extended thinking is enabled.
    pub content: Vec<BetaContentBlock>,

    /// The model that generated this message.
    pub model: Model,

    /// The reason the model stopped generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<BetaStopReason>,

    /// The stop sequence that caused generation to stop, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,

    /// Token usage statistics.
    pub usage: BetaUsage,
}

impl BetaMessage {
    /// Returns the concatenated text content from all text blocks.
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns all thinking content from thinking blocks.
    #[must_use]
    pub fn thinking(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.as_thinking())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns all tool use blocks from the response.
    #[must_use]
    pub fn tool_uses(&self) -> Vec<&ToolUseBlock> {
        self.content
            .iter()
            .filter_map(|block| block.as_tool_use())
            .collect()
    }

    /// Returns true if the model stopped due to tool use.
    #[must_use]
    pub fn has_tool_use(&self) -> bool {
        self.stop_reason == Some(BetaStopReason::ToolUse)
    }

    /// Returns true if the model reached a natural stopping point.
    #[must_use]
    pub fn is_end_turn(&self) -> bool {
        self.stop_reason == Some(BetaStopReason::EndTurn)
    }

    /// Returns true if the response contains thinking blocks.
    #[must_use]
    pub fn has_thinking(&self) -> bool {
        self.content.iter().any(BetaContentBlock::is_thinking)
    }
}

// =============================================================================
// Beta Content Block
// =============================================================================

/// Content block in a beta assistant response.
///
/// This extends the standard `ContentBlock` with additional beta types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BetaContentBlock {
    /// Text content.
    Text(TextBlock),

    /// Extended thinking content.
    Thinking(ThinkingBlock),

    /// Redacted thinking content.
    RedactedThinking(RedactedThinkingBlock),

    /// Tool use request.
    ToolUse(ToolUseBlock),

    /// Server-side tool use (e.g., web search).
    ServerToolUse(ServerToolUseBlock),

    /// Web search tool result.
    WebSearchToolResult(WebSearchToolResultBlock),
}

impl BetaContentBlock {
    /// Returns the text content if this is a text block.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(block) => Some(&block.text),
            _ => None,
        }
    }

    /// Returns the thinking content if this is a thinking block.
    #[must_use]
    pub fn as_thinking(&self) -> Option<&str> {
        match self {
            Self::Thinking(block) => Some(&block.thinking),
            _ => None,
        }
    }

    /// Returns the tool use block if this is a tool use.
    #[must_use]
    pub const fn as_tool_use(&self) -> Option<&ToolUseBlock> {
        match self {
            Self::ToolUse(block) => Some(block),
            _ => None,
        }
    }

    /// Returns true if this is a text block.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Returns true if this is a thinking block.
    #[must_use]
    pub const fn is_thinking(&self) -> bool {
        matches!(self, Self::Thinking(_))
    }

    /// Returns true if this is a tool use block.
    #[must_use]
    pub const fn is_tool_use(&self) -> bool {
        matches!(self, Self::ToolUse(_))
    }
}

// =============================================================================
// Beta Stop Reason
// =============================================================================

/// Reason why the model stopped generating in beta responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BetaStopReason {
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
    /// Model context window was exceeded.
    ModelContextWindowExceeded,
}

// =============================================================================
// Beta Usage
// =============================================================================

/// Token usage statistics for beta responses.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaUsage {
    /// Number of input tokens.
    pub input_tokens: i64,

    /// Number of output tokens.
    pub output_tokens: i64,

    /// Number of tokens used for cache creation.
    #[serde(default)]
    pub cache_creation_input_tokens: i64,

    /// Number of tokens read from cache.
    #[serde(default)]
    pub cache_read_input_tokens: i64,

    /// Server-side tool usage statistics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<BetaServerToolUsage>,
}

/// Server tool usage statistics in beta responses.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaServerToolUsage {
    /// Number of web search requests made.
    #[serde(default)]
    pub web_search_requests: i64,
}

// =============================================================================
// Beta Message Parameters
// =============================================================================

/// A message in a beta conversation (for requests).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaMessageParam {
    /// The role of the message author.
    pub role: Role,

    /// Content of the message.
    pub content: BetaMessageContent,
}

impl BetaMessageParam {
    /// Creates a new user message with text content.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: BetaMessageContent::Text(content.into()),
        }
    }

    /// Creates a new user message with content blocks.
    #[must_use]
    pub const fn user_with_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: Role::User,
            content: BetaMessageContent::Blocks(blocks),
        }
    }

    /// Creates a new assistant message with text content.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: BetaMessageContent::Text(content.into()),
        }
    }

    /// Creates a new assistant message with content blocks.
    #[must_use]
    pub const fn assistant_with_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: Role::Assistant,
            content: BetaMessageContent::Blocks(blocks),
        }
    }
}

/// Content of a beta message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BetaMessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<ContentBlockParam>),
}

// =============================================================================
// Beta Message Create Parameters
// =============================================================================

/// Parameters for creating a beta message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BetaMessageCreateParams {
    /// The model to use for generation.
    pub model: Model,

    /// The messages in the conversation.
    pub messages: Vec<BetaMessageParam>,

    /// Maximum number of tokens to generate.
    pub max_tokens: i64,

    /// Beta features to enable.
    #[serde(skip)]
    pub betas: Vec<BetaFeature>,

    /// System prompt for the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<BetaSystemPrompt>,

    /// Sampling temperature (0.0 to 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p sampling parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,

    /// Stop sequences that will halt generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Tools available for the model to use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<BetaToolUnion>>,

    /// How the model should choose tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<BetaToolChoice>,

    /// Request metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,

    /// Extended thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Service tier for request processing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<BetaServiceTier>,
}

impl BetaMessageCreateParams {
    /// Creates new beta message parameters.
    #[must_use]
    pub const fn new(model: Model, messages: Vec<BetaMessageParam>, max_tokens: i64) -> Self {
        Self {
            model,
            messages,
            max_tokens,
            betas: Vec::new(),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
            service_tier: None,
        }
    }

    /// Sets the beta features to enable.
    #[must_use]
    pub fn with_betas(mut self, betas: Vec<BetaFeature>) -> Self {
        self.betas = betas;
        self
    }

    /// Adds a beta feature.
    #[must_use]
    pub fn with_beta(mut self, beta: BetaFeature) -> Self {
        self.betas.push(beta);
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<BetaSystemPrompt>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets streaming mode.
    #[must_use]
    pub const fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Sets the tools.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<BetaToolUnion>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the tool choice.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: BetaToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets extended thinking.
    #[must_use]
    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Sets the metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Sets the service tier.
    #[must_use]
    pub const fn with_service_tier(mut self, tier: BetaServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }
}

// =============================================================================
// Beta Message Create Params Builder
// =============================================================================

/// Builder for creating beta message parameters.
#[derive(Debug, Clone)]
pub struct BetaMessageCreateParamsBuilder {
    model: Model,
    messages: Vec<BetaMessageParam>,
    max_tokens: i64,
    params: BetaMessageCreateParams,
}

impl BetaMessageCreateParamsBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new(model: Model, max_tokens: i64) -> Self {
        Self {
            model: model.clone(),
            messages: Vec::new(),
            max_tokens,
            params: BetaMessageCreateParams::new(model, Vec::new(), max_tokens),
        }
    }

    /// Adds a user message.
    #[must_use]
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(BetaMessageParam::user(content));
        self
    }

    /// Adds an assistant message.
    #[must_use]
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(BetaMessageParam::assistant(content));
        self
    }

    /// Adds a message.
    #[must_use]
    pub fn message(mut self, message: BetaMessageParam) -> Self {
        self.messages.push(message);
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn system(mut self, system: impl Into<BetaSystemPrompt>) -> Self {
        self.params.system = Some(system.into());
        self
    }

    /// Sets the temperature.
    #[must_use]
    pub const fn temperature(mut self, temperature: f64) -> Self {
        self.params.temperature = Some(temperature);
        self
    }

    /// Sets streaming mode.
    #[must_use]
    pub const fn stream(mut self, stream: bool) -> Self {
        self.params.stream = Some(stream);
        self
    }

    /// Sets the tools.
    #[must_use]
    pub fn tools(mut self, tools: Vec<BetaToolUnion>) -> Self {
        self.params.tools = Some(tools);
        self
    }

    /// Sets the tool choice.
    #[must_use]
    pub fn tool_choice(mut self, tool_choice: BetaToolChoice) -> Self {
        self.params.tool_choice = Some(tool_choice);
        self
    }

    /// Sets extended thinking.
    #[must_use]
    pub fn thinking(mut self, budget_tokens: i64) -> Self {
        self.params.thinking = Some(ThinkingConfig::enabled(budget_tokens));
        self
    }

    /// Adds a beta feature.
    #[must_use]
    pub fn beta(mut self, beta: BetaFeature) -> Self {
        self.params.betas.push(beta);
        self
    }

    /// Builds the parameters.
    #[must_use]
    pub fn build(mut self) -> BetaMessageCreateParams {
        self.params.model = self.model;
        self.params.messages = self.messages;
        self.params.max_tokens = self.max_tokens;
        self.params
    }
}

// =============================================================================
// Beta System Prompt
// =============================================================================

/// System prompt for a beta conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BetaSystemPrompt {
    /// Simple text system prompt.
    Text(String),
    /// Structured system prompt with cache control.
    Blocks(Vec<BetaTextBlockParam>),
}

impl From<&str> for BetaSystemPrompt {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<String> for BetaSystemPrompt {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

/// Text block parameter for beta system prompts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaTextBlockParam {
    /// Block type (always "text").
    #[serde(rename = "type")]
    pub block_type: String,

    /// Text content.
    pub text: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl BetaTextBlockParam {
    /// Creates a new text block.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            block_type: "text".to_string(),
            text: text.into(),
            cache_control: None,
        }
    }

    /// Adds cache control.
    #[must_use]
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

// =============================================================================
// Beta Tool Choice
// =============================================================================

/// Tool choice configuration for beta requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BetaToolChoice {
    /// Model automatically decides whether to use tools.
    Auto {
        /// Disable parallel tool use.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model must use at least one tool.
    Any {
        /// Disable parallel tool use.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model must use the specified tool.
    Tool {
        /// Name of the tool to use.
        name: String,

        /// Disable parallel tool use.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model is not allowed to use tools.
    None,
}

impl BetaToolChoice {
    /// Creates an auto tool choice.
    #[must_use]
    pub const fn auto() -> Self {
        Self::Auto {
            disable_parallel_tool_use: None,
        }
    }

    /// Creates an any tool choice.
    #[must_use]
    pub const fn any() -> Self {
        Self::Any {
            disable_parallel_tool_use: None,
        }
    }

    /// Creates a specific tool choice.
    #[must_use]
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool {
            name: name.into(),
            disable_parallel_tool_use: None,
        }
    }

    /// Creates a none tool choice.
    #[must_use]
    pub const fn none() -> Self {
        Self::None
    }
}

// =============================================================================
// Beta Service Tier
// =============================================================================

/// Service tier for beta request processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum BetaServiceTier {
    /// Automatic tier selection.
    #[default]
    Auto,
    /// Standard tier only.
    StandardOnly,
}

// =============================================================================
// Beta Message Count Tokens
// =============================================================================

/// Parameters for counting tokens in a beta message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaMessageCountTokensParams {
    /// The model to use for counting.
    pub model: Model,

    /// The messages to count.
    pub messages: Vec<BetaMessageParam>,

    /// Beta features to enable.
    #[serde(skip)]
    pub betas: Vec<BetaFeature>,

    /// System prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<BetaSystemPrompt>,

    /// Tools to include in the count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<BetaToolUnion>>,

    /// Tool choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<BetaToolChoice>,

    /// Thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// Response from the beta token counting endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaMessageTokensCount {
    /// The number of input tokens.
    pub input_tokens: i64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_config_enabled() {
        let config = ThinkingConfig::enabled(10000);
        assert!(config.is_enabled());
        assert_eq!(config.budget_tokens(), Some(10000));

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"enabled\""));
        assert!(json.contains("\"budget_tokens\":10000"));
    }

    #[test]
    fn test_thinking_config_disabled() {
        let config = ThinkingConfig::disabled();
        assert!(!config.is_enabled());
        assert_eq!(config.budget_tokens(), None);

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"disabled\""));
    }

    #[test]
    #[should_panic(expected = "budget_tokens must be at least 1024")]
    fn test_thinking_config_min_budget() {
        let _ = ThinkingConfig::enabled(500);
    }

    #[test]
    fn test_beta_message_param_user() {
        let msg = BetaMessageParam::user("Hello!");
        assert_eq!(msg.role, Role::User);
        match msg.content {
            BetaMessageContent::Text(text) => assert_eq!(text, "Hello!"),
            BetaMessageContent::Blocks(_) => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_beta_message_create_params() {
        let params = BetaMessageCreateParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![BetaMessageParam::user("Hello!")],
            1024,
        )
        .with_system("You are helpful.")
        .with_temperature(0.7)
        .with_thinking(ThinkingConfig::enabled(5000))
        .with_beta(BetaFeature::InterleavedThinking20250122);

        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.temperature, Some(0.7));
        assert!(params.thinking.is_some());
        assert_eq!(params.betas.len(), 1);
    }

    #[test]
    fn test_beta_message_create_builder() {
        let params = BetaMessageCreateParamsBuilder::new(Model::claude_sonnet_4_5_latest(), 1024)
            .system("You are helpful.")
            .user("Hello!")
            .temperature(0.7)
            .thinking(10000)
            .beta(BetaFeature::InterleavedThinking20250122)
            .build();

        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.messages.len(), 1);
        assert!(params.thinking.is_some());
    }

    #[test]
    fn test_beta_system_prompt_from_str() {
        let prompt: BetaSystemPrompt = "You are helpful.".into();
        match prompt {
            BetaSystemPrompt::Text(text) => assert_eq!(text, "You are helpful."),
            BetaSystemPrompt::Blocks(_) => panic!("Expected text"),
        }
    }

    #[test]
    fn test_beta_tool_choice_serialize() {
        let choice = BetaToolChoice::auto();
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"auto\""));

        let choice = BetaToolChoice::tool("web_search");
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"tool\""));
        assert!(json.contains("\"name\":\"web_search\""));
    }

    #[test]
    fn test_beta_stop_reason_serialize() {
        let reason = BetaStopReason::EndTurn;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"end_turn\"");

        let reason = BetaStopReason::ModelContextWindowExceeded;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"model_context_window_exceeded\"");
    }

    #[test]
    fn test_beta_content_block_text() {
        let block = BetaContentBlock::Text(TextBlock {
            text: "Hello".to_string(),
            citations: Vec::new(),
        });
        assert!(block.is_text());
        assert_eq!(block.as_text(), Some("Hello"));
    }

    #[test]
    fn test_beta_content_block_thinking() {
        let block = BetaContentBlock::Thinking(ThinkingBlock {
            thinking: "Let me think...".to_string(),
            signature: None,
        });
        assert!(block.is_thinking());
        assert_eq!(block.as_thinking(), Some("Let me think..."));
    }

    #[test]
    fn test_beta_service_tier() {
        let tier = BetaServiceTier::Auto;
        let json = serde_json::to_string(&tier).unwrap();
        assert_eq!(json, "\"auto\"");

        let tier = BetaServiceTier::StandardOnly;
        let json = serde_json::to_string(&tier).unwrap();
        assert_eq!(json, "\"standard_only\"");
    }

    #[test]
    fn test_beta_text_block_param() {
        let block = BetaTextBlockParam::new("Hello").with_cache_control(CacheControl::ephemeral());
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"cache_control\""));
    }

    #[test]
    fn test_beta_message_deserialize() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Hello!"}
            ],
            "model": "claude-sonnet-4-5-latest",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let message: BetaMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.id, "msg_123");
        assert!(message.has_thinking());
        assert_eq!(message.thinking(), "Let me think...");
        assert_eq!(message.text(), "Hello!");
    }
}
