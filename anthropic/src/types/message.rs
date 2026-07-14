//! Message types for the Anthropic API.
//!
//! This module defines types for messages in conversations, including
//! request parameters and response types.

use serde::{Deserialize, Serialize};

use super::content::{ContentBlock, ContentBlockParam};
use super::model::Model;
use super::shared::{CacheControl, Metadata, Role, StopReason};
use super::tool::{Tool, ToolChoice};
use super::usage::Usage;

// =============================================================================
// Message Response
// =============================================================================

/// A message response from the API.
///
/// This is returned when creating a message via the Messages API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Unique identifier for this message.
    pub id: String,

    /// Object type (always "message").
    #[serde(rename = "type")]
    pub message_type: String,

    /// The role of this message (always "assistant").
    pub role: Role,

    /// Content blocks in the response.
    pub content: Vec<ContentBlock>,

    /// The model that generated this message.
    pub model: Model,

    /// The reason the model stopped generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,

    /// The stop sequence that caused generation to stop, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,

    /// Token usage statistics.
    pub usage: Usage,
}

impl Message {
    /// Returns the concatenated text content from all text blocks.
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns all tool use blocks from the response.
    #[must_use]
    pub fn tool_uses(&self) -> Vec<&super::content::ToolUseBlock> {
        self.content
            .iter()
            .filter_map(|block| block.as_tool_use())
            .collect()
    }

    /// Returns true if the model stopped due to tool use.
    #[must_use]
    pub fn has_tool_use(&self) -> bool {
        self.stop_reason == Some(StopReason::ToolUse)
    }

    /// Returns true if the model reached a natural stopping point.
    #[must_use]
    pub fn is_end_turn(&self) -> bool {
        self.stop_reason == Some(StopReason::EndTurn)
    }
}

// =============================================================================
// Message Request Parameter
// =============================================================================

/// A message in a conversation (for requests).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageParam {
    /// The role of the message author.
    pub role: Role,

    /// Content of the message.
    pub content: MessageContent,
}

impl MessageParam {
    /// Creates a new user message with text content.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a new user message with content blocks.
    #[must_use]
    pub const fn user_with_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(blocks),
        }
    }

    /// Creates a new assistant message with text content.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a new assistant message with content blocks.
    #[must_use]
    pub const fn assistant_with_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Blocks(blocks),
        }
    }
}

/// Content of a message (either text or blocks).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<ContentBlockParam>),
}

// =============================================================================
// System Prompt
// =============================================================================

/// System prompt for a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple text system prompt.
    Text(String),
    /// Structured system prompt with cache control.
    Blocks(Vec<SystemPromptBlock>),
}

impl SystemPrompt {
    /// Creates a simple text system prompt.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Creates a system prompt with a cacheable block.
    #[must_use]
    pub fn with_cache_control(content: impl Into<String>, cache_control: CacheControl) -> Self {
        Self::Blocks(vec![SystemPromptBlock {
            block_type: "text".to_string(),
            text: content.into(),
            cache_control: Some(cache_control),
        }])
    }
}

impl From<&str> for SystemPrompt {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<String> for SystemPrompt {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

/// A block in a system prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemPromptBlock {
    /// Block type (always "text").
    #[serde(rename = "type")]
    pub block_type: String,

    /// Text content.
    pub text: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

// =============================================================================
// Message Create Parameters
// =============================================================================

/// Parameters for creating a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageCreateParams {
    /// The model to use for generation.
    pub model: Model,

    /// The messages in the conversation.
    pub messages: Vec<MessageParam>,

    /// Maximum number of tokens to generate.
    pub max_tokens: u32,

    /// System prompt for the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Sampling temperature (0.0 to 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p sampling parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Stop sequences that will halt generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Tools available for the model to use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// How the model should choose tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Request metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,

    /// Extended thinking configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Service tier for request processing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTierRequest>,
}

impl MessageCreateParams {
    /// Creates new message parameters.
    #[must_use]
    pub const fn new(model: Model, messages: Vec<MessageParam>, max_tokens: u32) -> Self {
        Self {
            model,
            messages,
            max_tokens,
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

    /// Sets the system prompt.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<SystemPrompt>) -> Self {
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
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the tool choice.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets extended thinking.
    #[must_use]
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig {
            thinking_type: "enabled".to_string(),
            budget_tokens,
        });
        self
    }

    /// Sets the metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Extended thinking configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Type of thinking (always "enabled" to enable).
    #[serde(rename = "type")]
    pub thinking_type: String,

    /// Maximum tokens for thinking.
    pub budget_tokens: u32,
}

/// Service tier request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ServiceTierRequest {
    /// Automatic tier selection.
    #[default]
    Auto,
    /// Standard tier.
    Standard,
    /// Priority tier.
    Priority,
}

// =============================================================================
// Message Create Builder
// =============================================================================

/// Builder for creating message parameters.
#[derive(Debug, Clone)]
pub struct MessageCreateParamsBuilder {
    model: Model,
    messages: Vec<MessageParam>,
    max_tokens: u32,
    params: MessageCreateParams,
}

impl MessageCreateParamsBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new(model: Model, max_tokens: u32) -> Self {
        Self {
            model: model.clone(),
            messages: Vec::new(),
            max_tokens,
            params: MessageCreateParams::new(model, Vec::new(), max_tokens),
        }
    }

    /// Adds a user message.
    #[must_use]
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(MessageParam::user(content));
        self
    }

    /// Adds an assistant message.
    #[must_use]
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(MessageParam::assistant(content));
        self
    }

    /// Adds a message.
    #[must_use]
    pub fn message(mut self, message: MessageParam) -> Self {
        self.messages.push(message);
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn system(mut self, system: impl Into<SystemPrompt>) -> Self {
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
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.params.tools = Some(tools);
        self
    }

    /// Sets the tool choice.
    #[must_use]
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.params.tool_choice = Some(tool_choice);
        self
    }

    /// Sets extended thinking.
    #[must_use]
    pub fn thinking(mut self, budget_tokens: u32) -> Self {
        self.params.thinking = Some(ThinkingConfig {
            thinking_type: "enabled".to_string(),
            budget_tokens,
        });
        self
    }

    /// Builds the parameters.
    #[must_use]
    pub fn build(mut self) -> MessageCreateParams {
        self.params.model = self.model;
        self.params.messages = self.messages;
        self.params.max_tokens = self.max_tokens;
        self.params
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_param_user() {
        let msg = MessageParam::user("Hello!");
        assert_eq!(msg.role, Role::User);
        match msg.content {
            MessageContent::Text(text) => assert_eq!(text, "Hello!"),
            MessageContent::Blocks(_) => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_message_param_assistant() {
        let msg = MessageParam::assistant("Hi there!");
        assert_eq!(msg.role, Role::Assistant);
    }

    #[test]
    fn test_message_param_serialize() {
        let msg = MessageParam::user("Hello!");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello!\""));
    }

    #[test]
    fn test_message_create_params() {
        let params = MessageCreateParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Hello!")],
            1024,
        )
        .with_system("You are helpful.")
        .with_temperature(0.7);

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"system\":\"You are helpful.\""));
    }

    #[test]
    fn test_message_create_builder() {
        let params = MessageCreateParamsBuilder::new(Model::claude_sonnet_4_5_latest(), 1024)
            .system("You are helpful.")
            .user("Hello!")
            .temperature(0.7)
            .build();

        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.messages.len(), 1);
        assert_eq!(params.temperature, Some(0.7));
    }

    #[test]
    fn test_system_prompt_text() {
        let prompt = SystemPrompt::text("You are helpful.");
        let json = serde_json::to_string(&prompt).unwrap();
        assert_eq!(json, "\"You are helpful.\"");
    }

    #[test]
    fn test_system_prompt_from_str() {
        let prompt: SystemPrompt = "You are helpful.".into();
        match prompt {
            SystemPrompt::Text(text) => assert_eq!(text, "You are helpful."),
            SystemPrompt::Blocks(_) => panic!("Expected text"),
        }
    }

    #[test]
    fn test_message_response_deserialize() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-5-latest",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        assert_eq!(message.id, "msg_123");
        assert_eq!(message.text(), "Hello!");
        assert!(message.is_end_turn());
    }

    #[test]
    fn test_thinking_config() {
        let config = ThinkingConfig {
            thinking_type: "enabled".to_string(),
            budget_tokens: 10000,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"enabled\""));
        assert!(json.contains("\"budget_tokens\":10000"));
    }
}
