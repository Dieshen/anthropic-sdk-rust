//! Type definitions for the Anthropic API.
//!
//! This module contains all the request and response types used by
//! the Anthropic API, organized into logical submodules.
//!
//! # Modules
//!
//! - [`model`] - Model identifiers and constants
//! - [`shared`] - Shared types used across the API
//! - [`content`] - Content block types for messages
//! - [`tool`] - Tool definitions and configuration
//! - [`message`] - Message types for conversations
//! - [`batch`] - Batch processing types
//! - [`usage`] - Token usage statistics
//!
//! # Examples
//!
//! Creating a simple message request:
//!
//! ```rust
//! use anthropic::types::{
//!     Model, MessageCreateParams, MessageParam,
//! };
//!
//! let params = MessageCreateParams::new(
//!     Model::claude_sonnet_4_5_latest(),
//!     vec![MessageParam::user("Hello!")],
//!     1024,
//! );
//! ```
//!
//! Using tools:
//!
//! ```rust
//! use anthropic::types::{
//!     Model, MessageCreateParams, MessageParam,
//!     ToolParam, ToolInputSchema, ToolChoice, Tool,
//! };
//!
//! let tool = ToolParam::new(
//!     "get_weather",
//!     ToolInputSchema::new()
//!         .with_properties(serde_json::json!({
//!             "location": {
//!                 "type": "string",
//!                 "description": "City name"
//!             }
//!         }))
//!         .with_required(vec!["location".to_string()]),
//! ).with_description("Get the current weather");
//!
//! let params = MessageCreateParams::new(
//!     Model::claude_sonnet_4_5_latest(),
//!     vec![MessageParam::user("What's the weather in Tokyo?")],
//!     1024,
//! )
//! .with_tools(vec![Tool::from(tool)]);
//! ```

pub mod batch;
pub mod content;
pub mod message;
pub mod model;
pub mod shared;
pub mod tool;
pub mod usage;

// Re-export commonly used types at the module level
pub use batch::{
    BatchCreateParams, BatchError, BatchListParams, BatchListResponse, BatchProcessingStatus,
    BatchRequest, BatchRequestCounts, BatchResult, BatchResultType, MessageBatch,
};

pub use content::{
    CharLocationCitation, Citation, CitationsConfig, ContentBlock, ContentBlockLocationCitation,
    ContentBlockParam, DocumentBlockParam, DocumentSource, ImageBlockParam, ImageSource,
    PageLocationCitation, RedactedThinkingBlock, RedactedThinkingBlockParam, ServerToolUseBlock,
    TextBlock, TextBlockParam, ThinkingBlock, ThinkingBlockParam, ToolResultBlockParam,
    ToolResultContent, ToolResultContentBlock, ToolUseBlock, ToolUseBlockParam, WebSearchResult,
    WebSearchResultContent, WebSearchResultLocationCitation, WebSearchToolResultBlock,
};

pub use message::{
    Message, MessageContent, MessageCreateParams, MessageCreateParamsBuilder, MessageParam,
    ServiceTierRequest, SystemPrompt, SystemPromptBlock, ThinkingConfig,
};

pub use model::Model;

pub use shared::{
    Base64Source, CacheControl, CacheControlType, CacheTtl, ExtraFields, FileSource,
    ImageMediaType, Metadata, Role, StopReason, UrlSource,
};

pub use tool::{
    BashTool, ComputerTool, ServerTool, TextEditorTool, Tool, ToolChoice, ToolInputSchema,
    ToolParam, ToolType, UserLocation, WebSearchTool,
};

pub use usage::{CacheCreation, MessageTokensCount, ServerToolUsage, ServiceTier, Usage};
