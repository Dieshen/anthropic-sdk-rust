//! Stream event types for the Anthropic API.
//!
//! This module defines the event types returned during streaming responses.
//! Events are delivered as Server-Sent Events (SSE) and represent incremental
//! updates to the message being generated.

use serde::{Deserialize, Serialize};

use crate::types::{
    ContentBlock, Message, ServerToolUseBlock, StopReason, TextBlock, ThinkingBlock, ToolUseBlock,
    WebSearchToolResultBlock,
};

// =============================================================================
// Stream Event (Main Union Type)
// =============================================================================

/// A streaming event from the Messages API.
///
/// Events are delivered in order and represent the progressive construction
/// of a message response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Start of a new message.
    ///
    /// Contains the initial message metadata including ID, model, and role.
    /// Content will be empty at this point.
    MessageStart {
        /// The initial message object.
        message: Message,
    },

    /// Start of a new content block.
    ///
    /// Indicates a new content block (text, tool use, etc.) is beginning.
    ContentBlockStart {
        /// Index of this content block in the message.
        index: u32,
        /// The initial content block (typically with empty/partial content).
        content_block: ContentBlockStartContent,
    },

    /// Incremental update to a content block.
    ///
    /// Contains partial content to append to the block at the given index.
    ContentBlockDelta {
        /// Index of the content block being updated.
        index: u32,
        /// The incremental content update.
        delta: ContentBlockDelta,
    },

    /// End of a content block.
    ///
    /// Indicates the content block at the given index is complete.
    ContentBlockStop {
        /// Index of the completed content block.
        index: u32,
    },

    /// Final message metadata.
    ///
    /// Contains the stop reason and final usage statistics.
    MessageDelta {
        /// Final message updates (stop reason, etc.).
        delta: MessageDelta,
        /// Final usage statistics.
        usage: MessageDeltaUsage,
    },

    /// End of the message stream.
    ///
    /// Indicates the message is complete.
    MessageStop,

    /// Server ping event.
    ///
    /// Keep-alive signal, can be ignored.
    Ping,

    /// Error event from the server.
    ///
    /// Indicates an error occurred during streaming.
    Error {
        /// Error details.
        error: StreamError,
    },
}

impl StreamEvent {
    /// Returns true if this is a message start event.
    #[must_use]
    pub fn is_message_start(&self) -> bool {
        matches!(self, Self::MessageStart { .. })
    }

    /// Returns true if this is a content block delta event.
    #[must_use]
    pub fn is_content_delta(&self) -> bool {
        matches!(self, Self::ContentBlockDelta { .. })
    }

    /// Returns true if this is a message stop event.
    #[must_use]
    pub fn is_message_stop(&self) -> bool {
        matches!(self, Self::MessageStop)
    }

    /// Returns true if this is a ping event.
    #[must_use]
    pub fn is_ping(&self) -> bool {
        matches!(self, Self::Ping)
    }

    /// Returns true if this is an error event.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Extracts text if this is a text delta event.
    #[must_use]
    pub fn as_text_delta(&self) -> Option<&str> {
        match self {
            Self::ContentBlockDelta {
                delta: ContentBlockDelta::TextDelta { text },
                ..
            } => Some(text),
            _ => None,
        }
    }

    /// Extracts thinking text if this is a thinking delta event.
    #[must_use]
    pub fn as_thinking_delta(&self) -> Option<&str> {
        match self {
            Self::ContentBlockDelta {
                delta: ContentBlockDelta::ThinkingDelta { thinking },
                ..
            } => Some(thinking),
            _ => None,
        }
    }

    /// Returns the block index if this event is associated with a content block.
    #[must_use]
    pub fn block_index(&self) -> Option<u32> {
        match self {
            Self::ContentBlockStart { index, .. }
            | Self::ContentBlockDelta { index, .. }
            | Self::ContentBlockStop { index } => Some(*index),
            _ => None,
        }
    }
}

// =============================================================================
// Content Block Start Content
// =============================================================================

/// Initial content block in a content_block_start event.
///
/// The content is typically empty or partial at this point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStartContent {
    /// Text block (initially empty).
    Text {
        /// Initial text (usually empty).
        text: String,
    },

    /// Extended thinking block.
    Thinking {
        /// Initial thinking content (usually empty).
        thinking: String,
    },

    /// Tool use block.
    ToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Name of the tool being invoked.
        name: String,
        /// Input will be streamed via delta events.
        input: serde_json::Value,
    },

    /// Server tool use block (e.g., web search).
    ServerToolUse {
        /// Unique ID for this tool use.
        id: String,
        /// Name of the server tool.
        name: String,
        /// Input parameters.
        input: serde_json::Value,
    },

    /// Web search tool result block.
    WebSearchToolResult {
        /// The tool use ID this result corresponds to.
        tool_use_id: String,
        /// Search result content.
        content: serde_json::Value,
    },

    /// Redacted thinking block.
    RedactedThinking {
        /// Redacted data.
        data: String,
    },
}

impl ContentBlockStartContent {
    /// Converts this start content into a full ContentBlock.
    ///
    /// The resulting block will have initial/empty content.
    #[must_use]
    pub fn into_content_block(self) -> ContentBlock {
        match self {
            Self::Text { text } => ContentBlock::Text(TextBlock {
                text,
                citations: Vec::new(),
            }),
            Self::Thinking { thinking } => ContentBlock::Thinking(ThinkingBlock {
                thinking,
                signature: None,
            }),
            Self::ToolUse { id, name, input } => {
                ContentBlock::ToolUse(ToolUseBlock { id, name, input })
            }
            Self::ServerToolUse { id, name, input } => {
                ContentBlock::ServerToolUse(ServerToolUseBlock { id, name, input })
            }
            Self::WebSearchToolResult {
                tool_use_id,
                content,
            } => ContentBlock::WebSearchToolResult(WebSearchToolResultBlock {
                tool_use_id,
                content: serde_json::from_value(content).unwrap_or_else(|_| {
                    crate::types::WebSearchResultContent::Error {
                        error: "Failed to parse content".to_string(),
                    }
                }),
            }),
            Self::RedactedThinking { data } => {
                ContentBlock::RedactedThinking(crate::types::RedactedThinkingBlock { data })
            }
        }
    }

    /// Returns true if this is a text block.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Returns true if this is a tool use block.
    #[must_use]
    pub fn is_tool_use(&self) -> bool {
        matches!(self, Self::ToolUse { .. })
    }

    /// Returns true if this is a thinking block.
    #[must_use]
    pub fn is_thinking(&self) -> bool {
        matches!(self, Self::Thinking { .. })
    }
}

// =============================================================================
// Content Block Delta
// =============================================================================

/// Incremental update to a content block.
///
/// These deltas are appended to the content block at the corresponding index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    /// Incremental text content.
    TextDelta {
        /// Text to append.
        text: String,
    },

    /// Incremental thinking content.
    ThinkingDelta {
        /// Thinking text to append.
        thinking: String,
    },

    /// Incremental tool input JSON.
    ///
    /// Tool input is streamed as partial JSON strings that should be
    /// concatenated and parsed when the block is complete.
    InputJsonDelta {
        /// Partial JSON string to append.
        partial_json: String,
    },

    /// Citation delta.
    CitationsDelta {
        /// Citation information.
        citation: serde_json::Value,
    },

    /// Signature delta (for thinking blocks).
    SignatureDelta {
        /// Signature to append.
        signature: String,
    },
}

impl ContentBlockDelta {
    /// Returns the text content if this is a text delta.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextDelta { text } => Some(text),
            _ => None,
        }
    }

    /// Returns the thinking content if this is a thinking delta.
    #[must_use]
    pub fn as_thinking(&self) -> Option<&str> {
        match self {
            Self::ThinkingDelta { thinking } => Some(thinking),
            _ => None,
        }
    }

    /// Returns the partial JSON if this is an input JSON delta.
    #[must_use]
    pub fn as_input_json(&self) -> Option<&str> {
        match self {
            Self::InputJsonDelta { partial_json } => Some(partial_json),
            _ => None,
        }
    }

    /// Returns true if this is a text delta.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::TextDelta { .. })
    }

    /// Returns true if this is a thinking delta.
    #[must_use]
    pub fn is_thinking(&self) -> bool {
        matches!(self, Self::ThinkingDelta { .. })
    }

    /// Returns true if this is an input JSON delta.
    #[must_use]
    pub fn is_input_json(&self) -> bool {
        matches!(self, Self::InputJsonDelta { .. })
    }
}

// =============================================================================
// Message Delta
// =============================================================================

/// Final message updates in a message_delta event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDelta {
    /// The reason generation stopped.
    pub stop_reason: Option<StopReason>,

    /// The stop sequence that caused generation to stop, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

// =============================================================================
// Message Delta Usage
// =============================================================================

/// Usage statistics in a message_delta event.
///
/// Contains the final output token count.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageDeltaUsage {
    /// Number of output tokens generated.
    pub output_tokens: i64,

    /// Number of tokens used for cache creation.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_creation_input_tokens: i64,

    /// Number of tokens read from cache.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_read_input_tokens: i64,
}

/// Helper function for serde skip_serializing_if
fn is_zero(val: &i64) -> bool {
    *val == 0
}

impl MessageDeltaUsage {
    /// Creates a new usage with the given output token count.
    #[must_use]
    pub fn new(output_tokens: i64) -> Self {
        Self {
            output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }
    }
}

// =============================================================================
// Stream Error
// =============================================================================

/// Error information from a streaming error event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamError {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,

    /// Human-readable error message.
    pub message: String,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.error_type, self.message)
    }
}

impl std::error::Error for StreamError {}

// =============================================================================
// Raw Event Types (for JSON parsing)
// =============================================================================

/// Raw message_start event from JSON.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RawMessageStartEvent {
    pub message: Message,
}

/// Raw content_block_start event from JSON.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RawContentBlockStartEvent {
    pub index: u32,
    pub content_block: ContentBlockStartContent,
}

/// Raw content_block_delta event from JSON.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RawContentBlockDeltaEvent {
    pub index: u32,
    pub delta: ContentBlockDelta,
}

/// Raw content_block_stop event from JSON.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RawContentBlockStopEvent {
    pub index: u32,
}

/// Raw message_delta event from JSON.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RawMessageDeltaEvent {
    pub delta: MessageDelta,
    pub usage: MessageDeltaUsage,
}

/// Raw error event from JSON.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RawErrorEvent {
    pub error: StreamError,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_message_start() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-latest",
                "usage": {"input_tokens": 10, "output_tokens": 1}
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(event.is_message_start());
    }

    #[test]
    fn test_stream_event_content_block_start_text() {
        let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 0);
                assert!(content_block.is_text());
            }
            _ => panic!("Expected ContentBlockStart"),
        }
    }

    #[test]
    fn test_stream_event_content_block_start_tool_use() {
        let json = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {}
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 1);
                assert!(content_block.is_tool_use());
            }
            _ => panic!("Expected ContentBlockStart"),
        }
    }

    #[test]
    fn test_stream_event_content_block_delta_text() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello, world!"
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(event.is_content_delta());
        assert_eq!(event.as_text_delta(), Some("Hello, world!"));
        assert_eq!(event.block_index(), Some(0));
    }

    #[test]
    fn test_stream_event_content_block_delta_input_json() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "input_json_delta",
                "partial_json": "{\"location\":"
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 1);
                assert!(delta.is_input_json());
                assert_eq!(delta.as_input_json(), Some("{\"location\":"));
            }
            _ => panic!("Expected ContentBlockDelta"),
        }
    }

    #[test]
    fn test_stream_event_content_block_delta_thinking() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "Let me think about this..."
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(
            event.as_thinking_delta(),
            Some("Let me think about this...")
        );
    }

    #[test]
    fn test_stream_event_content_block_stop() {
        let json = r#"{
            "type": "content_block_stop",
            "index": 0
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::ContentBlockStop { index } => assert_eq!(index, 0),
            _ => panic!("Expected ContentBlockStop"),
        }
    }

    #[test]
    fn test_stream_event_message_delta() {
        let json = r#"{
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn"
            },
            "usage": {
                "output_tokens": 50
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason, Some(StopReason::EndTurn));
                assert_eq!(usage.output_tokens, 50);
            }
            _ => panic!("Expected MessageDelta"),
        }
    }

    #[test]
    fn test_stream_event_message_stop() {
        let json = r#"{"type": "message_stop"}"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(event.is_message_stop());
    }

    #[test]
    fn test_stream_event_ping() {
        let json = r#"{"type": "ping"}"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(event.is_ping());
    }

    #[test]
    fn test_stream_event_error() {
        let json = r#"{
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Server is overloaded"
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        assert!(event.is_error());
        match event {
            StreamEvent::Error { error } => {
                assert_eq!(error.error_type, "overloaded_error");
                assert_eq!(error.message, "Server is overloaded");
            }
            _ => panic!("Expected Error"),
        }
    }

    #[test]
    fn test_content_block_delta_methods() {
        let text_delta = ContentBlockDelta::TextDelta {
            text: "hello".to_string(),
        };
        assert!(text_delta.is_text());
        assert!(!text_delta.is_thinking());
        assert!(!text_delta.is_input_json());
        assert_eq!(text_delta.as_text(), Some("hello"));
        assert_eq!(text_delta.as_thinking(), None);
        assert_eq!(text_delta.as_input_json(), None);

        let thinking_delta = ContentBlockDelta::ThinkingDelta {
            thinking: "thinking...".to_string(),
        };
        assert!(!thinking_delta.is_text());
        assert!(thinking_delta.is_thinking());
        assert_eq!(thinking_delta.as_thinking(), Some("thinking..."));

        let json_delta = ContentBlockDelta::InputJsonDelta {
            partial_json: "{\"key\":".to_string(),
        };
        assert!(json_delta.is_input_json());
        assert_eq!(json_delta.as_input_json(), Some("{\"key\":"));
    }

    #[test]
    fn test_content_block_start_into_content_block() {
        let text_start = ContentBlockStartContent::Text {
            text: "Hello".to_string(),
        };
        let block = text_start.into_content_block();
        assert!(block.is_text());
        assert_eq!(block.as_text(), Some("Hello"));

        let tool_start = ContentBlockStartContent::ToolUse {
            id: "toolu_123".to_string(),
            name: "get_weather".to_string(),
            input: serde_json::json!({}),
        };
        let block = tool_start.into_content_block();
        assert!(block.is_tool_use());
    }

    #[test]
    fn test_message_delta_usage() {
        let usage = MessageDeltaUsage::new(100);
        assert_eq!(usage.output_tokens, 100);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn test_stream_error_display() {
        let error = StreamError {
            error_type: "rate_limit_error".to_string(),
            message: "Rate limit exceeded".to_string(),
        };
        assert_eq!(error.to_string(), "rate_limit_error: Rate limit exceeded");
    }

    #[test]
    fn test_raw_event_parsing() {
        let json = r#"{"message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-latest","usage":{"input_tokens":10,"output_tokens":1}}}"#;
        let event: RawMessageStartEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.message.id, "msg_123");

        let json = r#"{"index":0,"content_block":{"type":"text","text":""}}"#;
        let event: RawContentBlockStartEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.index, 0);

        let json = r#"{"index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let event: RawContentBlockDeltaEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.index, 0);

        let json = r#"{"index":0}"#;
        let event: RawContentBlockStopEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.index, 0);

        let json = r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}"#;
        let event: RawMessageDeltaEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.delta.stop_reason, Some(StopReason::EndTurn));

        let json = r#"{"error":{"type":"error","message":"Something went wrong"}}"#;
        let event: RawErrorEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.error.message, "Something went wrong");
    }
}
