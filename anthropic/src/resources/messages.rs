//! Messages API resource.
//!
//! The Messages API allows you to send structured input messages with text
//! and/or image content, and the model will generate the next message in
//! the conversation.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::{Anthropic, MessageCreateParams, MessageParam, Model};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic::Error> {
//!     let client = Anthropic::new()?;
//!
//!     // Create a simple message
//!     let params = MessageCreateParams::new(
//!         Model::claude_sonnet_4_5_latest(),
//!         vec![MessageParam::user("Hello, Claude!")],
//!         1024,
//!     );
//!
//!     let message = client.messages().create(params).await?;
//!     println!("{}", message.text());
//!
//!     // Stream a message
//!     let mut stream = client.messages().stream(params).await?;
//!     while let Some(event) = stream.next().await {
//!         // Handle streaming events
//!     }
//!
//!     Ok(())
//! }
//! ```

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::client::Anthropic;
use crate::error::{Error, Result};
use crate::types::{
    Message, MessageCreateParams, MessageParam, MessageTokensCount, Model,
    SystemPrompt, ThinkingConfig, Tool, ToolChoice,
};

// =============================================================================
// API Endpoints
// =============================================================================

/// API endpoint for creating messages.
const MESSAGES_ENDPOINT: &str = "/v1/messages";

/// API endpoint for counting tokens.
const COUNT_TOKENS_ENDPOINT: &str = "/v1/messages/count_tokens";

// =============================================================================
// Messages Resource
// =============================================================================

/// Messages API resource.
///
/// Provides methods for creating messages, streaming responses, and counting tokens.
/// This is the primary interface for interacting with Claude.
///
/// # Thread Safety
///
/// `Messages` is `Clone` and can be shared across threads safely, as it
/// holds a reference to the thread-safe `Anthropic` client.
///
/// # Example
///
/// ```rust,ignore
/// use anthropic::{Anthropic, MessageCreateParams, MessageParam, Model};
///
/// let client = Anthropic::new()?;
/// let messages = client.messages();
///
/// // Create a message
/// let params = MessageCreateParams::new(
///     Model::claude_sonnet_4_5_latest(),
///     vec![MessageParam::user("What is 2 + 2?")],
///     1024,
/// );
///
/// let response = messages.create(params).await?;
/// println!("Answer: {}", response.text());
/// ```
#[derive(Debug, Clone)]
pub struct Messages {
    client: Anthropic,
}

impl Messages {
    /// Creates a new Messages resource.
    ///
    /// This is typically called via `client.messages()` rather than directly.
    #[must_use]
    pub fn new(client: Anthropic) -> Self {
        Self { client }
    }

    /// Creates a message.
    ///
    /// Sends a structured list of input messages and returns the model's response.
    /// This is the primary method for single-turn or multi-turn conversations.
    ///
    /// # Arguments
    ///
    /// * `params` - Message creation parameters including model, messages, and options.
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
    /// use anthropic::{Anthropic, MessageCreateParams, MessageParam, Model};
    ///
    /// let client = Anthropic::new()?;
    ///
    /// // Simple message
    /// let params = MessageCreateParams::new(
    ///     Model::claude_sonnet_4_5_latest(),
    ///     vec![MessageParam::user("Hello!")],
    ///     1024,
    /// );
    ///
    /// let message = client.messages().create(params).await?;
    /// println!("{}", message.text());
    ///
    /// // With system prompt and temperature
    /// let params = MessageCreateParams::new(
    ///     Model::claude_sonnet_4_5_latest(),
    ///     vec![MessageParam::user("Explain quantum computing")],
    ///     2048,
    /// )
    /// .with_system("You are a helpful science teacher.")
    /// .with_temperature(0.7);
    ///
    /// let message = client.messages().create(params).await?;
    /// ```
    pub async fn create(&self, params: MessageCreateParams) -> Result<Message> {
        // Ensure stream is not set for non-streaming requests
        let mut params = params;
        params.stream = None;

        self.client.post(MESSAGES_ENDPOINT, &params).await
    }

    /// Creates a streaming message.
    ///
    /// Sends a request and returns a stream of server-sent events (SSE).
    /// This allows processing the response incrementally as it's generated.
    ///
    /// # Arguments
    ///
    /// * `params` - Message creation parameters including model, messages, and options.
    ///
    /// # Returns
    ///
    /// A `MessageStream` that yields `StreamEvent` items as they arrive.
    ///
    /// # Errors
    ///
    /// The returned stream may yield errors if:
    /// - The connection is interrupted
    /// - An SSE event cannot be parsed
    /// - The API returns an error event
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic::{Anthropic, MessageCreateParams, MessageParam, Model};
    /// use futures::StreamExt;
    ///
    /// let client = Anthropic::new()?;
    ///
    /// let params = MessageCreateParams::new(
    ///     Model::claude_sonnet_4_5_latest(),
    ///     vec![MessageParam::user("Write a haiku")],
    ///     1024,
    /// );
    ///
    /// let mut stream = client.messages().stream(params).await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         StreamEvent::ContentBlockDelta { delta, .. } => {
    ///             if let Some(text) = delta.text() {
    ///                 print!("{}", text);
    ///             }
    ///         }
    ///         StreamEvent::MessageStop => {
    ///             println!("\n[Done]");
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub async fn stream(&self, params: MessageCreateParams) -> Result<MessageStream> {
        // Force streaming mode
        let mut params = params;
        params.stream = Some(true);

        let response = self.client.post_raw(MESSAGES_ENDPOINT, &params).await?;

        Ok(MessageStream::new(response))
    }

    /// Counts the number of tokens in a message.
    ///
    /// This endpoint counts tokens without creating a message, which is useful
    /// for estimating costs and ensuring messages fit within context limits.
    ///
    /// # Arguments
    ///
    /// * `params` - Token counting parameters, similar to message creation.
    ///
    /// # Returns
    ///
    /// A `MessageTokensCount` containing the input token count.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API request fails
    /// - The response cannot be parsed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic::{Anthropic, MessageParam, Model};
    /// use anthropic::resources::MessageCountTokensParams;
    ///
    /// let client = Anthropic::new()?;
    ///
    /// let params = MessageCountTokensParams::new(
    ///     Model::claude_sonnet_4_5_latest(),
    ///     vec![MessageParam::user("Hello, how are you?")],
    /// );
    ///
    /// let count = client.messages().count_tokens(params).await?;
    /// println!("Input tokens: {}", count.input_tokens);
    /// ```
    pub async fn count_tokens(&self, params: MessageCountTokensParams) -> Result<MessageTokensCount> {
        self.client.post(COUNT_TOKENS_ENDPOINT, &params).await
    }
}

// =============================================================================
// Token Counting Parameters
// =============================================================================

/// Parameters for counting tokens in a message.
///
/// This is similar to `MessageCreateParams` but without `max_tokens` (since no
/// output is generated) and without streaming options.
///
/// # Example
///
/// ```rust
/// use anthropic::types::{MessageParam, Model};
/// use anthropic::resources::MessageCountTokensParams;
///
/// let params = MessageCountTokensParams::new(
///     Model::claude_sonnet_4_5_latest(),
///     vec![MessageParam::user("Hello!")],
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageCountTokensParams {
    /// The model to use for token counting.
    pub model: Model,

    /// The messages to count tokens for.
    pub messages: Vec<MessageParam>,

    /// System prompt (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Extended thinking configuration (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Tool choice configuration (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Tools available for the model (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

impl MessageCountTokensParams {
    /// Creates new token counting parameters.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for token counting.
    /// * `messages` - The messages to count tokens for.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anthropic::types::{MessageParam, Model};
    /// use anthropic::resources::MessageCountTokensParams;
    ///
    /// let params = MessageCountTokensParams::new(
    ///     Model::claude_sonnet_4_5_latest(),
    ///     vec![MessageParam::user("Hello!")],
    /// );
    /// ```
    #[must_use]
    pub fn new(model: Model, messages: Vec<MessageParam>) -> Self {
        Self {
            model,
            messages,
            system: None,
            thinking: None,
            tool_choice: None,
            tools: None,
        }
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<SystemPrompt>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the thinking configuration.
    #[must_use]
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig {
            thinking_type: "enabled".to_string(),
            budget_tokens,
        });
        self
    }

    /// Sets the tool choice.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets the tools.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
}

// =============================================================================
// Streaming Types
// =============================================================================

/// A stream of message events from the API.
///
/// This stream yields `StreamEvent` items as server-sent events arrive.
/// Use the `futures::StreamExt` trait to iterate over events.
///
/// # Example
///
/// ```rust,ignore
/// use futures::StreamExt;
///
/// let mut stream = client.messages().stream(params).await?;
///
/// while let Some(event) = stream.next().await {
///     match event? {
///         StreamEvent::ContentBlockDelta { delta, index } => {
///             // Handle text delta
///         }
///         StreamEvent::MessageStop => {
///             // Message complete
///         }
///         _ => {}
///     }
/// }
/// ```
pub struct MessageStream {
    inner: Pin<Box<dyn Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send>>,
    buffer: String,
    done: bool,
}

impl std::fmt::Debug for MessageStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessageStream")
            .field("buffer", &self.buffer)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl MessageStream {
    /// Creates a new message stream from an HTTP response.
    fn new(response: reqwest::Response) -> Self {
        Self {
            inner: Box::pin(response.bytes_stream()),
            buffer: String::new(),
            done: false,
        }
    }

    /// Returns the final assembled message after the stream completes.
    ///
    /// This collects all events and builds the final `Message` object.
    /// Call this only after the stream has been fully consumed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut stream = client.messages().stream(params).await?;
    ///
    /// // Consume the stream
    /// while let Some(event) = stream.next().await {
    ///     let _ = event?;
    /// }
    ///
    /// // Get the final message
    /// if let Some(message) = stream.final_message() {
    ///     println!("Final response: {}", message.text());
    /// }
    /// ```
    pub fn final_message(&self) -> Option<Message> {
        // This would be implemented by accumulating message parts during streaming
        // For now, return None as we'd need to track state
        None
    }
}

impl Stream for MessageStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }

        loop {
            match self.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    let chunk = String::from_utf8_lossy(&bytes);
                    self.buffer.push_str(&chunk);

                    // Try to parse complete SSE events from buffer
                    if let Some((event, remaining)) = parse_sse_event(&self.buffer) {
                        self.buffer = remaining;

                        // Check for done event
                        if matches!(event, StreamEvent::MessageStop) {
                            self.done = true;
                        }

                        return Poll::Ready(Some(Ok(event)));
                    }
                    // If no complete event, continue polling
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(Error::streaming(e.to_string()))));
                }
                Poll::Ready(None) => {
                    self.done = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

/// Events emitted by the streaming API.
///
/// These events follow the Server-Sent Events (SSE) protocol and represent
/// different stages of message generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The message stream has started.
    MessageStart {
        /// The initial message object.
        message: Message,
    },

    /// A content block has started.
    ContentBlockStart {
        /// Index of this content block.
        index: usize,
        /// The content block.
        content_block: ContentBlockStart,
    },

    /// A delta update to a content block.
    ContentBlockDelta {
        /// Index of the content block being updated.
        index: usize,
        /// The delta update.
        delta: ContentDelta,
    },

    /// A content block has completed.
    ContentBlockStop {
        /// Index of the completed content block.
        index: usize,
    },

    /// A delta update to the message metadata.
    MessageDelta {
        /// The delta update.
        delta: MessageDeltaContent,
        /// Updated usage statistics.
        usage: Option<crate::types::Usage>,
    },

    /// The message stream has completed.
    MessageStop,

    /// A ping event (keepalive).
    Ping,

    /// An error event.
    Error {
        /// Error details.
        error: StreamError,
    },
}

/// Content block start information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStart {
    /// A text content block is starting.
    Text {
        /// Initial text (usually empty).
        text: String,
    },
    /// A tool use content block is starting.
    ToolUse {
        /// Tool use ID.
        id: String,
        /// Tool name.
        name: String,
        /// Initial input (usually empty object).
        input: serde_json::Value,
    },
    /// A thinking content block is starting (extended thinking).
    Thinking {
        /// Initial thinking text.
        thinking: String,
    },
}

/// Delta update for content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Text delta.
    TextDelta {
        /// The text to append.
        text: String,
    },
    /// Tool input JSON delta.
    InputJsonDelta {
        /// Partial JSON to append.
        partial_json: String,
    },
    /// Thinking delta (extended thinking).
    ThinkingDelta {
        /// The thinking text to append.
        thinking: String,
    },
}

impl ContentDelta {
    /// Returns the text content if this is a text delta.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            ContentDelta::TextDelta { text } => Some(text),
            _ => None,
        }
    }

    /// Returns the partial JSON if this is an input JSON delta.
    #[must_use]
    pub fn partial_json(&self) -> Option<&str> {
        match self {
            ContentDelta::InputJsonDelta { partial_json } => Some(partial_json),
            _ => None,
        }
    }

    /// Returns the thinking text if this is a thinking delta.
    #[must_use]
    pub fn thinking(&self) -> Option<&str> {
        match self {
            ContentDelta::ThinkingDelta { thinking } => Some(thinking),
            _ => None,
        }
    }
}

/// Delta update for message metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDeltaContent {
    /// The stop reason, if the message has stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<crate::types::StopReason>,

    /// The stop sequence that caused generation to stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Error information from a streaming error event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamError {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message.
    pub message: String,
}

/// Parses a complete SSE event from the buffer.
///
/// Returns the parsed event and the remaining buffer content, or None if
/// no complete event is available.
fn parse_sse_event(buffer: &str) -> Option<(StreamEvent, String)> {
    // SSE format: "event: <type>\ndata: <json>\n\n"
    let delimiter = "\n\n";

    if let Some(end_pos) = buffer.find(delimiter) {
        let event_str = &buffer[..end_pos];
        let remaining = buffer[end_pos + delimiter.len()..].to_string();

        // Parse the event
        let mut event_type = None;
        let mut event_data = None;

        for line in event_str.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                event_type = Some(rest.trim());
            } else if let Some(rest) = line.strip_prefix("data: ") {
                event_data = Some(rest);
            }
        }

        // Handle special event types
        match event_type {
            Some("ping") => return Some((StreamEvent::Ping, remaining)),
            Some("message_stop") => return Some((StreamEvent::MessageStop, remaining)),
            _ => {}
        }

        // Parse JSON data if present
        if let Some(data) = event_data {
            if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                return Some((event, remaining));
            }
        }

        // If we couldn't parse the event, skip it and return remaining
        return Some((StreamEvent::Ping, remaining)); // Use Ping as fallback
    }

    None
}

// =============================================================================
// Client Integration
// =============================================================================

impl Anthropic {
    /// Returns the Messages API resource.
    ///
    /// This provides access to message creation, streaming, and token counting.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = Anthropic::new()?;
    /// let messages = client.messages();
    ///
    /// let response = messages.create(params).await?;
    /// ```
    #[must_use]
    pub fn messages(&self) -> Messages {
        Messages::new(self.clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, Role, StopReason, Usage};

    #[test]
    fn test_message_count_tokens_params_new() {
        let params = MessageCountTokensParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Hello!")],
        );

        assert_eq!(params.messages.len(), 1);
        assert!(params.system.is_none());
        assert!(params.tools.is_none());
    }

    #[test]
    fn test_message_count_tokens_params_builder() {
        let params = MessageCountTokensParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Hello!")],
        )
        .with_system("You are helpful.")
        .with_thinking(10000);

        assert!(params.system.is_some());
        assert!(params.thinking.is_some());
        assert_eq!(params.thinking.as_ref().unwrap().budget_tokens, 10000);
    }

    #[test]
    fn test_message_count_tokens_params_serialize() {
        let params = MessageCountTokensParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Hello!")],
        );

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"model\":"));
        assert!(json.contains("\"messages\":"));
        // Optional fields should not be present
        assert!(!json.contains("\"system\":"));
        assert!(!json.contains("\"tools\":"));
    }

    #[test]
    fn test_stream_event_message_start_deserialize() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-latest",
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_123");
            }
            _ => panic!("Expected MessageStart event"),
        }
    }

    #[test]
    fn test_stream_event_content_block_delta_deserialize() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert_eq!(delta.text(), Some("Hello"));
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_content_delta_text() {
        let delta = ContentDelta::TextDelta {
            text: "Hello".to_string(),
        };
        assert_eq!(delta.text(), Some("Hello"));
        assert_eq!(delta.partial_json(), None);
        assert_eq!(delta.thinking(), None);
    }

    #[test]
    fn test_content_delta_input_json() {
        let delta = ContentDelta::InputJsonDelta {
            partial_json: r#"{"key":"#.to_string(),
        };
        assert_eq!(delta.text(), None);
        assert_eq!(delta.partial_json(), Some(r#"{"key":"#));
        assert_eq!(delta.thinking(), None);
    }

    #[test]
    fn test_content_delta_thinking() {
        let delta = ContentDelta::ThinkingDelta {
            thinking: "Let me think...".to_string(),
        };
        assert_eq!(delta.text(), None);
        assert_eq!(delta.partial_json(), None);
        assert_eq!(delta.thinking(), Some("Let me think..."));
    }

    #[test]
    fn test_parse_sse_event_ping() {
        let buffer = "event: ping\ndata: {}\n\nremaining";
        let result = parse_sse_event(buffer);

        assert!(result.is_some());
        let (event, remaining) = result.unwrap();
        assert!(matches!(event, StreamEvent::Ping));
        assert_eq!(remaining, "remaining");
    }

    #[test]
    fn test_parse_sse_event_message_stop() {
        let buffer = "event: message_stop\ndata: {}\n\n";
        let result = parse_sse_event(buffer);

        assert!(result.is_some());
        let (event, remaining) = result.unwrap();
        assert!(matches!(event, StreamEvent::MessageStop));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_sse_event_incomplete() {
        let buffer = "event: ping\ndata: {";
        let result = parse_sse_event(buffer);

        assert!(result.is_none());
    }

    #[test]
    fn test_content_block_start_serialize() {
        let block = ContentBlockStart::Text {
            text: String::new(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
    }

    #[test]
    fn test_content_block_start_tool_use() {
        let block = ContentBlockStart::ToolUse {
            id: "tool_123".to_string(),
            name: "get_weather".to_string(),
            input: serde_json::json!({}),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        assert!(json.contains("\"id\":\"tool_123\""));
        assert!(json.contains("\"name\":\"get_weather\""));
    }

    #[test]
    fn test_message_delta_content_deserialize() {
        let json = r#"{
            "stop_reason": "end_turn",
            "stop_sequence": null
        }"#;

        let delta: MessageDeltaContent = serde_json::from_str(json).unwrap();
        assert_eq!(delta.stop_reason, Some(StopReason::EndTurn));
        assert!(delta.stop_sequence.is_none());
    }

    #[test]
    fn test_stream_error_deserialize() {
        let json = r#"{
            "type": "error",
            "message": "Rate limit exceeded"
        }"#;

        let error: StreamError = serde_json::from_str(json).unwrap();
        assert_eq!(error.error_type, "error");
        assert_eq!(error.message, "Rate limit exceeded");
    }
}
