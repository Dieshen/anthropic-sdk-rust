//! Async stream wrapper for SSE message streaming.
//!
//! This module provides the `MessageStream` type which implements the `futures::Stream`
//! trait, allowing async iteration over streaming events from the Anthropic API.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::stream::Stream;
use pin_project_lite::pin_project;

use super::events::{
    ContentBlockDelta, ContentBlockStartContent, MessageDelta, MessageDeltaUsage,
    RawContentBlockDeltaEvent, RawContentBlockStartEvent, RawContentBlockStopEvent, RawErrorEvent,
    RawMessageDeltaEvent, RawMessageStartEvent, StreamEvent,
};
use super::sse::{SseDecoder, SseError, SseEvent};
use crate::types::{ContentBlock, Message, Usage};

// =============================================================================
// Error Type
// =============================================================================

/// Errors that can occur during message streaming.
#[derive(Debug)]
pub enum MessageStreamError {
    /// SSE decoding error.
    Sse(SseError),

    /// JSON parsing error.
    Json(serde_json::Error),

    /// IO/transport error.
    Io(std::io::Error),

    /// HTTP error from reqwest.
    Http(String),

    /// Server returned an error event.
    Server(String),

    /// Unexpected event type received.
    UnexpectedEvent(String),

    /// Stream ended unexpectedly.
    UnexpectedEnd,
}

impl std::fmt::Display for MessageStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sse(e) => write!(f, "SSE error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Http(e) => write!(f, "HTTP error: {e}"),
            Self::Server(e) => write!(f, "Server error: {e}"),
            Self::UnexpectedEvent(e) => write!(f, "Unexpected event type: {e}"),
            Self::UnexpectedEnd => write!(f, "Stream ended unexpectedly"),
        }
    }
}

impl std::error::Error for MessageStreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sse(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SseError> for MessageStreamError {
    fn from(err: SseError) -> Self {
        Self::Sse(err)
    }
}

impl From<serde_json::Error> for MessageStreamError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<std::io::Error> for MessageStreamError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

// =============================================================================
// Stream State
// =============================================================================

/// Current state of the message stream.
///
/// Tracks the accumulated message content as events are received.
#[derive(Debug, Clone)]
pub struct StreamState {
    /// The message being built (from message_start event).
    pub message: Option<Message>,

    /// Content blocks being accumulated.
    pub content_blocks: Vec<ContentBlock>,

    /// Accumulated text per content block (for convenience).
    pub text_buffers: Vec<String>,

    /// Accumulated tool input JSON per content block.
    pub input_json_buffers: Vec<String>,

    /// Final usage statistics.
    pub usage: Option<Usage>,

    /// Whether the stream has completed.
    pub is_complete: bool,
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamState {
    /// Creates a new empty stream state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            message: None,
            content_blocks: Vec::new(),
            text_buffers: Vec::new(),
            input_json_buffers: Vec::new(),
            usage: None,
            is_complete: false,
        }
    }

    /// Returns the concatenated text from all text blocks.
    #[must_use]
    pub fn text(&self) -> String {
        self.text_buffers.join("")
    }

    /// Returns the current message with accumulated content.
    ///
    /// Returns `None` if no message_start event has been received.
    #[must_use]
    pub fn current_message(&self) -> Option<Message> {
        self.message.as_ref().map(|msg| {
            let mut message = msg.clone();
            message.content = self.content_blocks.clone();
            if let Some(ref usage) = self.usage {
                message.usage = usage.clone();
            }
            message
        })
    }

    /// Updates the state based on a stream event.
    pub fn apply_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::MessageStart { message } => {
                self.message = Some(message.clone());
                self.usage = Some(message.usage.clone());
            }

            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let idx = *index as usize;

                // Ensure we have space for this index
                while self.content_blocks.len() <= idx {
                    self.content_blocks
                        .push(ContentBlock::Text(crate::types::TextBlock::new("")));
                    self.text_buffers.push(String::new());
                    self.input_json_buffers.push(String::new());
                }

                // Set the content block
                self.content_blocks[idx] = content_block.clone().into_content_block();
            }

            StreamEvent::ContentBlockDelta { index, delta } => {
                let idx = *index as usize;

                // Ensure buffers exist for this index
                while self.text_buffers.len() <= idx {
                    self.text_buffers.push(String::new());
                    self.input_json_buffers.push(String::new());
                }

                match delta {
                    ContentBlockDelta::TextDelta { text } => {
                        self.text_buffers[idx].push_str(text);

                        // Update the content block
                        if let Some(ContentBlock::Text(block)) = self.content_blocks.get_mut(idx) {
                            block.text.push_str(text);
                        }
                    }
                    ContentBlockDelta::ThinkingDelta { thinking } => {
                        // Update thinking block
                        if let Some(ContentBlock::Thinking(block)) =
                            self.content_blocks.get_mut(idx)
                        {
                            block.thinking.push_str(thinking);
                        }
                    }
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        self.input_json_buffers[idx].push_str(partial_json);
                    }
                    ContentBlockDelta::SignatureDelta { signature } => {
                        // Update thinking block signature
                        if let Some(ContentBlock::Thinking(block)) =
                            self.content_blocks.get_mut(idx)
                        {
                            let sig = block.signature.get_or_insert_with(String::new);
                            sig.push_str(signature);
                        }
                    }
                    ContentBlockDelta::CitationsDelta { .. } => {
                        // Citations are accumulated but not typically needed incrementally
                    }
                }
            }

            StreamEvent::ContentBlockStop { index } => {
                let idx = *index as usize;

                // Parse accumulated JSON for tool use blocks
                if idx < self.input_json_buffers.len() && !self.input_json_buffers[idx].is_empty() {
                    if let Some(ContentBlock::ToolUse(block)) = self.content_blocks.get_mut(idx) {
                        if let Ok(input) = serde_json::from_str(&self.input_json_buffers[idx]) {
                            block.input = input;
                        }
                    }
                }
            }

            StreamEvent::MessageDelta { delta, usage } => {
                // Update stop reason in message
                if let Some(ref mut msg) = self.message {
                    msg.stop_reason = delta.stop_reason.clone();
                    msg.stop_sequence = delta.stop_sequence.clone();
                }

                // Update usage
                if let Some(ref mut existing_usage) = self.usage {
                    existing_usage.output_tokens = usage.output_tokens;
                }
            }

            StreamEvent::MessageStop => {
                self.is_complete = true;
            }

            StreamEvent::Ping | StreamEvent::Error { .. } => {
                // No state update needed
            }
        }
    }
}

// =============================================================================
// Message Stream
// =============================================================================

pin_project! {
    /// Async stream of message events from the Anthropic API.
    ///
    /// Implements `futures::Stream` to yield `StreamEvent` items.
    /// The stream automatically decodes SSE data and parses JSON events.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = MessageStream::new(byte_stream);
    ///
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(event) => {
    ///             if let StreamEvent::ContentBlockDelta { delta, .. } = event {
    ///                 if let ContentBlockDelta::TextDelta { text } = delta {
    ///                     print!("{}", text);
    ///                 }
    ///             }
    ///         }
    ///         Err(e) => eprintln!("Stream error: {}", e),
    ///     }
    /// }
    ///
    /// // Get the final accumulated message
    /// let message = stream.state().current_message();
    /// ```
    #[derive(Debug)]
    pub struct MessageStream<S> {
        // The underlying byte stream
        #[pin]
        inner: S,

        // SSE decoder
        decoder: SseDecoder,

        // Buffered events ready to yield
        event_buffer: Vec<StreamEvent>,

        // Current buffer index
        buffer_index: usize,

        // Accumulated stream state
        state: StreamState,

        // Whether the stream has finished
        finished: bool,
    }
}

impl<S> MessageStream<S> {
    /// Creates a new message stream from a byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            decoder: SseDecoder::new(),
            event_buffer: Vec::new(),
            buffer_index: 0,
            state: StreamState::new(),
            finished: false,
        }
    }

    /// Returns a reference to the current stream state.
    #[must_use]
    pub fn state(&self) -> &StreamState {
        &self.state
    }

    /// Returns the accumulated text so far.
    #[must_use]
    pub fn text(&self) -> String {
        self.state.text()
    }

    /// Returns the current message with accumulated content.
    #[must_use]
    pub fn current_message(&self) -> Option<Message> {
        self.state.current_message()
    }

    /// Returns true if the stream has completed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.state.is_complete
    }

    /// Consumes the stream and returns the final message.
    ///
    /// Returns `None` if no message was received.
    #[must_use]
    pub fn into_message(self) -> Option<Message> {
        self.state.current_message()
    }
}

impl<S, E> Stream for MessageStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
    E: Into<MessageStreamError>,
{
    type Item = Result<StreamEvent, MessageStreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Return buffered events first
        if *this.buffer_index < this.event_buffer.len() {
            let event = this.event_buffer[*this.buffer_index].clone();
            *this.buffer_index += 1;

            // Update state
            this.state.apply_event(&event);

            return Poll::Ready(Some(Ok(event)));
        }

        // Clear the buffer if we've consumed everything
        if !this.event_buffer.is_empty() {
            this.event_buffer.clear();
            *this.buffer_index = 0;
        }

        // Check if we're already finished
        if *this.finished {
            return Poll::Ready(None);
        }

        // Poll the underlying stream for more data
        loop {
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    // Decode SSE events
                    let sse_events = match this.decoder.decode(&bytes) {
                        Ok(events) => events,
                        Err(e) => return Poll::Ready(Some(Err(e.into()))),
                    };

                    // Parse each SSE event into a StreamEvent
                    for sse_event in sse_events {
                        match parse_sse_event(&sse_event) {
                            Ok(Some(event)) => {
                                this.event_buffer.push(event);
                            }
                            Ok(None) => {
                                // Ping event, skip
                            }
                            Err(e) => {
                                return Poll::Ready(Some(Err(e)));
                            }
                        }
                    }

                    // If we got events, return the first one
                    if !this.event_buffer.is_empty() {
                        let event = this.event_buffer[0].clone();
                        *this.buffer_index = 1;

                        // Update state
                        this.state.apply_event(&event);

                        // Check for terminal events
                        if matches!(event, StreamEvent::MessageStop | StreamEvent::Error { .. }) {
                            *this.finished = true;
                        }

                        return Poll::Ready(Some(Ok(event)));
                    }

                    // No events yet, continue polling
                    continue;
                }

                Poll::Ready(Some(Err(e))) => {
                    *this.finished = true;
                    return Poll::Ready(Some(Err(e.into())));
                }

                Poll::Ready(None) => {
                    // Stream ended, check for any pending events
                    if let Some(event) = this.decoder.finalize() {
                        if let Ok(Some(stream_event)) = parse_sse_event(&event) {
                            this.state.apply_event(&stream_event);
                            *this.finished = true;
                            return Poll::Ready(Some(Ok(stream_event)));
                        }
                    }

                    *this.finished = true;
                    return Poll::Ready(None);
                }

                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

// =============================================================================
// SSE Event Parsing
// =============================================================================

/// Parses an SSE event into a StreamEvent.
///
/// Returns `Ok(None)` for ping events (which should be skipped).
fn parse_sse_event(sse_event: &SseEvent) -> Result<Option<StreamEvent>, MessageStreamError> {
    match sse_event.event_type.as_str() {
        "message_start" => {
            let raw: RawMessageStartEvent = serde_json::from_str(&sse_event.data)?;
            Ok(Some(StreamEvent::MessageStart {
                message: raw.message,
            }))
        }

        "content_block_start" => {
            let raw: RawContentBlockStartEvent = serde_json::from_str(&sse_event.data)?;
            Ok(Some(StreamEvent::ContentBlockStart {
                index: raw.index,
                content_block: raw.content_block,
            }))
        }

        "content_block_delta" => {
            let raw: RawContentBlockDeltaEvent = serde_json::from_str(&sse_event.data)?;
            Ok(Some(StreamEvent::ContentBlockDelta {
                index: raw.index,
                delta: raw.delta,
            }))
        }

        "content_block_stop" => {
            let raw: RawContentBlockStopEvent = serde_json::from_str(&sse_event.data)?;
            Ok(Some(StreamEvent::ContentBlockStop { index: raw.index }))
        }

        "message_delta" => {
            let raw: RawMessageDeltaEvent = serde_json::from_str(&sse_event.data)?;
            Ok(Some(StreamEvent::MessageDelta {
                delta: raw.delta,
                usage: raw.usage,
            }))
        }

        "message_stop" => Ok(Some(StreamEvent::MessageStop)),

        "ping" => Ok(None), // Skip ping events

        "error" => {
            let raw: RawErrorEvent = serde_json::from_str(&sse_event.data)?;
            Ok(Some(StreamEvent::Error { error: raw.error }))
        }

        other => Err(MessageStreamError::UnexpectedEvent(other.to_string())),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream::{self, StreamExt};

    fn make_sse_chunk(event_type: &str, data: &str) -> Bytes {
        Bytes::from(format!("event: {event_type}\ndata: {data}\n\n"))
    }

    #[tokio::test]
    async fn test_message_stream_simple() {
        let chunks: Vec<std::result::Result<Bytes, std::io::Error>> = vec![
            Ok(make_sse_chunk(
                "message_start",
                r#"{"message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-latest","usage":{"input_tokens":10,"output_tokens":1}}}"#,
            )),
            Ok(make_sse_chunk(
                "content_block_start",
                r#"{"index":0,"content_block":{"type":"text","text":""}}"#,
            )),
            Ok(make_sse_chunk(
                "content_block_delta",
                r#"{"index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            )),
            Ok(make_sse_chunk(
                "content_block_delta",
                r#"{"index":0,"delta":{"type":"text_delta","text":" world!"}}"#,
            )),
            Ok(make_sse_chunk("content_block_stop", r#"{"index":0}"#)),
            Ok(make_sse_chunk(
                "message_delta",
                r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#,
            )),
            Ok(make_sse_chunk("message_stop", r#"{"type":"message_stop"}"#)),
        ];

        let byte_stream = stream::iter(chunks);
        let mut message_stream = MessageStream::new(byte_stream);

        let mut events = Vec::new();
        while let Some(result) = message_stream.next().await {
            events.push(result.unwrap());
        }

        assert_eq!(events.len(), 7);
        assert!(events[0].is_message_start());
        assert!(events[6].is_message_stop());

        // Check accumulated text
        assert_eq!(message_stream.text(), "Hello world!");

        // Check final message
        let message = message_stream.into_message().unwrap();
        assert_eq!(message.id, "msg_123");
        assert_eq!(message.text(), "Hello world!");
    }

    #[tokio::test]
    async fn test_message_stream_with_tool_use() {
        let chunks: Vec<std::result::Result<Bytes, std::io::Error>> = vec![
            Ok(make_sse_chunk(
                "message_start",
                r#"{"message":{"id":"msg_456","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-latest","usage":{"input_tokens":10,"output_tokens":1}}}"#,
            )),
            Ok(make_sse_chunk(
                "content_block_start",
                r#"{"index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{}}}"#,
            )),
            Ok(make_sse_chunk(
                "content_block_delta",
                r#"{"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}"#,
            )),
            Ok(make_sse_chunk(
                "content_block_delta",
                r#"{"index":0,"delta":{"type":"input_json_delta","partial_json":"\"Tokyo\"}"}}"#,
            )),
            Ok(make_sse_chunk("content_block_stop", r#"{"index":0}"#)),
            Ok(make_sse_chunk(
                "message_delta",
                r#"{"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}"#,
            )),
            Ok(make_sse_chunk("message_stop", r#"{"type":"message_stop"}"#)),
        ];

        let byte_stream = stream::iter(chunks);
        let mut message_stream = MessageStream::new(byte_stream);

        // Consume all events
        while let Some(result) = message_stream.next().await {
            result.unwrap();
        }

        let message = message_stream.into_message().unwrap();
        assert!(message.has_tool_use());

        let tool_uses = message.tool_uses();
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].name, "get_weather");
        assert_eq!(tool_uses[0].input["location"], "Tokyo");
    }

    #[tokio::test]
    async fn test_message_stream_ping_events_skipped() {
        let chunks: Vec<std::result::Result<Bytes, std::io::Error>> = vec![
            Ok(make_sse_chunk(
                "message_start",
                r#"{"message":{"id":"msg_789","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-latest","usage":{"input_tokens":10,"output_tokens":1}}}"#,
            )),
            Ok(make_sse_chunk("ping", r#"{"type":"ping"}"#)),
            Ok(make_sse_chunk("ping", r#"{"type":"ping"}"#)),
            Ok(make_sse_chunk("message_stop", r#"{"type":"message_stop"}"#)),
        ];

        let byte_stream = stream::iter(chunks);
        let mut message_stream = MessageStream::new(byte_stream);

        let mut events = Vec::new();
        while let Some(result) = message_stream.next().await {
            events.push(result.unwrap());
        }

        // Should only have 2 events (ping events are skipped)
        assert_eq!(events.len(), 2);
        assert!(events[0].is_message_start());
        assert!(events[1].is_message_stop());
    }

    #[tokio::test]
    async fn test_message_stream_error_event() {
        let chunks: Vec<std::result::Result<Bytes, std::io::Error>> = vec![
            Ok(make_sse_chunk(
                "message_start",
                r#"{"message":{"id":"msg_err","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-latest","usage":{"input_tokens":10,"output_tokens":1}}}"#,
            )),
            Ok(make_sse_chunk(
                "error",
                r#"{"error":{"type":"overloaded_error","message":"Server is overloaded"}}"#,
            )),
        ];

        let byte_stream = stream::iter(chunks);
        let mut message_stream = MessageStream::new(byte_stream);

        let event1 = message_stream.next().await.unwrap().unwrap();
        assert!(event1.is_message_start());

        let event2 = message_stream.next().await.unwrap().unwrap();
        assert!(event2.is_error());

        // Stream should be finished after error
        assert!(message_stream.is_complete() || message_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_message_stream_chunked_data() {
        // Simulate data arriving in chunks that split across SSE boundaries
        let chunk1 = Bytes::from("event: message_start\ndata: {\"message\":{\"id\":\"msg_chunk\",");
        let chunk2 = Bytes::from("\"type\":\"message\",\"role\":\"assistant\",\"content\":[],");
        let chunk3 = Bytes::from("\"model\":\"claude-sonnet-4-5-latest\",\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n");
        let chunk4 = Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");

        let chunks: Vec<Result<Bytes, std::io::Error>> =
            vec![Ok(chunk1), Ok(chunk2), Ok(chunk3), Ok(chunk4)];

        let byte_stream = stream::iter(chunks);
        let mut message_stream = MessageStream::new(byte_stream);

        let mut events = Vec::new();
        while let Some(result) = message_stream.next().await {
            events.push(result.unwrap());
        }

        assert_eq!(events.len(), 2);
        assert!(events[0].is_message_start());
        assert!(events[1].is_message_stop());
    }

    #[test]
    fn test_stream_state_apply_events() {
        let mut state = StreamState::new();

        // Apply message_start
        state.apply_event(&StreamEvent::MessageStart {
            message: Message {
                id: "msg_test".to_string(),
                message_type: "message".to_string(),
                role: crate::types::Role::Assistant,
                content: Vec::new(),
                model: crate::types::Model::claude_sonnet_4_5_latest(),
                stop_reason: None,
                stop_sequence: None,
                usage: Usage::new(10, 0),
            },
        });

        assert!(state.message.is_some());

        // Apply content_block_start
        state.apply_event(&StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlockStartContent::Text {
                text: String::new(),
            },
        });

        assert_eq!(state.content_blocks.len(), 1);

        // Apply content_block_delta
        state.apply_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "Hello".to_string(),
            },
        });

        state.apply_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: " world!".to_string(),
            },
        });

        assert_eq!(state.text(), "Hello world!");

        // Apply message_stop
        state.apply_event(&StreamEvent::MessageStop);
        assert!(state.is_complete);
    }

    #[test]
    fn test_message_stream_error_display() {
        let sse_err = MessageStreamError::Sse(SseError::InvalidUtf8);
        assert!(sse_err.to_string().contains("SSE error"));

        let json_err = MessageStreamError::Json(serde_json::from_str::<()>("invalid").unwrap_err());
        assert!(json_err.to_string().contains("JSON error"));

        let server_err = MessageStreamError::Server("test error".to_string());
        assert_eq!(server_err.to_string(), "Server error: test error");

        let unexpected_err = MessageStreamError::UnexpectedEvent("foo".to_string());
        assert!(unexpected_err.to_string().contains("foo"));
    }

    #[test]
    fn test_parse_sse_event_all_types() {
        // message_start
        let event = SseEvent::new(
            "message_start",
            r#"{"message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-latest","usage":{"input_tokens":1,"output_tokens":1}}}"#,
        );
        let result = parse_sse_event(&event).unwrap();
        assert!(matches!(result, Some(StreamEvent::MessageStart { .. })));

        // content_block_start
        let event = SseEvent::new(
            "content_block_start",
            r#"{"index":0,"content_block":{"type":"text","text":""}}"#,
        );
        let result = parse_sse_event(&event).unwrap();
        assert!(matches!(
            result,
            Some(StreamEvent::ContentBlockStart { .. })
        ));

        // content_block_delta
        let event = SseEvent::new(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        );
        let result = parse_sse_event(&event).unwrap();
        assert!(matches!(
            result,
            Some(StreamEvent::ContentBlockDelta { .. })
        ));

        // content_block_stop
        let event = SseEvent::new("content_block_stop", r#"{"index":0}"#);
        let result = parse_sse_event(&event).unwrap();
        assert!(matches!(result, Some(StreamEvent::ContentBlockStop { .. })));

        // message_delta
        let event = SseEvent::new(
            "message_delta",
            r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}"#,
        );
        let result = parse_sse_event(&event).unwrap();
        assert!(matches!(result, Some(StreamEvent::MessageDelta { .. })));

        // message_stop
        let event = SseEvent::new("message_stop", r#"{"type":"message_stop"}"#);
        let result = parse_sse_event(&event).unwrap();
        assert!(matches!(result, Some(StreamEvent::MessageStop)));

        // ping
        let event = SseEvent::new("ping", r#"{"type":"ping"}"#);
        let result = parse_sse_event(&event).unwrap();
        assert!(result.is_none());

        // error
        let event = SseEvent::new(
            "error",
            r#"{"error":{"type":"test","message":"test error"}}"#,
        );
        let result = parse_sse_event(&event).unwrap();
        assert!(matches!(result, Some(StreamEvent::Error { .. })));

        // unknown event type
        let event = SseEvent::new("unknown_type", r#"{}"#);
        let result = parse_sse_event(&event);
        assert!(matches!(
            result,
            Err(MessageStreamError::UnexpectedEvent(_))
        ));
    }
}
