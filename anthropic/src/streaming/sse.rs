//! Server-Sent Events (SSE) decoder.
//!
//! This module implements an SSE decoder following the W3C `EventSource` specification.
//! It parses the `text/event-stream` format used by the Anthropic API.
//!
//! # SSE Format
//!
//! Events are formatted as:
//! ```text
//! event: message_start
//! data: {"type": "message_start", ...}
//!
//! event: content_block_delta
//! data: {"type": "content_block_delta", ...}
//! ```
//!
//! Multiple `data:` lines are concatenated with newlines.
//! Empty lines (double newline) dispatch the event.

use std::fmt;

/// An SSE event parsed from the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The event type (e.g., "`message_start`", "`content_block_delta`").
    pub event_type: String,
    /// The event data (typically JSON).
    pub data: String,
}

impl SseEvent {
    /// Creates a new SSE event.
    #[must_use]
    pub fn new(event_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            data: data.into(),
        }
    }

    /// Returns true if this is a ping event.
    #[must_use]
    pub fn is_ping(&self) -> bool {
        self.event_type == "ping"
    }

    /// Returns true if this is an error event.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.event_type == "error"
    }
}

/// Errors that can occur during SSE decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseError {
    /// Invalid UTF-8 in the stream.
    InvalidUtf8,
    /// Malformed SSE line.
    MalformedLine(String),
    /// Buffer exceeded maximum size.
    BufferOverflow,
    /// Unexpected end of stream.
    UnexpectedEof,
}

impl fmt::Display for SseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in SSE stream"),
            Self::MalformedLine(line) => write!(f, "malformed SSE line: {line}"),
            Self::BufferOverflow => write!(f, "SSE buffer exceeded maximum size"),
            Self::UnexpectedEof => write!(f, "unexpected end of SSE stream"),
        }
    }
}

impl std::error::Error for SseError {}

/// SSE decoder that processes a byte stream into events.
///
/// The decoder maintains internal state to handle multi-line data fields
/// and properly dispatches events on empty lines.
#[derive(Debug)]
pub struct SseDecoder {
    /// Buffer for incomplete lines.
    buffer: String,
    /// Current event type being accumulated.
    current_event_type: Option<String>,
    /// Current data being accumulated (may span multiple lines).
    current_data: Vec<String>,
    /// Maximum buffer size (default 16MB).
    max_buffer_size: usize,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SseDecoder {
    /// Default maximum buffer size (16 MB).
    pub const DEFAULT_MAX_BUFFER_SIZE: usize = 16 * 1024 * 1024;

    /// Creates a new SSE decoder with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: String::new(),
            current_event_type: None,
            current_data: Vec::new(),
            max_buffer_size: Self::DEFAULT_MAX_BUFFER_SIZE,
        }
    }

    /// Creates a new SSE decoder with a custom maximum buffer size.
    #[must_use]
    pub const fn with_max_buffer_size(max_size: usize) -> Self {
        Self {
            buffer: String::new(),
            current_event_type: None,
            current_data: Vec::new(),
            max_buffer_size: max_size,
        }
    }

    /// Decodes a chunk of bytes and returns any complete events.
    ///
    /// This method buffers incomplete data and returns events as they become
    /// available. Multiple events may be returned from a single chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if the input contains invalid UTF-8 or if the
    /// buffer exceeds the maximum size.
    pub fn decode(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, SseError> {
        // Convert bytes to string
        let text = std::str::from_utf8(chunk).map_err(|_| SseError::InvalidUtf8)?;

        // Append to buffer
        self.buffer.push_str(text);

        // Check buffer size
        if self.buffer.len() > self.max_buffer_size {
            self.reset();
            return Err(SseError::BufferOverflow);
        }

        let mut events = Vec::new();

        // Process complete lines
        while let Some(newline_pos) = self.buffer.find('\n') {
            // Extract the line (excluding newline)
            let line = self.buffer[..newline_pos].to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            // Handle CRLF line endings
            let line = line.strip_suffix('\r').unwrap_or(&line);

            // Process the line
            if let Some(event) = self.process_line(line) {
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Processes a single line of SSE data.
    ///
    /// Returns `Some(event)` if the line completes an event (empty line),
    /// `None` otherwise.
    fn process_line(&mut self, line: &str) -> Option<SseEvent> {
        // Empty line dispatches the current event
        if line.is_empty() {
            return self.dispatch_event();
        }

        // Comment lines start with ':' and are ignored
        if line.starts_with(':') {
            return None;
        }

        // Parse field:value
        let (field, value) = line.find(':').map_or((line, ""), |colon_pos| {
            let field = &line[..colon_pos];
            let value = &line[colon_pos + 1..];
            // Skip optional space after colon
            let value = value.strip_prefix(' ').unwrap_or(value);
            (field, value)
        });

        match field {
            "event" => {
                self.current_event_type = Some(value.to_string());
            }
            "data" => {
                self.current_data.push(value.to_string());
            }
            // "id" and "retry" are valid SSE fields but we don't use them
            // (id: sets the last event ID; retry: reconnection time in ms).
            // Unknown fields are ignored per the SSE spec, same as these,
            // so both are handled by the wildcard arm below.
            _ => {}
        }

        None
    }

    /// Dispatches the accumulated event if data is present.
    fn dispatch_event(&mut self) -> Option<SseEvent> {
        if self.current_data.is_empty() {
            // Reset state but don't emit event
            self.current_event_type = None;
            return None;
        }

        // Join data lines with newlines (per SSE spec)
        let data = self.current_data.join("\n");

        // Remove trailing newline if present (per SSE spec)
        let data = data.strip_suffix('\n').unwrap_or(&data).to_string();

        let event = SseEvent {
            event_type: self.current_event_type.take().unwrap_or_default(),
            data,
        };

        // Reset state
        self.current_data.clear();

        Some(event)
    }

    /// Resets the decoder state, clearing all buffers.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_event_type = None;
        self.current_data.clear();
    }

    /// Finalizes the stream and returns any pending event.
    ///
    /// This should be called when the stream ends to flush any
    /// accumulated data that wasn't followed by an empty line.
    #[must_use]
    pub fn finalize(&mut self) -> Option<SseEvent> {
        // Process any remaining data in the buffer
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            for line in remaining.lines() {
                self.process_line(line);
            }
        }

        // Dispatch any pending event
        self.dispatch_event()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_event() {
        let mut decoder = SseDecoder::new();
        let input = b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n";

        let events = decoder.decode(input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message_start");
        assert_eq!(events[0].data, "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_multiple_events() {
        let mut decoder = SseDecoder::new();
        let input = b"event: ping\ndata: {}\n\nevent: message_start\ndata: {\"foo\":\"bar\"}\n\n";

        let events = decoder.decode(input).unwrap();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "ping");
        assert_eq!(events[1].event_type, "message_start");
    }

    #[test]
    fn test_multi_line_data() {
        let mut decoder = SseDecoder::new();
        let input = b"event: test\ndata: line1\ndata: line2\ndata: line3\n\n";

        let events = decoder.decode(input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn test_chunked_input() {
        let mut decoder = SseDecoder::new();

        // First chunk (incomplete)
        let events1 = decoder.decode(b"event: test\nda").unwrap();
        assert!(events1.is_empty());

        // Second chunk (completes the event)
        let events2 = decoder.decode(b"ta: hello\n\n").unwrap();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].event_type, "test");
        assert_eq!(events2[0].data, "hello");
    }

    #[test]
    fn test_comment_lines_ignored() {
        let mut decoder = SseDecoder::new();
        let input = b": this is a comment\nevent: test\n: another comment\ndata: value\n\n";

        let events = decoder.decode(input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
        assert_eq!(events[0].data, "value");
    }

    #[test]
    fn test_colon_in_data() {
        let mut decoder = SseDecoder::new();
        let input = b"event: test\ndata: {\"url\": \"https://example.com\"}\n\n";

        let events = decoder.decode(input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"url\": \"https://example.com\"}");
    }

    #[test]
    fn test_optional_space_after_colon() {
        let mut decoder = SseDecoder::new();

        // With space
        let events1 = decoder.decode(b"event: test\ndata: value\n\n").unwrap();
        assert_eq!(events1[0].data, "value");

        decoder.reset();

        // Without space
        let events2 = decoder.decode(b"event:test\ndata:value\n\n").unwrap();
        assert_eq!(events2[0].data, "value");
    }

    #[test]
    fn test_crlf_line_endings() {
        let mut decoder = SseDecoder::new();
        let input = b"event: test\r\ndata: value\r\n\r\n";

        let events = decoder.decode(input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
        assert_eq!(events[0].data, "value");
    }

    #[test]
    fn test_empty_event_type() {
        let mut decoder = SseDecoder::new();
        let input = b"data: no event type\n\n";

        let events = decoder.decode(input).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "");
        assert_eq!(events[0].data, "no event type");
    }

    #[test]
    fn test_empty_data_not_dispatched() {
        let mut decoder = SseDecoder::new();
        let input = b"event: test\n\n";

        let events = decoder.decode(input).unwrap();

        // Event without data should not be dispatched
        assert!(events.is_empty());
    }

    #[test]
    fn test_finalize_pending_event() {
        let mut decoder = SseDecoder::new();

        // Input without final empty line
        let events = decoder.decode(b"event: test\ndata: value").unwrap();
        assert!(events.is_empty());

        // Finalize should dispatch the pending event
        let final_event = decoder.finalize();
        assert!(final_event.is_some());
        let event = final_event.unwrap();
        assert_eq!(event.event_type, "test");
        assert_eq!(event.data, "value");
    }

    #[test]
    fn test_invalid_utf8() {
        let mut decoder = SseDecoder::new();
        let invalid = [0xff, 0xfe];

        let result = decoder.decode(&invalid);
        assert!(matches!(result, Err(SseError::InvalidUtf8)));
    }

    #[test]
    fn test_buffer_overflow() {
        let mut decoder = SseDecoder::with_max_buffer_size(10);
        let large_input = b"data: this is way too long for the buffer\n\n";

        let result = decoder.decode(large_input);
        assert!(matches!(result, Err(SseError::BufferOverflow)));
    }

    #[test]
    fn test_reset() {
        let mut decoder = SseDecoder::new();

        // Start accumulating
        decoder.decode(b"event: test\ndata: part1").unwrap();

        // Reset
        decoder.reset();

        // New event should work normally
        let events = decoder.decode(b"event: other\ndata: value\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "other");
    }

    #[test]
    fn test_is_ping() {
        let event = SseEvent::new("ping", "{}");
        assert!(event.is_ping());

        let event = SseEvent::new("message_start", "{}");
        assert!(!event.is_ping());
    }

    #[test]
    fn test_is_error() {
        let event = SseEvent::new("error", "{}");
        assert!(event.is_error());

        let event = SseEvent::new("ping", "{}");
        assert!(!event.is_error());
    }

    #[test]
    fn test_anthropic_streaming_format() {
        // Test with a realistic Anthropic streaming response
        let mut decoder = SseDecoder::new();

        let input = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-opus-20240229\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n",
            "\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n",
            "\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world!\"}}\n",
            "\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n"
        );

        let events = decoder.decode(input.as_bytes()).unwrap();

        assert_eq!(events.len(), 8);
        assert_eq!(events[0].event_type, "message_start");
        assert_eq!(events[1].event_type, "content_block_start");
        assert_eq!(events[2].event_type, "ping");
        assert_eq!(events[3].event_type, "content_block_delta");
        assert_eq!(events[4].event_type, "content_block_delta");
        assert_eq!(events[5].event_type, "content_block_stop");
        assert_eq!(events[6].event_type, "message_delta");
        assert_eq!(events[7].event_type, "message_stop");
    }
}
