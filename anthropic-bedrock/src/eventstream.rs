//! AWS EventStream decoder for Bedrock streaming responses.
//!
//! AWS Bedrock uses the AWS EventStream binary protocol for streaming responses,
//! which is different from the SSE (Server-Sent Events) format used by the direct
//! Anthropic API. This module provides a decoder for parsing EventStream frames.
//!
//! # EventStream Format
//!
//! Each EventStream message has the following structure:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Prelude (12 bytes)                     │
//! ├───────────────────────┬─────────────────────────────────────┤
//! │  Total Length (4B)    │  Headers Length (4B)  │  CRC (4B)   │
//! ├───────────────────────┴─────────────────────────────────────┤
//! │                      Headers (variable)                     │
//! │  [name_len(1B)][name][type(1B)][value_len(2B)][value] ...   │
//! ├─────────────────────────────────────────────────────────────┤
//! │                      Payload (variable)                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    Message CRC (4 bytes)                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic_bedrock::eventstream::{EventStreamDecoder, Event};
//!
//! let mut decoder = EventStreamDecoder::new();
//!
//! // Feed raw bytes from the HTTP response
//! let events = decoder.decode(&raw_bytes)?;
//!
//! for event in events {
//!     match event {
//!         Event::Message(msg) => {
//!             // Handle the Claude streaming event
//!             println!("Event type: {:?}", msg.event_type);
//!             println!("Payload: {}", msg.payload);
//!         }
//!         Event::Exception(exc) => {
//!             eprintln!("Error: {}", exc.message);
//!         }
//!     }
//! }
//! ```
//!
//! # Bedrock Event Types
//!
//! Bedrock streaming responses contain events with the following `:event-type` headers:
//!
//! - `chunk` - Contains the actual streaming content (JSON payload with base64-encoded bytes)
//! - `exception` - Error event
//!
//! The `chunk` payload contains a JSON object with a `bytes` field that is base64-encoded.
//! When decoded, it contains the same streaming events as the direct Anthropic API.

use base64::Engine as _;
use bytes::BytesMut;
use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

// =============================================================================
// Constants
// =============================================================================

/// Minimum frame size (prelude + message CRC).
const MIN_FRAME_SIZE: usize = 16;

/// Prelude size (total_len + headers_len + prelude_crc).
const PRELUDE_SIZE: usize = 12;

/// Message CRC size.
const MESSAGE_CRC_SIZE: usize = 4;

/// Header type constants.
mod header_types {
    pub(super) const BOOL_TRUE: u8 = 0;
    pub(super) const BOOL_FALSE: u8 = 1;
    pub(super) const BYTE: u8 = 2;
    pub(super) const SHORT: u8 = 3;
    pub(super) const INT: u8 = 4;
    pub(super) const LONG: u8 = 5;
    pub(super) const BYTES: u8 = 6;
    pub(super) const STRING: u8 = 7;
    pub(super) const TIMESTAMP: u8 = 8;
    pub(super) const UUID: u8 = 9;
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during EventStream decoding.
#[derive(Debug, Error)]
pub enum EventStreamError {
    /// Not enough data to parse a complete frame.
    #[error("incomplete frame: need at least {needed} bytes, have {have}")]
    IncompleteFrame {
        /// Bytes needed.
        needed: usize,
        /// Bytes available.
        have: usize,
    },

    /// CRC checksum mismatch.
    #[error("CRC mismatch: expected {expected:#010x}, got {actual:#010x}")]
    CrcMismatch {
        /// Expected CRC.
        expected: u32,
        /// Actual computed CRC.
        actual: u32,
    },

    /// Invalid header format.
    #[error("invalid header: {0}")]
    InvalidHeader(String),

    /// Invalid frame structure.
    #[error("invalid frame: {0}")]
    InvalidFrame(String),

    /// Base64 decoding error.
    #[error("base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// UTF-8 decoding error.
    #[error("UTF-8 error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),
}

/// Result type for EventStream operations.
pub type Result<T> = std::result::Result<T, EventStreamError>;

// =============================================================================
// Event Types
// =============================================================================

/// A decoded EventStream event.
#[derive(Debug, Clone)]
pub enum Event {
    /// A message event containing Claude streaming data.
    Message(MessageEvent),
    /// An exception/error event.
    Exception(ExceptionEvent),
}

/// A message event from the EventStream.
#[derive(Debug, Clone)]
pub struct MessageEvent {
    /// The event type (e.g., "message_start", "content_block_delta").
    pub event_type: String,
    /// The decoded JSON payload.
    pub payload: String,
    /// Raw headers from the frame.
    pub headers: HashMap<String, HeaderValue>,
}

/// An exception event from the EventStream.
#[derive(Debug, Clone)]
pub struct ExceptionEvent {
    /// The exception type.
    pub exception_type: String,
    /// Error message.
    pub message: String,
}

/// Header value types in EventStream.
#[derive(Debug, Clone)]
pub enum HeaderValue {
    /// Boolean true.
    BoolTrue,
    /// Boolean false.
    BoolFalse,
    /// 8-bit signed integer.
    Byte(i8),
    /// 16-bit signed integer.
    Short(i16),
    /// 32-bit signed integer.
    Int(i32),
    /// 64-bit signed integer.
    Long(i64),
    /// Binary data.
    Bytes(Vec<u8>),
    /// UTF-8 string.
    String(String),
    /// Timestamp (milliseconds since epoch).
    Timestamp(i64),
    /// UUID.
    Uuid([u8; 16]),
}

impl HeaderValue {
    /// Returns the string value if this is a String variant.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

// =============================================================================
// Raw Frame
// =============================================================================

/// A raw EventStream frame before event interpretation.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Headers from the frame.
    pub headers: HashMap<String, HeaderValue>,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
}

// =============================================================================
// Bedrock Chunk Payload
// =============================================================================

/// The JSON structure of a Bedrock chunk payload.
#[derive(Debug, Deserialize)]
struct BedrockChunkPayload {
    /// Base64-encoded bytes containing the actual event data.
    bytes: String,
}

// =============================================================================
// EventStream Decoder
// =============================================================================

/// Decoder for AWS EventStream binary protocol.
///
/// This decoder maintains an internal buffer and can handle partial frames
/// that span multiple `decode` calls.
///
/// # Example
///
/// ```rust,ignore
/// use anthropic_bedrock::eventstream::EventStreamDecoder;
///
/// let mut decoder = EventStreamDecoder::new();
///
/// // Process chunks as they arrive
/// while let Some(chunk) = response.chunk().await? {
///     let events = decoder.decode(&chunk)?;
///     for event in events {
///         // Handle event...
///     }
/// }
/// ```
#[derive(Debug)]
pub struct EventStreamDecoder {
    buffer: BytesMut,
}

impl Default for EventStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventStreamDecoder {
    /// Creates a new EventStream decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(8192),
        }
    }

    /// Decodes raw bytes into EventStream events.
    ///
    /// This method accumulates data in an internal buffer and returns
    /// events as complete frames are received. Partial frames are buffered
    /// until more data arrives.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes from the HTTP response stream
    ///
    /// # Returns
    ///
    /// A vector of decoded events. May be empty if no complete frames
    /// are available yet.
    ///
    /// # Errors
    ///
    /// Returns an error if a frame is malformed or CRC validation fails.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<Event>> {
        self.buffer.extend_from_slice(data);
        let mut events = Vec::new();

        while let Some(frame) = self.try_decode_frame()? {
            if let Some(event) = self.frame_to_event(frame)? {
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Decodes raw bytes into raw frames without event interpretation.
    ///
    /// This is useful if you need access to the raw frame structure
    /// before Bedrock-specific processing.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes from the HTTP response stream
    ///
    /// # Returns
    ///
    /// A vector of decoded raw frames.
    pub fn decode_frames(&mut self, data: &[u8]) -> Result<Vec<RawFrame>> {
        self.buffer.extend_from_slice(data);
        let mut frames = Vec::new();

        while let Some(frame) = self.try_decode_frame()? {
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the number of buffered bytes.
    #[must_use]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// Clears the internal buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Tries to decode a single frame from the buffer.
    fn try_decode_frame(&mut self) -> Result<Option<RawFrame>> {
        // Need at least the prelude to determine frame size
        if self.buffer.len() < PRELUDE_SIZE {
            return Ok(None);
        }

        // Read total length from prelude (first 4 bytes)
        let total_length = u32::from_be_bytes([
            self.buffer[0],
            self.buffer[1],
            self.buffer[2],
            self.buffer[3],
        ]) as usize;

        // Validate minimum frame size
        if total_length < MIN_FRAME_SIZE {
            return Err(EventStreamError::InvalidFrame(format!(
                "frame too small: {} bytes",
                total_length
            )));
        }

        // Check if we have the complete frame
        if self.buffer.len() < total_length {
            return Ok(None);
        }

        // Extract the complete frame
        let frame_bytes = self.buffer.split_to(total_length);

        // Parse the frame
        self.parse_frame(&frame_bytes)
    }

    /// Parses a complete frame from bytes.
    fn parse_frame(&self, frame: &[u8]) -> Result<Option<RawFrame>> {
        let total_length = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let headers_length =
            u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
        let prelude_crc = u32::from_be_bytes([frame[8], frame[9], frame[10], frame[11]]);

        // Validate prelude CRC
        let computed_prelude_crc = crc32fast::hash(&frame[0..8]);
        if prelude_crc != computed_prelude_crc {
            return Err(EventStreamError::CrcMismatch {
                expected: prelude_crc,
                actual: computed_prelude_crc,
            });
        }

        // Validate message CRC
        let message_crc_offset = total_length - MESSAGE_CRC_SIZE;
        let message_crc = u32::from_be_bytes([
            frame[message_crc_offset],
            frame[message_crc_offset + 1],
            frame[message_crc_offset + 2],
            frame[message_crc_offset + 3],
        ]);

        let computed_message_crc = crc32fast::hash(&frame[0..message_crc_offset]);
        if message_crc != computed_message_crc {
            return Err(EventStreamError::CrcMismatch {
                expected: message_crc,
                actual: computed_message_crc,
            });
        }

        // Parse headers
        let headers_start = PRELUDE_SIZE;
        let headers_end = headers_start + headers_length;
        let headers = self.parse_headers(&frame[headers_start..headers_end])?;

        // Extract payload
        let payload_start = headers_end;
        let payload_end = message_crc_offset;
        let payload = frame[payload_start..payload_end].to_vec();

        Ok(Some(RawFrame { headers, payload }))
    }

    /// Parses headers from the header section.
    fn parse_headers(&self, mut data: &[u8]) -> Result<HashMap<String, HeaderValue>> {
        let mut headers = HashMap::new();

        while !data.is_empty() {
            // Read header name length (1 byte)
            if data.is_empty() {
                break;
            }
            let name_len = data[0] as usize;
            data = &data[1..];

            // Read header name
            if data.len() < name_len {
                return Err(EventStreamError::InvalidHeader(
                    "truncated header name".to_string(),
                ));
            }
            let name = String::from_utf8(data[..name_len].to_vec())?;
            data = &data[name_len..];

            // Read header type (1 byte)
            if data.is_empty() {
                return Err(EventStreamError::InvalidHeader(
                    "missing header type".to_string(),
                ));
            }
            let header_type = data[0];
            data = &data[1..];

            // Read header value based on type
            let (value, remaining) = self.parse_header_value(header_type, data)?;
            headers.insert(name, value);
            data = remaining;
        }

        Ok(headers)
    }

    /// Parses a header value based on its type.
    fn parse_header_value<'a>(
        &self,
        header_type: u8,
        data: &'a [u8],
    ) -> Result<(HeaderValue, &'a [u8])> {
        match header_type {
            header_types::BOOL_TRUE => Ok((HeaderValue::BoolTrue, data)),
            header_types::BOOL_FALSE => Ok((HeaderValue::BoolFalse, data)),
            header_types::BYTE => {
                if data.is_empty() {
                    return Err(EventStreamError::InvalidHeader(
                        "missing byte value".to_string(),
                    ));
                }
                Ok((HeaderValue::Byte(data[0] as i8), &data[1..]))
            }
            header_types::SHORT => {
                if data.len() < 2 {
                    return Err(EventStreamError::InvalidHeader(
                        "missing short value".to_string(),
                    ));
                }
                let value = i16::from_be_bytes([data[0], data[1]]);
                Ok((HeaderValue::Short(value), &data[2..]))
            }
            header_types::INT => {
                if data.len() < 4 {
                    return Err(EventStreamError::InvalidHeader(
                        "missing int value".to_string(),
                    ));
                }
                let value = i32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                Ok((HeaderValue::Int(value), &data[4..]))
            }
            header_types::LONG | header_types::TIMESTAMP => {
                if data.len() < 8 {
                    return Err(EventStreamError::InvalidHeader(
                        "missing long/timestamp value".to_string(),
                    ));
                }
                let value = i64::from_be_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]);
                let header_value = if header_type == header_types::TIMESTAMP {
                    HeaderValue::Timestamp(value)
                } else {
                    HeaderValue::Long(value)
                };
                Ok((header_value, &data[8..]))
            }
            header_types::BYTES => {
                if data.len() < 2 {
                    return Err(EventStreamError::InvalidHeader(
                        "missing bytes length".to_string(),
                    ));
                }
                let len = u16::from_be_bytes([data[0], data[1]]) as usize;
                if data.len() < 2 + len {
                    return Err(EventStreamError::InvalidHeader(
                        "truncated bytes value".to_string(),
                    ));
                }
                let value = data[2..2 + len].to_vec();
                Ok((HeaderValue::Bytes(value), &data[2 + len..]))
            }
            header_types::STRING => {
                if data.len() < 2 {
                    return Err(EventStreamError::InvalidHeader(
                        "missing string length".to_string(),
                    ));
                }
                let len = u16::from_be_bytes([data[0], data[1]]) as usize;
                if data.len() < 2 + len {
                    return Err(EventStreamError::InvalidHeader(
                        "truncated string value".to_string(),
                    ));
                }
                let value = String::from_utf8(data[2..2 + len].to_vec())?;
                Ok((HeaderValue::String(value), &data[2 + len..]))
            }
            header_types::UUID => {
                if data.len() < 16 {
                    return Err(EventStreamError::InvalidHeader(
                        "missing uuid value".to_string(),
                    ));
                }
                let mut uuid = [0u8; 16];
                uuid.copy_from_slice(&data[..16]);
                Ok((HeaderValue::Uuid(uuid), &data[16..]))
            }
            _ => Err(EventStreamError::InvalidHeader(format!(
                "unknown header type: {}",
                header_type
            ))),
        }
    }

    /// Converts a raw frame to an Event.
    fn frame_to_event(&self, frame: RawFrame) -> Result<Option<Event>> {
        // Get the message type header
        let message_type = frame
            .headers
            .get(":message-type")
            .and_then(|v| v.as_str())
            .unwrap_or("event");

        match message_type {
            "exception" => {
                // Parse exception
                let exception_type = frame
                    .headers
                    .get(":exception-type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let message = String::from_utf8(frame.payload)
                    .unwrap_or_else(|_| "Failed to decode exception message".to_string());

                Ok(Some(Event::Exception(ExceptionEvent {
                    exception_type,
                    message,
                })))
            }
            "event" => {
                // Get event type
                let event_type = frame
                    .headers
                    .get(":event-type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                // For Bedrock, the payload is JSON with a base64-encoded "bytes" field
                if event_type == "chunk" && !frame.payload.is_empty() {
                    let chunk: BedrockChunkPayload = serde_json::from_slice(&frame.payload)?;

                    // Decode the base64 bytes
                    let decoded_bytes =
                        base64::engine::general_purpose::STANDARD.decode(&chunk.bytes)?;
                    let payload = String::from_utf8(decoded_bytes)?;

                    // Try to extract the actual event type from the decoded payload
                    let actual_event_type = if let Ok(json) =
                        serde_json::from_str::<serde_json::Value>(&payload)
                    {
                        json.get("type")
                            .and_then(|t| t.as_str())
                            .map(String::from)
                            .unwrap_or(event_type)
                    } else {
                        event_type
                    };

                    Ok(Some(Event::Message(MessageEvent {
                        event_type: actual_event_type,
                        payload,
                        headers: frame.headers,
                    })))
                } else if !frame.payload.is_empty() {
                    // For non-chunk events, use payload directly
                    let payload = String::from_utf8(frame.payload)?;

                    Ok(Some(Event::Message(MessageEvent {
                        event_type,
                        payload,
                        headers: frame.headers,
                    })))
                } else {
                    // Empty payload events
                    Ok(Some(Event::Message(MessageEvent {
                        event_type,
                        payload: String::new(),
                        headers: frame.headers,
                    })))
                }
            }
            _ => {
                // Unknown message type - return as message for debugging
                let payload = String::from_utf8(frame.payload).unwrap_or_default();
                Ok(Some(Event::Message(MessageEvent {
                    event_type: format!("unknown:{}", message_type),
                    payload,
                    headers: frame.headers,
                })))
            }
        }
    }
}

// =============================================================================
// Stream Adapter
// =============================================================================

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A stream that decodes EventStream frames from a byte stream.
///
/// This adapter wraps a byte stream (like from an HTTP response) and
/// yields decoded events.
///
/// # Example
///
/// ```rust,ignore
/// use anthropic_bedrock::eventstream::EventStream;
/// use futures::StreamExt;
///
/// let byte_stream = response.bytes_stream();
/// let mut event_stream = EventStream::new(byte_stream);
///
/// while let Some(result) = event_stream.next().await {
///     match result {
///         Ok(event) => {
///             // Process event
///         }
///         Err(e) => {
///             eprintln!("Error: {}", e);
///         }
///     }
/// }
/// ```
pub struct EventStream<S>
where
    S: std::fmt::Debug,
{
    inner: S,
    decoder: EventStreamDecoder,
    pending_events: Vec<Event>,
}

impl<S: std::fmt::Debug> std::fmt::Debug for EventStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream")
            .field("inner", &self.inner)
            .field("decoder", &self.decoder)
            .field("pending_events", &self.pending_events.len())
            .finish()
    }
}

impl<S: std::fmt::Debug> EventStream<S> {
    /// Creates a new EventStream from a byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            decoder: EventStreamDecoder::new(),
            pending_events: Vec::new(),
        }
    }
}

impl<S, E> Stream for EventStream<S>
where
    S: Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin + std::fmt::Debug,
    E: std::error::Error + 'static,
{
    type Item = Result<Event>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // First, check if we have pending events
        if !self.pending_events.is_empty() {
            return Poll::Ready(Some(Ok(self.pending_events.remove(0))));
        }

        // Try to get more data from the inner stream
        let this = &mut *self;

        loop {
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    match this.decoder.decode(&bytes) {
                        Ok(events) => {
                            if !events.is_empty() {
                                // Store all but the first event
                                this.pending_events.extend(events.into_iter().skip(1));

                                // Return the first event if we decoded any
                                if let Some(event) = this.pending_events.first().cloned() {
                                    this.pending_events.remove(0);
                                    return Poll::Ready(Some(Ok(event)));
                                }
                            }
                            // If no events decoded, continue polling for more data
                        }
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
                Poll::Ready(Some(Err(_))) => {
                    return Poll::Ready(Some(Err(EventStreamError::InvalidFrame(
                        "stream error".to_string(),
                    ))));
                }
                Poll::Ready(None) => {
                    // Stream ended - check if there are remaining buffered events
                    if !this.pending_events.is_empty() {
                        return Poll::Ready(Some(Ok(this.pending_events.remove(0))));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Creates a simple EventStream frame for testing.
///
/// This is primarily useful for testing and debugging.
#[must_use]
pub fn create_test_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
    // Build headers
    let mut headers = Vec::new();

    // :message-type header
    let msg_type_name = ":message-type";
    headers.push(msg_type_name.len() as u8);
    headers.extend_from_slice(msg_type_name.as_bytes());
    headers.push(header_types::STRING);
    let msg_type_value = "event";
    headers.extend_from_slice(&(msg_type_value.len() as u16).to_be_bytes());
    headers.extend_from_slice(msg_type_value.as_bytes());

    // :event-type header
    let event_type_name = ":event-type";
    headers.push(event_type_name.len() as u8);
    headers.extend_from_slice(event_type_name.as_bytes());
    headers.push(header_types::STRING);
    headers.extend_from_slice(&(event_type.len() as u16).to_be_bytes());
    headers.extend_from_slice(event_type.as_bytes());

    // Calculate lengths
    let headers_length = headers.len();
    let total_length = PRELUDE_SIZE + headers_length + payload.len() + MESSAGE_CRC_SIZE;

    // Build frame
    let mut frame = Vec::with_capacity(total_length);

    // Prelude (without CRC yet)
    frame.extend_from_slice(&(total_length as u32).to_be_bytes());
    frame.extend_from_slice(&(headers_length as u32).to_be_bytes());

    // Calculate and add prelude CRC
    let prelude_crc = crc32fast::hash(&frame[0..8]);
    frame.extend_from_slice(&prelude_crc.to_be_bytes());

    // Headers
    frame.extend_from_slice(&headers);

    // Payload
    frame.extend_from_slice(payload);

    // Calculate and add message CRC
    let message_crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&message_crc.to_be_bytes());

    frame
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_frame() {
        let payload = b"test payload";
        let frame = create_test_frame("test_event", payload);

        // Verify frame structure
        let total_length =
            u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        assert_eq!(frame.len(), total_length);
    }

    #[test]
    fn test_decoder_basic() {
        let mut decoder = EventStreamDecoder::new();

        let payload = br#"{"test": "data"}"#;
        let frame = create_test_frame("test_event", payload);

        let events = decoder.decode(&frame).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            Event::Message(msg) => {
                assert_eq!(msg.event_type, "test_event");
                assert_eq!(msg.payload, r#"{"test": "data"}"#);
            }
            _ => panic!("Expected message event"),
        }
    }

    #[test]
    fn test_decoder_partial_frame() {
        let mut decoder = EventStreamDecoder::new();

        let payload = br#"{"test": "data"}"#;
        let frame = create_test_frame("test_event", payload);

        // Feed partial frame
        let mid = frame.len() / 2;
        let events1 = decoder.decode(&frame[..mid]).unwrap();
        assert!(events1.is_empty());

        // Feed rest of frame
        let events2 = decoder.decode(&frame[mid..]).unwrap();
        assert_eq!(events2.len(), 1);
    }

    #[test]
    fn test_decoder_multiple_frames() {
        let mut decoder = EventStreamDecoder::new();

        let frame1 = create_test_frame("event1", b"payload1");
        let frame2 = create_test_frame("event2", b"payload2");

        let mut combined = frame1.clone();
        combined.extend_from_slice(&frame2);

        let events = decoder.decode(&combined).unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_decoder_crc_validation() {
        let mut decoder = EventStreamDecoder::new();

        let frame = create_test_frame("test", b"data");
        let mut corrupted = frame.clone();
        // Corrupt a byte in the payload
        corrupted[20] ^= 0xFF;

        let result = decoder.decode(&corrupted);
        assert!(matches!(result, Err(EventStreamError::CrcMismatch { .. })));
    }

    #[test]
    fn test_bedrock_chunk_decoding() {
        let mut decoder = EventStreamDecoder::new();

        // Create a Bedrock-style chunk with base64-encoded content
        let inner_event = r#"{"type":"message_start","message":{"id":"msg_123"}}"#;
        let encoded = base64::engine::general_purpose::STANDARD.encode(inner_event.as_bytes());
        let chunk_payload = format!(r#"{{"bytes":"{}"}}"#, encoded);

        let frame = create_test_frame("chunk", chunk_payload.as_bytes());
        let events = decoder.decode(&frame).unwrap();

        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Message(msg) => {
                assert_eq!(msg.event_type, "message_start");
                assert!(msg.payload.contains("msg_123"));
            }
            _ => panic!("Expected message event"),
        }
    }

    #[test]
    fn test_header_value_as_str() {
        let string_value = HeaderValue::String("test".to_string());
        assert_eq!(string_value.as_str(), Some("test"));

        let int_value = HeaderValue::Int(42);
        assert_eq!(int_value.as_str(), None);
    }

    #[test]
    fn test_decoder_empty_payload() {
        let mut decoder = EventStreamDecoder::new();

        let frame = create_test_frame("ping", &[]);
        let events = decoder.decode(&frame).unwrap();

        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Message(msg) => {
                assert_eq!(msg.event_type, "ping");
                assert!(msg.payload.is_empty());
            }
            _ => panic!("Expected message event"),
        }
    }

    #[test]
    fn test_decoder_is_empty() {
        let mut decoder = EventStreamDecoder::new();
        assert!(decoder.is_empty());
        assert_eq!(decoder.buffered_len(), 0);

        // Feed partial frame data (not enough to decode, just buffers it)
        let _ = decoder.decode(&[0, 0, 0, 20]); // Partial prelude indicating 20 byte frame
        assert!(!decoder.is_empty());
        assert_eq!(decoder.buffered_len(), 4);

        decoder.clear();
        assert!(decoder.is_empty());
    }

    #[test]
    fn test_decoder_decode_frames() {
        let mut decoder = EventStreamDecoder::new();

        let frame1 = create_test_frame("event1", b"data1");
        let frame2 = create_test_frame("event2", b"data2");

        let mut combined = frame1.clone();
        combined.extend_from_slice(&frame2);

        let frames = decoder.decode_frames(&combined).unwrap();
        assert_eq!(frames.len(), 2);

        // Check headers
        assert!(frames[0].headers.contains_key(":event-type"));
        assert!(frames[1].headers.contains_key(":event-type"));
    }
}
