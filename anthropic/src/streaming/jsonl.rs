//! JSONL (JSON Lines) streaming decoder.
//!
//! This module provides a streaming decoder for JSONL format, which is used
//! by the Anthropic API to return batch results. Each line in a JSONL response
//! is a complete JSON object that can be parsed independently.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::streaming::JsonlStream;
//!
//! // Stream batch results from an HTTP response
//! let mut stream = JsonlStream::new(response);
//!
//! while let Some(result) = stream.next().await {
//!     match result {
//!         Ok(batch_result) => println!("Got result: {:?}", batch_result),
//!         Err(e) => eprintln!("Error: {}", e),
//!     }
//! }
//! ```

use bytes::Bytes;
use futures_core::Stream;
use pin_project_lite::pin_project;
use serde::de::DeserializeOwned;
use std::io::{self, BufRead, BufReader, Cursor};
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::error::{Error, Result};

// =============================================================================
// JSONL Stream
// =============================================================================

pin_project! {
    /// A streaming decoder for JSONL (JSON Lines) format.
    ///
    /// This stream reads lines from an HTTP response body and parses each line
    /// as a separate JSON object. It's used for streaming batch results from
    /// the Anthropic API.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to deserialize each line into
    /// * `S` - The underlying byte stream type
    #[derive(Debug)]
    pub struct JsonlStream<T, S> {
        #[pin]
        inner: S,
        buffer: Vec<u8>,
        _marker: PhantomData<T>,
    }
}

impl<T, S> JsonlStream<T, S>
where
    T: DeserializeOwned,
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Unpin,
{
    /// Creates a new JSONL stream from an HTTP response body stream.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying byte stream from the HTTP response
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::with_capacity(8192), // Pre-allocate reasonable buffer
            _marker: PhantomData,
        }
    }

    /// Creates a new JSONL stream from an HTTP response.
    ///
    /// # Arguments
    ///
    /// * `response` - The reqwest Response to stream from
    #[must_use]
    pub fn from_response(
        response: reqwest::Response,
    ) -> JsonlStream<T, impl Stream<Item = std::result::Result<Bytes, reqwest::Error>>> {
        JsonlStream::new(response.bytes_stream())
    }
}

impl<T, S> Stream for JsonlStream<T, S>
where
    T: DeserializeOwned,
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<T>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // Try to find a complete line in the buffer
        loop {
            // Look for newline in existing buffer
            if let Some(newline_pos) = this.buffer.iter().position(|&b| b == b'\n') {
                // Extract the line (excluding the newline)
                let line: Vec<u8> = this.buffer.drain(..=newline_pos).collect();
                let line = &line[..line.len() - 1]; // Remove trailing newline

                // Skip empty lines
                if line.is_empty() || line.iter().all(|&b| b.is_ascii_whitespace()) {
                    continue;
                }

                // Parse the JSON
                match serde_json::from_slice(line) {
                    Ok(value) => return Poll::Ready(Some(Ok(value))),
                    Err(e) => return Poll::Ready(Some(Err(Error::Json(e)))),
                }
            }

            // Need more data - poll the underlying stream
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    this.buffer.extend_from_slice(&chunk);
                    // Continue loop to check for newline
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(Error::Http(e))));
                }
                Poll::Ready(None) => {
                    // Stream ended - check if there's remaining data in buffer
                    if this.buffer.is_empty()
                        || this.buffer.iter().all(|&b| b.is_ascii_whitespace())
                    {
                        return Poll::Ready(None);
                    }
                    // Try to parse remaining buffer as final line
                    let line = std::mem::take(this.buffer);
                    match serde_json::from_slice(&line) {
                        Ok(value) => return Poll::Ready(Some(Ok(value))),
                        Err(e) => return Poll::Ready(Some(Err(Error::Json(e)))),
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// =============================================================================
// Synchronous JSONL Reader
// =============================================================================

/// A synchronous JSONL reader for processing batch results.
///
/// This reader processes JSONL data from a byte slice or reader,
/// parsing each line as a separate JSON object.
///
/// # Example
///
/// ```rust
/// use anthropic::streaming::JsonlReader;
/// use serde::Deserialize;
///
/// #[derive(Debug, Deserialize)]
/// struct Item {
///     id: String,
///     value: i32,
/// }
///
/// let data = r#"{"id": "1", "value": 10}
/// {"id": "2", "value": 20}
/// "#;
///
/// let reader = JsonlReader::<Item>::from_bytes(data.as_bytes());
/// for item in reader {
///     println!("Item: {:?}", item);
/// }
/// ```
#[derive(Debug)]
pub struct JsonlReader<T, R = Cursor<Vec<u8>>> {
    reader: BufReader<R>,
    _marker: PhantomData<T>,
}

impl<T> JsonlReader<T, Cursor<Vec<u8>>>
where
    T: DeserializeOwned,
{
    /// Creates a new JSONL reader from a byte slice.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            reader: BufReader::new(Cursor::new(data.to_vec())),
            _marker: PhantomData,
        }
    }

    /// Creates a new JSONL reader from a string.
    // Intentionally not `std::str::FromStr`: this constructor is infallible
    // (no `Err` type) and takes a full multi-line JSONL document rather than
    // parsing a single value, so it doesn't fit that trait's contract.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(data: &str) -> Self {
        Self::from_bytes(data.as_bytes())
    }
}

impl<T, R> JsonlReader<T, R>
where
    T: DeserializeOwned,
    R: io::Read,
{
    /// Creates a new JSONL reader from any reader.
    pub fn from_reader(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            _marker: PhantomData,
        }
    }
}

impl<T, R> Iterator for JsonlReader<T, R>
where
    T: DeserializeOwned,
    R: io::Read,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None, // EOF
                Ok(_) => {
                    // Skip empty lines
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Parse the JSON
                    match serde_json::from_str(trimmed) {
                        Ok(value) => return Some(Ok(value)),
                        Err(e) => return Some(Err(Error::Json(e))),
                    }
                }
                Err(e) => {
                    return Some(Err(Error::Streaming(format!(
                        "IO error reading JSONL: {e}"
                    ))));
                }
            }
        }
    }
}

// =============================================================================
// JSONL Encoder
// =============================================================================

/// Encodes items as JSONL (JSON Lines) format.
///
/// Each item is serialized as a single-line JSON object followed by a newline.
///
/// # Example
///
/// ```rust
/// use anthropic::streaming::JsonlEncoder;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Item {
///     id: String,
///     value: i32,
/// }
///
/// let mut encoder = JsonlEncoder::new();
/// encoder.push(&Item { id: "1".to_string(), value: 10 }).unwrap();
/// encoder.push(&Item { id: "2".to_string(), value: 20 }).unwrap();
///
/// let output = encoder.into_string();
/// assert!(output.contains(r#"{"id":"1","value":10}"#));
/// ```
#[derive(Debug, Default)]
pub struct JsonlEncoder {
    buffer: Vec<u8>,
}

impl JsonlEncoder {
    /// Creates a new JSONL encoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new encoder with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Appends an item to the JSONL output.
    ///
    /// # Errors
    ///
    /// Returns an error if the item cannot be serialized to JSON.
    pub fn push<T: serde::Serialize>(&mut self, item: &T) -> Result<()> {
        serde_json::to_writer(&mut self.buffer, item)?;
        self.buffer.push(b'\n');
        Ok(())
    }

    /// Returns the encoded JSONL as bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Consumes the encoder and returns the bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    /// Consumes the encoder and returns a string.
    ///
    /// # Panics
    ///
    /// Panics if the buffer contains invalid UTF-8 (should not happen for valid JSON).
    #[must_use]
    pub fn into_string(self) -> String {
        String::from_utf8(self.buffer).expect("JSON should be valid UTF-8")
    }

    /// Returns the current length of the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parses a JSONL string into a vector of items.
///
/// This is a convenience function for parsing JSONL data when you want
/// all results at once rather than streaming.
///
/// # Example
///
/// ```rust
/// use anthropic::streaming::parse_jsonl;
/// use serde::Deserialize;
///
/// #[derive(Debug, Deserialize, PartialEq)]
/// struct Item {
///     id: i32,
/// }
///
/// let data = r#"{"id": 1}
/// {"id": 2}
/// {"id": 3}"#;
///
/// let items: Vec<Item> = parse_jsonl(data).unwrap();
/// assert_eq!(items.len(), 3);
/// ```
///
/// # Errors
///
/// Returns an error if any non-empty line fails to deserialize as `T`
/// (invalid JSON, or JSON that doesn't match `T`'s shape).
pub fn parse_jsonl<T: DeserializeOwned>(data: &str) -> Result<Vec<T>> {
    JsonlReader::from_str(data).collect()
}

/// Encodes items as a JSONL string.
///
/// # Example
///
/// ```rust
/// use anthropic::streaming::encode_jsonl;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Item { id: i32 }
///
/// let items = vec![Item { id: 1 }, Item { id: 2 }];
/// let jsonl = encode_jsonl(&items).unwrap();
/// assert!(jsonl.contains(r#"{"id":1}"#));
/// ```
///
/// # Errors
///
/// Returns an error if any item fails to serialize to JSON.
pub fn encode_jsonl<T: serde::Serialize>(items: &[T]) -> Result<String> {
    let mut encoder = JsonlEncoder::with_capacity(items.len() * 64);
    for item in items {
        encoder.push(item)?;
    }
    Ok(encoder.into_string())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestItem {
        id: String,
        value: i32,
    }

    #[test]
    fn test_jsonl_reader_from_bytes() {
        let data = r#"{"id": "1", "value": 10}
{"id": "2", "value": 20}
{"id": "3", "value": 30}
"#;

        let reader = JsonlReader::<TestItem>::from_bytes(data.as_bytes());
        let items: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].id, "1");
        assert_eq!(items[0].value, 10);
        assert_eq!(items[2].id, "3");
        assert_eq!(items[2].value, 30);
    }

    #[test]
    fn test_jsonl_reader_skips_empty_lines() {
        let data = r#"{"id": "1", "value": 10}

{"id": "2", "value": 20}

"#;

        let reader = JsonlReader::<TestItem>::from_str(data);
        let items: Vec<_> = reader.collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_jsonl_reader_handles_invalid_json() {
        let data = r#"{"id": "1", "value": 10}
not valid json
{"id": "2", "value": 20}
"#;

        let reader = JsonlReader::<TestItem>::from_str(data);
        let results: Vec<_> = reader.collect();

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok());
    }

    #[test]
    fn test_jsonl_encoder() {
        let mut encoder = JsonlEncoder::new();
        encoder
            .push(&TestItem {
                id: "1".to_string(),
                value: 10,
            })
            .unwrap();
        encoder
            .push(&TestItem {
                id: "2".to_string(),
                value: 20,
            })
            .unwrap();

        let output = encoder.into_string();
        assert!(output.contains(r#"{"id":"1","value":10}"#));
        assert!(output.contains(r#"{"id":"2","value":20}"#));
        assert!(output.ends_with('\n'));
    }

    #[test]
    fn test_parse_jsonl() {
        let data = r#"{"id": "1", "value": 10}
{"id": "2", "value": 20}"#;

        let items: Vec<TestItem> = parse_jsonl(data).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "1");
        assert_eq!(items[1].id, "2");
    }

    #[test]
    fn test_encode_jsonl() {
        let items = vec![
            TestItem {
                id: "1".to_string(),
                value: 10,
            },
            TestItem {
                id: "2".to_string(),
                value: 20,
            },
        ];

        let output = encode_jsonl(&items).unwrap();
        let lines: Vec<_> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"id\":\"1\""));
        assert!(lines[1].contains("\"id\":\"2\""));
    }

    #[test]
    fn test_encoder_capacity() {
        let encoder = JsonlEncoder::with_capacity(1024);
        assert!(encoder.is_empty());
        assert_eq!(encoder.len(), 0);
    }

    #[test]
    fn test_encoder_clear() {
        let mut encoder = JsonlEncoder::new();
        encoder
            .push(&TestItem {
                id: "1".to_string(),
                value: 10,
            })
            .unwrap();
        assert!(!encoder.is_empty());

        encoder.clear();
        assert!(encoder.is_empty());
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![
            TestItem {
                id: "a".to_string(),
                value: 1,
            },
            TestItem {
                id: "b".to_string(),
                value: 2,
            },
            TestItem {
                id: "c".to_string(),
                value: 3,
            },
        ];

        let encoded = encode_jsonl(&original).unwrap();
        let decoded: Vec<TestItem> = parse_jsonl(&encoded).unwrap();

        assert_eq!(original, decoded);
    }
}
