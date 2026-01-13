//! Streaming support for the Anthropic API.
//!
//! This module provides types and utilities for handling streaming responses
//! from the Anthropic API, including:
//!
//! - **SSE (Server-Sent Events)** for message streaming
//! - **JSONL (JSON Lines)** for batch results streaming
//!
//! # SSE Streaming Overview
//!
//! When streaming is enabled, the API returns a series of events:
//! - `message_start` - Initial message metadata
//! - `content_block_start` - Start of a content block
//! - `content_block_delta` - Incremental content updates
//! - `content_block_stop` - End of a content block
//! - `message_delta` - Final message metadata (stop reason, usage)
//! - `message_stop` - End of the message stream
//!
//! # Example: SSE Streaming
//!
//! ```rust,ignore
//! use anthropic::streaming::MessageStream;
//! use futures::StreamExt;
//!
//! let stream = client.messages().create_stream(params).await?;
//!
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         StreamEvent::ContentBlockDelta { delta, .. } => {
//!             if let ContentBlockDelta::TextDelta { text } = delta {
//!                 print!("{}", text);
//!             }
//!         }
//!         StreamEvent::MessageStop => break,
//!         _ => {}
//!     }
//! }
//! ```
//!
//! # Example: JSONL Batch Results
//!
//! ```rust,ignore
//! use anthropic::streaming::JsonlStream;
//! use futures::StreamExt;
//!
//! let stream = client.batches().results_stream(batch_id).await?;
//!
//! while let Some(result) = stream.next().await {
//!     match result {
//!         Ok(batch_result) => println!("Got: {:?}", batch_result),
//!         Err(e) => eprintln!("Error: {}", e),
//!     }
//! }
//! ```

mod events;
pub mod jsonl;
mod sse;
mod stream;

pub use events::{
    ContentBlockDelta, ContentBlockStartContent, MessageDelta, MessageDeltaUsage, StreamEvent,
};
pub use jsonl::{encode_jsonl, parse_jsonl, JsonlEncoder, JsonlReader, JsonlStream};
pub use sse::{SseDecoder, SseEvent, SseError};
pub use stream::{MessageStream, MessageStreamError, StreamState};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _ = std::any::type_name::<StreamEvent>();
        let _ = std::any::type_name::<ContentBlockDelta>();
        let _ = std::any::type_name::<SseDecoder>();
        let _ = std::any::type_name::<MessageStream<futures::stream::Empty<Result<bytes::Bytes, std::io::Error>>>>();
    }
}
