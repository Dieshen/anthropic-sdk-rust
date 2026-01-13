//! # Anthropic Bedrock Integration
//!
//! AWS Bedrock integration for the Anthropic SDK, providing access to Claude models
//! through AWS infrastructure with AWS Signature V4 authentication.
//!
//! ## Overview
//!
//! This crate enables using Claude models hosted on AWS Bedrock. It provides:
//!
//! - [`BedrockClient`] for making API calls to Claude models on Bedrock
//! - AWS Signature V4 request signing via the [`auth`] module
//! - Full support for message creation with tools, system prompts, and more
//! - Streaming support (returns raw `EventStream` data)
//!
//! ## Model IDs
//!
//! Bedrock uses a different model ID format than the direct Anthropic API:
//!
//! | Direct API | Bedrock |
//! |------------|---------|
//! | `claude-3-sonnet-20240229` | `anthropic.claude-3-sonnet-20240229-v1:0` |
//! | `claude-3-haiku-20240307` | `anthropic.claude-3-haiku-20240307-v1:0` |
//! | `claude-3-opus-20240229` | `anthropic.claude-3-opus-20240229-v1:0` |
//!
//! Use the constants in [`models`] for convenience.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use anthropic_bedrock::{BedrockClient, CreateMessageRequest, models};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic_bedrock::Error> {
//!     // Create client using AWS credentials from environment
//!     // Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
//!     // Optional: AWS_REGION, AWS_SESSION_TOKEN
//!     let client = BedrockClient::new()?;
//!
//!     // Create a message
//!     let response = client.create_message(
//!         models::CLAUDE_3_SONNET,
//!         CreateMessageRequest::builder(1024)
//!             .system("You are a helpful assistant.")
//!             .user("What is Rust?")
//!             .build()
//!     ).await?;
//!
//!     println!("{}", response.text());
//!     Ok(())
//! }
//! ```
//!
//! ## Configuration
//!
//! ### From Environment Variables
//!
//! The simplest way to configure the client is using environment variables:
//!
//! ```bash
//! export AWS_ACCESS_KEY_ID=AKIA...
//! export AWS_SECRET_ACCESS_KEY=secret...
//! export AWS_REGION=us-east-1  # Optional, defaults to us-east-1
//! export AWS_SESSION_TOKEN=...  # Optional, for temporary credentials
//! ```
//!
//! ### Explicit Configuration
//!
//! For more control, use the builder:
//!
//! ```rust,ignore
//! use anthropic_bedrock::{BedrockClient, auth::AwsCredentials};
//! use std::time::Duration;
//!
//! let client = BedrockClient::builder()
//!     .region("us-west-2")
//!     .credentials(AwsCredentials::new(
//!         "AKIA...",
//!         "secret...",
//!         None, // or Some("session-token") for temporary credentials
//!     ))
//!     .timeout(Duration::from_secs(60))
//!     .build()?;
//! ```
//!
//! ## Streaming
//!
//! Bedrock uses AWS `EventStream` format for streaming responses, which is different
//! from the SSE format used by the direct Anthropic API. This crate provides a
//! built-in [`eventstream`] module for decoding these frames.
//!
//! ```rust,ignore
//! use anthropic_bedrock::{BedrockClient, eventstream::{EventStreamDecoder, Event}};
//!
//! let stream = client.create_message_stream(
//!     models::CLAUDE_3_SONNET,
//!     request
//! ).await?;
//!
//! // Decode the EventStream response
//! let bytes = stream.bytes().await?;
//! let mut decoder = EventStreamDecoder::new();
//! let events = decoder.decode(&bytes)?;
//!
//! for event in events {
//!     match event {
//!         Event::Message(msg) => {
//!             println!("Event: {} - {}", msg.event_type, msg.payload);
//!         }
//!         Event::Exception(exc) => {
//!             eprintln!("Error: {}", exc.message);
//!         }
//!     }
//! }
//! ```
//!
//! ## Error Handling
//!
//! ```rust,ignore
//! match client.create_message(model, request).await {
//!     Ok(response) => println!("{}", response.text()),
//!     Err(Error::Config(msg)) => eprintln!("Configuration error: {}", msg),
//!     Err(Error::Signing(msg)) => eprintln!("AWS signing failed: {}", msg),
//!     Err(Error::Api(err)) => eprintln!("API error: {} - {}", err.status, err.message),
//!     Err(e) => eprintln!("Other error: {}", e),
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]
#![allow(clippy::module_name_repetitions)]

// Modules
pub mod auth;
pub mod client;
mod error;
pub mod eventstream;

// Re-export main types
pub use auth::AwsCredentials;
pub use client::{
    // Client
    BedrockClient,
    BedrockClientBuilder,
    BedrockConfig,
    BedrockConfigBuilder,
    BedrockStreamResponse,
    // Request/Response types
    CreateMessageRequest,
    CreateMessageRequestBuilder,
    CreateMessageResponse,
    // Message types
    ContentBlock,
    ImageSource,
    Message,
    MessageContent,
    ResponseContentBlock,
    Role,
    StopReason,
    Usage,
    // Tool types
    Tool,
    ToolChoice,
    // Model constants
    models,
    // Constants
    ANTHROPIC_VERSION,
    DEFAULT_BEDROCK_VERSION,
};
pub use error::{BedrockApiError, Error, Result};

/// SDK version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_exports() {
        // Verify key types are exported
        let _ = std::any::TypeId::of::<BedrockClient>();
        let _ = std::any::TypeId::of::<AwsCredentials>();
        let _ = std::any::TypeId::of::<CreateMessageRequest>();
        let _ = std::any::TypeId::of::<Error>();
    }
}
