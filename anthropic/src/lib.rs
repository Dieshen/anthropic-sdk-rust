//! # Anthropic SDK for Rust
//!
//! Official Rust client for the Anthropic API.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use anthropic::{Anthropic, Model};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic::Error> {
//!     let client = Anthropic::new()?;
//!
//!     // Create a message
//!     let response = client.messages().create(
//!         MessageCreateParams::builder()
//!             .model(Model::CLAUDE_SONNET_4_5_LATEST)
//!             .max_tokens(1024)
//!             .messages(vec![MessageParam::user("Hello!")])
//!             .build()?
//!     ).await?;
//!
//!     println!("{}", response.content_text());
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - `tokio-runtime` (default): Use tokio async runtime
//! - `async-std-runtime`: Use async-std runtime
//! - `rustls-tls` (default): Use rustls for TLS
//! - `native-tls`: Use native TLS implementation
//! - `bedrock`: Enable AWS Bedrock integration
//! - `vertex`: Enable Google Vertex AI integration
//! - `beta`: Enable beta API features

#![forbid(unsafe_code)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

// Core modules
pub mod client;
pub mod config;
pub mod error;
pub mod http;
pub mod types;

// Modules to be implemented in Phase 3+
// pub mod resources;
// pub mod streaming;

// Beta features (feature-gated)
// #[cfg(feature = "beta")]
// pub mod beta;

/// SDK version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export commonly used types
pub use client::{Anthropic, AnthropicBuilder};
pub use config::{ClientConfig, ClientConfigBuilder};
pub use error::{ApiError, ApiErrorType, Error, Result};

// Re-export types module at top level
pub use types::{
    // Models
    Model,
    // Messages
    Message, MessageContent, MessageCreateParams, MessageCreateParamsBuilder, MessageParam,
    SystemPrompt, ThinkingConfig,
    // Content
    ContentBlock, ContentBlockParam, TextBlock, TextBlockParam,
    ToolUseBlock, ToolResultBlockParam,
    // Tools
    Tool, ToolParam, ToolChoice, ToolInputSchema,
    // Shared
    Role, StopReason, CacheControl, Metadata,
    // Usage
    Usage,
    // Batch
    MessageBatch, BatchRequest, BatchCreateParams, BatchResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
