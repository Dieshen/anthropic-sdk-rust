//! # Anthropic Vertex AI Integration
//!
//! Google Vertex AI integration for the Anthropic SDK.
//!
//! This crate provides access to Claude models through Google Cloud's
//! Vertex AI platform, including OAuth2 authentication and region routing.
//!
//! ## Features
//!
//! - **Google Cloud Authentication**: Supports service account keys, Application
//!   Default Credentials (ADC), and GCE metadata service
//! - **Automatic Token Refresh**: OAuth2 tokens are automatically refreshed
//! - **Streaming Support**: Full support for streaming responses
//! - **Similar API**: API design mirrors the main Anthropic SDK
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use anthropic_vertex::{VertexClient, VertexModel, VertexMessageParams, MessageParam};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic_vertex::Error> {
//!     // Create client using Application Default Credentials
//!     let client = VertexClient::new("us-central1", "my-project-id").await?;
//!
//!     // Create a message
//!     let response = client.create_message(VertexMessageParams::new(
//!         VertexModel::Claude35SonnetV2,
//!         1024,
//!         vec![MessageParam::user("Hello, Claude!")],
//!     )).await?;
//!
//!     println!("{}", response.text());
//!     Ok(())
//! }
//! ```
//!
//! ## Authentication
//!
//! ### Application Default Credentials (Recommended)
//!
//! The simplest way to authenticate is using ADC:
//!
//! ```rust,ignore
//! // Uses ADC - checks GOOGLE_APPLICATION_CREDENTIALS, gcloud CLI, or GCE metadata
//! let client = VertexClient::new("us-central1", "my-project").await?;
//! ```
//!
//! ### Service Account Key File
//!
//! For explicit service account authentication:
//!
//! ```rust,ignore
//! use anthropic_vertex::auth::GoogleCredentials;
//!
//! let credentials = GoogleCredentials::from_service_account_file("key.json").await?;
//! let client = VertexClient::with_credentials("us-central1", "my-project", credentials)?;
//! ```
//!
//! ### Builder Pattern
//!
//! For full control over client configuration:
//!
//! ```rust,ignore
//! use std::time::Duration;
//!
//! let client = VertexClient::builder()
//!     .region("europe-west4")
//!     .project_id("my-project")
//!     .timeout(Duration::from_secs(300))
//!     .max_retries(5)
//!     .build()
//!     .await?;
//! ```
//!
//! ## Streaming
//!
//! For streaming responses:
//!
//! ```rust,ignore
//! use futures::StreamExt;
//!
//! let mut stream = client.create_message_stream(
//!     VertexMessageParams::new(
//!         VertexModel::Claude35SonnetV2,
//!         1024,
//!         vec![MessageParam::user("Tell me a story.")],
//!     ).with_stream()
//! ).await?;
//!
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         StreamEvent::ContentBlockDelta { delta, .. } => {
//!             if let ContentDelta::TextDelta { text } = delta {
//!                 print!("{}", text);
//!             }
//!         }
//!         StreamEvent::MessageStop => break,
//!         _ => {}
//!     }
//! }
//! ```
//!
//! ## Available Models
//!
//! The following Claude models are available on Vertex AI:
//!
//! | Model | Vertex Model ID |
//! |-------|-----------------|
//! | Claude 3 Opus | `claude-3-opus@20240229` |
//! | Claude 3 Sonnet | `claude-3-sonnet@20240229` |
//! | Claude 3 Haiku | `claude-3-haiku@20240307` |
//! | Claude 3.5 Sonnet | `claude-3-5-sonnet@20240620` |
//! | Claude 3.5 Sonnet v2 | `claude-3-5-sonnet-v2@20241022` |
//! | Claude 3.5 Haiku | `claude-3-5-haiku@20241022` |
//! | Claude Sonnet 4 | `claude-sonnet-4@20250514` |
//!
//! ## Regions
//!
//! Vertex AI is available in multiple regions. Common regions include:
//!
//! - `us-central1` (Iowa)
//! - `us-east4` (Virginia)
//! - `us-west1` (Oregon)
//! - `europe-west4` (Netherlands)
//! - `asia-northeast1` (Tokyo)
//!
//! Use `global` for the global endpoint.
//!
//! ## Error Handling
//!
//! ```rust,ignore
//! match client.create_message(params).await {
//!     Ok(response) => println!("{}", response.text()),
//!     Err(Error::Api { status, message, .. }) => {
//!         eprintln!("API error ({}): {}", status, message);
//!     }
//!     Err(Error::Auth(msg)) => {
//!         eprintln!("Authentication error: {}", msg);
//!     }
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]
// Allow these clippy lints for cleaner code - can be tightened later
#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    clippy::module_name_repetitions
)]

pub mod auth;
pub mod client;

// Re-export main types at crate root
pub use auth::GoogleCredentials;
pub use client::{
    // Client
    VertexClient,
    VertexClientBuilder,
    VertexConfig,
    // Models
    VertexModel,
    // Request/Response types
    VertexMessageParams,
    VertexMessage,
    VertexCountTokensParams,
    TokenCount,
    // Message types
    MessageParam,
    MessageContent,
    Role,
    SystemPrompt,
    SystemPromptBlock,
    StopReason,
    // Content types
    ContentBlock,
    ContentBlockParam,
    ToolUseBlock,
    ImageSource,
    ToolResultContent,
    ToolResultBlock,
    // Tool types
    Tool,
    ToolInputSchema,
    ToolChoice,
    // Misc types
    Metadata,
    CacheControl,
    ThinkingConfig,
    Usage,
    // Streaming types
    StreamEvent,
    ContentDelta,
    MessageDeltaContent,
    DeltaUsage,
    StreamError,
    // Constants
    VERTEX_ANTHROPIC_VERSION,
    DEFAULT_TIMEOUT_SECS,
    DEFAULT_MAX_RETRIES,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// =============================================================================
// Error Types
// =============================================================================

/// Error types for the Vertex AI integration.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Authentication error.
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// HTTP/network error.
    #[error("HTTP error: {0}")]
    Http(String),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(String),

    /// API error from Vertex AI.
    #[error("API error ({status}): {message}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Error type.
        error_type: String,
        /// Error message.
        message: String,
    },

    /// Invalid request error.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Streaming error.
    #[error("Streaming error: {0}")]
    Streaming(String),
}

/// Result type alias for Vertex AI operations.
pub type Result<T> = std::result::Result<T, Error>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_error_display() {
        let auth_err = Error::Auth("Invalid credentials".to_string());
        assert!(auth_err.to_string().contains("Authentication error"));

        let api_err = Error::Api {
            status: 400,
            error_type: "invalid_request".to_string(),
            message: "Bad request".to_string(),
        };
        assert!(api_err.to_string().contains("400"));
        assert!(api_err.to_string().contains("Bad request"));
    }

    #[test]
    fn test_reexports() {
        // Verify that key types are properly re-exported
        let _model = VertexModel::Claude35SonnetV2;
        let _msg = MessageParam::user("test");
        let _role = Role::User;
    }
}
