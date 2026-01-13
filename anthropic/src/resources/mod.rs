//! API resources for the Anthropic SDK.
//!
//! This module provides strongly-typed resource interfaces for interacting
//! with the Anthropic API. Each resource corresponds to an API endpoint
//! and provides methods for creating, retrieving, and managing API objects.
//!
//! # Resources
//!
//! - [`Messages`]: Create and stream messages via the Messages API
//! - [`Models`]: List and retrieve model information
//! - [`Batches`]: Process multiple messages efficiently via the Batches API
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
//!     // Create a message
//!     let params = MessageCreateParams::new(
//!         Model::claude_sonnet_4_5_latest(),
//!         vec![MessageParam::user("Hello!")],
//!         1024,
//!     );
//!
//!     let message = client.messages().create(params).await?;
//!     println!("{}", message.text());
//!
//!     // List available models
//!     let models = client.models().list(Default::default()).await?;
//!     for model in models.data {
//!         println!("{}: {}", model.id, model.display_name);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod batches;
pub mod messages;
pub mod models;

pub use batches::{BatchPaginator, Batches, DeletedMessageBatch};
pub use messages::{
    ContentBlockStart, ContentDelta, MessageCountTokensParams, MessageDeltaContent, Messages,
    MessageStream, StreamError, StreamEvent,
};
pub use models::{ModelInfo, Models, ModelsGetParams, ModelsListParams, ModelsListResponse};
