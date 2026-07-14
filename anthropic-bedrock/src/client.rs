//! Bedrock client for interacting with Claude models on AWS.
//!
//! This module provides the [`BedrockClient`] type for making requests to
//! Claude models hosted on AWS Bedrock.
//!
//! # Overview
//!
//! The Bedrock client provides a similar API to the main Anthropic client,
//! but routes requests through AWS Bedrock infrastructure with AWS Signature V4
//! authentication.
//!
//! # Model IDs
//!
//! Bedrock uses a different model ID format than the direct Anthropic API:
//!
//! | Direct API | Bedrock |
//! |------------|---------|
//! | `claude-3-sonnet-20240229` | `anthropic.claude-3-sonnet-20240229-v1:0` |
//! | `claude-3-haiku-20240307` | `anthropic.claude-3-haiku-20240307-v1:0` |
//! | `claude-3-opus-20240229` | `anthropic.claude-3-opus-20240229-v1:0` |
//! | `claude-3-5-sonnet-20240620` | `anthropic.claude-3-5-sonnet-20240620-v1:0` |
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic_bedrock::{BedrockClient, BedrockConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic_bedrock::Error> {
//!     // Create client with default credentials from environment
//!     let client = BedrockClient::new()?;
//!
//!     // Or with explicit configuration
//!     let client = BedrockClient::builder()
//!         .region("us-west-2")
//!         .credentials(AwsCredentials::new("AKIA...", "secret", None))
//!         .build()?;
//!
//!     // Create a message
//!     let response = client.create_message(
//!         CreateMessageRequest::builder()
//!             .model("anthropic.claude-3-sonnet-20240229-v1:0")
//!             .max_tokens(1024)
//!             .messages(vec![Message::user("Hello!")])
//!             .build()?
//!     ).await?;
//!
//!     println!("{}", response.text());
//!     Ok(())
//! }
//! ```

#![allow(clippy::future_not_send)]

use bytes::Bytes;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, trace};

use crate::auth::{get_region_from_env, sign_request, AwsCredentials, SigningRequest};
use crate::error::{BedrockApiError, Error, Result};

// =============================================================================
// Constants
// =============================================================================

/// Default Bedrock API version.
pub const DEFAULT_BEDROCK_VERSION: &str = "bedrock-2023-05-31";

/// Anthropic version to use with Bedrock.
pub const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// Default request timeout.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Default maximum retries.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// User agent for SDK requests.
const USER_AGENT: &str = concat!("anthropic-bedrock-rust/", env!("CARGO_PKG_VERSION"));

// =============================================================================
// Model IDs
// =============================================================================

/// Predefined Bedrock model IDs for Claude models.
///
/// These constants provide the correct Bedrock model ID format for each Claude model.
pub mod models {
    /// Claude 3 Opus on Bedrock.
    pub const CLAUDE_3_OPUS: &str = "anthropic.claude-3-opus-20240229-v1:0";
    /// Claude 3 Sonnet on Bedrock.
    pub const CLAUDE_3_SONNET: &str = "anthropic.claude-3-sonnet-20240229-v1:0";
    /// Claude 3 Haiku on Bedrock.
    pub const CLAUDE_3_HAIKU: &str = "anthropic.claude-3-haiku-20240307-v1:0";
    /// Claude 3.5 Sonnet on Bedrock.
    pub const CLAUDE_3_5_SONNET: &str = "anthropic.claude-3-5-sonnet-20240620-v1:0";
    /// Claude 3.5 Sonnet v2 on Bedrock.
    pub const CLAUDE_3_5_SONNET_V2: &str = "anthropic.claude-3-5-sonnet-20241022-v2:0";
    /// Claude 3.5 Haiku on Bedrock.
    pub const CLAUDE_3_5_HAIKU: &str = "anthropic.claude-3-5-haiku-20241022-v1:0";
}

// =============================================================================
// Bedrock Configuration
// =============================================================================

/// Configuration for the Bedrock client.
#[derive(Debug, Clone)]
pub struct BedrockConfig {
    /// AWS region for Bedrock.
    region: String,
    /// AWS credentials.
    credentials: AwsCredentials,
    /// Request timeout.
    timeout: Duration,
    /// Maximum number of retries.
    max_retries: u32,
    /// Custom headers to include in requests.
    custom_headers: HeaderMap,
}

impl BedrockConfig {
    /// Creates a new configuration builder.
    #[must_use]
    pub fn builder() -> BedrockConfigBuilder {
        BedrockConfigBuilder::default()
    }

    /// Creates a configuration from environment variables.
    ///
    /// Reads AWS credentials and region from standard environment variables:
    /// - `AWS_ACCESS_KEY_ID`
    /// - `AWS_SECRET_ACCESS_KEY`
    /// - `AWS_SESSION_TOKEN` (optional)
    /// - `AWS_REGION` or `AWS_DEFAULT_REGION`
    ///
    /// # Errors
    ///
    /// Returns an error if required credentials are not found.
    pub fn from_env() -> Result<Self> {
        let credentials = AwsCredentials::from_env()?;
        let region = get_region_from_env();

        Ok(Self {
            region,
            credentials,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_retries: DEFAULT_MAX_RETRIES,
            custom_headers: HeaderMap::new(),
        })
    }

    /// Returns the AWS region.
    #[must_use]
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Returns the AWS credentials.
    #[must_use]
    pub const fn credentials(&self) -> &AwsCredentials {
        &self.credentials
    }

    /// Returns the request timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the maximum number of retries.
    #[must_use]
    pub const fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Returns the Bedrock runtime endpoint URL.
    #[must_use]
    pub fn endpoint(&self) -> String {
        format!("https://bedrock-runtime.{}.amazonaws.com", self.region)
    }

    /// Returns the host for Bedrock runtime.
    #[must_use]
    pub fn host(&self) -> String {
        format!("bedrock-runtime.{}.amazonaws.com", self.region)
    }
}

/// Builder for [`BedrockConfig`].
#[derive(Debug, Default)]
pub struct BedrockConfigBuilder {
    region: Option<String>,
    credentials: Option<AwsCredentials>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    custom_headers: HeaderMap,
}

impl BedrockConfigBuilder {
    /// Sets the AWS region.
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the AWS credentials.
    #[must_use]
    pub fn credentials(mut self, credentials: AwsCredentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the maximum number of retries.
    #[must_use]
    pub const fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Adds a custom header to all requests.
    #[must_use]
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        if let (Ok(name), Ok(value)) = (
            HeaderName::try_from(name.as_ref()),
            HeaderValue::try_from(value.as_ref()),
        ) {
            self.custom_headers.insert(name, value);
        }
        self
    }

    /// Builds the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if credentials are not provided and cannot be loaded from environment.
    pub fn build(self) -> Result<BedrockConfig> {
        let credentials = match self.credentials {
            Some(c) => c,
            None => AwsCredentials::from_env()?,
        };

        let region = self.region.unwrap_or_else(get_region_from_env);

        Ok(BedrockConfig {
            region,
            credentials,
            timeout: self
                .timeout
                .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS)),
            max_retries: self.max_retries.unwrap_or(DEFAULT_MAX_RETRIES),
            custom_headers: self.custom_headers,
        })
    }
}

// =============================================================================
// Message Types
// =============================================================================

/// Role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User role.
    User,
    /// Assistant role.
    Assistant,
}

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message author.
    pub role: Role,
    /// The content of the message.
    pub content: MessageContent,
}

impl Message {
    /// Creates a new user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a new assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a message with content blocks.
    #[must_use]
    pub const fn with_blocks(role: Role, blocks: Vec<ContentBlock>) -> Self {
        Self {
            role,
            content: MessageContent::Blocks(blocks),
        }
    }
}

/// Message content (text or blocks).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<ContentBlock>),
}

/// A content block in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Image content.
    Image {
        /// Image source.
        source: ImageSource,
    },
    /// Tool use block.
    ToolUse {
        /// Tool use ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input.
        input: serde_json::Value,
    },
    /// Tool result block.
    ToolResult {
        /// Tool use ID.
        tool_use_id: String,
        /// Tool result content.
        content: String,
    },
}

/// Image source for image content blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64 encoded image.
    Base64 {
        /// Media type (e.g., "image/png").
        media_type: String,
        /// Base64 encoded data.
        data: String,
    },
}

/// Stop reason for message generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model reached a natural stopping point.
    EndTurn,
    /// Model reached max tokens limit.
    MaxTokens,
    /// Model generated a stop sequence.
    StopSequence,
    /// Model wants to use a tool.
    ToolUse,
}

/// Token usage information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens.
    pub input_tokens: u32,
    /// Number of output tokens.
    pub output_tokens: u32,
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// Request to create a message.
#[derive(Debug, Clone, Serialize)]
pub struct CreateMessageRequest {
    /// Anthropic version (set automatically).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_version: Option<String>,

    /// The messages in the conversation.
    pub messages: Vec<Message>,

    /// Maximum number of tokens to generate.
    pub max_tokens: u32,

    /// System prompt (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Sampling temperature (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Tools available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// How the model should choose tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

impl CreateMessageRequest {
    /// Creates a new request builder.
    #[must_use]
    pub const fn builder(max_tokens: u32) -> CreateMessageRequestBuilder {
        CreateMessageRequestBuilder::new(max_tokens)
    }
}

/// Builder for [`CreateMessageRequest`].
#[derive(Debug, Clone)]
pub struct CreateMessageRequestBuilder {
    max_tokens: u32,
    messages: Vec<Message>,
    system: Option<String>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
}

impl CreateMessageRequestBuilder {
    /// Creates a new builder.
    #[must_use]
    pub const fn new(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            messages: Vec::new(),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
        }
    }

    /// Adds a message to the conversation.
    #[must_use]
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Adds a user message.
    #[must_use]
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::user(content));
        self
    }

    /// Adds an assistant message.
    #[must_use]
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(content));
        self
    }

    /// Sets the messages.
    #[must_use]
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the temperature.
    #[must_use]
    pub const fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets top-p.
    #[must_use]
    pub const fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets top-k.
    #[must_use]
    pub const fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets stop sequences.
    #[must_use]
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Sets the tools.
    #[must_use]
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the tool choice.
    #[must_use]
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Builds the request.
    #[must_use]
    pub fn build(self) -> CreateMessageRequest {
        CreateMessageRequest {
            anthropic_version: Some(ANTHROPIC_VERSION.to_string()),
            messages: self.messages,
            max_tokens: self.max_tokens,
            system: self.system,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences,
            tools: self.tools,
            tool_choice: self.tool_choice,
        }
    }
}

/// Response from creating a message.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateMessageResponse {
    /// Unique message ID.
    pub id: String,
    /// Object type (always "message").
    #[serde(rename = "type")]
    pub message_type: String,
    /// The role (always "assistant").
    pub role: Role,
    /// Content blocks in the response.
    pub content: Vec<ResponseContentBlock>,
    /// The model that generated the response.
    pub model: String,
    /// Why generation stopped.
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// The stop sequence if one was hit.
    #[serde(default)]
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: Usage,
}

impl CreateMessageResponse {
    /// Returns the concatenated text from all text blocks.
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ResponseContentBlock::Text { text } => Some(text.as_str()),
                ResponseContentBlock::ToolUse { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns all tool use blocks.
    #[must_use]
    pub fn tool_uses(&self) -> Vec<&ResponseContentBlock> {
        self.content
            .iter()
            .filter(|block| matches!(block, ResponseContentBlock::ToolUse { .. }))
            .collect()
    }

    /// Returns true if the model wants to use a tool.
    #[must_use]
    pub fn has_tool_use(&self) -> bool {
        self.stop_reason == Some(StopReason::ToolUse)
    }
}

/// Content block in a response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    /// Text content.
    Text {
        /// The text.
        text: String,
    },
    /// Tool use.
    ToolUse {
        /// Tool use ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input.
        input: serde_json::Value,
    },
}

/// Tool definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Input schema (JSON Schema).
    pub input_schema: serde_json::Value,
}

impl Tool {
    /// Creates a new tool.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            input_schema,
        }
    }
}

/// Tool choice specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model decides whether to use tools.
    Auto,
    /// Model must use a tool.
    Any,
    /// Model must use a specific tool.
    Tool {
        /// The tool name.
        name: String,
    },
}

// =============================================================================
// Bedrock Client
// =============================================================================

/// Client for interacting with Claude models on AWS Bedrock.
///
/// This client provides similar functionality to the main Anthropic client,
/// but routes requests through AWS Bedrock with Signature V4 authentication.
///
/// # Example
///
/// ```rust,ignore
/// use anthropic_bedrock::BedrockClient;
///
/// #[tokio::main]
/// async fn main() -> Result<(), anthropic_bedrock::Error> {
///     let client = BedrockClient::new().await?;
///
///     let response = client.create_message(
///         "anthropic.claude-3-sonnet-20240229-v1:0",
///         CreateMessageRequest::builder(1024)
///             .user("Hello!")
///             .build()
///     ).await?;
///
///     println!("{}", response.text());
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct BedrockClient {
    inner: Arc<BedrockClientInner>,
}

#[derive(Debug)]
struct BedrockClientInner {
    http_client: reqwest::Client,
    config: BedrockConfig,
}

impl BedrockClient {
    /// Creates a new client with default configuration from environment.
    ///
    /// # Errors
    ///
    /// Returns an error if AWS credentials cannot be loaded from environment.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic_bedrock::BedrockClient;
    ///
    /// // Ensure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optionally
    /// // AWS_REGION are set in your environment
    /// let client = BedrockClient::new()?;
    /// ```
    pub fn new() -> Result<Self> {
        let config = BedrockConfig::from_env()?;
        Self::with_config(config)
    }

    /// Creates a new client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The Bedrock configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client fails to initialize.
    pub fn with_config(config: BedrockConfig) -> Result<Self> {
        let mut default_headers = HeaderMap::new();
        default_headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        // Merge custom headers
        for (name, value) in &config.custom_headers {
            default_headers.insert(name.clone(), value.clone());
        }

        let http_client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .default_headers(default_headers)
            .timeout(config.timeout)
            .connect_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(Error::Http)?;

        Ok(Self {
            inner: Arc::new(BedrockClientInner {
                http_client,
                config,
            }),
        })
    }

    /// Creates a new client builder.
    #[must_use]
    pub fn builder() -> BedrockClientBuilder {
        BedrockClientBuilder::default()
    }

    /// Returns the configured AWS region.
    #[must_use]
    pub fn region(&self) -> &str {
        self.inner.config.region()
    }

    /// Returns the Bedrock endpoint URL.
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.inner.config.endpoint()
    }

    /// Creates a message using the specified model.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The Bedrock model ID (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
    /// * `request` - The message request parameters
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Request signing fails
    /// - Network request fails
    /// - API returns an error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic_bedrock::{BedrockClient, CreateMessageRequest, models};
    ///
    /// let client = BedrockClient::new().await?;
    ///
    /// let response = client.create_message(
    ///     models::CLAUDE_3_SONNET,
    ///     CreateMessageRequest::builder(1024)
    ///         .system("You are a helpful assistant.")
    ///         .user("What is Rust?")
    ///         .build()
    /// ).await?;
    ///
    /// println!("{}", response.text());
    /// ```
    pub async fn create_message(
        &self,
        model_id: &str,
        request: CreateMessageRequest,
    ) -> Result<CreateMessageResponse> {
        let path = format!("/model/{}/invoke", url_encode_model_id(model_id));

        self.invoke_model(&path, &request).await
    }

    /// Creates a streaming message using the specified model.
    ///
    /// # Note
    ///
    /// Bedrock uses AWS `EventStream` format for streaming, which is different
    /// from the SSE format used by the direct Anthropic API. This method
    /// returns the raw response bytes - parsing `EventStream` is complex and
    /// may require additional processing.
    ///
    /// For production streaming use, consider using the AWS SDK's Bedrock
    /// Runtime client which provides built-in `EventStream` decoding.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The Bedrock model ID
    /// * `request` - The message request parameters
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn create_message_stream(
        &self,
        model_id: &str,
        request: CreateMessageRequest,
    ) -> Result<BedrockStreamResponse> {
        let path = format!(
            "/model/{}/invoke-with-response-stream",
            url_encode_model_id(model_id)
        );

        self.invoke_model_stream(&path, &request).await
    }

    /// Invokes a Bedrock model and returns the parsed response.
    async fn invoke_model<T, R>(&self, path: &str, request: &T) -> Result<R>
    where
        T: Serialize,
        R: serde::de::DeserializeOwned,
    {
        let body = serde_json::to_vec(request)?;
        let response = self.send_signed_request(path, &body).await?;

        // Check for errors
        let status = response.status();
        if !status.is_success() {
            return Err(self.parse_error_response(status, response).await);
        }

        // Parse response
        let response_body = response.bytes().await?;
        trace!(body_len = response_body.len(), "Received response body");

        let parsed: R = serde_json::from_slice(&response_body)?;
        Ok(parsed)
    }

    /// Invokes a Bedrock model with streaming response.
    async fn invoke_model_stream<T>(&self, path: &str, request: &T) -> Result<BedrockStreamResponse>
    where
        T: Serialize,
    {
        let body = serde_json::to_vec(request)?;
        let response = self.send_signed_request(path, &body).await?;

        // Check for errors
        let status = response.status();
        if !status.is_success() {
            return Err(self.parse_error_response(status, response).await);
        }

        Ok(BedrockStreamResponse { response })
    }

    /// Sends a signed request to Bedrock.
    async fn send_signed_request(&self, path: &str, body: &[u8]) -> Result<reqwest::Response> {
        let config = &self.inner.config;
        let host = config.host();
        let url = format!("https://{host}{path}");

        debug!(
            url = %url,
            body_len = body.len(),
            region = %config.region(),
            "Sending signed request to Bedrock"
        );

        // Sign the request
        let signing_request = SigningRequest {
            credentials: config.credentials(),
            region: config.region(),
            method: "POST",
            uri: path,
            host: &host,
            headers: &[("content-type", "application/json")],
            body,
            signing_time: None,
        };
        let signed_headers = sign_request(&signing_request)?;

        // Build the request
        let mut request_builder = self.inner.http_client.post(&url).body(body.to_vec());

        // Add signed headers
        for (name, value) in signed_headers {
            request_builder = request_builder.header(name, value);
        }

        // Send request
        let response = request_builder.send().await?;

        Ok(response)
    }

    /// Parses an error response from Bedrock.
    async fn parse_error_response(&self, status: StatusCode, response: reqwest::Response) -> Error {
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Failed to read error body".to_string());

        debug!(
            status = %status,
            body = %body,
            "Received error response from Bedrock"
        );

        // Try to parse as Bedrock error
        if let Ok(error) = serde_json::from_str::<BedrockErrorResponse>(&body) {
            return Error::Api(BedrockApiError {
                status,
                error_type: error.error_type.unwrap_or_else(|| "unknown".to_string()),
                message: error.message,
            });
        }

        // Fallback to generic error
        Error::Api(BedrockApiError {
            status,
            error_type: "unknown".to_string(),
            message: body,
        })
    }
}

/// Builder for [`BedrockClient`].
#[derive(Debug, Default)]
pub struct BedrockClientBuilder {
    config_builder: BedrockConfigBuilder,
}

impl BedrockClientBuilder {
    /// Sets the AWS region.
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.config_builder = self.config_builder.region(region);
        self
    }

    /// Sets the AWS credentials.
    #[must_use]
    pub fn credentials(mut self, credentials: AwsCredentials) -> Self {
        self.config_builder = self.config_builder.credentials(credentials);
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config_builder = self.config_builder.timeout(timeout);
        self
    }

    /// Sets the maximum number of retries.
    #[must_use]
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.config_builder = self.config_builder.max_retries(max_retries);
        self
    }

    /// Adds a custom header.
    #[must_use]
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        self.config_builder = self.config_builder.header(name, value);
        self
    }

    /// Builds the client.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or HTTP client fails to initialize.
    pub fn build(self) -> Result<BedrockClient> {
        let config = self.config_builder.build()?;
        BedrockClient::with_config(config)
    }
}

// =============================================================================
// Streaming Response
// =============================================================================

/// A streaming response from Bedrock.
///
/// Bedrock uses AWS `EventStream` format for streaming, which encodes events
/// as binary frames. This struct wraps the raw HTTP response.
///
/// # `EventStream` Format
///
/// Each `EventStream` message contains:
/// - Prelude: Total byte length and headers length (8 bytes)
/// - Headers: Key-value pairs with type information
/// - Payload: The actual data (base64-encoded JSON for Bedrock)
/// - CRC: Checksum for validation
///
/// # Example
///
/// ```rust,ignore
/// let stream_response = client.create_message_stream(
///     models::CLAUDE_3_SONNET,
///     request
/// ).await?;
///
/// // Get the raw bytes for EventStream processing
/// let bytes = stream_response.bytes().await?;
/// // Parse EventStream frames...
/// ```
#[derive(Debug)]
pub struct BedrockStreamResponse {
    response: reqwest::Response,
}

impl BedrockStreamResponse {
    /// Returns the HTTP status code.
    #[must_use]
    pub fn status(&self) -> StatusCode {
        self.response.status()
    }

    /// Consumes the response and returns the body as bytes.
    ///
    /// The bytes are in AWS `EventStream` format and need to be decoded.
    ///
    /// # Errors
    ///
    /// Returns an error if reading the response body fails.
    pub async fn bytes(self) -> Result<Bytes> {
        Ok(self.response.bytes().await?)
    }

    /// Returns the underlying response for custom processing.
    #[must_use]
    pub fn into_response(self) -> reqwest::Response {
        self.response
    }
}

// =============================================================================
// Helper Types
// =============================================================================

/// Raw Bedrock error response.
#[derive(Debug, Deserialize)]
struct BedrockErrorResponse {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: String,
}

/// URL-encodes a model ID for use in paths.
fn url_encode_model_id(model_id: &str) -> String {
    // Model IDs can contain colons which need URL encoding
    model_id.replace(':', "%3A")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello!");
        assert_eq!(msg.role, Role::User);
        match msg.content {
            MessageContent::Text(text) => assert_eq!(text, "Hello!"),
            MessageContent::Blocks(_) => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("Hi there!");
        assert_eq!(msg.role, Role::Assistant);
    }

    #[test]
    fn test_create_message_request_builder() {
        let request = CreateMessageRequest::builder(1024)
            .system("You are helpful.")
            .user("Hello!")
            .temperature(0.7)
            .build();

        assert_eq!(request.max_tokens, 1024);
        assert_eq!(request.system, Some("You are helpful.".to_string()));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.messages.len(), 1);
    }

    #[test]
    fn test_create_message_request_serialization() {
        let request = CreateMessageRequest::builder(1024).user("Hello!").build();

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(json.contains("\"messages\""));
        assert!(json.contains("\"anthropic_version\""));
    }

    #[test]
    fn test_response_text_extraction() {
        let response = CreateMessageResponse {
            id: "msg_123".to_string(),
            message_type: "message".to_string(),
            role: Role::Assistant,
            content: vec![
                ResponseContentBlock::Text {
                    text: "Hello, ".to_string(),
                },
                ResponseContentBlock::Text {
                    text: "world!".to_string(),
                },
            ],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        assert_eq!(response.text(), "Hello, world!");
    }

    #[test]
    fn test_tool_creation() {
        let tool = Tool::new(
            "calculator",
            Some("Performs calculations".to_string()),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }),
        );

        assert_eq!(tool.name, "calculator");
        assert_eq!(tool.description, Some("Performs calculations".to_string()));
    }

    #[test]
    fn test_tool_choice_serialization() {
        let auto = ToolChoice::Auto;
        let json = serde_json::to_string(&auto).unwrap();
        assert!(json.contains("\"type\":\"auto\""));

        let specific = ToolChoice::Tool {
            name: "calculator".to_string(),
        };
        let json = serde_json::to_string(&specific).unwrap();
        assert!(json.contains("\"type\":\"tool\""));
        assert!(json.contains("\"name\":\"calculator\""));
    }

    #[test]
    fn test_url_encode_model_id() {
        assert_eq!(
            url_encode_model_id("anthropic.claude-3-sonnet-20240229-v1:0"),
            "anthropic.claude-3-sonnet-20240229-v1%3A0"
        );
    }

    #[test]
    fn test_model_constants() {
        assert!(models::CLAUDE_3_OPUS.contains("opus"));
        assert!(models::CLAUDE_3_SONNET.contains("sonnet"));
        assert!(models::CLAUDE_3_HAIKU.contains("haiku"));
        assert!(models::CLAUDE_3_5_SONNET.contains("claude-3-5-sonnet"));
    }

    #[test]
    fn test_config_endpoint() {
        let config = BedrockConfig {
            region: "us-west-2".to_string(),
            credentials: AwsCredentials::new("AKIA", "secret", None),
            timeout: Duration::from_secs(60),
            max_retries: 2,
            custom_headers: HeaderMap::new(),
        };

        assert_eq!(
            config.endpoint(),
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        );
        assert_eq!(config.host(), "bedrock-runtime.us-west-2.amazonaws.com");
    }

    #[test]
    fn test_content_block_serialization() {
        let text_block = ContentBlock::Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&text_block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_stop_reason_serialization() {
        let reason = StopReason::EndTurn;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"end_turn\"");

        let reason = StopReason::ToolUse;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"tool_use\"");
    }

    #[test]
    fn test_response_has_tool_use() {
        let response = CreateMessageResponse {
            id: "msg_123".to_string(),
            message_type: "message".to_string(),
            role: Role::Assistant,
            content: vec![ResponseContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "calculator".to_string(),
                input: serde_json::json!({"expression": "2+2"}),
            }],
            model: "claude-3-sonnet".to_string(),
            stop_reason: Some(StopReason::ToolUse),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        assert!(response.has_tool_use());
        assert_eq!(response.tool_uses().len(), 1);
    }
}
