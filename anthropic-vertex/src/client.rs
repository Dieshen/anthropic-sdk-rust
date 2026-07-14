//! Vertex AI client for Anthropic models.
//!
//! This module provides a client for accessing Claude models through
//! Google Cloud's Vertex AI platform.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic_vertex::{VertexClient, VertexModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic_vertex::Error> {
//!     let client = VertexClient::new("us-central1", "my-project-id").await?;
//!
//!     let response = client.create_message(
//!         VertexModel::Claude3Sonnet,
//!         1024,
//!         vec![MessageParam::user("Hello!")],
//!     ).await?;
//!
//!     println!("{}", response.text());
//!     Ok(())
//! }
//! ```

use bytes::Bytes;
use futures::stream::Stream;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

use crate::auth::GoogleCredentials;
use crate::{Error, Result};

// =============================================================================
// Constants
// =============================================================================

/// Default Anthropic version for Vertex AI.
pub const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// Default request timeout for Vertex AI.
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Default maximum retry attempts.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// User agent for SDK requests.
const USER_AGENT: &str = concat!("anthropic-vertex-rust/", env!("CARGO_PKG_VERSION"));

// =============================================================================
// Vertex Models
// =============================================================================

/// Available Claude models on Vertex AI.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum VertexModel {
    /// Claude 3 Opus
    Claude3Opus,
    /// Claude 3 Sonnet
    Claude3Sonnet,
    /// Claude 3 Haiku
    Claude3Haiku,
    /// Claude 3.5 Sonnet
    Claude35Sonnet,
    /// Claude 3.5 Sonnet v2
    #[default]
    Claude35SonnetV2,
    /// Claude 3.5 Haiku
    Claude35Haiku,
    /// Claude Sonnet 4
    ClaudeSonnet4,
    /// Custom model ID
    Custom(String),
}

impl VertexModel {
    /// Returns the Vertex AI model ID.
    pub fn model_id(&self) -> &str {
        match self {
            Self::Claude3Opus => "claude-3-opus@20240229",
            Self::Claude3Sonnet => "claude-3-sonnet@20240229",
            Self::Claude3Haiku => "claude-3-haiku@20240307",
            Self::Claude35Sonnet => "claude-3-5-sonnet@20240620",
            Self::Claude35SonnetV2 => "claude-3-5-sonnet-v2@20241022",
            Self::Claude35Haiku => "claude-3-5-haiku@20241022",
            Self::ClaudeSonnet4 => "claude-sonnet-4@20250514",
            Self::Custom(id) => id,
        }
    }

    /// Creates a custom model from a model ID string.
    pub fn custom(model_id: impl Into<String>) -> Self {
        Self::Custom(model_id.into())
    }
}

impl std::fmt::Display for VertexModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model_id())
    }
}

impl From<&str> for VertexModel {
    fn from(s: &str) -> Self {
        match s {
            "claude-3-opus@20240229" => Self::Claude3Opus,
            "claude-3-sonnet@20240229" => Self::Claude3Sonnet,
            "claude-3-haiku@20240307" => Self::Claude3Haiku,
            "claude-3-5-sonnet@20240620" => Self::Claude35Sonnet,
            "claude-3-5-sonnet-v2@20241022" => Self::Claude35SonnetV2,
            "claude-3-5-haiku@20241022" => Self::Claude35Haiku,
            "claude-sonnet-4@20250514" => Self::ClaudeSonnet4,
            other => Self::Custom(other.to_string()),
        }
    }
}

// =============================================================================
// Vertex Client Configuration
// =============================================================================

/// Configuration for the Vertex AI client.
#[derive(Debug, Clone)]
pub struct VertexConfig {
    /// Google Cloud region (e.g., "us-central1").
    pub region: String,
    /// Google Cloud project ID.
    pub project_id: String,
    /// Request timeout duration.
    pub timeout: Duration,
    /// Maximum retry attempts.
    pub max_retries: u32,
    /// Additional headers to include in requests.
    pub default_headers: HeaderMap,
}

impl VertexConfig {
    /// Creates a new Vertex configuration.
    pub fn new(region: impl Into<String>, project_id: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            project_id: project_id.into(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_retries: DEFAULT_MAX_RETRIES,
            default_headers: HeaderMap::new(),
        }
    }

    /// Sets the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the maximum retry attempts.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Adds a custom header.
    pub fn with_header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        if let (Ok(name), Ok(value)) = (
            HeaderName::try_from(name.as_ref()),
            HeaderValue::try_from(value.as_ref()),
        ) {
            self.default_headers.insert(name, value);
        }
        self
    }

    /// Returns the base URL for the Vertex AI endpoint.
    pub fn base_url(&self) -> String {
        if self.region == "global" {
            "https://aiplatform.googleapis.com".to_string()
        } else {
            format!("https://{}-aiplatform.googleapis.com", self.region)
        }
    }
}

// =============================================================================
// Vertex Client
// =============================================================================

/// Client for accessing Claude models through Google Vertex AI.
///
/// This client provides an API similar to the main Anthropic client but
/// routes requests through Google Cloud's Vertex AI platform.
///
/// # Authentication
///
/// The client uses Google Cloud credentials for authentication. Credentials
/// can be provided via:
/// - Service account JSON key file
/// - Application Default Credentials (ADC)
/// - GCE metadata service (when running on Google Cloud)
///
/// # Example
///
/// ```rust,ignore
/// use anthropic_vertex::{VertexClient, VertexModel};
///
/// // Create client with ADC (recommended)
/// let client = VertexClient::new("us-central1", "my-project").await?;
///
/// // Or with explicit credentials
/// let client = VertexClient::with_credentials(
///     "us-central1",
///     "my-project",
///     GoogleCredentials::from_service_account_file("key.json").await?,
/// )?;
/// ```
#[derive(Debug, Clone)]
pub struct VertexClient {
    /// Internal shared state.
    inner: Arc<VertexClientInner>,
}

/// Internal state for the Vertex client.
#[derive(Debug)]
struct VertexClientInner {
    /// Configuration.
    config: VertexConfig,
    /// Google Cloud credentials.
    credentials: GoogleCredentials,
    /// HTTP client.
    http_client: reqwest::Client,
}

impl VertexClient {
    /// Creates a new Vertex AI client using Application Default Credentials.
    ///
    /// # Arguments
    ///
    /// * `region` - Google Cloud region (e.g., "us-central1", "europe-west4")
    /// * `project_id` - Google Cloud project ID
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No valid credentials can be found
    /// - The HTTP client fails to initialize
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = VertexClient::new("us-central1", "my-project").await?;
    /// ```
    pub async fn new(region: impl Into<String>, project_id: impl Into<String>) -> Result<Self> {
        let config = VertexConfig::new(region, project_id);
        let credentials = GoogleCredentials::from_adc().await?;
        Self::with_credentials_and_config(credentials, config)
    }

    /// Creates a new Vertex AI client with explicit credentials.
    ///
    /// # Arguments
    ///
    /// * `region` - Google Cloud region
    /// * `project_id` - Google Cloud project ID
    /// * `credentials` - Google Cloud credentials
    pub fn with_credentials(
        region: impl Into<String>,
        project_id: impl Into<String>,
        credentials: GoogleCredentials,
    ) -> Result<Self> {
        let config = VertexConfig::new(region, project_id);
        Self::with_credentials_and_config(credentials, config)
    }

    /// Creates a new Vertex AI client with custom configuration.
    pub fn with_config(config: VertexConfig, credentials: GoogleCredentials) -> Result<Self> {
        Self::with_credentials_and_config(credentials, config)
    }

    /// Creates a new Vertex AI client with credentials and configuration.
    fn with_credentials_and_config(
        credentials: GoogleCredentials,
        config: VertexConfig,
    ) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );

        // Merge custom headers
        for (name, value) in &config.default_headers {
            headers.insert(name.clone(), value.clone());
        }

        let http_client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .default_headers(headers)
            .timeout(config.timeout)
            .connect_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()
            .map_err(|e| Error::Http(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(VertexClientInner {
                config,
                credentials,
                http_client,
            }),
        })
    }

    /// Creates a new builder for the Vertex client.
    pub fn builder() -> VertexClientBuilder {
        VertexClientBuilder::new()
    }

    /// Returns the configured region.
    pub fn region(&self) -> &str {
        &self.inner.config.region
    }

    /// Returns the configured project ID.
    pub fn project_id(&self) -> &str {
        &self.inner.config.project_id
    }

    /// Returns the base URL for API requests.
    pub fn base_url(&self) -> String {
        self.inner.config.base_url()
    }

    /// Creates a message using the Claude model.
    ///
    /// # Arguments
    ///
    /// * `params` - Message creation parameters
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = client.create_message(VertexMessageParams {
    ///     model: VertexModel::Claude3Sonnet,
    ///     max_tokens: 1024,
    ///     messages: vec![MessageParam::user("Hello!")],
    ///     ..Default::default()
    /// }).await?;
    /// ```
    pub async fn create_message(&self, params: VertexMessageParams) -> Result<VertexMessage> {
        let stream = params.stream.unwrap_or(false);
        if stream {
            return Err(Error::InvalidRequest(
                "Use create_message_stream for streaming requests".to_string(),
            ));
        }

        let model_id = params.model.model_id().to_string();
        let path = self.build_predict_path(&model_id, false);

        // Build the request body (remove model from body for Vertex)
        let body = Self::build_request_body(&params);

        self.post(&path, &body).await
    }

    /// Creates a streaming message using the Claude model.
    ///
    /// Returns a stream of server-sent events.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut stream = client.create_message_stream(VertexMessageParams {
    ///     model: VertexModel::Claude3Sonnet,
    ///     max_tokens: 1024,
    ///     messages: vec![MessageParam::user("Hello!")],
    ///     stream: Some(true),
    ///     ..Default::default()
    /// }).await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         Ok(data) => println!("{:?}", data),
    ///         Err(e) => eprintln!("Error: {}", e),
    ///     }
    /// }
    /// ```
    pub async fn create_message_stream(
        &self,
        mut params: VertexMessageParams,
    ) -> Result<impl Stream<Item = Result<StreamEvent>>> {
        params.stream = Some(true);

        let model_id = params.model.model_id().to_string();
        let path = self.build_predict_path(&model_id, true);

        // Build the request body
        let body = Self::build_request_body(&params);

        self.post_stream(&path, &body).await
    }

    /// Counts tokens for a message request.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let count = client.count_tokens(VertexCountTokensParams {
    ///     model: VertexModel::Claude3Sonnet,
    ///     messages: vec![MessageParam::user("Hello!")],
    ///     ..Default::default()
    /// }).await?;
    ///
    /// println!("Input tokens: {}", count.input_tokens);
    /// ```
    pub async fn count_tokens(&self, params: VertexCountTokensParams) -> Result<TokenCount> {
        let path = format!(
            "/v1/projects/{}/locations/{}/publishers/anthropic/models/count-tokens:rawPredict",
            self.inner.config.project_id, self.inner.config.region
        );

        let body = VertexCountTokensRequest {
            anthropic_version: VERTEX_ANTHROPIC_VERSION.to_string(),
            messages: params.messages,
            system: params.system,
            tools: params.tools,
        };

        self.post(&path, &body).await
    }

    /// Builds the predict endpoint path.
    fn build_predict_path(&self, model_id: &str, stream: bool) -> String {
        let specifier = if stream {
            "streamRawPredict"
        } else {
            "rawPredict"
        };

        format!(
            "/v1/projects/{}/locations/{}/publishers/anthropic/models/{}:{}",
            self.inner.config.project_id, self.inner.config.region, model_id, specifier
        )
    }

    /// Builds the request body for Vertex AI.
    fn build_request_body(params: &VertexMessageParams) -> VertexMessageRequest {
        VertexMessageRequest {
            anthropic_version: VERTEX_ANTHROPIC_VERSION.to_string(),
            max_tokens: params.max_tokens,
            messages: params.messages.clone(),
            system: params.system.clone(),
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            stop_sequences: params.stop_sequences.clone(),
            stream: params.stream,
            tools: params.tools.clone(),
            tool_choice: params.tool_choice.clone(),
            metadata: params.metadata.clone(),
            thinking: params.thinking.clone(),
        }
    }

    /// Makes a POST request with JSON body.
    async fn post<T, B>(&self, path: &str, body: &B) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize + Sync,
    {
        let url = format!("{}{}", self.inner.config.base_url(), path);

        // Get fresh token
        let token = self.inner.credentials.get_token().await?;

        let response = self
            .inner
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .json(body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status, body));
        }

        response
            .json()
            .await
            .map_err(|e| Error::Json(e.to_string()))
    }

    /// Makes a streaming POST request.
    async fn post_stream<B>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<impl Stream<Item = Result<StreamEvent>>>
    where
        B: serde::Serialize + Sync,
    {
        let url = format!("{}{}", self.inner.config.base_url(), path);

        // Get fresh token
        let token = self.inner.credentials.get_token().await?;

        let response = self
            .inner
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .json(body)
            .send()
            .await
            .map_err(|e| Error::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Self::parse_error(status, body));
        }

        // Create a stream from the response bytes
        let byte_stream = response.bytes_stream();

        Ok(SseStream::new(byte_stream))
    }

    /// Parses an error response.
    fn parse_error(status: StatusCode, body: String) -> Error {
        // Try to parse as Vertex/Anthropic error format
        if let Ok(error_response) = serde_json::from_str::<VertexErrorResponse>(&body) {
            return Error::Api {
                status: status.as_u16(),
                error_type: error_response.error.error_type,
                message: error_response.error.message,
            };
        }

        // Try Google Cloud error format
        if let Ok(gcp_error) = serde_json::from_str::<GcpErrorResponse>(&body) {
            return Error::Api {
                status: status.as_u16(),
                error_type: "gcp_error".to_string(),
                message: gcp_error.error.message,
            };
        }

        Error::Api {
            status: status.as_u16(),
            error_type: "unknown".to_string(),
            message: if body.is_empty() {
                status
                    .canonical_reason()
                    .unwrap_or("Unknown error")
                    .to_string()
            } else {
                body
            },
        }
    }
}

// =============================================================================
// Vertex Client Builder
// =============================================================================

/// Builder for creating a [`VertexClient`].
#[derive(Debug, Default)]
pub struct VertexClientBuilder {
    region: Option<String>,
    project_id: Option<String>,
    credentials: Option<GoogleCredentials>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    headers: HeaderMap,
}

impl VertexClientBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the Google Cloud region.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the Google Cloud project ID.
    pub fn project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    /// Sets the Google Cloud credentials.
    pub fn credentials(mut self, credentials: GoogleCredentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Sets the request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the maximum retry attempts.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Adds a custom header.
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        if let (Ok(name), Ok(value)) = (
            HeaderName::try_from(name.as_ref()),
            HeaderValue::try_from(value.as_ref()),
        ) {
            self.headers.insert(name, value);
        }
        self
    }

    /// Builds the Vertex client using ADC for credentials if not provided.
    pub async fn build(self) -> Result<VertexClient> {
        let region = self
            .region
            .ok_or_else(|| Error::Config("region is required".to_string()))?;
        let project_id = self
            .project_id
            .ok_or_else(|| Error::Config("project_id is required".to_string()))?;

        let credentials = match self.credentials {
            Some(creds) => creds,
            None => GoogleCredentials::from_adc().await?,
        };

        let mut config = VertexConfig::new(region, project_id);

        if let Some(timeout) = self.timeout {
            config = config.with_timeout(timeout);
        }

        if let Some(max_retries) = self.max_retries {
            config = config.with_max_retries(max_retries);
        }

        config.default_headers = self.headers;

        VertexClient::with_credentials_and_config(credentials, config)
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// Parameters for creating a message on Vertex AI.
#[derive(Debug, Clone, Default)]
pub struct VertexMessageParams {
    /// The model to use.
    pub model: VertexModel,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Messages in the conversation.
    pub messages: Vec<MessageParam>,
    /// System prompt.
    pub system: Option<SystemPrompt>,
    /// Sampling temperature (0.0 to 1.0).
    pub temperature: Option<f64>,
    /// Top-p sampling parameter.
    pub top_p: Option<f64>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream the response.
    pub stream: Option<bool>,
    /// Tools available for the model.
    pub tools: Option<Vec<Tool>>,
    /// Tool choice configuration.
    pub tool_choice: Option<ToolChoice>,
    /// Request metadata.
    pub metadata: Option<Metadata>,
    /// Extended thinking configuration.
    pub thinking: Option<ThinkingConfig>,
}

impl VertexMessageParams {
    /// Creates new message parameters.
    pub fn new(model: VertexModel, max_tokens: u32, messages: Vec<MessageParam>) -> Self {
        Self {
            model,
            max_tokens,
            messages,
            ..Default::default()
        }
    }

    /// Sets the system prompt.
    pub fn with_system(mut self, system: impl Into<SystemPrompt>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Enables streaming.
    pub fn with_stream(mut self) -> Self {
        self.stream = Some(true);
        self
    }

    /// Sets the tools.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
}

/// Internal request format for Vertex AI.
#[derive(Debug, Clone, Serialize)]
struct VertexMessageRequest {
    anthropic_version: String,
    max_tokens: u32,
    messages: Vec<MessageParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

/// Message response from Vertex AI.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VertexMessage {
    /// Unique message ID.
    pub id: String,
    /// Response type (always "message").
    #[serde(rename = "type")]
    pub message_type: String,
    /// Role of the message author.
    pub role: Role,
    /// Content blocks.
    pub content: Vec<ContentBlock>,
    /// Model that generated the response.
    pub model: String,
    /// Reason generation stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Stop sequence that triggered stop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// Token usage statistics.
    pub usage: Usage,
}

impl VertexMessage {
    /// Returns the concatenated text content.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns all tool use blocks.
    pub fn tool_uses(&self) -> Vec<&ToolUseBlock> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse(tool_use) => Some(tool_use),
                _ => None,
            })
            .collect()
    }

    /// Returns true if generation stopped due to tool use.
    pub fn has_tool_use(&self) -> bool {
        self.stop_reason == Some(StopReason::ToolUse)
    }
}

// =============================================================================
// Token Counting
// =============================================================================

/// Parameters for counting tokens.
#[derive(Debug, Clone, Default)]
pub struct VertexCountTokensParams {
    /// The model to use for counting.
    pub model: VertexModel,
    /// Messages to count.
    pub messages: Vec<MessageParam>,
    /// System prompt.
    pub system: Option<SystemPrompt>,
    /// Tools.
    pub tools: Option<Vec<Tool>>,
}

/// Internal request for token counting.
#[derive(Debug, Clone, Serialize)]
struct VertexCountTokensRequest {
    anthropic_version: String,
    messages: Vec<MessageParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

/// Token count response.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenCount {
    /// Number of input tokens.
    pub input_tokens: u32,
}

// =============================================================================
// Message Types (simplified versions for Vertex)
// =============================================================================

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageParam {
    /// Role of the message author.
    pub role: Role,
    /// Content of the message.
    pub content: MessageContent,
}

impl MessageParam {
    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a user message with content blocks.
    pub fn user_with_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(blocks),
        }
    }

    /// Creates an assistant message with content blocks.
    pub fn assistant_with_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Blocks(blocks),
        }
    }
}

/// Message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text.
    Text(String),
    /// Structured blocks.
    Blocks(Vec<ContentBlockParam>),
}

/// Role in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User role.
    User,
    /// Assistant role.
    Assistant,
}

/// System prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple text.
    Text(String),
    /// Structured blocks.
    Blocks(Vec<SystemPromptBlock>),
}

impl From<&str> for SystemPrompt {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<String> for SystemPrompt {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

/// System prompt block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPromptBlock {
    /// Block type.
    #[serde(rename = "type")]
    pub block_type: String,
    /// Text content.
    pub text: String,
    /// Cache control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Stop reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// End of turn.
    EndTurn,
    /// Hit max tokens.
    MaxTokens,
    /// Hit stop sequence.
    StopSequence,
    /// Tool use.
    ToolUse,
}

/// Content block in response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text block.
    Text {
        /// The text content.
        text: String,
    },
    /// Tool use block.
    ToolUse(ToolUseBlock),
    /// Thinking block.
    Thinking {
        /// Thinking content.
        thinking: String,
    },
}

/// Tool use block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    /// Tool use ID.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool input.
    pub input: serde_json::Value,
}

/// Content block parameter for requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockParam {
    /// Text block.
    Text {
        /// The text content.
        text: String,
    },
    /// Image block.
    Image {
        /// Image source.
        source: ImageSource,
    },
    /// Tool result block.
    ToolResult {
        /// Tool use ID.
        tool_use_id: String,
        /// Result content.
        content: ToolResultContent,
    },
}

/// Image source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64 encoded image.
    Base64 {
        /// Media type.
        media_type: String,
        /// Base64 data.
        data: String,
    },
    /// URL image.
    Url {
        /// Image URL.
        url: String,
    },
}

/// Tool result content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple text result.
    Text(String),
    /// Structured blocks.
    Blocks(Vec<ToolResultBlock>),
}

/// Tool result block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    /// Text result.
    Text {
        /// The text.
        text: String,
    },
    /// Image result.
    Image {
        /// Image source.
        source: ImageSource,
    },
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Input schema.
    pub input_schema: ToolInputSchema,
}

/// Tool input schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    /// Schema type (always "object").
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Properties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
    /// Required properties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl ToolInputSchema {
    /// Creates a new tool input schema.
    pub fn new() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: None,
            required: None,
        }
    }

    /// Sets the properties.
    pub fn with_properties(mut self, properties: serde_json::Value) -> Self {
        self.properties = Some(properties);
        self
    }

    /// Sets the required properties.
    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = Some(required);
        self
    }
}

impl Default for ToolInputSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Auto selection.
    Auto,
    /// Any tool.
    Any,
    /// Specific tool.
    Tool {
        /// Tool name.
        name: String,
    },
}

/// Request metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// User ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Cache control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache type.
    #[serde(rename = "type")]
    pub cache_type: String,
}

/// Extended thinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Thinking type.
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// Budget in tokens.
    pub budget_tokens: u32,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens.
    pub input_tokens: u32,
    /// Output tokens.
    pub output_tokens: u32,
    /// Cache creation tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Cache read tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

// =============================================================================
// Error Response Types
// =============================================================================

/// Vertex/Anthropic error response.
#[derive(Debug, Deserialize)]
struct VertexErrorResponse {
    error: VertexError,
}

#[derive(Debug, Deserialize)]
struct VertexError {
    #[serde(rename = "type", default)]
    error_type: String,
    message: String,
}

/// GCP error response format.
#[derive(Debug, Deserialize)]
struct GcpErrorResponse {
    error: GcpError,
}

#[derive(Debug, Deserialize)]
struct GcpError {
    message: String,
    #[allow(dead_code)]
    code: Option<i32>,
    #[allow(dead_code)]
    status: Option<String>,
}

// =============================================================================
// Streaming
// =============================================================================

/// Server-sent event from streaming response.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Message start event.
    MessageStart {
        /// The message.
        message: VertexMessage,
    },
    /// Content block start.
    ContentBlockStart {
        /// Block index.
        index: u32,
        /// Content block.
        content_block: ContentBlock,
    },
    /// Content block delta.
    ContentBlockDelta {
        /// Block index.
        index: u32,
        /// Delta content.
        delta: ContentDelta,
    },
    /// Content block stop.
    ContentBlockStop {
        /// Block index.
        index: u32,
    },
    /// Message delta.
    MessageDelta {
        /// Delta.
        delta: MessageDeltaContent,
        /// Usage.
        usage: Option<DeltaUsage>,
    },
    /// Message stop.
    MessageStop,
    /// Ping event.
    Ping,
    /// Error event.
    Error {
        /// Error details.
        error: StreamError,
    },
}

/// Content delta in streaming.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Text delta.
    TextDelta {
        /// Text content.
        text: String,
    },
    /// Input JSON delta.
    InputJsonDelta {
        /// Partial JSON.
        partial_json: String,
    },
    /// Thinking delta.
    ThinkingDelta {
        /// Thinking content.
        thinking: String,
    },
}

/// Message delta content.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDeltaContent {
    /// Stop reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Stop sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Delta usage.
#[derive(Debug, Clone, Deserialize)]
pub struct DeltaUsage {
    /// Output tokens.
    pub output_tokens: u32,
}

/// Stream error.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamError {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message.
    pub message: String,
}

/// SSE stream wrapper.
struct SseStream<S> {
    inner: S,
    buffer: String,
}

impl<S> SseStream<S>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Unpin,
{
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
        }
    }
}

impl<S> Stream for SseStream<S>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<StreamEvent>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            // Try to parse events from buffer first
            if let Some(event) = self.parse_next_event() {
                return std::task::Poll::Ready(Some(event));
            }

            // Need more data
            match Pin::new(&mut self.inner).poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        self.buffer.push_str(text);
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(Error::Http(e.to_string()))));
                }
                std::task::Poll::Ready(None) => {
                    // Stream ended, check buffer one more time
                    if let Some(event) = self.parse_next_event() {
                        return std::task::Poll::Ready(Some(event));
                    }
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

impl<S> SseStream<S> {
    /// Parses the next event from the buffer.
    fn parse_next_event(&mut self) -> Option<Result<StreamEvent>> {
        // Look for complete SSE messages (terminated by double newline)
        while let Some(pos) = self.buffer.find("\n\n") {
            let message = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            // Parse SSE format - extract data field (event type is embedded in JSON)
            let mut data = String::new();

            for line in message.lines() {
                if let Some(value) = line.strip_prefix("data: ") {
                    data = value.to_string();
                }
                // Note: event type is parsed from the JSON "type" field, not SSE event field
            }

            // Skip empty events or just newlines
            if data.is_empty() {
                continue;
            }

            // Parse the JSON data
            match serde_json::from_str::<StreamEvent>(&data) {
                Ok(event) => return Some(Ok(event)),
                Err(e) => {
                    warn!(error = %e, data = %data, "Failed to parse stream event");
                    // Continue trying to parse more events
                }
            }
        }

        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_model_ids() {
        assert_eq!(
            VertexModel::Claude3Opus.model_id(),
            "claude-3-opus@20240229"
        );
        assert_eq!(
            VertexModel::Claude3Sonnet.model_id(),
            "claude-3-sonnet@20240229"
        );
        assert_eq!(
            VertexModel::Claude35SonnetV2.model_id(),
            "claude-3-5-sonnet-v2@20241022"
        );
        assert_eq!(
            VertexModel::custom("custom-model").model_id(),
            "custom-model"
        );
    }

    #[test]
    fn test_vertex_model_from_str() {
        assert_eq!(
            VertexModel::from("claude-3-opus@20240229"),
            VertexModel::Claude3Opus
        );
        assert_eq!(
            VertexModel::from("custom-model"),
            VertexModel::Custom("custom-model".to_string())
        );
    }

    #[test]
    fn test_vertex_config_base_url() {
        let config = VertexConfig::new("us-central1", "my-project");
        assert_eq!(
            config.base_url(),
            "https://us-central1-aiplatform.googleapis.com"
        );

        let global_config = VertexConfig::new("global", "my-project");
        assert_eq!(
            global_config.base_url(),
            "https://aiplatform.googleapis.com"
        );
    }

    #[test]
    fn test_message_param_creation() {
        let user_msg = MessageParam::user("Hello!");
        assert_eq!(user_msg.role, Role::User);
        match user_msg.content {
            MessageContent::Text(text) => assert_eq!(text, "Hello!"),
            MessageContent::Blocks(_) => panic!("Expected text content"),
        }

        let assistant_msg = MessageParam::assistant("Hi there!");
        assert_eq!(assistant_msg.role, Role::Assistant);
    }

    #[test]
    fn test_vertex_message_params() {
        let params = VertexMessageParams::new(
            VertexModel::Claude35SonnetV2,
            1024,
            vec![MessageParam::user("Hello!")],
        )
        .with_system("You are helpful.")
        .with_temperature(0.7);

        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.temperature, Some(0.7));
        assert!(params.system.is_some());
    }

    #[test]
    fn test_tool_input_schema() {
        let schema = ToolInputSchema::new()
            .with_properties(serde_json::json!({
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            }))
            .with_required(vec!["location".to_string()]);

        assert_eq!(schema.schema_type, "object");
        assert!(schema.properties.is_some());
        assert_eq!(schema.required, Some(vec!["location".to_string()]));
    }

    #[test]
    fn test_message_serialization() {
        let msg = MessageParam::user("Hello!");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello!\""));
    }

    #[test]
    fn test_vertex_message_request_serialization() {
        let request = VertexMessageRequest {
            anthropic_version: VERTEX_ANTHROPIC_VERSION.to_string(),
            max_tokens: 1024,
            messages: vec![MessageParam::user("Hello!")],
            system: Some(SystemPrompt::Text("Be helpful.".to_string())),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("anthropic_version"));
        assert!(json.contains("vertex-2023-10-16"));
        assert!(json.contains("max_tokens"));
        assert!(!json.contains("model")); // Model should NOT be in body
    }

    #[test]
    fn test_stream_event_deserialization() {
        let json = r#"{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-v2@20241022","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#;

        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_123");
            }
            _ => panic!("Expected MessageStart event"),
        }
    }

    #[test]
    fn test_content_delta_deserialization() {
        let json = r#"{"type":"text_delta","text":"Hello"}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();
        match delta {
            ContentDelta::TextDelta { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected TextDelta"),
        }
    }

    #[test]
    fn test_vertex_message_text_extraction() {
        let message = VertexMessage {
            id: "msg_123".to_string(),
            message_type: "message".to_string(),
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Hello ".to_string(),
                },
                ContentBlock::Text {
                    text: "World!".to_string(),
                },
            ],
            model: "claude-3-5-sonnet-v2@20241022".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        assert_eq!(message.text(), "Hello World!");
    }

    #[test]
    fn test_system_prompt_from() {
        let prompt: SystemPrompt = "Be helpful".into();
        match prompt {
            SystemPrompt::Text(text) => assert_eq!(text, "Be helpful"),
            SystemPrompt::Blocks(_) => panic!("Expected text"),
        }
    }

    #[test]
    fn test_default_vertex_model() {
        let model = VertexModel::default();
        assert_eq!(model, VertexModel::Claude35SonnetV2);
    }
}
