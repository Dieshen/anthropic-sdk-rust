//! Content block types for messages.
//!
//! This module defines the various content block types that can appear in
//! messages, both in requests (params) and responses.
//!
//! # Content Block Types
//!
//! Response content blocks:
//! - [`TextBlock`] - Text content with optional citations
//! - [`ThinkingBlock`] - Extended thinking output
//! - [`RedactedThinkingBlock`] - Redacted thinking content
//! - [`ToolUseBlock`] - Tool invocation
//! - [`ServerToolUseBlock`] - Server-side tool use (web search)
//! - [`WebSearchToolResultBlock`] - Web search results
//!
//! Request content blocks:
//! - [`TextBlockParam`] - Text content for requests
//! - [`ImageBlockParam`] - Image content for requests
//! - [`DocumentBlockParam`] - Document content for requests
//! - [`ToolResultBlockParam`] - Tool result for requests
//! - [`ToolUseBlockParam`] - Tool use for requests (in assistant messages)

use serde::{Deserialize, Serialize};

use super::shared::CacheControl;

// =============================================================================
// Response Content Blocks (Union)
// =============================================================================

/// Content block in an assistant response.
///
/// This is a discriminated union based on the `type` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content.
    Text(TextBlock),

    /// Extended thinking content.
    Thinking(ThinkingBlock),

    /// Redacted thinking content.
    RedactedThinking(RedactedThinkingBlock),

    /// Tool use request.
    ToolUse(ToolUseBlock),

    /// Server-side tool use (e.g., web search).
    ServerToolUse(ServerToolUseBlock),

    /// Web search tool result.
    WebSearchToolResult(WebSearchToolResultBlock),
}

impl ContentBlock {
    /// Returns the text content if this is a text block.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(block) => Some(&block.text),
            _ => None,
        }
    }

    /// Returns true if this is a text block.
    #[must_use]
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Returns true if this is a tool use block.
    #[must_use]
    pub fn is_tool_use(&self) -> bool {
        matches!(self, Self::ToolUse(_))
    }

    /// Returns the tool use block if this is one.
    #[must_use]
    pub fn as_tool_use(&self) -> Option<&ToolUseBlock> {
        match self {
            Self::ToolUse(block) => Some(block),
            _ => None,
        }
    }

    /// Returns true if this is a thinking block.
    #[must_use]
    pub fn is_thinking(&self) -> bool {
        matches!(self, Self::Thinking(_))
    }
}

// =============================================================================
// Text Block
// =============================================================================

/// Text content block in a response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBlock {
    /// The text content.
    pub text: String,

    /// Citations for the text content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<Citation>,
}

impl TextBlock {
    /// Creates a new text block.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            citations: Vec::new(),
        }
    }
}

// =============================================================================
// Thinking Blocks
// =============================================================================

/// Extended thinking content block.
///
/// Contains the model's reasoning process when extended thinking is enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingBlock {
    /// The thinking content.
    pub thinking: String,

    /// Cryptographic signature for verification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Redacted thinking content block.
///
/// Contains thinking content that has been redacted for policy reasons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RedactedThinkingBlock {
    /// The redacted data.
    pub data: String,
}

// =============================================================================
// Tool Use Block
// =============================================================================

/// Tool use request block.
///
/// Indicates that the model wants to invoke a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUseBlock {
    /// Unique identifier for this tool use.
    pub id: String,

    /// Name of the tool to invoke.
    pub name: String,

    /// Input parameters for the tool (JSON object).
    pub input: serde_json::Value,
}

impl ToolUseBlock {
    /// Creates a new tool use block.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Attempts to deserialize the input as a specific type.
    pub fn parse_input<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(self.input.clone())
    }
}

// =============================================================================
// Server Tool Use Block
// =============================================================================

/// Server-side tool use block.
///
/// Used for built-in server tools like web search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolUseBlock {
    /// Unique identifier for this tool use.
    pub id: String,

    /// Name of the server tool (e.g., "web_search").
    pub name: String,

    /// Input parameters for the tool.
    pub input: serde_json::Value,
}

// =============================================================================
// Web Search Result Block
// =============================================================================

/// Web search tool result block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchToolResultBlock {
    /// The tool use ID this result corresponds to.
    pub tool_use_id: String,

    /// The search result content.
    pub content: WebSearchResultContent,
}

/// Content of a web search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebSearchResultContent {
    /// Search results.
    SearchResults {
        /// List of search results.
        results: Vec<WebSearchResult>,
    },
    /// Error during search.
    Error {
        /// Error message.
        error: String,
    },
}

/// Individual web search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchResult {
    /// Title of the search result.
    pub title: String,
    /// URL of the search result.
    pub url: String,
    /// Snippet from the search result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

// =============================================================================
// Citations
// =============================================================================

/// Citation for text content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Citation {
    /// Character position citation.
    CharLocation(CharLocationCitation),
    /// Page number citation.
    PageLocation(PageLocationCitation),
    /// Content block index citation.
    ContentBlockLocation(ContentBlockLocationCitation),
    /// Web search result citation.
    WebSearchResultLocation(WebSearchResultLocationCitation),
}

/// Citation with character position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharLocationCitation {
    /// The cited text.
    pub cited_text: String,
    /// Index of the document.
    pub document_index: i64,
    /// Document title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,
    /// Start character position.
    pub start_char_index: i64,
    /// End character position.
    pub end_char_index: i64,
}

/// Citation with page number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageLocationCitation {
    /// The cited text.
    pub cited_text: String,
    /// Index of the document.
    pub document_index: i64,
    /// Document title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,
    /// Page number (1-indexed).
    pub page_number: i64,
}

/// Citation with content block index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockLocationCitation {
    /// The cited text.
    pub cited_text: String,
    /// Index of the document.
    pub document_index: i64,
    /// Document title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_title: Option<String>,
    /// Content block index.
    pub content_block_index: i64,
}

/// Citation from web search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchResultLocationCitation {
    /// The cited text.
    pub cited_text: String,
    /// URL of the source.
    pub url: String,
    /// Title of the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

// =============================================================================
// Request Content Blocks (Params)
// =============================================================================

/// Content block in a request message.
///
/// This is a discriminated union based on the `type` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockParam {
    /// Text content.
    Text(TextBlockParam),

    /// Image content.
    Image(ImageBlockParam),

    /// Document content.
    Document(DocumentBlockParam),

    /// Tool result (response to tool use).
    ToolResult(ToolResultBlockParam),

    /// Tool use (in assistant messages for context).
    ToolUse(ToolUseBlockParam),

    /// Thinking content (in assistant messages for context).
    Thinking(ThinkingBlockParam),

    /// Redacted thinking (in assistant messages for context).
    RedactedThinking(RedactedThinkingBlockParam),
}

impl ContentBlockParam {
    /// Creates a text content block.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextBlockParam::new(text))
    }

    /// Creates a tool result content block.
    #[must_use]
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult(ToolResultBlockParam::new(tool_use_id, content))
    }
}

// =============================================================================
// Text Block Param
// =============================================================================

/// Text content block for requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBlockParam {
    /// The text content.
    pub text: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl TextBlockParam {
    /// Creates a new text block param.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Adds cache control to this block.
    #[must_use]
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

// =============================================================================
// Image Block Param
// =============================================================================

/// Image content block for requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageBlockParam {
    /// The image source.
    pub source: ImageSource,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Source for image content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 {
        /// Media type (e.g., "image/png").
        media_type: String,
        /// Base64-encoded data.
        data: String,
    },
    /// URL reference.
    Url {
        /// Image URL.
        url: String,
    },
    /// File reference.
    File {
        /// File ID.
        file_id: String,
    },
}

impl ImageBlockParam {
    /// Creates an image block from base64 data.
    #[must_use]
    pub fn from_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
            cache_control: None,
        }
    }

    /// Creates an image block from a URL.
    #[must_use]
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: ImageSource::Url { url: url.into() },
            cache_control: None,
        }
    }

    /// Creates an image block from a file ID.
    #[must_use]
    pub fn from_file(file_id: impl Into<String>) -> Self {
        Self {
            source: ImageSource::File {
                file_id: file_id.into(),
            },
            cache_control: None,
        }
    }
}

// =============================================================================
// Document Block Param
// =============================================================================

/// Document content block for requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentBlockParam {
    /// The document source.
    pub source: DocumentSource,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    /// Optional title for the document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Context about the document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,

    /// Citations configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
}

/// Source for document content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base64-encoded document data.
    Base64 {
        /// Media type (e.g., "application/pdf").
        media_type: String,
        /// Base64-encoded data.
        data: String,
    },
    /// URL reference.
    Url {
        /// Document URL.
        url: String,
    },
    /// File reference.
    File {
        /// File ID.
        file_id: String,
    },
    /// Plain text content.
    Text {
        /// Text content.
        text: String,
    },
}

/// Citations configuration for documents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CitationsConfig {
    /// Whether citations are enabled.
    pub enabled: bool,
}

impl DocumentBlockParam {
    /// Creates a document block from base64 data.
    #[must_use]
    pub fn from_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            source: DocumentSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
            cache_control: None,
            title: None,
            context: None,
            citations: None,
        }
    }

    /// Creates a document block from a URL.
    #[must_use]
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: DocumentSource::Url { url: url.into() },
            cache_control: None,
            title: None,
            context: None,
            citations: None,
        }
    }

    /// Creates a document block from a file ID.
    #[must_use]
    pub fn from_file(file_id: impl Into<String>) -> Self {
        Self {
            source: DocumentSource::File {
                file_id: file_id.into(),
            },
            cache_control: None,
            title: None,
            context: None,
            citations: None,
        }
    }

    /// Creates a document block from plain text.
    #[must_use]
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            source: DocumentSource::Text { text: text.into() },
            cache_control: None,
            title: None,
            context: None,
            citations: None,
        }
    }

    /// Sets the title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

// =============================================================================
// Tool Result Block Param
// =============================================================================

/// Tool result content block for requests.
///
/// Used to provide the result of a tool invocation back to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultBlockParam {
    /// The ID of the tool use this is a result for.
    pub tool_use_id: String,

    /// The result content.
    pub content: ToolResultContent,

    /// Whether this result indicates an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Content for a tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple text result.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<ToolResultContentBlock>),
}

/// Content block in a tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    /// Text content.
    Text {
        /// The text content.
        text: String,
    },
    /// Image content.
    Image {
        /// The image source.
        source: ImageSource,
    },
}

impl ToolResultBlockParam {
    /// Creates a new tool result with text content.
    #[must_use]
    pub fn new(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: ToolResultContent::Text(content.into()),
            is_error: None,
            cache_control: None,
        }
    }

    /// Creates a new tool result with structured content blocks.
    #[must_use]
    pub fn with_blocks(
        tool_use_id: impl Into<String>,
        blocks: Vec<ToolResultContentBlock>,
    ) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: ToolResultContent::Blocks(blocks),
            is_error: None,
            cache_control: None,
        }
    }

    /// Creates an error tool result.
    #[must_use]
    pub fn error(tool_use_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: ToolResultContent::Text(error.into()),
            is_error: Some(true),
            cache_control: None,
        }
    }
}

// =============================================================================
// Tool Use Block Param (for assistant messages in context)
// =============================================================================

/// Tool use block param for including in assistant messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUseBlockParam {
    /// Unique identifier for this tool use.
    pub id: String,

    /// Name of the tool.
    pub name: String,

    /// Input parameters for the tool.
    pub input: serde_json::Value,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

// =============================================================================
// Thinking Block Params (for assistant messages in context)
// =============================================================================

/// Thinking block param for including in assistant messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingBlockParam {
    /// The thinking content.
    pub thinking: String,

    /// Signature for verification.
    pub signature: String,
}

/// Redacted thinking block param for including in assistant messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RedactedThinkingBlockParam {
    /// The redacted data.
    pub data: String,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_block_text() {
        let block = ContentBlock::Text(TextBlock::new("Hello, world!"));
        assert!(block.is_text());
        assert_eq!(block.as_text(), Some("Hello, world!"));
    }

    #[test]
    fn test_content_block_serialize_text() {
        let block = ContentBlock::Text(TextBlock::new("Hello"));
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_content_block_deserialize_text() {
        let json = r#"{"type":"text","text":"Hello, world!"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        assert!(block.is_text());
        assert_eq!(block.as_text(), Some("Hello, world!"));
    }

    #[test]
    fn test_content_block_deserialize_tool_use() {
        let json = r#"{"type":"tool_use","id":"tool_123","name":"get_weather","input":{"location":"Tokyo"}}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        assert!(block.is_tool_use());
        let tool_use = block.as_tool_use().unwrap();
        assert_eq!(tool_use.id, "tool_123");
        assert_eq!(tool_use.name, "get_weather");
    }

    #[test]
    fn test_content_block_param_text() {
        let block = ContentBlockParam::text("Hello");
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
    }

    #[test]
    fn test_tool_result_block_param() {
        let result = ToolResultBlockParam::new("tool_123", "Success!");
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"tool_use_id\":\"tool_123\""));
        assert!(json.contains("\"content\":\"Success!\""));
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResultBlockParam::error("tool_123", "Something went wrong");
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn test_image_block_param_base64() {
        let block = ImageBlockParam::from_base64("image/png", "iVBORw0KGgo...");
        let content_block = ContentBlockParam::Image(block);
        let json = serde_json::to_string(&content_block).unwrap();
        assert!(json.contains("\"type\":\"image\""));
        assert!(json.contains("\"media_type\":\"image/png\""));
    }

    #[test]
    fn test_document_block_param() {
        let block = DocumentBlockParam::from_url("https://example.com/doc.pdf")
            .with_title("Example Document");
        let content_block = ContentBlockParam::Document(block);
        let json = serde_json::to_string(&content_block).unwrap();
        assert!(json.contains("\"type\":\"document\""));
        assert!(json.contains("\"title\":\"Example Document\""));
    }
}
