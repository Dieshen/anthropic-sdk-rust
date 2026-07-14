//! Beta tool definitions for the Anthropic API.
//!
//! This module defines beta server-side tools including:
//! - Web Search (`web_search_20250305`)
//! - Computer Use tools (`bash_20250124`, `text_editor_20250124`, `computer_20250124`)
//! - Code Execution (`code_execution_20250522`)

use serde::{Deserialize, Serialize};

use crate::types::shared::CacheControl;

// =============================================================================
// Beta Tool Union
// =============================================================================

/// Union type for all beta tools.
///
/// This includes both client-defined tools and server-side tools like
/// web search and computer use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BetaToolUnion {
    /// Custom user-defined tool.
    Custom(BetaToolParam),
    /// Server-side beta tool.
    Server(BetaTool),
}

impl From<BetaToolParam> for BetaToolUnion {
    fn from(tool: BetaToolParam) -> Self {
        Self::Custom(tool)
    }
}

impl From<BetaTool> for BetaToolUnion {
    fn from(tool: BetaTool) -> Self {
        Self::Server(tool)
    }
}

/// Custom tool parameter definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaToolParam {
    /// Name of the tool.
    pub name: String,

    /// Description of what this tool does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for the tool's input parameters.
    pub input_schema: BetaToolInputSchema,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// JSON Schema for beta tool input parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetaToolInputSchema {
    /// Schema type (always "object").
    #[serde(rename = "type")]
    pub schema_type: String,

    /// Property definitions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,

    /// Required property names.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,

    /// Additional fields for JSON Schema extensions.
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for BetaToolInputSchema {
    fn default() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: None,
            required: Vec::new(),
            extra: std::collections::HashMap::new(),
        }
    }
}

// =============================================================================
// Server-Side Beta Tools
// =============================================================================

/// Server-side beta tool definitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BetaTool {
    /// Web search tool (2025-03-05 version).
    #[serde(rename = "web_search_20250305")]
    WebSearch(WebSearchTool20250305),

    /// Bash tool (2025-01-24 version) for computer use.
    #[serde(rename = "bash_20250124")]
    Bash(BashTool20250124),

    /// Text editor tool (2025-01-24 version) for computer use.
    #[serde(rename = "text_editor_20250124")]
    TextEditor(TextEditorTool20250124),

    /// Computer control tool (2025-01-24 version).
    #[serde(rename = "computer_20250124")]
    Computer(ComputerTool20250124),

    /// Code execution tool (2025-05-22 version).
    #[serde(rename = "code_execution_20250522")]
    CodeExecution(CodeExecutionTool),
}

// =============================================================================
// Web Search Tool
// =============================================================================

/// Web search tool configuration (2025-03-05 version).
///
/// Allows Claude to search the web for information during message generation.
///
/// # Example
///
/// ```rust
/// use anthropic::beta::WebSearchTool20250305;
///
/// let tool = WebSearchTool20250305::new()
///     .with_max_uses(5)
///     .with_allowed_domains(vec!["example.com".to_string()]);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchTool20250305 {
    /// Name of the tool (always "`web_search`").
    #[serde(default = "default_web_search_name")]
    pub name: String,

    /// Maximum number of search queries to perform.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<i64>,

    /// Allowed domains for search results.
    /// Cannot be used alongside `blocked_domains`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,

    /// Blocked domains for search results.
    /// Cannot be used alongside `allowed_domains`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,

    /// User location for search context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_web_search_name() -> String {
    "web_search".to_string()
}

impl WebSearchTool20250305 {
    /// Creates a new web search tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of search queries.
    #[must_use]
    pub const fn with_max_uses(mut self, max_uses: i64) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    /// Sets the allowed domains for search results.
    #[must_use]
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self.blocked_domains = None; // Clear blocked domains
        self
    }

    /// Sets the blocked domains for search results.
    #[must_use]
    pub fn with_blocked_domains(mut self, domains: Vec<String>) -> Self {
        self.blocked_domains = Some(domains);
        self.allowed_domains = None; // Clear allowed domains
        self
    }

    /// Sets the user location for search context.
    #[must_use]
    pub fn with_user_location(mut self, location: WebSearchUserLocation) -> Self {
        self.user_location = Some(location);
        self
    }

    /// Sets cache control for this tool.
    #[must_use]
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// User location for web search context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchUserLocation {
    /// Location type (always "approximate").
    #[serde(rename = "type", default = "default_approximate")]
    pub location_type: String,

    /// City name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,

    /// Region/state name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,

    /// Country code (ISO 3166-1 alpha-2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,

    /// Timezone (IANA format, e.g., "`America/New_York`").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

fn default_approximate() -> String {
    "approximate".to_string()
}

impl WebSearchUserLocation {
    /// Creates an approximate location with no details.
    #[must_use]
    pub fn approximate() -> Self {
        Self {
            location_type: "approximate".to_string(),
            city: None,
            region: None,
            country: None,
            timezone: None,
        }
    }

    /// Sets the city.
    #[must_use]
    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    /// Sets the region.
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the country code.
    #[must_use]
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Sets the timezone.
    #[must_use]
    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}

// =============================================================================
// Web Search Result Types
// =============================================================================

/// Web search tool result block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchToolResultBlock {
    /// Type of block (always "`web_search_tool_result`").
    #[serde(rename = "type")]
    pub block_type: String,

    /// The tool use ID this result corresponds to.
    pub tool_use_id: String,

    /// The search result content.
    pub content: WebSearchToolResultContent,
}

/// Content of a web search tool result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WebSearchToolResultContent {
    /// Successful search results.
    Results(Vec<WebSearchResultBlock>),
    /// Error during search.
    Error(WebSearchToolResultError),
}

/// Individual web search result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchResultBlock {
    /// Type of block (always "`web_search_result`").
    #[serde(rename = "type", default = "default_web_search_result")]
    pub block_type: String,

    /// URL of the search result.
    pub url: String,

    /// Title of the search result.
    pub title: String,

    /// Encrypted index for citations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_index: Option<String>,

    /// Page age information.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,

    /// Content blocks from the page.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<WebSearchContentBlock>,
}

fn default_web_search_result() -> String {
    "web_search_result".to_string()
}

/// Content block within a web search result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebSearchContentBlock {
    /// Text content from the page.
    Text {
        /// The text content.
        text: String,
    },
}

/// Error from web search tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchToolResultError {
    /// Type (always "`web_search_tool_result_error`").
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error code.
    pub error_code: WebSearchToolResultErrorCode,
}

/// Error codes for web search failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchToolResultErrorCode {
    /// Invalid tool input was provided.
    InvalidToolInput,
    /// Service is unavailable.
    Unavailable,
    /// Maximum uses exceeded.
    MaxUsesExceeded,
    /// Too many requests.
    TooManyRequests,
    /// Query is too long.
    QueryTooLong,
}

// =============================================================================
// Computer Use Tools
// =============================================================================

/// Bash tool for computer use (2025-01-24 version).
///
/// Allows Claude to execute bash commands in a controlled environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BashTool20250124 {
    /// Name of the tool (always "bash").
    #[serde(default = "default_bash_name")]
    pub name: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_bash_name() -> String {
    "bash".to_string()
}

impl Default for BashTool20250124 {
    fn default() -> Self {
        Self {
            name: default_bash_name(),
            cache_control: None,
        }
    }
}

impl BashTool20250124 {
    /// Creates a new bash tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets cache control for this tool.
    #[must_use]
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// Text editor tool for computer use (2025-01-24 version).
///
/// Allows Claude to view and edit text files using a `str_replace_editor`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextEditorTool20250124 {
    /// Name of the tool (always "`str_replace_editor`").
    #[serde(default = "default_text_editor_name")]
    pub name: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_text_editor_name() -> String {
    "str_replace_editor".to_string()
}

impl Default for TextEditorTool20250124 {
    fn default() -> Self {
        Self {
            name: default_text_editor_name(),
            cache_control: None,
        }
    }
}

impl TextEditorTool20250124 {
    /// Creates a new text editor tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets cache control for this tool.
    #[must_use]
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// Computer control tool for computer use (2025-01-24 version).
///
/// Allows Claude to control a computer's mouse and keyboard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputerTool20250124 {
    /// Name of the tool (always "computer").
    #[serde(default = "default_computer_name")]
    pub name: String,

    /// Display width in pixels.
    pub display_width_px: i64,

    /// Display height in pixels.
    pub display_height_px: i64,

    /// Display number for multi-monitor setups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_number: Option<i64>,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_computer_name() -> String {
    "computer".to_string()
}

impl ComputerTool20250124 {
    /// Creates a new computer control tool with the specified display dimensions.
    #[must_use]
    pub fn new(display_width_px: i64, display_height_px: i64) -> Self {
        Self {
            name: default_computer_name(),
            display_width_px,
            display_height_px,
            display_number: None,
            cache_control: None,
        }
    }

    /// Sets the display number for multi-monitor setups.
    #[must_use]
    pub const fn with_display_number(mut self, display_number: i64) -> Self {
        self.display_number = Some(display_number);
        self
    }

    /// Sets cache control for this tool.
    #[must_use]
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

// =============================================================================
// Code Execution Tool
// =============================================================================

/// Code execution tool (2025-05-22 version).
///
/// Allows Claude to execute code in a sandboxed environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeExecutionTool {
    /// Name of the tool (always "`code_execution`").
    #[serde(default = "default_code_execution_name")]
    pub name: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_code_execution_name() -> String {
    "code_execution".to_string()
}

impl Default for CodeExecutionTool {
    fn default() -> Self {
        Self {
            name: default_code_execution_name(),
            cache_control: None,
        }
    }
}

impl CodeExecutionTool {
    /// Creates a new code execution tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets cache control for this tool.
    #[must_use]
    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// Code execution result block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeExecutionResultBlock {
    /// Type (always "`code_execution_result`").
    #[serde(rename = "type")]
    pub block_type: String,

    /// Return code from execution.
    pub return_code: i64,

    /// Standard output.
    pub stdout: String,

    /// Standard error.
    pub stderr: String,

    /// Output files or content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<CodeExecutionOutputBlock>,
}

/// Output block from code execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeExecutionOutputBlock {
    /// Type (always "`code_execution_output`").
    #[serde(rename = "type")]
    pub block_type: String,

    /// File ID of the output.
    pub file_id: String,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_tool_serialize() {
        let tool = WebSearchTool20250305::new()
            .with_max_uses(5)
            .with_allowed_domains(vec!["example.com".to_string()]);

        let beta_tool = BetaTool::WebSearch(tool);
        let json = serde_json::to_string(&beta_tool).unwrap();

        assert!(json.contains("\"type\":\"web_search_20250305\""));
        assert!(json.contains("\"max_uses\":5"));
        assert!(json.contains("\"allowed_domains\":[\"example.com\"]"));
    }

    #[test]
    fn test_web_search_user_location() {
        let location = WebSearchUserLocation::approximate()
            .with_city("San Francisco")
            .with_country("US")
            .with_timezone("America/Los_Angeles");

        let json = serde_json::to_string(&location).unwrap();
        assert!(json.contains("\"type\":\"approximate\""));
        assert!(json.contains("\"city\":\"San Francisco\""));
        assert!(json.contains("\"country\":\"US\""));
    }

    #[test]
    fn test_bash_tool_serialize() {
        let tool = BashTool20250124::new();
        let beta_tool = BetaTool::Bash(tool);
        let json = serde_json::to_string(&beta_tool).unwrap();

        assert!(json.contains("\"type\":\"bash_20250124\""));
        assert!(json.contains("\"name\":\"bash\""));
    }

    #[test]
    fn test_text_editor_tool_serialize() {
        let tool = TextEditorTool20250124::new();
        let beta_tool = BetaTool::TextEditor(tool);
        let json = serde_json::to_string(&beta_tool).unwrap();

        assert!(json.contains("\"type\":\"text_editor_20250124\""));
        assert!(json.contains("\"name\":\"str_replace_editor\""));
    }

    #[test]
    fn test_computer_tool_serialize() {
        let tool = ComputerTool20250124::new(1920, 1080).with_display_number(1);
        let beta_tool = BetaTool::Computer(tool);
        let json = serde_json::to_string(&beta_tool).unwrap();

        assert!(json.contains("\"type\":\"computer_20250124\""));
        assert!(json.contains("\"display_width_px\":1920"));
        assert!(json.contains("\"display_height_px\":1080"));
        assert!(json.contains("\"display_number\":1"));
    }

    #[test]
    fn test_code_execution_tool_serialize() {
        let tool = CodeExecutionTool::new();
        let beta_tool = BetaTool::CodeExecution(tool);
        let json = serde_json::to_string(&beta_tool).unwrap();

        assert!(json.contains("\"type\":\"code_execution_20250522\""));
        assert!(json.contains("\"name\":\"code_execution\""));
    }

    #[test]
    fn test_web_search_error_codes() {
        let error = WebSearchToolResultError {
            error_type: "web_search_tool_result_error".to_string(),
            error_code: WebSearchToolResultErrorCode::MaxUsesExceeded,
        };
        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("\"error_code\":\"max_uses_exceeded\""));
    }

    #[test]
    fn test_beta_tool_union_from_server_tool() {
        let web_search = WebSearchTool20250305::new();
        let tool = BetaTool::WebSearch(web_search);
        let union: BetaToolUnion = tool.into();

        match union {
            BetaToolUnion::Server(_) => (),
            BetaToolUnion::Custom(_) => panic!("Expected Server variant"),
        }
    }

    #[test]
    fn test_web_search_result_block() {
        let result = WebSearchResultBlock {
            block_type: "web_search_result".to_string(),
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            encrypted_index: Some("enc123".to_string()),
            page_age: None,
            content: vec![WebSearchContentBlock::Text {
                text: "Sample content".to_string(),
            }],
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"url\":\"https://example.com\""));
        assert!(json.contains("\"title\":\"Example\""));
    }
}
