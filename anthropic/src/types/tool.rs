//! Tool definition types for the Anthropic API.
//!
//! This module defines types for tool definitions, tool choices,
//! and server-side tools like web search.

use serde::{Deserialize, Serialize};

use super::shared::CacheControl;

// =============================================================================
// Tool Definition (Request)
// =============================================================================

/// Tool definition for requests.
///
/// Defines a tool that the model can use during message generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolParam {
    /// Name of the tool.
    ///
    /// This is how the tool will be called by the model.
    pub name: String,

    /// Description of what this tool does.
    ///
    /// The more detailed the description, the better the model will use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for the tool's input parameters.
    pub input_schema: ToolInputSchema,

    /// Tool type (always "custom" for user-defined tools).
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<ToolType>,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl ToolParam {
    /// Creates a new tool definition.
    #[must_use]
    pub fn new(name: impl Into<String>, input_schema: ToolInputSchema) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
            tool_type: None,
            cache_control: None,
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets cache control.
    #[must_use]
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// Tool type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    /// Custom user-defined tool.
    Custom,
}

impl Default for ToolType {
    fn default() -> Self {
        Self::Custom
    }
}

// =============================================================================
// Tool Input Schema
// =============================================================================

/// JSON Schema for tool input parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInputSchema {
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

impl ToolInputSchema {
    /// Creates a new tool input schema.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: None,
            required: Vec::new(),
            extra: std::collections::HashMap::new(),
        }
    }

    /// Sets the properties.
    #[must_use]
    pub fn with_properties(mut self, properties: serde_json::Value) -> Self {
        self.properties = Some(properties);
        self
    }

    /// Sets the required fields.
    #[must_use]
    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = required;
        self
    }

    /// Creates a schema from a JSON value.
    pub fn from_json(value: serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value)
    }
}

impl Default for ToolInputSchema {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tool Choice
// =============================================================================

/// Tool choice configuration for requests.
///
/// Controls how the model selects which tools to use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model automatically decides whether to use tools.
    Auto {
        /// Disable parallel tool use (default: false).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model must use at least one tool.
    Any {
        /// Disable parallel tool use (default: false).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model must use the specified tool.
    Tool {
        /// Name of the tool to use.
        name: String,

        /// Disable parallel tool use (default: false).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model is not allowed to use tools.
    None,
}

impl ToolChoice {
    /// Creates an auto tool choice.
    #[must_use]
    pub fn auto() -> Self {
        Self::Auto {
            disable_parallel_tool_use: None,
        }
    }

    /// Creates an any tool choice.
    #[must_use]
    pub fn any() -> Self {
        Self::Any {
            disable_parallel_tool_use: None,
        }
    }

    /// Creates a specific tool choice.
    #[must_use]
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool {
            name: name.into(),
            disable_parallel_tool_use: None,
        }
    }

    /// Creates a none tool choice.
    #[must_use]
    pub fn none() -> Self {
        Self::None
    }
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::auto()
    }
}

// =============================================================================
// Server Tools
// =============================================================================

/// Server-side tool union for requests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerTool {
    /// Web search tool.
    #[serde(rename = "web_search_20250305")]
    WebSearch(WebSearchTool),

    /// Bash tool (computer use).
    #[serde(rename = "bash_20250124")]
    Bash(BashTool),

    /// Text editor tool (computer use).
    #[serde(rename = "text_editor_20250124")]
    TextEditor(TextEditorTool),

    /// Computer use tool.
    #[serde(rename = "computer_20250124")]
    Computer(ComputerTool),
}

/// Web search tool configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WebSearchTool {
    /// Name of the tool (always "web_search").
    #[serde(default = "default_web_search_name")]
    pub name: String,

    /// Maximum number of search queries to perform.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<i64>,

    /// Allowed domains for search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,

    /// Blocked domains for search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,

    /// User location for search context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_web_search_name() -> String {
    "web_search".to_string()
}

impl WebSearchTool {
    /// Creates a new web search tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of uses.
    #[must_use]
    pub fn with_max_uses(mut self, max_uses: i64) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    /// Sets the allowed domains.
    #[must_use]
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self
    }

    /// Sets the blocked domains.
    #[must_use]
    pub fn with_blocked_domains(mut self, domains: Vec<String>) -> Self {
        self.blocked_domains = Some(domains);
        self
    }
}

/// User location for web search context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserLocation {
    /// Location type.
    #[serde(rename = "type")]
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

    /// Timezone (IANA format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl UserLocation {
    /// Creates an approximate location.
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
}

/// Bash tool for computer use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BashTool {
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

/// Text editor tool for computer use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextEditorTool {
    /// Name of the tool (always "str_replace_editor").
    #[serde(default = "default_text_editor_name")]
    pub name: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_text_editor_name() -> String {
    "str_replace_editor".to_string()
}

/// Computer use tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputerTool {
    /// Name of the tool (always "computer").
    #[serde(default = "default_computer_name")]
    pub name: String,

    /// Display width in pixels.
    pub display_width_px: i64,

    /// Display height in pixels.
    pub display_height_px: i64,

    /// Display number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_number: Option<i64>,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

fn default_computer_name() -> String {
    "computer".to_string()
}

impl ComputerTool {
    /// Creates a new computer tool.
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
}

// =============================================================================
// Tool Union (combines custom and server tools)
// =============================================================================

/// Union type for all tools (custom and server).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Tool {
    /// Custom user-defined tool.
    Custom(ToolParam),
    /// Server-side tool.
    Server(ServerTool),
}

impl From<ToolParam> for Tool {
    fn from(tool: ToolParam) -> Self {
        Self::Custom(tool)
    }
}

impl From<ServerTool> for Tool {
    fn from(tool: ServerTool) -> Self {
        Self::Server(tool)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_param_serialize() {
        let tool = ToolParam::new(
            "get_weather",
            ToolInputSchema::new()
                .with_properties(serde_json::json!({
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                }))
                .with_required(vec!["location".to_string()]),
        )
        .with_description("Get the current weather");

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(json.contains("\"description\":\"Get the current weather\""));
        assert!(json.contains("\"type\":\"object\""));
    }

    #[test]
    fn test_tool_choice_auto() {
        let choice = ToolChoice::auto();
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"auto\""));
    }

    #[test]
    fn test_tool_choice_tool() {
        let choice = ToolChoice::tool("get_weather");
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"tool\""));
        assert!(json.contains("\"name\":\"get_weather\""));
    }

    #[test]
    fn test_tool_choice_none() {
        let choice = ToolChoice::none();
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"none\""));
    }

    #[test]
    fn test_web_search_tool() {
        let tool = WebSearchTool::new()
            .with_max_uses(5)
            .with_allowed_domains(vec!["example.com".to_string()]);

        let server_tool = ServerTool::WebSearch(tool);
        let json = serde_json::to_string(&server_tool).unwrap();
        assert!(json.contains("\"type\":\"web_search_20250305\""));
        assert!(json.contains("\"max_uses\":5"));
    }

    #[test]
    fn test_tool_input_schema() {
        let schema = ToolInputSchema::new()
            .with_properties(serde_json::json!({
                "query": {"type": "string"}
            }))
            .with_required(vec!["query".to_string()]);

        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains("\"type\":\"object\""));
        assert!(json.contains("\"required\":[\"query\"]"));
    }

    #[test]
    fn test_tool_deserialize() {
        let json = r#"{"name":"test","description":"A test tool","input_schema":{"type":"object","properties":{"x":{"type":"number"}}}}"#;
        let tool: ToolParam = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "test");
        assert_eq!(tool.description, Some("A test tool".to_string()));
    }
}
