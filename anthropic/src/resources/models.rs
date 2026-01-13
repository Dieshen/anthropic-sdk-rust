//! Models API resource.
//!
//! This module provides access to the Anthropic Models API, which allows you to
//! list available models and retrieve information about specific models.
//!
//! The Models API response can be used to determine which models are available
//! for use in the API, and to resolve model aliases to model IDs.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::Anthropic;
//! use anthropic::resources::models::{Models, ModelsListParams};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anthropic::Error> {
//!     let client = Anthropic::new()?;
//!
//!     // List all available models
//!     let response = client.models().list(ModelsListParams::default()).await?;
//!     for model in &response.data {
//!         println!("{}: {}", model.id, model.display_name);
//!     }
//!
//!     // Get a specific model by ID
//!     let model = client.models().get("claude-sonnet-4-5-latest", Default::default()).await?;
//!     println!("Model: {} created at {}", model.display_name, model.created_at);
//!
//!     // Paginate through models
//!     let params = ModelsListParams::new().with_limit(10);
//!     let first_page = client.models().list(params).await?;
//!
//!     if first_page.has_more {
//!         if let Some(last_id) = first_page.last_id {
//!             let next_params = ModelsListParams::new()
//!                 .with_limit(10)
//!                 .with_after_id(&last_id);
//!             let next_page = client.models().list(next_params).await?;
//!             // Process next page...
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::client::Anthropic;
use crate::error::Result;

// =============================================================================
// Constants
// =============================================================================

/// API endpoint for models.
const MODELS_PATH: &str = "v1/models";

// =============================================================================
// ModelInfo
// =============================================================================

/// Information about a model.
///
/// This struct contains metadata about an available model, including its
/// identifier, display name, creation time, and type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier.
    ///
    /// This is the ID you use when making API requests, e.g., "claude-sonnet-4-5-20250929".
    pub id: String,

    /// A human-readable name for the model.
    ///
    /// This is a user-friendly name like "Claude Sonnet 4.5".
    pub display_name: String,

    /// RFC 3339 datetime string representing when the model was released.
    ///
    /// May be set to an epoch value if the release date is unknown.
    pub created_at: DateTime<Utc>,

    /// Object type.
    ///
    /// For Models, this is always "model".
    #[serde(rename = "type")]
    pub model_type: ModelType,
}

impl ModelInfo {
    /// Returns true if this model ID contains the given substring.
    ///
    /// Useful for filtering models by family or version.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anthropic::resources::models::ModelInfo;
    /// use chrono::Utc;
    ///
    /// let model = ModelInfo {
    ///     id: "claude-sonnet-4-5-20250929".to_string(),
    ///     display_name: "Claude Sonnet 4.5".to_string(),
    ///     created_at: Utc::now(),
    ///     model_type: anthropic::resources::models::ModelType::Model,
    /// };
    ///
    /// assert!(model.id_contains("sonnet"));
    /// assert!(model.id_contains("4-5"));
    /// assert!(!model.id_contains("opus"));
    /// ```
    #[must_use]
    pub fn id_contains(&self, pattern: &str) -> bool {
        self.id.contains(pattern)
    }

    /// Returns true if this is a Sonnet model.
    #[must_use]
    pub fn is_sonnet(&self) -> bool {
        self.id_contains("sonnet")
    }

    /// Returns true if this is an Opus model.
    #[must_use]
    pub fn is_opus(&self) -> bool {
        self.id_contains("opus")
    }

    /// Returns true if this is a Haiku model.
    #[must_use]
    pub fn is_haiku(&self) -> bool {
        self.id_contains("haiku")
    }
}

// =============================================================================
// ModelType
// =============================================================================

/// The type of a model object.
///
/// Currently, all models have type "model".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    /// A model object.
    Model,
}

impl Default for ModelType {
    fn default() -> Self {
        Self::Model
    }
}

// =============================================================================
// ModelsGetParams
// =============================================================================

/// Parameters for getting a specific model.
///
/// Currently, the only parameter is an optional list of beta features to enable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelsGetParams {
    /// Optional beta features to enable.
    ///
    /// These are passed in the `anthropic-beta` header.
    #[serde(skip)]
    pub betas: Vec<String>,
}

impl ModelsGetParams {
    /// Creates new get parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a beta feature.
    #[must_use]
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    /// Adds multiple beta features.
    #[must_use]
    pub fn with_betas(mut self, betas: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.betas.extend(betas.into_iter().map(Into::into));
        self
    }
}

// =============================================================================
// ModelsListParams
// =============================================================================

/// Parameters for listing models.
///
/// Supports pagination with cursor-based navigation using `after_id` and `before_id`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelsListParams {
    /// ID of the object to use as a cursor for pagination.
    ///
    /// When provided, returns the page of results immediately after this object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_id: Option<String>,

    /// ID of the object to use as a cursor for pagination.
    ///
    /// When provided, returns the page of results immediately before this object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_id: Option<String>,

    /// Number of items to return per page.
    ///
    /// Defaults to 20. Ranges from 1 to 1000.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,

    /// Optional beta features to enable.
    ///
    /// These are passed in the `anthropic-beta` header.
    #[serde(skip)]
    pub betas: Vec<String>,
}

impl ModelsListParams {
    /// Creates new list parameters with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the after_id cursor for forward pagination.
    ///
    /// Returns results immediately after the specified ID.
    #[must_use]
    pub fn with_after_id(mut self, after_id: impl Into<String>) -> Self {
        self.after_id = Some(after_id.into());
        self
    }

    /// Sets the before_id cursor for backward pagination.
    ///
    /// Returns results immediately before the specified ID.
    #[must_use]
    pub fn with_before_id(mut self, before_id: impl Into<String>) -> Self {
        self.before_id = Some(before_id.into());
        self
    }

    /// Sets the maximum number of items to return per page.
    ///
    /// Valid range is 1 to 1000. Defaults to 20.
    #[must_use]
    pub fn with_limit(mut self, limit: i64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Adds a beta feature.
    #[must_use]
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    /// Adds multiple beta features.
    #[must_use]
    pub fn with_betas(mut self, betas: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.betas.extend(betas.into_iter().map(Into::into));
        self
    }
}

// =============================================================================
// ModelsListResponse
// =============================================================================

/// Response from listing models.
///
/// Contains the list of models and pagination information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelsListResponse {
    /// The list of models.
    pub data: Vec<ModelInfo>,

    /// Whether there are more results available.
    pub has_more: bool,

    /// ID of the first model in the current page.
    ///
    /// Can be used with `before_id` to navigate to the previous page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,

    /// ID of the last model in the current page.
    ///
    /// Can be used with `after_id` to navigate to the next page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

impl ModelsListResponse {
    /// Returns the number of models in this response.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if this response contains no models.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns an iterator over the models.
    pub fn iter(&self) -> impl Iterator<Item = &ModelInfo> {
        self.data.iter()
    }

    /// Returns parameters to fetch the next page, if there is one.
    ///
    /// Returns `None` if there are no more results (`has_more` is false)
    /// or if `last_id` is not available.
    #[must_use]
    pub fn next_page_params(&self) -> Option<ModelsListParams> {
        if self.has_more {
            self.last_id.as_ref().map(|last_id| {
                ModelsListParams::new().with_after_id(last_id)
            })
        } else {
            None
        }
    }

    /// Returns parameters to fetch the previous page.
    ///
    /// Returns `None` if `first_id` is not available.
    #[must_use]
    pub fn prev_page_params(&self) -> Option<ModelsListParams> {
        self.first_id.as_ref().map(|first_id| {
            ModelsListParams::new().with_before_id(first_id)
        })
    }

    /// Filters models by a predicate function.
    ///
    /// Returns a new vector containing only models that match the predicate.
    #[must_use]
    pub fn filter<F>(&self, predicate: F) -> Vec<&ModelInfo>
    where
        F: Fn(&ModelInfo) -> bool,
    {
        self.data.iter().filter(|m| predicate(m)).collect()
    }

    /// Finds a model by ID.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&ModelInfo> {
        self.data.iter().find(|m| m.id == id)
    }
}

impl IntoIterator for ModelsListResponse {
    type Item = ModelInfo;
    type IntoIter = std::vec::IntoIter<ModelInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> IntoIterator for &'a ModelsListResponse {
    type Item = &'a ModelInfo;
    type IntoIter = std::slice::Iter<'a, ModelInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// =============================================================================
// Models Resource
// =============================================================================

/// The Models API resource.
///
/// Provides methods to list and retrieve model information from the Anthropic API.
///
/// # Example
///
/// ```rust,ignore
/// use anthropic::Anthropic;
///
/// let client = Anthropic::new()?;
///
/// // List all models
/// let response = client.models().list(Default::default()).await?;
/// for model in response.data {
///     println!("{}: {}", model.id, model.display_name);
/// }
///
/// // Get a specific model
/// let model = client.models().get("claude-sonnet-4-5-latest", Default::default()).await?;
/// ```
#[derive(Debug, Clone)]
pub struct Models {
    client: Anthropic,
}

impl Models {
    /// Creates a new Models resource with the given client.
    #[must_use]
    pub fn new(client: Anthropic) -> Self {
        Self { client }
    }

    /// Get a specific model by ID.
    ///
    /// The Models API response can be used to determine information about a specific
    /// model or resolve a model alias to a model ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The unique identifier of the model to retrieve.
    /// * `params` - Optional parameters (currently only beta features).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model_id is empty
    /// - The model does not exist (404)
    /// - Authentication fails (401)
    /// - The request fails for any other reason
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = client.models().get("claude-sonnet-4-5-latest", Default::default()).await?;
    /// println!("Model: {} ({})", model.display_name, model.id);
    /// println!("Created: {}", model.created_at);
    /// ```
    pub async fn get(&self, model_id: &str, _params: ModelsGetParams) -> Result<ModelInfo> {
        if model_id.is_empty() {
            return Err(crate::error::Error::RequestBuild(
                "model_id cannot be empty".to_string(),
            ));
        }

        let path = format!("{}/{}", MODELS_PATH, model_id);
        self.client.get(&path).await
    }

    /// List available models.
    ///
    /// The Models API response can be used to determine which models are available
    /// for use in the API. More recently released models are listed first.
    ///
    /// # Arguments
    ///
    /// * `params` - Pagination and filtering parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Authentication fails (401)
    /// - The limit parameter is out of range (400)
    /// - The request fails for any other reason
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic::resources::models::ModelsListParams;
    ///
    /// // List with default parameters
    /// let response = client.models().list(Default::default()).await?;
    ///
    /// // List with pagination
    /// let params = ModelsListParams::new().with_limit(10);
    /// let response = client.models().list(params).await?;
    ///
    /// // Iterate through pages
    /// while let Some(next_params) = response.next_page_params() {
    ///     let next_page = client.models().list(next_params).await?;
    ///     // Process next_page...
    /// }
    /// ```
    pub async fn list(&self, params: ModelsListParams) -> Result<ModelsListResponse> {
        self.client.get_with_query(MODELS_PATH, &params).await
    }

    /// List all models, automatically handling pagination.
    ///
    /// This method fetches all available models by automatically requesting
    /// subsequent pages until all models have been retrieved.
    ///
    /// **Warning**: This may make multiple API requests and return a large
    /// number of models. Consider using `list()` with pagination parameters
    /// for better control over resource usage.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the API requests fail.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let all_models = client.models().list_all().await?;
    /// println!("Total models: {}", all_models.len());
    /// ```
    pub async fn list_all(&self) -> Result<Vec<ModelInfo>> {
        let mut all_models = Vec::new();
        let mut params = ModelsListParams::new();

        loop {
            let response = self.list(params).await?;
            all_models.extend(response.data);

            if !response.has_more {
                break;
            }

            // Get the next page using the last_id cursor
            match response.last_id {
                Some(last_id) => {
                    params = ModelsListParams::new().with_after_id(last_id);
                }
                None => break,
            }
        }

        Ok(all_models)
    }
}

// =============================================================================
// Client Extension
// =============================================================================

impl Anthropic {
    /// Returns the Models API resource.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = Anthropic::new()?;
    /// let models = client.models().list(Default::default()).await?;
    /// ```
    #[must_use]
    pub fn models(&self) -> Models {
        Models::new(self.clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, TimeZone};

    fn sample_model_info() -> ModelInfo {
        ModelInfo {
            id: "claude-sonnet-4-5-20250929".to_string(),
            display_name: "Claude Sonnet 4.5".to_string(),
            created_at: Utc.with_ymd_and_hms(2025, 9, 29, 0, 0, 0).unwrap(),
            model_type: ModelType::Model,
        }
    }

    #[test]
    fn test_model_info_serialize() {
        let model = sample_model_info();
        let json = serde_json::to_string(&model).unwrap();

        assert!(json.contains("\"id\":\"claude-sonnet-4-5-20250929\""));
        assert!(json.contains("\"display_name\":\"Claude Sonnet 4.5\""));
        assert!(json.contains("\"type\":\"model\""));
        assert!(json.contains("\"created_at\":"));
    }

    #[test]
    fn test_model_info_deserialize() {
        let json = r#"{
            "id": "claude-opus-4-5-20251101",
            "display_name": "Claude Opus 4.5",
            "created_at": "2025-11-01T00:00:00Z",
            "type": "model"
        }"#;

        let model: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(model.id, "claude-opus-4-5-20251101");
        assert_eq!(model.display_name, "Claude Opus 4.5");
        assert_eq!(model.model_type, ModelType::Model);
        assert_eq!(model.created_at.year(), 2025);
        assert_eq!(model.created_at.month(), 11);
    }

    #[test]
    fn test_model_info_helpers() {
        let sonnet = ModelInfo {
            id: "claude-sonnet-4-5-20250929".to_string(),
            display_name: "Claude Sonnet 4.5".to_string(),
            created_at: Utc::now(),
            model_type: ModelType::Model,
        };

        assert!(sonnet.is_sonnet());
        assert!(!sonnet.is_opus());
        assert!(!sonnet.is_haiku());
        assert!(sonnet.id_contains("4-5"));

        let opus = ModelInfo {
            id: "claude-opus-4-5-20251101".to_string(),
            display_name: "Claude Opus 4.5".to_string(),
            created_at: Utc::now(),
            model_type: ModelType::Model,
        };

        assert!(!opus.is_sonnet());
        assert!(opus.is_opus());
        assert!(!opus.is_haiku());

        let haiku = ModelInfo {
            id: "claude-haiku-4-5-20251001".to_string(),
            display_name: "Claude Haiku 4.5".to_string(),
            created_at: Utc::now(),
            model_type: ModelType::Model,
        };

        assert!(!haiku.is_sonnet());
        assert!(!haiku.is_opus());
        assert!(haiku.is_haiku());
    }

    #[test]
    fn test_model_type_serialize() {
        assert_eq!(
            serde_json::to_string(&ModelType::Model).unwrap(),
            "\"model\""
        );
    }

    #[test]
    fn test_model_type_deserialize() {
        let model_type: ModelType = serde_json::from_str("\"model\"").unwrap();
        assert_eq!(model_type, ModelType::Model);
    }

    #[test]
    fn test_models_list_params_default() {
        let params = ModelsListParams::default();
        assert!(params.after_id.is_none());
        assert!(params.before_id.is_none());
        assert!(params.limit.is_none());
        assert!(params.betas.is_empty());
    }

    #[test]
    fn test_models_list_params_builder() {
        let params = ModelsListParams::new()
            .with_after_id("model_123")
            .with_limit(50)
            .with_beta("some-beta-feature");

        assert_eq!(params.after_id, Some("model_123".to_string()));
        assert!(params.before_id.is_none());
        assert_eq!(params.limit, Some(50));
        assert_eq!(params.betas, vec!["some-beta-feature"]);
    }

    #[test]
    fn test_models_list_params_serialize() {
        let params = ModelsListParams::new()
            .with_after_id("model_123")
            .with_limit(10);

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"after_id\":\"model_123\""));
        assert!(json.contains("\"limit\":10"));
        // betas should not be serialized to JSON (skip attribute)
        assert!(!json.contains("betas"));
    }

    #[test]
    fn test_models_list_params_serialize_skips_none() {
        let params = ModelsListParams::new().with_limit(20);

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"limit\":20"));
        assert!(!json.contains("after_id"));
        assert!(!json.contains("before_id"));
    }

    #[test]
    fn test_models_get_params() {
        let params = ModelsGetParams::new()
            .with_beta("beta-1")
            .with_betas(vec!["beta-2", "beta-3"]);

        assert_eq!(params.betas, vec!["beta-1", "beta-2", "beta-3"]);
    }

    #[test]
    fn test_models_list_response_deserialize() {
        let json = r#"{
            "data": [
                {
                    "id": "claude-sonnet-4-5-20250929",
                    "display_name": "Claude Sonnet 4.5",
                    "created_at": "2025-09-29T00:00:00Z",
                    "type": "model"
                },
                {
                    "id": "claude-opus-4-5-20251101",
                    "display_name": "Claude Opus 4.5",
                    "created_at": "2025-11-01T00:00:00Z",
                    "type": "model"
                }
            ],
            "has_more": true,
            "first_id": "claude-sonnet-4-5-20250929",
            "last_id": "claude-opus-4-5-20251101"
        }"#;

        let response: ModelsListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.len(), 2);
        assert!(!response.is_empty());
        assert!(response.has_more);
        assert_eq!(response.first_id, Some("claude-sonnet-4-5-20250929".to_string()));
        assert_eq!(response.last_id, Some("claude-opus-4-5-20251101".to_string()));

        assert_eq!(response.data[0].id, "claude-sonnet-4-5-20250929");
        assert_eq!(response.data[1].id, "claude-opus-4-5-20251101");
    }

    #[test]
    fn test_models_list_response_next_page_params() {
        let response = ModelsListResponse {
            data: vec![sample_model_info()],
            has_more: true,
            first_id: Some("first".to_string()),
            last_id: Some("last".to_string()),
        };

        let next_params = response.next_page_params();
        assert!(next_params.is_some());
        let params = next_params.unwrap();
        assert_eq!(params.after_id, Some("last".to_string()));
    }

    #[test]
    fn test_models_list_response_no_more_pages() {
        let response = ModelsListResponse {
            data: vec![sample_model_info()],
            has_more: false,
            first_id: Some("first".to_string()),
            last_id: Some("last".to_string()),
        };

        assert!(response.next_page_params().is_none());
    }

    #[test]
    fn test_models_list_response_prev_page_params() {
        let response = ModelsListResponse {
            data: vec![sample_model_info()],
            has_more: false,
            first_id: Some("first".to_string()),
            last_id: Some("last".to_string()),
        };

        let prev_params = response.prev_page_params();
        assert!(prev_params.is_some());
        let params = prev_params.unwrap();
        assert_eq!(params.before_id, Some("first".to_string()));
    }

    #[test]
    fn test_models_list_response_filter() {
        let response = ModelsListResponse {
            data: vec![
                ModelInfo {
                    id: "claude-sonnet-4-5".to_string(),
                    display_name: "Sonnet".to_string(),
                    created_at: Utc::now(),
                    model_type: ModelType::Model,
                },
                ModelInfo {
                    id: "claude-opus-4-5".to_string(),
                    display_name: "Opus".to_string(),
                    created_at: Utc::now(),
                    model_type: ModelType::Model,
                },
            ],
            has_more: false,
            first_id: None,
            last_id: None,
        };

        let sonnets = response.filter(|m| m.is_sonnet());
        assert_eq!(sonnets.len(), 1);
        assert!(sonnets[0].is_sonnet());
    }

    #[test]
    fn test_models_list_response_find_by_id() {
        let response = ModelsListResponse {
            data: vec![
                ModelInfo {
                    id: "claude-sonnet-4-5".to_string(),
                    display_name: "Sonnet".to_string(),
                    created_at: Utc::now(),
                    model_type: ModelType::Model,
                },
                ModelInfo {
                    id: "claude-opus-4-5".to_string(),
                    display_name: "Opus".to_string(),
                    created_at: Utc::now(),
                    model_type: ModelType::Model,
                },
            ],
            has_more: false,
            first_id: None,
            last_id: None,
        };

        let found = response.find_by_id("claude-opus-4-5");
        assert!(found.is_some());
        assert_eq!(found.unwrap().display_name, "Opus");

        let not_found = response.find_by_id("claude-haiku-4-5");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_models_list_response_into_iter() {
        let response = ModelsListResponse {
            data: vec![sample_model_info(), sample_model_info()],
            has_more: false,
            first_id: None,
            last_id: None,
        };

        let models: Vec<ModelInfo> = response.into_iter().collect();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_models_list_response_iter_ref() {
        let response = ModelsListResponse {
            data: vec![sample_model_info(), sample_model_info()],
            has_more: false,
            first_id: None,
            last_id: None,
        };

        let count = response.iter().count();
        assert_eq!(count, 2);

        // Can still use response after iteration
        assert_eq!(response.len(), 2);
    }

    #[test]
    fn test_empty_list_response() {
        let response = ModelsListResponse {
            data: vec![],
            has_more: false,
            first_id: None,
            last_id: None,
        };

        assert!(response.is_empty());
        assert_eq!(response.len(), 0);
        assert!(response.next_page_params().is_none());
        assert!(response.prev_page_params().is_none());
    }
}
