//! Beta Skills API for the Anthropic SDK.
//!
//! This module provides access to the Skills API, which allows you to create,
//! manage, and use reusable packages of functionality with Claude. Skills are
//! versioned bundles that can be loaded into Claude's execution container.
//!
//! # Beta Status
//!
//! The Skills API is currently in beta and requires the `skills-2025-10-02`
//! beta header. This is handled automatically when using the Skills resource.
//!
//! # Concepts
//!
//! - **Skill**: A reusable package of functionality (instructions, tools, files)
//! - **Version**: An immutable snapshot of a skill, identified by Unix timestamp
//! - **SKILL.md**: Manifest file describing the skill's configuration
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::Anthropic;
//! use anthropic::beta::skills::{Skills, SkillCreateParams};
//!
//! let client = Anthropic::new()?;
//! let skills = client.beta_skills();
//!
//! // Create a skill
//! let skill = skills.create(SkillCreateParams::new(
//!     "my-skill",
//!     "A helpful assistant skill",
//!     skill_bundle_data,
//! )).await?;
//!
//! // List skills
//! let list = skills.list(Default::default()).await?;
//!
//! // Delete a skill
//! skills.delete(&skill.id).await?;
//! ```

use std::sync::Arc;

use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// =============================================================================
// Beta Header
// =============================================================================

/// Beta header value for the Skills API.
pub const SKILLS_API_BETA_HEADER: &str = "skills-2025-10-02";

// =============================================================================
// Skill Types
// =============================================================================

/// Type of skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillType {
    /// User-created custom skill.
    Custom,
    /// Anthropic-provided built-in skill.
    Anthropic,
}

/// Metadata for a skill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier for the skill.
    pub id: String,

    /// Object type (always "skill").
    #[serde(rename = "type")]
    pub skill_type_marker: String,

    /// Human-readable name of the skill.
    pub name: String,

    /// Description of what the skill does.
    pub description: String,

    /// Type of skill (custom or anthropic).
    #[serde(rename = "skill_type")]
    pub skill_kind: SkillType,

    /// ISO 8601 timestamp of when the skill was created.
    pub created_at: String,

    /// ISO 8601 timestamp of when the skill was last updated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,

    /// The latest version number (Unix timestamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_version: Option<u64>,
}

/// Response when deleting a skill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletedSkill {
    /// ID of the deleted skill.
    pub id: String,

    /// Object type (always "skill_deleted").
    #[serde(rename = "type")]
    pub deleted_type: String,
}

// =============================================================================
// Skill Version Types
// =============================================================================

/// Metadata for a skill version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillVersion {
    /// Version number (Unix epoch timestamp).
    pub version: u64,

    /// Object type (always "skill_version").
    #[serde(rename = "type")]
    pub version_type: String,

    /// ID of the parent skill.
    pub skill_id: String,

    /// ISO 8601 timestamp of when this version was created.
    pub created_at: String,

    /// SHA-256 hash of the version contents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,

    /// Size of the version bundle in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

/// Response when deleting a skill version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletedSkillVersion {
    /// ID of the parent skill.
    pub skill_id: String,

    /// Version number that was deleted.
    pub version: u64,

    /// Object type (always "skill_version_deleted").
    #[serde(rename = "type")]
    pub deleted_type: String,
}

// =============================================================================
// List Types
// =============================================================================

/// Parameters for listing skills.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillsListParams {
    /// Maximum number of skills to return (1-100).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,

    /// Cursor for pagination.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_id: Option<String>,

    /// Filter by skill type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill_type: Option<SkillType>,
}

impl SkillsListParams {
    /// Creates new list parameters with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of skills to return.
    #[must_use]
    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sets the pagination cursor.
    #[must_use]
    pub fn with_after_id(mut self, after_id: impl Into<String>) -> Self {
        self.after_id = Some(after_id.into());
        self
    }

    /// Filters to only custom skills.
    #[must_use]
    pub fn custom_only(mut self) -> Self {
        self.skill_type = Some(SkillType::Custom);
        self
    }

    /// Filters to only Anthropic-provided skills.
    #[must_use]
    pub fn anthropic_only(mut self) -> Self {
        self.skill_type = Some(SkillType::Anthropic);
        self
    }
}

/// Response when listing skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsListResponse {
    /// List of skills.
    pub data: Vec<Skill>,

    /// Whether there are more skills to fetch.
    pub has_more: bool,

    /// ID of the first skill in the list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,

    /// ID of the last skill in the list (use for pagination).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

impl SkillsListResponse {
    /// Returns pagination parameters for fetching the next page.
    #[must_use]
    pub fn next_page_params(&self) -> Option<SkillsListParams> {
        if self.has_more {
            self.last_id.as_ref().map(|id| {
                SkillsListParams::new().with_after_id(id.clone())
            })
        } else {
            None
        }
    }
}

/// Parameters for listing skill versions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionsListParams {
    /// Maximum number of versions to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,

    /// Return versions after this timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_version: Option<u64>,
}

impl VersionsListParams {
    /// Creates new list parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of versions to return.
    #[must_use]
    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sets the pagination cursor.
    #[must_use]
    pub fn with_after_version(mut self, version: u64) -> Self {
        self.after_version = Some(version);
        self
    }
}

/// Response when listing skill versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionsListResponse {
    /// List of versions.
    pub data: Vec<SkillVersion>,

    /// Whether there are more versions to fetch.
    pub has_more: bool,

    /// First version in the list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_version: Option<u64>,

    /// Last version in the list (use for pagination).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_version: Option<u64>,
}

// =============================================================================
// Create/Update Types
// =============================================================================

/// Parameters for creating a skill.
#[derive(Debug, Clone)]
pub struct SkillCreateParams {
    /// Human-readable name for the skill.
    pub name: String,

    /// Description of the skill.
    pub description: String,

    /// The skill bundle data (zip or tar.gz containing SKILL.md).
    pub bundle: Vec<u8>,

    /// MIME type of the bundle.
    pub bundle_mime_type: String,
}

impl SkillCreateParams {
    /// Creates new skill creation parameters.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable name
    /// * `description` - What the skill does
    /// * `bundle` - Skill bundle data (zip or tar.gz)
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        bundle: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            bundle: bundle.into(),
            bundle_mime_type: "application/zip".to_string(),
        }
    }

    /// Creates parameters with a tar.gz bundle.
    #[must_use]
    pub fn with_tar_gz(
        name: impl Into<String>,
        description: impl Into<String>,
        bundle: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            bundle: bundle.into(),
            bundle_mime_type: "application/gzip".to_string(),
        }
    }
}

/// Parameters for creating a new skill version.
#[derive(Debug, Clone)]
pub struct VersionCreateParams {
    /// The skill bundle data for this version.
    pub bundle: Vec<u8>,

    /// MIME type of the bundle.
    pub bundle_mime_type: String,
}

impl VersionCreateParams {
    /// Creates new version parameters with a zip bundle.
    #[must_use]
    pub fn zip(bundle: impl Into<Vec<u8>>) -> Self {
        Self {
            bundle: bundle.into(),
            bundle_mime_type: "application/zip".to_string(),
        }
    }

    /// Creates new version parameters with a tar.gz bundle.
    #[must_use]
    pub fn tar_gz(bundle: impl Into<Vec<u8>>) -> Self {
        Self {
            bundle: bundle.into(),
            bundle_mime_type: "application/gzip".to_string(),
        }
    }
}

// =============================================================================
// Skills Client
// =============================================================================

/// Internal client for skills operations.
#[derive(Debug)]
pub(crate) struct SkillsClient {
    http_client: reqwest::Client,
    base_url: url::Url,
}

impl SkillsClient {
    /// Creates a new skills client.
    pub(crate) fn new(http_client: reqwest::Client, base_url: url::Url) -> Self {
        Self {
            http_client,
            base_url,
        }
    }

    /// Builds a full URL from a path.
    fn build_url(&self, path: &str) -> Result<url::Url> {
        let path = path.trim_start_matches('/');
        self.base_url.join(path).map_err(Error::Url)
    }

    /// Makes a GET request with the beta header.
    async fn get<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = self.build_url(path)?;
        let response = self
            .http_client
            .get(url)
            .header("anthropic-beta", SKILLS_API_BETA_HEADER)
            .send()
            .await?;
        self.handle_response(response).await
    }

    /// Makes a GET request with query parameters.
    async fn get_with_query<T, Q>(&self, path: &str, query: &Q) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        Q: serde::Serialize + ?Sized,
    {
        let url = self.build_url(path)?;
        let response = self
            .http_client
            .get(url)
            .query(query)
            .header("anthropic-beta", SKILLS_API_BETA_HEADER)
            .send()
            .await?;
        self.handle_response(response).await
    }

    /// Makes a DELETE request.
    async fn delete<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = self.build_url(path)?;
        let response = self
            .http_client
            .delete(url)
            .header("anthropic-beta", SKILLS_API_BETA_HEADER)
            .send()
            .await?;
        self.handle_response(response).await
    }

    /// Makes a multipart upload request.
    async fn upload_multipart<T>(&self, path: &str, form: Form) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = self.build_url(path)?;
        let response = self
            .http_client
            .post(url)
            .multipart(form)
            .header("anthropic-beta", SKILLS_API_BETA_HEADER)
            .send()
            .await?;
        self.handle_response(response).await
    }

    /// Handles an HTTP response.
    async fn handle_response<T>(&self, response: reqwest::Response) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let status = response.status();
        if status.is_success() {
            let data = response.json::<T>().await?;
            Ok(data)
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(Error::Streaming(format!("HTTP {}: {}", status, body)))
        }
    }
}

// =============================================================================
// Skills Resource
// =============================================================================

/// Resource for interacting with the Skills API.
///
/// Skills are reusable, versioned packages of functionality that can be
/// loaded into Claude's execution container. They allow you to bundle
/// instructions, tools, and files into shareable units.
///
/// # Example
///
/// ```rust,ignore
/// let client = Anthropic::new()?;
/// let skills = client.beta_skills();
///
/// // Create a skill
/// let skill = skills.create(SkillCreateParams::new(
///     "code-reviewer",
///     "A skill for reviewing code",
///     bundle_data,
/// )).await?;
///
/// // Create a new version
/// let version = skills.create_version(&skill.id, VersionCreateParams::zip(
///     updated_bundle_data,
/// )).await?;
///
/// // Use the skill in a message request (via skill loader)
/// ```
#[derive(Debug, Clone)]
pub struct Skills {
    client: Arc<SkillsClient>,
}

impl Skills {
    /// Creates a new Skills resource.
    pub(crate) fn new(http_client: reqwest::Client, base_url: url::Url) -> Self {
        Self {
            client: Arc::new(SkillsClient::new(http_client, base_url)),
        }
    }

    /// Creates a new skill.
    ///
    /// # Arguments
    ///
    /// * `params` - Skill creation parameters
    ///
    /// # Returns
    ///
    /// Returns the created skill metadata.
    pub async fn create(&self, params: SkillCreateParams) -> Result<Skill> {
        let bundle_part = Part::bytes(params.bundle)
            .file_name("skill.zip")
            .mime_str(&params.bundle_mime_type)
            .map_err(|e| Error::config(format!("Invalid MIME type: {}", e)))?;

        let form = Form::new()
            .text("name", params.name)
            .text("description", params.description)
            .part("bundle", bundle_part);

        self.client.upload_multipart("v1/skills", form).await
    }

    /// Retrieves a skill by ID.
    ///
    /// # Arguments
    ///
    /// * `skill_id` - The unique identifier of the skill
    pub async fn get(&self, skill_id: &str) -> Result<Skill> {
        if skill_id.is_empty() {
            return Err(Error::config("skill_id cannot be empty"));
        }
        self.client.get(&format!("v1/skills/{}", skill_id)).await
    }

    /// Lists skills with optional filtering.
    ///
    /// # Arguments
    ///
    /// * `params` - Pagination and filtering parameters
    pub async fn list(&self, params: SkillsListParams) -> Result<SkillsListResponse> {
        self.client.get_with_query("v1/skills", &params).await
    }

    /// Deletes a skill and all its versions.
    ///
    /// # Arguments
    ///
    /// * `skill_id` - The unique identifier of the skill to delete
    pub async fn delete(&self, skill_id: &str) -> Result<DeletedSkill> {
        if skill_id.is_empty() {
            return Err(Error::config("skill_id cannot be empty"));
        }
        self.client.delete(&format!("v1/skills/{}", skill_id)).await
    }

    // =========================================================================
    // Versions
    // =========================================================================

    /// Creates a new version of a skill.
    ///
    /// # Arguments
    ///
    /// * `skill_id` - The skill to create a version for
    /// * `params` - Version creation parameters
    pub async fn create_version(
        &self,
        skill_id: &str,
        params: VersionCreateParams,
    ) -> Result<SkillVersion> {
        if skill_id.is_empty() {
            return Err(Error::config("skill_id cannot be empty"));
        }

        let bundle_part = Part::bytes(params.bundle)
            .file_name("version.zip")
            .mime_str(&params.bundle_mime_type)
            .map_err(|e| Error::config(format!("Invalid MIME type: {}", e)))?;

        let form = Form::new().part("bundle", bundle_part);

        self.client
            .upload_multipart(&format!("v1/skills/{}/versions", skill_id), form)
            .await
    }

    /// Retrieves a specific version of a skill.
    ///
    /// # Arguments
    ///
    /// * `skill_id` - The skill ID
    /// * `version` - The version number (Unix timestamp)
    pub async fn get_version(&self, skill_id: &str, version: u64) -> Result<SkillVersion> {
        if skill_id.is_empty() {
            return Err(Error::config("skill_id cannot be empty"));
        }
        self.client
            .get(&format!("v1/skills/{}/versions/{}", skill_id, version))
            .await
    }

    /// Lists versions of a skill.
    ///
    /// # Arguments
    ///
    /// * `skill_id` - The skill ID
    /// * `params` - Pagination parameters
    pub async fn list_versions(
        &self,
        skill_id: &str,
        params: VersionsListParams,
    ) -> Result<VersionsListResponse> {
        if skill_id.is_empty() {
            return Err(Error::config("skill_id cannot be empty"));
        }
        self.client
            .get_with_query(&format!("v1/skills/{}/versions", skill_id), &params)
            .await
    }

    /// Deletes a specific version of a skill.
    ///
    /// # Arguments
    ///
    /// * `skill_id` - The skill ID
    /// * `version` - The version number to delete
    pub async fn delete_version(
        &self,
        skill_id: &str,
        version: u64,
    ) -> Result<DeletedSkillVersion> {
        if skill_id.is_empty() {
            return Err(Error::config("skill_id cannot be empty"));
        }
        self.client
            .delete(&format!("v1/skills/{}/versions/{}", skill_id, version))
            .await
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_create_params() {
        let params = SkillCreateParams::new("my-skill", "A test skill", vec![1, 2, 3]);
        assert_eq!(params.name, "my-skill");
        assert_eq!(params.description, "A test skill");
        assert_eq!(params.bundle, vec![1, 2, 3]);
        assert_eq!(params.bundle_mime_type, "application/zip");
    }

    #[test]
    fn test_skill_create_params_tar_gz() {
        let params = SkillCreateParams::with_tar_gz("my-skill", "desc", vec![1, 2, 3]);
        assert_eq!(params.bundle_mime_type, "application/gzip");
    }

    #[test]
    fn test_version_create_params() {
        let params = VersionCreateParams::zip(vec![1, 2, 3]);
        assert_eq!(params.bundle_mime_type, "application/zip");

        let params = VersionCreateParams::tar_gz(vec![4, 5, 6]);
        assert_eq!(params.bundle_mime_type, "application/gzip");
    }

    #[test]
    fn test_skills_list_params_builder() {
        let params = SkillsListParams::new()
            .with_limit(50)
            .with_after_id("skill_abc")
            .custom_only();

        assert_eq!(params.limit, Some(50));
        assert_eq!(params.after_id, Some("skill_abc".to_string()));
        assert_eq!(params.skill_type, Some(SkillType::Custom));
    }

    #[test]
    fn test_skill_type_serialize() {
        assert_eq!(
            serde_json::to_string(&SkillType::Custom).unwrap(),
            r#""custom""#
        );
        assert_eq!(
            serde_json::to_string(&SkillType::Anthropic).unwrap(),
            r#""anthropic""#
        );
    }

    #[test]
    fn test_skill_deserialize() {
        let json = r#"{
            "id": "skill_abc123",
            "type": "skill",
            "name": "code-reviewer",
            "description": "Reviews code for issues",
            "skill_type": "custom",
            "created_at": "2025-01-01T00:00:00Z",
            "latest_version": 1704067200
        }"#;

        let skill: Skill = serde_json::from_str(json).unwrap();
        assert_eq!(skill.id, "skill_abc123");
        assert_eq!(skill.name, "code-reviewer");
        assert_eq!(skill.skill_kind, SkillType::Custom);
        assert_eq!(skill.latest_version, Some(1704067200));
    }

    #[test]
    fn test_skill_version_deserialize() {
        let json = r#"{
            "version": 1704067200,
            "type": "skill_version",
            "skill_id": "skill_abc123",
            "created_at": "2025-01-01T00:00:00Z",
            "content_hash": "abc123",
            "size_bytes": 1024
        }"#;

        let version: SkillVersion = serde_json::from_str(json).unwrap();
        assert_eq!(version.version, 1704067200);
        assert_eq!(version.skill_id, "skill_abc123");
        assert_eq!(version.content_hash, Some("abc123".to_string()));
        assert_eq!(version.size_bytes, Some(1024));
    }

    #[test]
    fn test_deleted_skill_deserialize() {
        let json = r#"{
            "id": "skill_abc123",
            "type": "skill_deleted"
        }"#;

        let deleted: DeletedSkill = serde_json::from_str(json).unwrap();
        assert_eq!(deleted.id, "skill_abc123");
        assert_eq!(deleted.deleted_type, "skill_deleted");
    }

    #[test]
    fn test_skills_list_response_pagination() {
        let response = SkillsListResponse {
            data: vec![],
            has_more: true,
            first_id: Some("skill_001".to_string()),
            last_id: Some("skill_010".to_string()),
        };

        let next = response.next_page_params().unwrap();
        assert_eq!(next.after_id, Some("skill_010".to_string()));
    }

    #[test]
    fn test_versions_list_params() {
        let params = VersionsListParams::new()
            .with_limit(25)
            .with_after_version(1704067200);

        assert_eq!(params.limit, Some(25));
        assert_eq!(params.after_version, Some(1704067200));
    }
}
