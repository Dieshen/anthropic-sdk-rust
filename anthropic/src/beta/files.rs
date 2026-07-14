//! Beta Files API for the Anthropic SDK.
//!
//! This module provides access to the Files API, which allows you to upload,
//! manage, and reference files for use with Claude messages. Files are stored
//! server-side and can be referenced by ID in message content blocks.
//!
//! # Beta Status
//!
//! The Files API is currently in beta and requires the `files-api-2025-04-14`
//! beta header. This is handled automatically when using the Files resource.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::Anthropic;
//! use anthropic::beta::files::{Files, FileUploadParams};
//!
//! let client = Anthropic::new()?;
//! let files = client.beta_files();
//!
//! // Upload a file
//! let file = files.upload(
//!     FileUploadParams::new(file_data, "document.pdf", "application/pdf")
//! ).await?;
//!
//! println!("Uploaded file ID: {}", file.id);
//!
//! // List files
//! let list = files.list(Default::default()).await?;
//! for f in list.data {
//!     println!("File: {} ({} bytes)", f.filename, f.size_bytes);
//! }
//!
//! // Delete a file
//! files.delete(&file.id).await?;
//! ```
//!
//! # Using Files in Messages
//!
//! Once uploaded, files can be referenced in message content blocks using
//! their file ID:
//!
//! ```rust,ignore
//! use anthropic::types::{ContentBlockParam, DocumentBlockParam, FileSource};
//!
//! let content = ContentBlockParam::Document(DocumentBlockParam {
//!     source: FileSource::file_id(&file.id),
//!     cache_control: None,
//! });
//! ```

use std::sync::Arc;

use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// =============================================================================
// Beta Header
// =============================================================================

/// Beta header value for the Files API.
pub const FILES_API_BETA_HEADER: &str = "files-api-2025-04-14";

// =============================================================================
// File Metadata Types
// =============================================================================

/// Metadata for an uploaded file.
///
/// This represents a file stored on Anthropic's servers that can be
/// referenced in message content blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileMetadata {
    /// Unique identifier for the file.
    ///
    /// This ID is used to reference the file in message content blocks.
    pub id: String,

    /// Object type (always "file").
    #[serde(rename = "type")]
    pub file_type: String,

    /// Original filename of the uploaded file.
    pub filename: String,

    /// MIME type of the file.
    pub mime_type: String,

    /// Size of the file in bytes.
    pub size_bytes: u64,

    /// ISO 8601 timestamp of when the file was created.
    pub created_at: String,

    /// ISO 8601 timestamp of when the file will be automatically deleted.
    ///
    /// Files are typically retained for a limited period.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,

    /// Whether the file can be downloaded.
    #[serde(default)]
    pub downloadable: bool,
}

/// Response when deleting a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletedFile {
    /// ID of the deleted file.
    pub id: String,

    /// Object type (always "file_deleted").
    #[serde(rename = "type")]
    pub deleted_type: String,
}

// =============================================================================
// List Files Types
// =============================================================================

/// Parameters for listing files.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilesListParams {
    /// Maximum number of files to return (1-100).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,

    /// Cursor for pagination (from previous response).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_id: Option<String>,

    /// Filter by MIME type prefix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl FilesListParams {
    /// Creates new list parameters with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of files to return.
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

    /// Sets the MIME type filter.
    #[must_use]
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }
}

/// Response when listing files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesListResponse {
    /// List of file metadata objects.
    pub data: Vec<FileMetadata>,

    /// Whether there are more files to fetch.
    pub has_more: bool,

    /// ID of the first file in the list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,

    /// ID of the last file in the list (use for pagination).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

impl FilesListResponse {
    /// Returns pagination parameters for fetching the next page.
    #[must_use]
    pub fn next_page_params(&self) -> Option<FilesListParams> {
        if self.has_more {
            self.last_id
                .as_ref()
                .map(|id| FilesListParams::new().with_after_id(id.clone()))
        } else {
            None
        }
    }

    /// Returns true if there are more pages available.
    #[must_use]
    pub fn has_next_page(&self) -> bool {
        self.has_more && self.last_id.is_some()
    }
}

// =============================================================================
// Upload Types
// =============================================================================

/// Parameters for uploading a file.
#[derive(Debug, Clone)]
pub struct FileUploadParams {
    /// File content as bytes.
    pub data: Vec<u8>,

    /// Original filename.
    pub filename: String,

    /// MIME type of the file.
    pub mime_type: String,
}

impl FileUploadParams {
    /// Creates new upload parameters.
    ///
    /// # Arguments
    ///
    /// * `data` - The file content as bytes
    /// * `filename` - The original filename
    /// * `mime_type` - The MIME type (e.g., "application/pdf", "image/png")
    #[must_use]
    pub fn new(
        data: impl Into<Vec<u8>>,
        filename: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Self {
        Self {
            data: data.into(),
            filename: filename.into(),
            mime_type: mime_type.into(),
        }
    }

    /// Creates upload parameters for a PDF file.
    #[must_use]
    pub fn pdf(data: impl Into<Vec<u8>>, filename: impl Into<String>) -> Self {
        Self::new(data, filename, "application/pdf")
    }

    /// Creates upload parameters for a PNG image.
    #[must_use]
    pub fn png(data: impl Into<Vec<u8>>, filename: impl Into<String>) -> Self {
        Self::new(data, filename, "image/png")
    }

    /// Creates upload parameters for a JPEG image.
    #[must_use]
    pub fn jpeg(data: impl Into<Vec<u8>>, filename: impl Into<String>) -> Self {
        Self::new(data, filename, "image/jpeg")
    }

    /// Creates upload parameters for a text file.
    #[must_use]
    pub fn text(data: impl Into<Vec<u8>>, filename: impl Into<String>) -> Self {
        Self::new(data, filename, "text/plain")
    }
}

// =============================================================================
// Files Client
// =============================================================================

/// Internal client for files operations.
#[derive(Debug)]
pub(crate) struct FilesClient {
    http_client: reqwest::Client,
    base_url: url::Url,
}

impl FilesClient {
    /// Creates a new files client.
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
            .header("anthropic-beta", FILES_API_BETA_HEADER)
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
            .header("anthropic-beta", FILES_API_BETA_HEADER)
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
            .header("anthropic-beta", FILES_API_BETA_HEADER)
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
            .header("anthropic-beta", FILES_API_BETA_HEADER)
            .send()
            .await?;
        self.handle_response(response).await
    }

    /// Gets raw bytes from a response.
    async fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let url = self.build_url(path)?;
        let response = self
            .http_client
            .get(url)
            .header("anthropic-beta", FILES_API_BETA_HEADER)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            Ok(response.bytes().await?.to_vec())
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(Error::Streaming(format!("HTTP {}: {}", status, body)))
        }
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
// Files Resource
// =============================================================================

/// Resource for interacting with the Files API.
///
/// The Files API allows you to upload, manage, and reference files for use
/// with Claude messages. Files can be used as document or image content
/// blocks by referencing their ID.
///
/// # Example
///
/// ```rust,ignore
/// let client = Anthropic::new()?;
/// let files = client.beta_files();
///
/// // Upload a PDF document
/// let pdf_data = std::fs::read("document.pdf")?;
/// let file = files.upload(
///     FileUploadParams::pdf(pdf_data, "document.pdf")
/// ).await?;
///
/// // Use the file in a message
/// let message = client.messages().create(
///     MessageCreateParams::builder()
///         .model(Model::claude_sonnet_4_5_latest())
///         .max_tokens(1024)
///         .messages(vec![
///             MessageParam::user_with_content(vec![
///                 ContentBlockParam::Document(DocumentBlockParam {
///                     source: FileSource::file_id(&file.id),
///                     cache_control: None,
///                 }),
///                 ContentBlockParam::Text("Summarize this document".into()),
///             ])
///         ])
///         .build()?
/// ).await?;
/// ```
#[derive(Debug, Clone)]
pub struct Files {
    client: Arc<FilesClient>,
}

impl Files {
    /// Creates a new Files resource.
    pub(crate) fn new(http_client: reqwest::Client, base_url: url::Url) -> Self {
        Self {
            client: Arc::new(FilesClient::new(http_client, base_url)),
        }
    }

    /// Uploads a file to Anthropic's servers.
    ///
    /// The uploaded file can be referenced by its ID in message content blocks.
    ///
    /// # Arguments
    ///
    /// * `params` - Upload parameters including file data, filename, and MIME type
    ///
    /// # Returns
    ///
    /// Returns metadata for the uploaded file including its ID.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("image.png")?;
    /// let file = files.upload(
    ///     FileUploadParams::png(data, "image.png")
    /// ).await?;
    /// println!("File ID: {}", file.id);
    /// ```
    pub async fn upload(&self, params: FileUploadParams) -> Result<FileMetadata> {
        let part = Part::bytes(params.data)
            .file_name(params.filename)
            .mime_str(&params.mime_type)
            .map_err(|e| Error::config(format!("Invalid MIME type: {}", e)))?;

        let form = Form::new().part("file", part);

        self.client.upload_multipart("v1/files", form).await
    }

    /// Retrieves metadata for a file by ID.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The unique identifier of the file
    ///
    /// # Returns
    ///
    /// Returns the file metadata if found.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let file = files.get("file_abc123").await?;
    /// println!("Filename: {}", file.filename);
    /// ```
    pub async fn get(&self, file_id: &str) -> Result<FileMetadata> {
        if file_id.is_empty() {
            return Err(Error::config("file_id cannot be empty"));
        }
        self.client.get(&format!("v1/files/{}", file_id)).await
    }

    /// Lists uploaded files with optional filtering.
    ///
    /// # Arguments
    ///
    /// * `params` - Pagination and filtering parameters
    ///
    /// # Returns
    ///
    /// Returns a paginated list of file metadata.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // List first 10 PDF files
    /// let list = files.list(
    ///     FilesListParams::new()
    ///         .with_limit(10)
    ///         .with_mime_type("application/pdf")
    /// ).await?;
    ///
    /// for file in list.data {
    ///     println!("{}: {} bytes", file.filename, file.size_bytes);
    /// }
    /// ```
    pub async fn list(&self, params: FilesListParams) -> Result<FilesListResponse> {
        self.client.get_with_query("v1/files", &params).await
    }

    /// Lists all files using automatic pagination.
    ///
    /// This method fetches all pages and returns all files.
    /// For large collections, consider using `list()` with pagination.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let all_files = files.list_all().await?;
    /// println!("Total files: {}", all_files.len());
    /// ```
    pub async fn list_all(&self) -> Result<Vec<FileMetadata>> {
        let mut all_files = Vec::new();
        let mut params = FilesListParams::new();

        loop {
            let response = self.list(params).await?;
            let next = response.next_page_params();
            all_files.extend(response.data);

            if let Some(next_params) = next {
                params = next_params;
            } else {
                break;
            }
        }

        Ok(all_files)
    }

    /// Downloads the content of a file.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The unique identifier of the file
    ///
    /// # Returns
    ///
    /// Returns the file content as bytes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let content = files.download("file_abc123").await?;
    /// std::fs::write("downloaded.pdf", content)?;
    /// ```
    pub async fn download(&self, file_id: &str) -> Result<Vec<u8>> {
        if file_id.is_empty() {
            return Err(Error::config("file_id cannot be empty"));
        }
        self.client
            .get_bytes(&format!("v1/files/{}/content", file_id))
            .await
    }

    /// Deletes a file.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The unique identifier of the file to delete
    ///
    /// # Returns
    ///
    /// Returns confirmation of deletion.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let deleted = files.delete("file_abc123").await?;
    /// println!("Deleted file: {}", deleted.id);
    /// ```
    pub async fn delete(&self, file_id: &str) -> Result<DeletedFile> {
        if file_id.is_empty() {
            return Err(Error::config("file_id cannot be empty"));
        }
        self.client.delete(&format!("v1/files/{}", file_id)).await
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_upload_params_new() {
        let params = FileUploadParams::new(vec![1, 2, 3], "test.pdf", "application/pdf");
        assert_eq!(params.data, vec![1, 2, 3]);
        assert_eq!(params.filename, "test.pdf");
        assert_eq!(params.mime_type, "application/pdf");
    }

    #[test]
    fn test_file_upload_params_pdf() {
        let params = FileUploadParams::pdf(vec![1, 2, 3], "test.pdf");
        assert_eq!(params.mime_type, "application/pdf");
    }

    #[test]
    fn test_file_upload_params_png() {
        let params = FileUploadParams::png(vec![1, 2, 3], "test.png");
        assert_eq!(params.mime_type, "image/png");
    }

    #[test]
    fn test_file_upload_params_jpeg() {
        let params = FileUploadParams::jpeg(vec![1, 2, 3], "test.jpg");
        assert_eq!(params.mime_type, "image/jpeg");
    }

    #[test]
    fn test_files_list_params_builder() {
        let params = FilesListParams::new()
            .with_limit(50)
            .with_after_id("file_abc123")
            .with_mime_type("application/pdf");

        assert_eq!(params.limit, Some(50));
        assert_eq!(params.after_id, Some("file_abc123".to_string()));
        assert_eq!(params.mime_type, Some("application/pdf".to_string()));
    }

    #[test]
    fn test_files_list_response_next_page() {
        let response = FilesListResponse {
            data: vec![],
            has_more: true,
            first_id: Some("file_001".to_string()),
            last_id: Some("file_010".to_string()),
        };

        assert!(response.has_next_page());
        let next = response.next_page_params().unwrap();
        assert_eq!(next.after_id, Some("file_010".to_string()));
    }

    #[test]
    fn test_files_list_response_no_more_pages() {
        let response = FilesListResponse {
            data: vec![],
            has_more: false,
            first_id: Some("file_001".to_string()),
            last_id: Some("file_005".to_string()),
        };

        assert!(!response.has_next_page());
        assert!(response.next_page_params().is_none());
    }

    #[test]
    fn test_file_metadata_deserialize() {
        let json = r#"{
            "id": "file_abc123",
            "type": "file",
            "filename": "document.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "created_at": "2025-01-01T00:00:00Z",
            "downloadable": true
        }"#;

        let file: FileMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(file.id, "file_abc123");
        assert_eq!(file.filename, "document.pdf");
        assert_eq!(file.mime_type, "application/pdf");
        assert_eq!(file.size_bytes, 1024);
        assert!(file.downloadable);
    }

    #[test]
    fn test_deleted_file_deserialize() {
        let json = r#"{
            "id": "file_abc123",
            "type": "file_deleted"
        }"#;

        let deleted: DeletedFile = serde_json::from_str(json).unwrap();
        assert_eq!(deleted.id, "file_abc123");
        assert_eq!(deleted.deleted_type, "file_deleted");
    }
}
