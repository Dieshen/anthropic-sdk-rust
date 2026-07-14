//! Message Batches API resource.
//!
//! This module provides the [`Batches`] resource for interacting with the
//! Anthropic Message Batches API. Batches allow you to process multiple
//! message requests efficiently, with results delivered asynchronously.
//!
//! # Overview
//!
//! The Message Batches API processes multiple requests in parallel, with
//! results typically available within 24 hours. This is ideal for:
//!
//! - Large-scale data processing
//! - Background tasks that don't need immediate results
//! - Cost-effective bulk operations
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic::{Anthropic, BatchCreateParams, BatchRequest, MessageCreateParams, Model};
//!
//! let client = Anthropic::new()?;
//!
//! // Create batch requests
//! let requests = vec![
//!     BatchRequest::new("req-1", MessageCreateParams::new(
//!         Model::claude_sonnet_4_5_latest(),
//!         vec![MessageParam::user("What is 2+2?")],
//!         1024,
//!     )),
//!     BatchRequest::new("req-2", MessageCreateParams::new(
//!         Model::claude_sonnet_4_5_latest(),
//!         vec![MessageParam::user("What is 3+3?")],
//!         1024,
//!     )),
//! ];
//!
//! // Submit the batch
//! let batch = client.batches().create(BatchCreateParams::new(requests)).await?;
//! println!("Batch ID: {}", batch.id);
//!
//! // Poll for completion
//! loop {
//!     let status = client.batches().get(&batch.id).await?;
//!     if status.is_ended() {
//!         break;
//!     }
//!     tokio::time::sleep(Duration::from_secs(60)).await;
//! }
//!
//! // Stream results
//! use futures::StreamExt;
//! let mut stream = client.batches().results_stream(&batch.id).await?;
//! while let Some(result) = stream.next().await {
//!     println!("Result: {:?}", result?);
//! }
//! ```

use futures::{Future, Stream};
use reqwest::Method;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::streaming::JsonlStream;
use crate::types::{
    BatchCreateParams, BatchListParams, BatchListResponse, BatchResult, MessageBatch,
};

// =============================================================================
// Deleted Batch Response
// =============================================================================

/// Response from deleting a message batch.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeletedMessageBatch {
    /// ID of the deleted batch.
    pub id: String,

    /// Object type (always "`message_batch_deleted`").
    #[serde(rename = "type")]
    pub deleted_type: String,
}

// =============================================================================
// Batches Resource
// =============================================================================

/// Resource for interacting with the Message Batches API.
///
/// This resource provides methods for creating, monitoring, and managing
/// message batches. Batches allow efficient processing of multiple requests
/// with results delivered asynchronously.
///
/// # Thread Safety
///
/// `Batches` is `Clone` and can be shared across threads safely.
#[derive(Debug, Clone)]
pub struct Batches {
    /// Reference to the parent client.
    client: Arc<BatchesClient>,
}

/// Internal client interface for batches operations.
#[derive(Debug)]
pub(crate) struct BatchesClient {
    http_client: reqwest::Client,
    base_url: url::Url,
}

impl BatchesClient {
    /// Creates a new batches client.
    pub(crate) const fn new(http_client: reqwest::Client, base_url: url::Url) -> Self {
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

    /// Makes a GET request.
    async fn get<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = self.build_url(path)?;
        let response = self.http_client.get(url).send().await?;
        self.handle_response(response).await
    }

    /// Makes a GET request with query parameters.
    // `Q` is only bound by `Serialize`, not `Sync`, so the `&Q` reference
    // captured by this async fn makes the returned future `!Send`. Adding a
    // `Sync` bound would be a breaking change for callers of this crate-
    // internal helper, so we accept the `!Send` future instead.
    #[allow(clippy::future_not_send)]
    async fn get_with_query<T, Q>(&self, path: &str, query: &Q) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        Q: serde::Serialize + ?Sized,
    {
        let url = self.build_url(path)?;
        let response = self.http_client.get(url).query(query).send().await?;
        self.handle_response(response).await
    }

    /// Makes a POST request with a JSON body.
    // `B` is only bound by `Serialize`, not `Sync`; see `get_with_query`
    // above for why this makes the returned future `!Send`.
    #[allow(clippy::future_not_send)]
    async fn post<T, B>(&self, path: &str, body: &B) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {
        let url = self.build_url(path)?;
        let response = self.http_client.post(url).json(body).send().await?;
        self.handle_response(response).await
    }

    /// Makes a POST request without a body.
    async fn post_empty<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = self.build_url(path)?;
        let response = self.http_client.post(url).send().await?;
        self.handle_response(response).await
    }

    /// Makes a DELETE request.
    async fn delete<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = self.build_url(path)?;
        let response = self.http_client.delete(url).send().await?;
        self.handle_response(response).await
    }

    /// Makes a GET request and returns the raw response for streaming.
    async fn get_stream(&self, path: &str) -> Result<reqwest::Response> {
        let url = self.build_url(path)?;
        let response = self
            .http_client
            .request(Method::GET, url)
            .header("Accept", "application/x-jsonl")
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Streaming(format!(
                "HTTP {status} when fetching results: {body}"
            )));
        }

        Ok(response)
    }

    /// Handles an HTTP response, parsing JSON or returning an error.
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
            Err(Error::Streaming(format!("HTTP {status}: {body}")))
        }
    }
}

impl Batches {
    /// Creates a new Batches resource.
    pub(crate) fn new(http_client: reqwest::Client, base_url: url::Url) -> Self {
        Self {
            client: Arc::new(BatchesClient::new(http_client, base_url)),
        }
    }

    /// Creates a new message batch.
    ///
    /// Submits a batch of message requests for processing. The batch begins
    /// processing immediately and can take up to 24 hours to complete.
    ///
    /// # Arguments
    ///
    /// * `params` - The batch creation parameters containing the requests
    ///
    /// # Returns
    ///
    /// Returns the created [`MessageBatch`] with its ID and initial status.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The request parameters are invalid
    /// - Authentication fails
    /// - The API is unavailable
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let requests = vec![
    ///     BatchRequest::new("req-1", params1),
    ///     BatchRequest::new("req-2", params2),
    /// ];
    ///
    /// let batch = client.batches().create(BatchCreateParams::new(requests)).await?;
    /// println!("Created batch: {}", batch.id);
    /// ```
    pub async fn create(&self, params: BatchCreateParams) -> Result<MessageBatch> {
        self.client.post("v1/messages/batches", &params).await
    }

    /// Retrieves a message batch by ID.
    ///
    /// This endpoint is idempotent and can be used to poll for batch completion.
    /// Once the batch has finished processing, the `results_url` field will be
    /// populated.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The unique identifier of the batch
    ///
    /// # Returns
    ///
    /// Returns the [`MessageBatch`] with current status and request counts.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch ID is not found
    /// - Authentication fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let batch = client.batches().get("batch_abc123").await?;
    /// println!("Status: {:?}", batch.processing_status);
    /// println!("Succeeded: {}", batch.request_counts.succeeded);
    /// ```
    pub async fn get(&self, batch_id: &str) -> Result<MessageBatch> {
        if batch_id.is_empty() {
            return Err(Error::config("batch_id cannot be empty"));
        }
        self.client
            .get(&format!("v1/messages/batches/{batch_id}"))
            .await
    }

    /// Lists all message batches.
    ///
    /// Returns batches in reverse chronological order (most recent first).
    /// Use pagination parameters to iterate through large result sets.
    ///
    /// # Arguments
    ///
    /// * `params` - Optional pagination parameters
    ///
    /// # Returns
    ///
    /// Returns a [`BatchListResponse`] containing the batches and pagination info.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Authentication fails
    /// - The `after_id` cursor in `params` refers to a batch that no longer exists
    /// - The API is unavailable or returns a malformed response
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // List first page
    /// let response = client.batches().list(BatchListParams::new()).await?;
    ///
    /// for batch in &response.data {
    ///     println!("Batch: {} - {:?}", batch.id, batch.processing_status);
    /// }
    ///
    /// // Get next page if available
    /// if response.has_more {
    ///     let next = client.batches().list(
    ///         BatchListParams::new().with_after_id(response.last_id.unwrap())
    ///     ).await?;
    /// }
    /// ```
    pub async fn list(&self, params: BatchListParams) -> Result<BatchListResponse> {
        self.client
            .get_with_query("v1/messages/batches", &params)
            .await
    }

    /// Lists all message batches with automatic pagination.
    ///
    /// Returns an async iterator that automatically fetches subsequent pages
    /// as needed. This is the recommended way to iterate through all batches.
    ///
    /// # Arguments
    ///
    /// * `params` - Initial pagination parameters (limit will be used per page)
    ///
    /// # Returns
    ///
    /// Returns a [`BatchPaginator`] that implements async iteration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut paginator = client.batches().list_auto_paging(BatchListParams::new());
    ///
    /// while let Some(batch) = paginator.next().await {
    ///     let batch = batch?;
    ///     println!("Batch: {}", batch.id);
    /// }
    /// ```
    #[must_use]
    pub fn list_auto_paging(&self, params: BatchListParams) -> BatchPaginator {
        BatchPaginator::new(self.clone(), params)
    }

    /// Cancels a message batch.
    ///
    /// Batches may be canceled any time before processing ends. Once cancellation
    /// is initiated, the batch enters a `canceling` state. The system may complete
    /// any in-progress, non-interruptible requests before finalizing cancellation.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The unique identifier of the batch to cancel
    ///
    /// # Returns
    ///
    /// Returns the updated [`MessageBatch`] with `canceling` status.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch ID is not found
    /// - The batch has already ended
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let batch = client.batches().cancel("batch_abc123").await?;
    /// assert!(batch.is_canceling());
    /// ```
    pub async fn cancel(&self, batch_id: &str) -> Result<MessageBatch> {
        if batch_id.is_empty() {
            return Err(Error::config("batch_id cannot be empty"));
        }
        self.client
            .post_empty(&format!("v1/messages/batches/{batch_id}/cancel"))
            .await
    }

    /// Deletes a message batch.
    ///
    /// Message batches can only be deleted once they've finished processing.
    /// If you'd like to delete an in-progress batch, you must first cancel it.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The unique identifier of the batch to delete
    ///
    /// # Returns
    ///
    /// Returns a [`DeletedMessageBatch`] confirming deletion.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch ID is not found
    /// - The batch is still processing (cancel it first)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let deleted = client.batches().delete("batch_abc123").await?;
    /// println!("Deleted batch: {}", deleted.id);
    /// ```
    pub async fn delete(&self, batch_id: &str) -> Result<DeletedMessageBatch> {
        if batch_id.is_empty() {
            return Err(Error::config("batch_id cannot be empty"));
        }
        self.client
            .delete(&format!("v1/messages/batches/{batch_id}"))
            .await
    }

    /// Retrieves batch results as a collected vector.
    ///
    /// Downloads and parses all results from a completed batch. This method
    /// waits for all results to be downloaded before returning.
    ///
    /// For large batches, consider using [`results_stream`](Self::results_stream)
    /// to process results incrementally.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The unique identifier of the completed batch
    ///
    /// # Returns
    ///
    /// Returns a vector of [`BatchResult`] containing all request results.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch ID is not found
    /// - The batch has not finished processing
    /// - Results parsing fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = client.batches().results("batch_abc123").await?;
    ///
    /// for result in results {
    ///     match result.result {
    ///         BatchResultType::Succeeded { message } => {
    ///             println!("{}: {}", result.custom_id, message.content_text());
    ///         }
    ///         BatchResultType::Errored { error } => {
    ///             println!("{}: Error - {}", result.custom_id, error.message);
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub async fn results(&self, batch_id: &str) -> Result<Vec<BatchResult>> {
        if batch_id.is_empty() {
            return Err(Error::config("batch_id cannot be empty"));
        }

        let response = self
            .client
            .get_stream(&format!("v1/messages/batches/{batch_id}/results"))
            .await?;

        let body = response.text().await?;
        crate::streaming::parse_jsonl(&body)
    }

    /// Streams batch results as JSONL.
    ///
    /// Returns an async stream that yields batch results one at a time as they
    /// are downloaded. This is memory-efficient for large batches.
    ///
    /// Results are not guaranteed to be in the same order as requests. Use the
    /// `custom_id` field to match results to requests.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The unique identifier of the completed batch
    ///
    /// # Returns
    ///
    /// Returns a [`JsonlStream`] that yields [`BatchResult`] items.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial request fails. Individual items in
    /// the stream may also yield errors if parsing fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = client.batches().results_stream("batch_abc123").await?;
    ///
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(batch_result) => {
    ///             println!("Got result for: {}", batch_result.custom_id);
    ///         }
    ///         Err(e) => {
    ///             eprintln!("Error parsing result: {}", e);
    ///         }
    ///     }
    /// }
    /// ```
    pub async fn results_stream(
        &self,
        batch_id: &str,
    ) -> Result<
        JsonlStream<
            BatchResult,
            impl Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>,
        >,
    > {
        if batch_id.is_empty() {
            return Err(Error::config("batch_id cannot be empty"));
        }

        let response = self
            .client
            .get_stream(&format!("v1/messages/batches/{batch_id}/results"))
            .await?;

        Ok(JsonlStream::<BatchResult, _>::new(response.bytes_stream()))
    }

    /// Polls a batch until it completes or the timeout is reached.
    ///
    /// This is a convenience method that repeatedly polls the batch status
    /// until processing ends. It's useful for scripts and CLI tools.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The unique identifier of the batch
    /// * `poll_interval` - Duration between status checks
    /// * `timeout` - Maximum time to wait for completion
    ///
    /// # Returns
    ///
    /// Returns the final [`MessageBatch`] status when processing ends.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The timeout is exceeded
    /// - Any status check fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::time::Duration;
    ///
    /// let batch = client.batches().poll_until_complete(
    ///     "batch_abc123",
    ///     Duration::from_secs(60),    // Check every minute
    ///     Duration::from_hours(24),   // Wait up to 24 hours
    /// ).await?;
    ///
    /// println!("Batch completed: {:?}", batch.processing_status);
    /// ```
    pub async fn poll_until_complete(
        &self,
        batch_id: &str,
        poll_interval: std::time::Duration,
        timeout: std::time::Duration,
    ) -> Result<MessageBatch> {
        let start = std::time::Instant::now();

        loop {
            let batch = self.get(batch_id).await?;

            if batch.is_ended() {
                return Ok(batch);
            }

            if start.elapsed() >= timeout {
                return Err(Error::Timeout(timeout));
            }

            tokio::time::sleep(poll_interval).await;
        }
    }
}

// =============================================================================
// Pagination
// =============================================================================

/// Async paginator for batch listing.
///
/// This paginator automatically fetches subsequent pages of results as you
/// iterate through them. It implements [`Stream`] for async iteration.
#[derive(Debug)]
pub struct BatchPaginator {
    batches: Batches,
    params: BatchListParams,
    current_page: Option<BatchListResponse>,
    current_index: usize,
    exhausted: bool,
}

impl BatchPaginator {
    /// Creates a new paginator.
    const fn new(batches: Batches, params: BatchListParams) -> Self {
        Self {
            batches,
            params,
            current_page: None,
            current_index: 0,
            exhausted: false,
        }
    }

    /// Fetches the next page of results.
    async fn fetch_next_page(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        let response = self.batches.list(self.params.clone()).await?;

        if response.data.is_empty() {
            self.exhausted = true;
            return Ok(false);
        }

        // Update params for next page
        if let Some(last_id) = &response.last_id {
            self.params = self.params.clone().with_after_id(last_id.clone());
        }

        // Check if there are more pages
        if !response.has_more {
            self.exhausted = true;
        }

        self.current_page = Some(response);
        self.current_index = 0;
        Ok(true)
    }

    /// Returns the next batch, fetching a new page if needed.
    pub async fn next(&mut self) -> Option<Result<MessageBatch>> {
        // Check if we have items in the current page
        if let Some(ref page) = self.current_page {
            if self.current_index < page.data.len() {
                let batch = page.data[self.current_index].clone();
                self.current_index += 1;
                return Some(Ok(batch));
            }
        }

        // Need to fetch next page
        if self.exhausted {
            return None;
        }

        match self.fetch_next_page().await {
            Ok(true) => {
                if let Some(ref page) = self.current_page {
                    if !page.data.is_empty() {
                        let batch = page.data[0].clone();
                        self.current_index = 1;
                        return Some(Ok(batch));
                    }
                }
                None
            }
            Ok(false) => None,
            Err(e) => Some(Err(e)),
        }
    }

    /// Collects all remaining batches into a vector.
    ///
    /// # Errors
    ///
    /// Returns an error if fetching any page fails, for the same reasons as
    /// [`Batches::list`]: authentication failure, an invalid pagination
    /// cursor, or an unavailable/malformed API response. Any batches already
    /// collected before the failing page are discarded.
    pub async fn collect_all(&mut self) -> Result<Vec<MessageBatch>> {
        let mut results = Vec::new();

        while let Some(result) = self.next().await {
            results.push(result?);
        }

        Ok(results)
    }
}

// Implement Stream trait for BatchPaginator
impl Stream for BatchPaginator {
    type Item = Result<MessageBatch>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Check if we have items in the current page
        if let Some(ref page) = this.current_page {
            if this.current_index < page.data.len() {
                let batch = page.data[this.current_index].clone();
                this.current_index += 1;
                return std::task::Poll::Ready(Some(Ok(batch)));
            }
        }

        if this.exhausted {
            return std::task::Poll::Ready(None);
        }

        // Create a future for fetching the next page
        let batches = this.batches.clone();
        let params = this.params.clone();

        // We need to use a boxed future here for the async operation
        let mut future = Box::pin(async move { batches.list(params).await });

        // Poll the future
        match future.as_mut().poll(cx) {
            std::task::Poll::Ready(Ok(response)) => {
                if response.data.is_empty() {
                    this.exhausted = true;
                    return std::task::Poll::Ready(None);
                }

                // Update params for next page
                if let Some(last_id) = &response.last_id {
                    this.params = this.params.clone().with_after_id(last_id.clone());
                }

                if !response.has_more {
                    this.exhausted = true;
                }

                let batch = response.data[0].clone();
                this.current_page = Some(response);
                this.current_index = 1;
                std::task::Poll::Ready(Some(Ok(batch)))
            }
            std::task::Poll::Ready(Err(e)) => std::task::Poll::Ready(Some(Err(e))),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

// =============================================================================
// Client Integration
// =============================================================================

use crate::client::Anthropic;

impl Anthropic {
    /// Returns the Message Batches API resource.
    ///
    /// This provides access to batch creation, monitoring, and result retrieval.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic::{Anthropic, BatchCreateParams, BatchRequest};
    ///
    /// let client = Anthropic::new()?;
    /// let batches = client.batches();
    ///
    /// // Create a batch
    /// let batch = batches.create(params).await?;
    ///
    /// // Check status
    /// let status = batches.get(&batch.id).await?;
    ///
    /// // Get results when complete
    /// let results = batches.results(&batch.id).await?;
    /// ```
    #[must_use]
    pub fn batches(&self) -> Batches {
        Batches::new(self.http_client(), self.base_url().clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BatchProcessingStatus, BatchRequest, BatchRequestCounts, MessageCreateParams, MessageParam,
        Model,
    };
    use chrono::Utc;

    fn create_test_batch() -> MessageBatch {
        MessageBatch {
            id: "batch_123".to_string(),
            batch_type: "message_batch".to_string(),
            processing_status: BatchProcessingStatus::InProgress,
            request_counts: BatchRequestCounts::default(),
            created_at: Utc::now(),
            ended_at: None,
            expires_at: Utc::now() + chrono::Duration::hours(24),
            cancel_initiated_at: None,
            archived_at: None,
            results_url: None,
        }
    }

    fn create_test_params() -> MessageCreateParams {
        MessageCreateParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Hello!")],
            1024,
        )
    }

    #[test]
    fn test_batch_create_params() {
        let params = create_test_params();
        let request = BatchRequest::new("req-001", params);
        let batch_params = BatchCreateParams::new(vec![request]);

        assert_eq!(batch_params.requests.len(), 1);
        assert_eq!(batch_params.requests[0].custom_id, "req-001");
    }

    #[test]
    fn test_batch_list_params_builder() {
        let params = BatchListParams::new()
            .with_limit(50)
            .with_after_id("batch_abc")
            .with_before_id("batch_xyz");

        assert_eq!(params.limit, Some(50));
        assert_eq!(params.after_id, Some("batch_abc".to_string()));
        assert_eq!(params.before_id, Some("batch_xyz".to_string()));
    }

    #[test]
    fn test_message_batch_status_helpers() {
        let mut batch = create_test_batch();

        assert!(batch.is_processing());
        assert!(!batch.is_ended());
        assert!(!batch.is_canceling());

        batch.processing_status = BatchProcessingStatus::Ended;
        assert!(!batch.is_processing());
        assert!(batch.is_ended());

        batch.processing_status = BatchProcessingStatus::Canceling;
        assert!(batch.is_canceling());
    }

    #[test]
    fn test_deleted_batch_serialization() {
        let deleted = DeletedMessageBatch {
            id: "batch_123".to_string(),
            deleted_type: "message_batch_deleted".to_string(),
        };

        let json = serde_json::to_string(&deleted).unwrap();
        assert!(json.contains("batch_123"));
        assert!(json.contains("message_batch_deleted"));

        let parsed: DeletedMessageBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "batch_123");
    }

    #[test]
    fn test_batch_list_response_deserialization() {
        let json = r#"{
            "data": [],
            "has_more": false,
            "first_id": null,
            "last_id": null
        }"#;

        let response: BatchListResponse = serde_json::from_str(json).unwrap();
        assert!(response.data.is_empty());
        assert!(!response.has_more);
    }

    #[test]
    fn test_batch_list_response_with_data() {
        let json = r#"{
            "data": [{
                "id": "batch_123",
                "type": "message_batch",
                "processing_status": "in_progress",
                "request_counts": {
                    "processing": 5,
                    "succeeded": 0,
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0
                },
                "created_at": "2024-01-01T00:00:00Z",
                "expires_at": "2024-01-02T00:00:00Z"
            }],
            "has_more": true,
            "first_id": "batch_123",
            "last_id": "batch_123"
        }"#;

        let response: BatchListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(response.has_more);
        assert_eq!(response.first_id, Some("batch_123".to_string()));
        assert_eq!(response.last_id, Some("batch_123".to_string()));
    }

    #[test]
    fn test_batch_request_counts() {
        let counts = BatchRequestCounts {
            processing: 10,
            succeeded: 5,
            errored: 2,
            canceled: 1,
            expired: 0,
        };

        assert_eq!(counts.total(), 18);
    }
}
