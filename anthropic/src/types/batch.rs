//! Batch types for the Anthropic Message Batches API.
//!
//! This module defines types for creating and managing message batches,
//! which allow for efficient processing of multiple messages.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::message::{Message, MessageCreateParams};

// =============================================================================
// Message Batch
// =============================================================================

/// A message batch response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageBatch {
    /// Unique identifier for this batch.
    pub id: String,

    /// Object type (always "message_batch").
    #[serde(rename = "type")]
    pub batch_type: String,

    /// Processing status of the batch.
    pub processing_status: BatchProcessingStatus,

    /// Request counts by status.
    pub request_counts: BatchRequestCounts,

    /// Time when the batch was created.
    pub created_at: DateTime<Utc>,

    /// Time when processing ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,

    /// Time when the batch expires.
    pub expires_at: DateTime<Utc>,

    /// Time when cancellation was initiated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancel_initiated_at: Option<DateTime<Utc>>,

    /// Time when the batch was archived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived_at: Option<DateTime<Utc>>,

    /// URL to download results (available after processing ends).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub results_url: Option<String>,
}

impl MessageBatch {
    /// Returns true if the batch is still processing.
    #[must_use]
    pub fn is_processing(&self) -> bool {
        self.processing_status == BatchProcessingStatus::InProgress
    }

    /// Returns true if the batch has ended.
    #[must_use]
    pub fn is_ended(&self) -> bool {
        self.processing_status == BatchProcessingStatus::Ended
    }

    /// Returns true if the batch is being cancelled.
    #[must_use]
    pub fn is_canceling(&self) -> bool {
        self.processing_status == BatchProcessingStatus::Canceling
    }
}

/// Processing status of a message batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchProcessingStatus {
    /// Batch is currently being processed.
    InProgress,
    /// Batch cancellation has been initiated.
    Canceling,
    /// Batch processing has ended.
    Ended,
}

/// Request counts for a message batch.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    /// Number of requests currently processing.
    pub processing: i64,
    /// Number of successfully completed requests.
    pub succeeded: i64,
    /// Number of failed requests.
    pub errored: i64,
    /// Number of cancelled requests.
    pub canceled: i64,
    /// Number of expired requests.
    pub expired: i64,
}

impl BatchRequestCounts {
    /// Returns the total number of requests.
    #[must_use]
    pub fn total(&self) -> i64 {
        self.processing + self.succeeded + self.errored + self.canceled + self.expired
    }
}

// =============================================================================
// Batch Create Parameters
// =============================================================================

/// Parameters for creating a message batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchCreateParams {
    /// The requests to include in the batch.
    pub requests: Vec<BatchRequest>,
}

impl BatchCreateParams {
    /// Creates new batch parameters with the given requests.
    #[must_use]
    pub fn new(requests: Vec<BatchRequest>) -> Self {
        Self { requests }
    }
}

/// A single request in a batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Custom identifier for this request.
    pub custom_id: String,

    /// The message parameters.
    pub params: MessageCreateParams,
}

impl BatchRequest {
    /// Creates a new batch request.
    #[must_use]
    pub fn new(custom_id: impl Into<String>, params: MessageCreateParams) -> Self {
        Self {
            custom_id: custom_id.into(),
            params,
        }
    }
}

// =============================================================================
// Batch Results
// =============================================================================

/// A single result from a batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchResult {
    /// Custom identifier for this request.
    pub custom_id: String,

    /// The result of the request.
    pub result: BatchResultType,
}

/// The type of result for a batch request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BatchResultType {
    /// Successful result.
    Succeeded {
        /// The message response.
        message: Message,
    },
    /// Error result.
    Errored {
        /// The error details.
        error: BatchError,
    },
    /// Cancelled result.
    Canceled,
    /// Expired result.
    Expired,
}

impl BatchResultType {
    /// Returns the message if this is a successful result.
    #[must_use]
    pub fn message(&self) -> Option<&Message> {
        match self {
            Self::Succeeded { message } => Some(message),
            _ => None,
        }
    }

    /// Returns true if this result was successful.
    #[must_use]
    pub fn is_succeeded(&self) -> bool {
        matches!(self, Self::Succeeded { .. })
    }

    /// Returns true if this result was an error.
    #[must_use]
    pub fn is_errored(&self) -> bool {
        matches!(self, Self::Errored { .. })
    }
}

/// Error details for a batch request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchError {
    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error message.
    pub message: String,
}

// =============================================================================
// Batch List Parameters
// =============================================================================

/// Parameters for listing message batches.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchListParams {
    /// Maximum number of batches to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,

    /// Cursor for pagination (before this ID).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_id: Option<String>,

    /// Cursor for pagination (after this ID).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_id: Option<String>,
}

impl BatchListParams {
    /// Creates new list parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the limit.
    #[must_use]
    pub fn with_limit(mut self, limit: i64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sets the before_id cursor.
    #[must_use]
    pub fn with_before_id(mut self, before_id: impl Into<String>) -> Self {
        self.before_id = Some(before_id.into());
        self
    }

    /// Sets the after_id cursor.
    #[must_use]
    pub fn with_after_id(mut self, after_id: impl Into<String>) -> Self {
        self.after_id = Some(after_id.into());
        self
    }
}

/// Response from listing message batches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchListResponse {
    /// The list of batches.
    pub data: Vec<MessageBatch>,

    /// Whether there are more results.
    pub has_more: bool,

    /// ID of the first batch in the list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,

    /// ID of the last batch in the list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::MessageParam;
    use crate::types::model::Model;

    #[test]
    fn test_batch_processing_status_serialize() {
        assert_eq!(
            serde_json::to_string(&BatchProcessingStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&BatchProcessingStatus::Ended).unwrap(),
            "\"ended\""
        );
    }

    #[test]
    fn test_batch_request_counts() {
        let counts = BatchRequestCounts {
            processing: 5,
            succeeded: 10,
            errored: 2,
            canceled: 1,
            expired: 0,
        };
        assert_eq!(counts.total(), 18);
    }

    #[test]
    fn test_batch_request() {
        let params = MessageCreateParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Hello!")],
            1024,
        );

        let request = BatchRequest::new("req-001", params);
        assert_eq!(request.custom_id, "req-001");

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"custom_id\":\"req-001\""));
    }

    #[test]
    fn test_batch_result_succeeded() {
        let json = r#"{
            "custom_id": "req-001",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "model": "claude-sonnet-4-5-latest",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                }
            }
        }"#;

        let result: BatchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.custom_id, "req-001");
        assert!(result.result.is_succeeded());
        assert!(result.result.message().is_some());
    }

    #[test]
    fn test_batch_result_errored() {
        let json = r#"{
            "custom_id": "req-002",
            "result": {
                "type": "errored",
                "error": {
                    "type": "invalid_request",
                    "message": "Invalid input"
                }
            }
        }"#;

        let result: BatchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.custom_id, "req-002");
        assert!(result.result.is_errored());
    }

    #[test]
    fn test_batch_list_params() {
        let params = BatchListParams::new()
            .with_limit(10)
            .with_after_id("batch_123");

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"limit\":10"));
        assert!(json.contains("\"after_id\":\"batch_123\""));
    }
}
