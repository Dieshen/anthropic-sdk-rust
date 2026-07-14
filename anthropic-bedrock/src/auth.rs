//! AWS Signature V4 authentication for Bedrock requests.
//!
//! This module provides AWS Signature V4 request signing for authenticating
//! requests to AWS Bedrock. It uses pure Rust crypto libraries (sha2, hmac)
//! to avoid native compilation dependencies.
//!
//! # Overview
//!
//! AWS Signature V4 is a protocol for signing AWS API requests. This module
//! implements the signing algorithm to authenticate requests to the Bedrock
//! runtime service.
//!
//! # Example
//!
//! ```rust,ignore
//! use anthropic_bedrock::auth::{AwsCredentials, sign_request, SigningRequest};
//!
//! // Load credentials from environment
//! let credentials = AwsCredentials::from_env()?;
//!
//! // Sign a request
//! let request = SigningRequest {
//!     credentials: &credentials,
//!     region: "us-east-1",
//!     method: "POST",
//!     uri: "/model/anthropic.claude-3-sonnet/invoke",
//!     host: "bedrock-runtime.us-east-1.amazonaws.com",
//!     headers: &[("content-type", "application/json")],
//!     body: b"{}",
//!     signing_time: None,
//! };
//! let signed_headers = sign_request(&request)?;
//! ```

use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use tracing::{debug, trace};

use crate::error::{Error, Result};

// =============================================================================
// Constants
// =============================================================================

/// AWS service name for Bedrock Runtime.
pub const SERVICE_NAME: &str = "bedrock";

/// Default AWS region if not specified.
pub const DEFAULT_REGION: &str = "us-east-1";

/// Environment variable for AWS access key ID.
pub const ENV_AWS_ACCESS_KEY_ID: &str = "AWS_ACCESS_KEY_ID";

/// Environment variable for AWS secret access key.
pub const ENV_AWS_SECRET_ACCESS_KEY: &str = "AWS_SECRET_ACCESS_KEY";

/// Environment variable for AWS session token (optional, for temporary credentials).
pub const ENV_AWS_SESSION_TOKEN: &str = "AWS_SESSION_TOKEN";

/// Environment variable for AWS region.
pub const ENV_AWS_REGION: &str = "AWS_REGION";

/// Alternative environment variable for AWS region.
pub const ENV_AWS_DEFAULT_REGION: &str = "AWS_DEFAULT_REGION";

/// AWS4 signing algorithm identifier.
const AWS4_ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// Date format for AWS signatures (ISO 8601 basic).
const DATE_FORMAT: &str = "%Y%m%dT%H%M%SZ";

/// Date format for credential scope.
const DATE_FORMAT_SHORT: &str = "%Y%m%d";

// =============================================================================
// AWS Credentials
// =============================================================================

/// AWS credentials for authenticating Bedrock requests.
///
/// This struct holds the AWS access key ID, secret access key, and optional
/// session token for temporary credentials (e.g., from STS `AssumeRole`).
///
/// # Security
///
/// Credentials are stored in memory. The `Debug` implementation intentionally
/// hides the secret access key and session token to prevent accidental exposure
/// in logs.
///
/// # Example
///
/// ```rust
/// use anthropic_bedrock::auth::AwsCredentials;
///
/// // Explicit credentials
/// let creds = AwsCredentials::new(
///     "AKIAIOSFODNN7EXAMPLE",
///     "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
///     None,
/// );
///
/// // With session token (temporary credentials)
/// let creds = AwsCredentials::new(
///     "ASIAXXX...",
///     "secret...",
///     Some("session-token...".to_string()),
/// );
/// ```
#[derive(Clone)]
pub struct AwsCredentials {
    /// AWS access key ID.
    access_key_id: String,
    /// AWS secret access key.
    secret_access_key: String,
    /// Optional session token for temporary credentials.
    session_token: Option<String>,
}

impl std::fmt::Debug for AwsCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsCredentials")
            .field("access_key_id", &self.access_key_id)
            .field("secret_access_key", &"[REDACTED]")
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

impl AwsCredentials {
    /// Creates new AWS credentials.
    ///
    /// # Arguments
    ///
    /// * `access_key_id` - The AWS access key ID
    /// * `secret_access_key` - The AWS secret access key
    /// * `session_token` - Optional session token for temporary credentials
    ///
    /// # Example
    ///
    /// ```rust
    /// use anthropic_bedrock::auth::AwsCredentials;
    ///
    /// let creds = AwsCredentials::new(
    ///     "AKIAIOSFODNN7EXAMPLE",
    ///     "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ///     None,
    /// );
    /// ```
    #[must_use]
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
        }
    }

    /// Creates credentials from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `AWS_ACCESS_KEY_ID` (required)
    /// - `AWS_SECRET_ACCESS_KEY` (required)
    /// - `AWS_SESSION_TOKEN` (optional, for temporary credentials)
    ///
    /// # Errors
    ///
    /// Returns an error if `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY`
    /// is not set or is empty.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anthropic_bedrock::auth::AwsCredentials;
    ///
    /// // Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set
    /// let creds = AwsCredentials::from_env()?;
    /// ```
    pub fn from_env() -> Result<Self> {
        let access_key_id = std::env::var(ENV_AWS_ACCESS_KEY_ID).map_err(|_| {
            Error::config(format!(
                "Missing required environment variable: {ENV_AWS_ACCESS_KEY_ID}"
            ))
        })?;

        if access_key_id.is_empty() {
            return Err(Error::config(format!(
                "Environment variable {ENV_AWS_ACCESS_KEY_ID} is empty"
            )));
        }

        let secret_access_key = std::env::var(ENV_AWS_SECRET_ACCESS_KEY).map_err(|_| {
            Error::config(format!(
                "Missing required environment variable: {ENV_AWS_SECRET_ACCESS_KEY}"
            ))
        })?;

        if secret_access_key.is_empty() {
            return Err(Error::config(format!(
                "Environment variable {ENV_AWS_SECRET_ACCESS_KEY} is empty"
            )));
        }

        let session_token = std::env::var(ENV_AWS_SESSION_TOKEN)
            .ok()
            .filter(|s| !s.is_empty());

        debug!(
            access_key_id_len = access_key_id.len(),
            has_session_token = session_token.is_some(),
            "Loaded AWS credentials from environment"
        );

        Ok(Self {
            access_key_id,
            secret_access_key,
            session_token,
        })
    }

    /// Returns the AWS access key ID.
    #[must_use]
    pub fn access_key_id(&self) -> &str {
        &self.access_key_id
    }

    /// Returns the AWS secret access key.
    ///
    /// # Security Warning
    ///
    /// This method exposes the secret access key. Use with caution and
    /// avoid logging or displaying the returned value.
    #[must_use]
    pub fn secret_access_key(&self) -> &str {
        &self.secret_access_key
    }

    /// Returns the session token, if present.
    #[must_use]
    pub fn session_token(&self) -> Option<&str> {
        self.session_token.as_deref()
    }

    /// Returns `true` if these are temporary credentials (have a session token).
    #[must_use]
    pub const fn is_temporary(&self) -> bool {
        self.session_token.is_some()
    }
}

// =============================================================================
// AWS Region
// =============================================================================

/// Gets the AWS region from environment variables.
///
/// Checks the following environment variables in order:
/// 1. `AWS_REGION`
/// 2. `AWS_DEFAULT_REGION`
///
/// If neither is set, returns the default region (`us-east-1`).
///
/// # Example
///
/// ```rust
/// use anthropic_bedrock::auth::get_region_from_env;
///
/// let region = get_region_from_env();
/// println!("Using region: {}", region);
/// ```
#[must_use]
pub fn get_region_from_env() -> String {
    std::env::var(ENV_AWS_REGION)
        .or_else(|_| std::env::var(ENV_AWS_DEFAULT_REGION))
        .unwrap_or_else(|_| DEFAULT_REGION.to_string())
}

// =============================================================================
// Request Signing (AWS Signature V4)
// =============================================================================

/// Parameters for signing an HTTP request.
///
/// This struct bundles all the parameters needed for AWS Signature V4 signing.
#[derive(Debug)]
pub struct SigningRequest<'a> {
    /// AWS credentials to use for signing.
    pub credentials: &'a AwsCredentials,
    /// AWS region (e.g., "us-east-1").
    pub region: &'a str,
    /// HTTP method (e.g., "POST").
    pub method: &'a str,
    /// Request URI path (e.g., "/model/anthropic.claude-3-sonnet/invoke").
    pub uri: &'a str,
    /// The host header value.
    pub host: &'a str,
    /// Additional headers to include in the signature.
    pub headers: &'a [(&'a str, &'a str)],
    /// Request body bytes.
    pub body: &'a [u8],
    /// Optional specific signing time (defaults to now).
    pub signing_time: Option<DateTime<Utc>>,
}

/// Signs an HTTP request using AWS Signature V4.
///
/// This function computes the AWS Signature V4 for a request to the Bedrock
/// service and returns the headers that must be added to the request.
///
/// # Arguments
///
/// * `request` - The signing request parameters
///
/// # Returns
///
/// A vector of (header name, header value) pairs that should be added to the request.
///
/// # Errors
///
/// Returns an error if signing fails.
///
/// # Example
///
/// ```rust,ignore
/// use anthropic_bedrock::auth::{AwsCredentials, sign_request, SigningRequest};
///
/// let credentials = AwsCredentials::from_env()?;
/// let request = SigningRequest {
///     credentials: &credentials,
///     region: "us-east-1",
///     method: "POST",
///     uri: "/model/anthropic.claude-3-sonnet/invoke",
///     host: "bedrock-runtime.us-east-1.amazonaws.com",
///     headers: &[("content-type", "application/json")],
///     body: b"{}",
///     signing_time: None,
/// };
/// let signed_headers = sign_request(&request)?;
/// ```
pub fn sign_request(request: &SigningRequest<'_>) -> Result<Vec<(String, String)>> {
    trace!(
        method = request.method,
        uri = request.uri,
        region = request.region,
        body_len = request.body.len(),
        "Signing request"
    );

    let now = request.signing_time.unwrap_or_else(Utc::now);
    let date_time = now.format(DATE_FORMAT).to_string();
    let date = now.format(DATE_FORMAT_SHORT).to_string();

    // Build canonical headers
    let mut canonical_headers: BTreeMap<String, String> = BTreeMap::new();
    canonical_headers.insert("host".to_string(), request.host.to_string());
    canonical_headers.insert("x-amz-date".to_string(), date_time.clone());

    for (name, value) in request.headers {
        canonical_headers.insert((*name).to_lowercase(), (*value).to_string());
    }

    // Add security token if present
    if let Some(token) = &request.credentials.session_token {
        canonical_headers.insert("x-amz-security-token".to_string(), token.clone());
    }

    // Create signed headers list
    let signed_headers: Vec<&str> = canonical_headers.keys().map(String::as_str).collect();
    let signed_headers_str = signed_headers.join(";");

    // Build canonical headers string
    let canonical_headers_str: String = canonical_headers
        .iter()
        .map(|(k, v)| format!("{k}:{}\n", v.trim()))
        .collect();

    // Calculate payload hash
    let payload_hash = sha256_hex(request.body);

    // Build canonical request
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        request.method,
        request.uri,
        "", // query string (empty for Bedrock invoke)
        canonical_headers_str,
        signed_headers_str,
        payload_hash
    );

    trace!(canonical_request = %canonical_request, "Built canonical request");

    // Calculate hash of canonical request
    let canonical_request_hash = sha256_hex(canonical_request.as_bytes());

    // Build credential scope
    let credential_scope = format!("{date}/{}/{SERVICE_NAME}/aws4_request", request.region);

    // Build string to sign
    let string_to_sign =
        format!("{AWS4_ALGORITHM}\n{date_time}\n{credential_scope}\n{canonical_request_hash}");

    trace!(string_to_sign = %string_to_sign, "Built string to sign");

    // Calculate signing key
    let signing_key = derive_signing_key(
        request.credentials.secret_access_key(),
        &date,
        request.region,
    )?;

    // Calculate signature
    let signature = hmac_sha256_hex(&signing_key, string_to_sign.as_bytes())?;

    // Build authorization header
    let authorization = format!(
        "{AWS4_ALGORITHM} Credential={}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}",
        request.credentials.access_key_id(),
    );

    // Build result headers
    let mut result_headers = vec![
        ("authorization".to_string(), authorization),
        ("x-amz-date".to_string(), date_time),
        ("x-amz-content-sha256".to_string(), payload_hash),
    ];

    // Add security token if present
    if let Some(token) = &request.credentials.session_token {
        result_headers.push(("x-amz-security-token".to_string(), token.clone()));
    }

    debug!(
        header_count = result_headers.len(),
        "Request signed successfully"
    );

    Ok(result_headers)
}

/// Derives the signing key for AWS Signature V4.
///
/// The signing key is derived through a series of HMAC operations:
/// 1. HMAC("AWS4" + secret_key, date)
/// 2. HMAC(result1, region)
/// 3. HMAC(result2, service)
/// 4. HMAC(result3, "aws4_request")
fn derive_signing_key(secret_key: &str, date: &str, region: &str) -> Result<Vec<u8>> {
    let k_secret = format!("AWS4{secret_key}");
    let k_date = hmac_sha256(k_secret.as_bytes(), date.as_bytes())?;
    let k_region = hmac_sha256(&k_date, region.as_bytes())?;
    let k_service = hmac_sha256(&k_region, SERVICE_NAME.as_bytes())?;
    let k_signing = hmac_sha256(&k_service, b"aws4_request")?;
    Ok(k_signing)
}

/// Computes SHA-256 hash and returns hex-encoded string.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

/// Computes HMAC-SHA256 and returns raw bytes.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|e| Error::signing(format!("Failed to create HMAC: {e}")))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

/// Computes HMAC-SHA256 and returns hex-encoded string.
fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> Result<String> {
    let result = hmac_sha256(key, data)?;
    Ok(hex::encode(result))
}

/// Represents a signed request ready to be sent.
#[derive(Debug)]
pub struct SignedRequest {
    /// The signed headers to add to the request.
    pub headers: Vec<(String, String)>,
    /// The signing time used.
    pub signed_at: DateTime<Utc>,
}

impl SignedRequest {
    /// Creates a new signed request.
    #[must_use]
    pub const fn new(headers: Vec<(String, String)>, signed_at: DateTime<Utc>) -> Self {
        Self { headers, signed_at }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_new() {
        let creds = AwsCredentials::new(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        );

        assert_eq!(creds.access_key_id(), "AKIAIOSFODNN7EXAMPLE");
        assert_eq!(
            creds.secret_access_key(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        );
        assert!(creds.session_token().is_none());
        assert!(!creds.is_temporary());
    }

    #[test]
    fn test_credentials_with_session_token() {
        let creds = AwsCredentials::new("ASIAXXX", "secret", Some("session-token".to_string()));

        assert!(creds.session_token().is_some());
        assert_eq!(creds.session_token().unwrap(), "session-token");
        assert!(creds.is_temporary());
    }

    #[test]
    fn test_credentials_debug_hides_secrets() {
        let creds = AwsCredentials::new(
            "AKIAIOSFODNN7EXAMPLE",
            "super-secret-key",
            Some("session-token".to_string()),
        );

        let debug_output = format!("{creds:?}");

        // Should contain access key ID (not sensitive)
        assert!(debug_output.contains("AKIAIOSFODNN7EXAMPLE"));

        // Should NOT contain secrets
        assert!(!debug_output.contains("super-secret-key"));
        assert!(!debug_output.contains("session-token"));

        // Should indicate redaction
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn test_get_region_from_env_default() {
        // Clear any existing region env vars for this test
        let _guard = EnvVarGuard::new(&[ENV_AWS_REGION, ENV_AWS_DEFAULT_REGION]);

        let region = get_region_from_env();
        assert_eq!(region, DEFAULT_REGION);
    }

    #[test]
    fn test_get_region_from_env_aws_region() {
        let _guard = EnvVarGuard::new(&[ENV_AWS_REGION, ENV_AWS_DEFAULT_REGION]);
        std::env::set_var(ENV_AWS_REGION, "eu-west-1");

        let region = get_region_from_env();
        assert_eq!(region, "eu-west-1");
    }

    #[test]
    fn test_get_region_from_env_aws_default_region() {
        let _guard = EnvVarGuard::new(&[ENV_AWS_REGION, ENV_AWS_DEFAULT_REGION]);
        std::env::set_var(ENV_AWS_DEFAULT_REGION, "ap-southeast-1");

        let region = get_region_from_env();
        assert_eq!(region, "ap-southeast-1");
    }

    #[test]
    fn test_get_region_aws_region_takes_precedence() {
        let _guard = EnvVarGuard::new(&[ENV_AWS_REGION, ENV_AWS_DEFAULT_REGION]);
        std::env::set_var(ENV_AWS_REGION, "us-west-2");
        std::env::set_var(ENV_AWS_DEFAULT_REGION, "eu-central-1");

        let region = get_region_from_env();
        assert_eq!(region, "us-west-2");
    }

    #[test]
    fn test_sha256_hex() {
        // Known test vector
        let result = sha256_hex(b"");
        assert_eq!(
            result,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let result = sha256_hex(b"hello");
        assert_eq!(
            result,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_hmac_sha256() {
        // Test HMAC computation
        let key = b"secret";
        let data = b"message";
        let result = hmac_sha256(key, data).unwrap();
        assert_eq!(result.len(), 32); // SHA256 produces 32 bytes

        let hex_result = hex::encode(&result);
        assert_eq!(
            hex_result,
            "8b5f48702995c1598c573db1e21866a9b825d4a794d169d7060a03605796360b"
        );
    }

    #[test]
    fn test_sign_request_basic() {
        let creds = AwsCredentials::new(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        );

        let request = SigningRequest {
            credentials: &creds,
            region: "us-east-1",
            method: "POST",
            uri: "/model/anthropic.claude-3-sonnet/invoke",
            host: "bedrock-runtime.us-east-1.amazonaws.com",
            headers: &[("content-type", "application/json")],
            body: b"{}",
            signing_time: None,
        };

        let headers = sign_request(&request);

        assert!(headers.is_ok());
        let headers = headers.unwrap();

        // Should have authorization and x-amz-date headers at minimum
        let header_names: Vec<_> = headers.iter().map(|(n, _)| n.to_lowercase()).collect();
        assert!(header_names.contains(&"authorization".to_string()));
        assert!(header_names.contains(&"x-amz-date".to_string()));
        assert!(header_names.contains(&"x-amz-content-sha256".to_string()));

        // Authorization should have correct format
        let auth_header = headers
            .iter()
            .find(|(n, _)| n.to_lowercase() == "authorization")
            .unwrap();
        assert!(auth_header.1.starts_with("AWS4-HMAC-SHA256"));
        assert!(auth_header.1.contains("Credential="));
        assert!(auth_header.1.contains("SignedHeaders="));
        assert!(auth_header.1.contains("Signature="));
    }

    #[test]
    fn test_sign_request_with_session_token() {
        let creds = AwsCredentials::new("ASIAXXX", "secret", Some("session-token".to_string()));

        let request = SigningRequest {
            credentials: &creds,
            region: "us-east-1",
            method: "POST",
            uri: "/model/anthropic.claude-3-sonnet/invoke",
            host: "bedrock-runtime.us-east-1.amazonaws.com",
            headers: &[("content-type", "application/json")],
            body: b"{}",
            signing_time: None,
        };

        let headers = sign_request(&request);

        assert!(headers.is_ok());
        let headers = headers.unwrap();

        // Should have x-amz-security-token header for session credentials
        let header_names: Vec<_> = headers.iter().map(|(n, _)| n.to_lowercase()).collect();
        assert!(header_names.contains(&"x-amz-security-token".to_string()));

        // Verify the token value
        let token_header = headers
            .iter()
            .find(|(n, _)| n.to_lowercase() == "x-amz-security-token")
            .unwrap();
        assert_eq!(token_header.1, "session-token");
    }

    #[test]
    fn test_sign_request_with_specific_time() {
        use chrono::TimeZone;

        let creds = AwsCredentials::new(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        );

        let signing_time = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();

        let request = SigningRequest {
            credentials: &creds,
            region: "us-east-1",
            method: "POST",
            uri: "/model/anthropic.claude-3-sonnet/invoke",
            host: "bedrock-runtime.us-east-1.amazonaws.com",
            headers: &[("content-type", "application/json")],
            body: b"{}",
            signing_time: Some(signing_time),
        };

        let headers = sign_request(&request);

        assert!(headers.is_ok());
        let headers = headers.unwrap();

        // Check the date header matches our specified time
        let date_header = headers
            .iter()
            .find(|(n, _)| n.to_lowercase() == "x-amz-date")
            .unwrap();
        assert_eq!(date_header.1, "20240115T120000Z");
    }

    #[test]
    fn test_derive_signing_key() {
        // Test the signing key derivation
        let key = derive_signing_key("secret", "20240115", "us-east-1").unwrap();
        assert_eq!(key.len(), 32); // SHA256 output
    }

    #[test]
    fn test_signed_request_struct() {
        let headers = vec![
            (
                "authorization".to_string(),
                "AWS4-HMAC-SHA256...".to_string(),
            ),
            ("x-amz-date".to_string(), "20240101T000000Z".to_string()),
        ];
        let signed_at = Utc::now();

        let signed = SignedRequest::new(headers.clone(), signed_at);
        assert_eq!(signed.headers.len(), 2);
        assert_eq!(signed.signed_at, signed_at);
    }

    // Helper to restore environment variables after tests
    struct EnvVarGuard {
        vars: Vec<(String, Option<String>)>,
    }

    impl EnvVarGuard {
        fn new(var_names: &[&str]) -> Self {
            let vars = var_names
                .iter()
                .map(|name| {
                    let value = std::env::var(*name).ok();
                    std::env::remove_var(*name);
                    (name.to_string(), value)
                })
                .collect();
            Self { vars }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for (name, value) in &self.vars {
                match value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}
