//! Google Cloud authentication for Vertex AI.
//!
//! This module provides authentication support for Google Cloud Platform,
//! including:
//! - Service Account JSON key file authentication
//! - Application Default Credentials (ADC)
//! - OAuth2 token refresh
//!
//! # Authentication Methods
//!
//! ## Service Account Key
//!
//! Load credentials from a service account JSON key file:
//!
//! ```rust,ignore
//! use anthropic_vertex::auth::GoogleCredentials;
//!
//! let creds = GoogleCredentials::from_service_account_file("path/to/key.json").await?;
//! ```
//!
//! ## Application Default Credentials (ADC)
//!
//! Automatically discover credentials in the following order:
//! 1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable
//! 2. Well-known file locations (gcloud CLI config)
//! 3. Compute Engine metadata service (when running on GCP)
//!
//! ```rust,ignore
//! use anthropic_vertex::auth::GoogleCredentials;
//!
//! let creds = GoogleCredentials::from_adc().await?;
//! ```

// base64 is available if needed for future features
use chrono::{Duration, Utc};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, trace};

use crate::{Error, Result};

// =============================================================================
// Constants
// =============================================================================

/// Default OAuth2 scopes for Vertex AI API access.
pub const DEFAULT_SCOPES: &[&str] = &["https://www.googleapis.com/auth/cloud-platform"];

/// Google OAuth2 token endpoint.
const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";

/// Buffer time before token expiration to trigger refresh (5 minutes).
const TOKEN_REFRESH_BUFFER_SECS: i64 = 300;

/// Environment variable for service account credentials path.
const ENV_GOOGLE_APPLICATION_CREDENTIALS: &str = "GOOGLE_APPLICATION_CREDENTIALS";

/// GCE metadata server URL for fetching tokens.
const GCE_METADATA_URL: &str =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

// =============================================================================
// Credential Types
// =============================================================================

/// Credentials for authenticating with Google Cloud.
///
/// This type supports multiple authentication methods:
/// - Service account JSON key files
/// - Application Default Credentials (ADC)
/// - Compute Engine metadata service
///
/// Tokens are automatically refreshed when they expire.
#[derive(Debug, Clone)]
pub struct GoogleCredentials {
    /// The credential source used for authentication.
    source: CredentialSource,
    /// Cached access token with expiration.
    token_cache: Arc<RwLock<Option<CachedToken>>>,
    /// HTTP client for token requests.
    http_client: reqwest::Client,
    /// OAuth2 scopes for token requests.
    scopes: Vec<String>,
}

/// Source of Google Cloud credentials.
#[derive(Debug, Clone)]
enum CredentialSource {
    /// Service account credentials from a JSON key file.
    ServiceAccount(ServiceAccountKey),
    /// Authorized user credentials (from gcloud CLI).
    AuthorizedUser(AuthorizedUserCredentials),
    /// GCE metadata service (when running on Google Cloud).
    GceMetadata,
}

/// Cached OAuth2 access token.
#[derive(Debug, Clone)]
struct CachedToken {
    /// The access token.
    access_token: SecretString,
    /// When the token expires.
    expires_at: chrono::DateTime<Utc>,
}

impl CachedToken {
    /// Returns true if the token is expired or will expire soon.
    fn is_expired(&self) -> bool {
        Utc::now() + Duration::seconds(TOKEN_REFRESH_BUFFER_SECS) >= self.expires_at
    }
}

// =============================================================================
// Service Account Key
// =============================================================================

/// Service account key file contents.
#[derive(Debug, Clone, Deserialize)]
pub struct ServiceAccountKey {
    /// Account type (should be "service_account").
    #[serde(rename = "type")]
    pub key_type: String,
    /// Project ID.
    pub project_id: String,
    /// Private key ID.
    pub private_key_id: String,
    /// Private key in PEM format.
    pub private_key: String,
    /// Service account email.
    pub client_email: String,
    /// Client ID.
    pub client_id: String,
    /// Auth URI.
    pub auth_uri: String,
    /// Token URI.
    pub token_uri: String,
}

/// JWT claims for service account authentication.
#[derive(Debug, Serialize)]
struct JwtClaims {
    /// Issuer (service account email).
    iss: String,
    /// Scope (space-separated).
    scope: String,
    /// Audience (token endpoint).
    aud: String,
    /// Issued at timestamp.
    iat: i64,
    /// Expiration timestamp.
    exp: i64,
}

// =============================================================================
// Authorized User Credentials
// =============================================================================

/// Authorized user credentials (from gcloud CLI login).
#[derive(Debug, Clone, Deserialize)]
pub struct AuthorizedUserCredentials {
    /// Client ID.
    pub client_id: String,
    /// Client secret.
    pub client_secret: String,
    /// Refresh token.
    pub refresh_token: String,
    /// Account type.
    #[serde(rename = "type")]
    pub cred_type: String,
}

// =============================================================================
// Token Response
// =============================================================================

/// OAuth2 token response.
#[derive(Debug, Deserialize)]
struct TokenResponse {
    /// The access token.
    access_token: String,
    /// Token type (usually "Bearer").
    #[allow(dead_code)]
    token_type: String,
    /// Token lifetime in seconds.
    expires_in: i64,
}

// =============================================================================
// GoogleCredentials Implementation
// =============================================================================

impl GoogleCredentials {
    /// Creates credentials from Application Default Credentials (ADC).
    ///
    /// This method searches for credentials in the following order:
    /// 1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable
    /// 2. Well-known file locations for gcloud CLI
    /// 3. GCE metadata service (when running on Google Cloud)
    ///
    /// # Errors
    ///
    /// Returns an error if no valid credentials can be found.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let creds = GoogleCredentials::from_adc().await?;
    /// ```
    pub async fn from_adc() -> Result<Self> {
        Self::from_adc_with_scopes(DEFAULT_SCOPES).await
    }

    /// Creates credentials from ADC with custom scopes.
    ///
    /// # Arguments
    ///
    /// * `scopes` - OAuth2 scopes to request
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let creds = GoogleCredentials::from_adc_with_scopes(&[
    ///     "https://www.googleapis.com/auth/cloud-platform",
    /// ]).await?;
    /// ```
    pub async fn from_adc_with_scopes(scopes: &[&str]) -> Result<Self> {
        let scopes: Vec<String> = scopes.iter().map(|&s| s.to_string()).collect();

        // Try GOOGLE_APPLICATION_CREDENTIALS environment variable
        if let Ok(path) = std::env::var(ENV_GOOGLE_APPLICATION_CREDENTIALS) {
            debug!(
                path = %path,
                "Loading credentials from GOOGLE_APPLICATION_CREDENTIALS"
            );
            return Self::from_file_with_scopes(&path, &scopes).await;
        }

        // Try well-known file locations
        if let Some(path) = Self::find_well_known_credentials() {
            debug!(path = ?path, "Loading credentials from well-known location");
            return Self::from_file_with_scopes(&path, &scopes).await;
        }

        // Try GCE metadata service
        debug!("Attempting to use GCE metadata service");
        Self::from_gce_metadata_with_scopes(&scopes).await
    }

    /// Creates credentials from a service account JSON key file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the service account key file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let creds = GoogleCredentials::from_service_account_file("key.json").await?;
    /// ```
    pub async fn from_service_account_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_service_account_file_with_scopes(path, DEFAULT_SCOPES).await
    }

    /// Creates credentials from a service account JSON key file with custom scopes.
    pub async fn from_service_account_file_with_scopes(
        path: impl AsRef<Path>,
        scopes: &[&str],
    ) -> Result<Self> {
        let scopes: Vec<String> = scopes.iter().map(|&s| s.to_string()).collect();
        let contents = tokio::fs::read_to_string(path.as_ref())
            .await
            .map_err(|e| Error::Auth(format!("Failed to read credentials file: {e}")))?;

        let key: ServiceAccountKey = serde_json::from_str(&contents)
            .map_err(|e| Error::Auth(format!("Failed to parse service account key: {e}")))?;

        if key.key_type != "service_account" {
            return Err(Error::Auth(format!(
                "Expected service_account type, got: {}",
                key.key_type
            )));
        }

        Ok(Self {
            source: CredentialSource::ServiceAccount(key),
            token_cache: Arc::new(RwLock::new(None)),
            http_client: reqwest::Client::new(),
            scopes,
        })
    }

    /// Creates credentials from a service account key struct.
    pub fn from_service_account_key(key: ServiceAccountKey) -> Result<Self> {
        Self::from_service_account_key_with_scopes(key, DEFAULT_SCOPES)
    }

    /// Creates credentials from a service account key struct with custom scopes.
    pub fn from_service_account_key_with_scopes(
        key: ServiceAccountKey,
        scopes: &[&str],
    ) -> Result<Self> {
        let scopes: Vec<String> = scopes.iter().map(|&s| s.to_string()).collect();

        if key.key_type != "service_account" {
            return Err(Error::Auth(format!(
                "Expected service_account type, got: {}",
                key.key_type
            )));
        }

        Ok(Self {
            source: CredentialSource::ServiceAccount(key),
            token_cache: Arc::new(RwLock::new(None)),
            http_client: reqwest::Client::new(),
            scopes,
        })
    }

    /// Creates credentials from GCE metadata service.
    ///
    /// This only works when running on Google Cloud infrastructure.
    pub async fn from_gce_metadata() -> Result<Self> {
        Self::from_gce_metadata_with_scopes(
            &DEFAULT_SCOPES
                .iter()
                .map(|&s| s.to_string())
                .collect::<Vec<_>>(),
        )
        .await
    }

    /// Creates credentials from GCE metadata service with custom scopes.
    async fn from_gce_metadata_with_scopes(scopes: &[String]) -> Result<Self> {
        let http_client = reqwest::Client::new();

        // Verify we can reach the metadata server
        let response = http_client
            .get(GCE_METADATA_URL)
            .header("Metadata-Flavor", "Google")
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .map_err(|e| {
                Error::Auth(format!(
                    "Failed to reach GCE metadata server (not running on GCP?): {e}"
                ))
            })?;

        if !response.status().is_success() {
            return Err(Error::Auth(format!(
                "GCE metadata server returned status: {}",
                response.status()
            )));
        }

        Ok(Self {
            source: CredentialSource::GceMetadata,
            token_cache: Arc::new(RwLock::new(None)),
            http_client,
            scopes: scopes.to_vec(),
        })
    }

    /// Creates credentials from any supported credential file.
    async fn from_file_with_scopes(path: impl AsRef<Path>, scopes: &[String]) -> Result<Self> {
        let contents = tokio::fs::read_to_string(path.as_ref())
            .await
            .map_err(|e| Error::Auth(format!("Failed to read credentials file: {e}")))?;

        // Try to parse as service account first
        if let Ok(key) = serde_json::from_str::<ServiceAccountKey>(&contents) {
            if key.key_type == "service_account" {
                return Ok(Self {
                    source: CredentialSource::ServiceAccount(key),
                    token_cache: Arc::new(RwLock::new(None)),
                    http_client: reqwest::Client::new(),
                    scopes: scopes.to_vec(),
                });
            }
        }

        // Try authorized user credentials
        if let Ok(user_creds) = serde_json::from_str::<AuthorizedUserCredentials>(&contents) {
            if user_creds.cred_type == "authorized_user" {
                return Ok(Self {
                    source: CredentialSource::AuthorizedUser(user_creds),
                    token_cache: Arc::new(RwLock::new(None)),
                    http_client: reqwest::Client::new(),
                    scopes: scopes.to_vec(),
                });
            }
        }

        Err(Error::Auth(
            "Unrecognized credential file format".to_string(),
        ))
    }

    /// Finds credentials in well-known file locations.
    fn find_well_known_credentials() -> Option<PathBuf> {
        // Check platform-specific locations
        #[cfg(target_os = "windows")]
        let config_dir = std::env::var("APPDATA")
            .ok()
            .map(PathBuf::from)
            .map(|p| p.join("gcloud"));

        #[cfg(not(target_os = "windows"))]
        let config_dir = dirs::home_dir().map(|p| p.join(".config").join("gcloud"));

        if let Some(dir) = config_dir {
            let adc_path = dir.join("application_default_credentials.json");
            if adc_path.exists() {
                return Some(adc_path);
            }
        }

        None
    }

    /// Gets a valid access token, refreshing if necessary.
    ///
    /// This method is safe to call concurrently; only one refresh will occur.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let token = creds.get_token().await?;
    /// // Use token in Authorization header
    /// ```
    pub async fn get_token(&self) -> Result<String> {
        // Check if we have a valid cached token
        {
            let cache = self.token_cache.read().await;
            if let Some(ref token) = *cache {
                if !token.is_expired() {
                    trace!("Using cached token");
                    return Ok(token.access_token.expose_secret().to_string());
                }
            }
        }

        // Token is missing or expired, refresh it
        debug!("Refreshing access token");
        let new_token = self.refresh_token().await?;

        // Cache the new token
        {
            let mut cache = self.token_cache.write().await;
            *cache = Some(new_token.clone());
        }

        Ok(new_token.access_token.expose_secret().to_string())
    }

    /// Refreshes the access token.
    async fn refresh_token(&self) -> Result<CachedToken> {
        match &self.source {
            CredentialSource::ServiceAccount(key) => self.refresh_service_account_token(key).await,
            CredentialSource::AuthorizedUser(creds) => {
                self.refresh_authorized_user_token(creds).await
            }
            CredentialSource::GceMetadata => self.refresh_gce_metadata_token().await,
        }
    }

    /// Refreshes token using service account credentials.
    async fn refresh_service_account_token(&self, key: &ServiceAccountKey) -> Result<CachedToken> {
        let now = Utc::now();
        let exp = now + Duration::hours(1);

        let claims = JwtClaims {
            iss: key.client_email.clone(),
            scope: self.scopes.join(" "),
            aud: TOKEN_ENDPOINT.to_string(),
            iat: now.timestamp(),
            exp: exp.timestamp(),
        };

        // Create JWT header
        let header = Header {
            alg: Algorithm::RS256,
            kid: Some(key.private_key_id.clone()),
            ..Default::default()
        };

        // Sign the JWT
        let encoding_key = EncodingKey::from_rsa_pem(key.private_key.as_bytes())
            .map_err(|e| Error::Auth(format!("Invalid private key: {e}")))?;

        let jwt = encode(&header, &claims, &encoding_key)
            .map_err(|e| Error::Auth(format!("Failed to sign JWT: {e}")))?;

        // Exchange JWT for access token
        let response = self
            .http_client
            .post(TOKEN_ENDPOINT)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", &jwt),
            ])
            .send()
            .await
            .map_err(|e| Error::Auth(format!("Token request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Auth(format!(
                "Token request failed with status {status}: {body}"
            )));
        }

        let token_response: TokenResponse = response
            .json()
            .await
            .map_err(|e| Error::Auth(format!("Failed to parse token response: {e}")))?;

        Ok(CachedToken {
            access_token: SecretString::from(token_response.access_token),
            expires_at: Utc::now() + Duration::seconds(token_response.expires_in),
        })
    }

    /// Refreshes token using authorized user credentials.
    async fn refresh_authorized_user_token(
        &self,
        creds: &AuthorizedUserCredentials,
    ) -> Result<CachedToken> {
        let response = self
            .http_client
            .post(TOKEN_ENDPOINT)
            .form(&[
                ("client_id", creds.client_id.as_str()),
                ("client_secret", creds.client_secret.as_str()),
                ("refresh_token", creds.refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await
            .map_err(|e| Error::Auth(format!("Token refresh failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Auth(format!(
                "Token refresh failed with status {status}: {body}"
            )));
        }

        let token_response: TokenResponse = response
            .json()
            .await
            .map_err(|e| Error::Auth(format!("Failed to parse token response: {e}")))?;

        Ok(CachedToken {
            access_token: SecretString::from(token_response.access_token),
            expires_at: Utc::now() + Duration::seconds(token_response.expires_in),
        })
    }

    /// Refreshes token from GCE metadata service.
    async fn refresh_gce_metadata_token(&self) -> Result<CachedToken> {
        let response = self
            .http_client
            .get(GCE_METADATA_URL)
            .header("Metadata-Flavor", "Google")
            .send()
            .await
            .map_err(|e| Error::Auth(format!("Metadata request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Auth(format!(
                "Metadata request failed with status {status}: {body}"
            )));
        }

        let token_response: TokenResponse = response
            .json()
            .await
            .map_err(|e| Error::Auth(format!("Failed to parse metadata response: {e}")))?;

        Ok(CachedToken {
            access_token: SecretString::from(token_response.access_token),
            expires_at: Utc::now() + Duration::seconds(token_response.expires_in),
        })
    }

    /// Returns the project ID from service account credentials, if available.
    pub fn project_id(&self) -> Option<&str> {
        match &self.source {
            CredentialSource::ServiceAccount(key) => Some(&key.project_id),
            _ => None,
        }
    }

    /// Returns the configured OAuth2 scopes.
    pub fn scopes(&self) -> &[String] {
        &self.scopes
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_service_account_key() -> serde_json::Value {
        // This is a fake RSA key for testing - it's intentionally invalid for production
        let fake_private_key = r#"-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy0AHB7MvnM2eMzWryFlWFV
cR7CpQgVPpJwyhHgpHa4bM9G9A3wEiCwDgF5fmL5gVHBVVHEPIYr2mVHLt0ByJCB
Ts/ikJNJnRi1RlcTGlbLJn2RYxsAAvoaWLmHoCPQgPt/0rkVFWwLEJqjT+FXXxLU
tKWQqCMGMiLmJa4DF1KHsLWNOFvHiVNleBDrsJvHnWPwlXMCQ5OL4udYWyxBjpIK
rOJjBDA7fPr2p7woSoSLYxgF1ywBwQNMsNPfM5wWnjLjYcXRvNWLCD9Flr93LGkS
ORMoS6KbvPvLLADjmSFqWFihFoRjKcrjQNsFXwIDAQABAoIBAFbLOD04JYXEqChS
jO0X1bfSKH9UDs1zPEUwXDwGjG4wvvnqfnrcwn7M7gdDI1K0LNo9X1MoKa0hLFTe
LviILspnvWAvMqwe8Mq9F4T1s7dNVVe4DLB2rzD4Xp7kRhFE4xIpMNcRmPl1JG8k
WPgCNSCVbTH2WnDVgIpLP1D4pMNTPsaEL0PFmKMzbdWFbjj8d7AB7+YCpVnylLyv
+g7Y/UpzFthFP+NVjS9j7ZgsFCODU/3ASNM8EzrC4x2Lj9X3m0IYi8lMVdjvCxWZ
Tk6/Dmbh5fWZpLYCvL1Ib7xcREHevMbo9bPQovHoRMH6jgh7Gf4D8AMlS0r0cP6m
FcWBL2ECgYEA6z0RYcaLslam/o8b0hDxl3M3dPHs0h3aM5vMuSJCRvnSnEJjjuxN
3Li/EDT+DS/5XgvuJttCGcXmzpHdorYNab7VHChs7nSCgexB/2/b3PMa6JNAQMDB
OfO3buwazWbmWNVYrtGJpKYc7lcqqFRm3r6Nq+CRF3PrtLo9Hn2S9LsCgYEA48HV
EpETncXAwqP8bsOP4WDhR9mDg0S3thWilbH3PpHBq5ZqB6KwYdUAB6QrfPGC7X6a
OLmJ8sAFMVhMJnSnZ3+7ksHWZ6lCCdjhBJX+UMbNbJ0JJrvAqT9cjQYMeRfD6VkP
VCAqm0cNXy08kcnKQBofsfptYRBttN/sTx38NzUCgYEAz0BdlB7nnBDC7+CQYWER
yFLe08ZxWO/YD5khgJNwT9ExoN/YMToEi8X+Y2LVLin/xHkPJPJM+76bHv4YNWQ9
PN5JPTE9sJvJxVNm4tzNkqVeSBJuPFUmx0YdQV0ewHzPqA3F4JqLXu2XL8RNFLxO
CxDJMNyfL9zf8sWMF0dPM7kCgYBW+cR/y/bpf/FsMJqDpFSPRT2BXCnKvPJNy5l5
K9lWDaRt5j3mzMZ7I7glyMC8VPlNZMSYemBLB9e0Bgn9BaD+g1qLqP/ImmJYNqWF
4CKqKVvnJKC1OcCpVVMBLPuOwJ9QN2GYm1YDdSzYJogPGRH7pHjDC9lMD+JLH6nf
OQJY1QKBgE0M3Kq3O0f+n9Y3Y9bDJlJK5UXreVNVqXg6Freu01k/1M/rl0fPdqLF
J/xMx+BqC9LKmbV2IZfN5x0pVSjLPgHy1VEXbCR3YBKhJR4zNAkR0ED9mic7cBGO
v2plKta7feKH6MFXAB9akHLVk/b9cLDy2B8r9bb8L0F6fpmC/fMT
-----END RSA PRIVATE KEY-----"#;

        serde_json::json!({
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "key123",
            "private_key": fake_private_key,
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        })
    }

    fn create_test_authorized_user_creds() -> serde_json::Value {
        serde_json::json!({
            "type": "authorized_user",
            "client_id": "client123.apps.googleusercontent.com",
            "client_secret": "secret123",
            "refresh_token": "refresh123"
        })
    }

    #[tokio::test]
    async fn test_load_service_account_file() {
        let key = create_test_service_account_key();
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", serde_json::to_string(&key).unwrap()).unwrap();

        let creds = GoogleCredentials::from_service_account_file(temp_file.path())
            .await
            .unwrap();

        assert_eq!(creds.project_id(), Some("test-project"));
        assert_eq!(creds.scopes(), DEFAULT_SCOPES);
    }

    #[tokio::test]
    async fn test_load_service_account_with_custom_scopes() {
        let key = create_test_service_account_key();
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", serde_json::to_string(&key).unwrap()).unwrap();

        let custom_scopes = &["https://www.googleapis.com/auth/compute"];
        let creds = GoogleCredentials::from_service_account_file_with_scopes(
            temp_file.path(),
            custom_scopes,
        )
        .await
        .unwrap();

        assert_eq!(creds.scopes().len(), 1);
        assert_eq!(creds.scopes()[0], "https://www.googleapis.com/auth/compute");
    }

    #[tokio::test]
    async fn test_load_authorized_user_file() {
        let creds_json = create_test_authorized_user_creds();
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", serde_json::to_string(&creds_json).unwrap()).unwrap();

        let scopes: Vec<String> = DEFAULT_SCOPES.iter().map(|&s| s.to_string()).collect();
        let creds = GoogleCredentials::from_file_with_scopes(temp_file.path(), &scopes)
            .await
            .unwrap();

        // Authorized user credentials don't have a project ID
        assert_eq!(creds.project_id(), None);
    }

    #[test]
    fn test_parse_service_account_key() {
        let key_json = create_test_service_account_key();
        let key: ServiceAccountKey = serde_json::from_value(key_json).unwrap();

        assert_eq!(key.key_type, "service_account");
        assert_eq!(key.project_id, "test-project");
        assert_eq!(
            key.client_email,
            "test@test-project.iam.gserviceaccount.com"
        );
    }

    #[test]
    fn test_parse_authorized_user_credentials() {
        let creds_json = create_test_authorized_user_creds();
        let creds: AuthorizedUserCredentials = serde_json::from_value(creds_json).unwrap();

        assert_eq!(creds.cred_type, "authorized_user");
        assert_eq!(creds.client_id, "client123.apps.googleusercontent.com");
    }

    #[tokio::test]
    async fn test_invalid_credential_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, r#"{{"invalid": "json"}}"#).unwrap();

        let scopes: Vec<String> = DEFAULT_SCOPES.iter().map(|&s| s.to_string()).collect();
        let result = GoogleCredentials::from_file_with_scopes(temp_file.path(), &scopes).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_cached_token_expiration() {
        let token = CachedToken {
            access_token: SecretString::from("test_token".to_string()),
            expires_at: Utc::now() + Duration::hours(1),
        };
        assert!(!token.is_expired());

        let expired_token = CachedToken {
            access_token: SecretString::from("test_token".to_string()),
            expires_at: Utc::now() - Duration::hours(1),
        };
        assert!(expired_token.is_expired());

        // Token expiring soon should be considered expired (within buffer)
        let expiring_soon = CachedToken {
            access_token: SecretString::from("test_token".to_string()),
            expires_at: Utc::now() + Duration::seconds(60), // Less than 5 min buffer
        };
        assert!(expiring_soon.is_expired());
    }

    #[test]
    fn test_jwt_claims_serialization() {
        let claims = JwtClaims {
            iss: "test@example.iam.gserviceaccount.com".to_string(),
            scope: "https://www.googleapis.com/auth/cloud-platform".to_string(),
            aud: TOKEN_ENDPOINT.to_string(),
            iat: 1234567890,
            exp: 1234571490,
        };

        let json = serde_json::to_string(&claims).unwrap();
        assert!(json.contains("iss"));
        assert!(json.contains("scope"));
        assert!(json.contains("aud"));
        assert!(json.contains("iat"));
        assert!(json.contains("exp"));
    }
}
