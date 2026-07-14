//! Retry logic with exponential backoff for the Anthropic API.
//!
//! This module provides configurable retry behavior for handling transient
//! failures and rate limiting. It implements exponential backoff with jitter
//! to prevent thundering herd problems.
//!
//! # Default Configuration
//!
//! - Max retries: 2
//! - Initial delay: 500ms
//! - Max delay: 8 seconds
//! - Multiplier: 2x
//!
//! # Example
//!
//! ```rust
//! use anthropic::http::{RetryConfig, ExponentialBackoff, RetryPolicy};
//!
//! // Use default configuration
//! let config = RetryConfig::default();
//! let policy = ExponentialBackoff::new(config);
//!
//! // Custom configuration
//! let custom_config = RetryConfig::builder()
//!     .max_retries(5)
//!     .initial_delay_ms(1000)
//!     .max_delay_ms(30000)
//!     .multiplier(1.5)
//!     .build();
//! ```

use crate::error::Error;
use std::time::Duration;

/// Default maximum number of retry attempts.
pub const DEFAULT_MAX_RETRIES: u32 = 2;
/// Default initial delay in milliseconds.
pub const DEFAULT_INITIAL_DELAY_MS: u64 = 500;
/// Default maximum delay in milliseconds.
pub const DEFAULT_MAX_DELAY_MS: u64 = 8000;
/// Default backoff multiplier.
pub const DEFAULT_MULTIPLIER: f64 = 2.0;
/// Maximum jitter factor (0.0 to 1.0).
pub const DEFAULT_JITTER_FACTOR: f64 = 0.25;

/// Configuration for retry behavior.
///
/// This struct defines the parameters for exponential backoff retry logic.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay before the first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier applied to the delay after each retry.
    pub multiplier: f64,
    /// Jitter factor to randomize delays (0.0 to 1.0).
    pub jitter_factor: f64,
    /// Whether to respect the Retry-After header.
    pub respect_retry_after: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            initial_delay: Duration::from_millis(DEFAULT_INITIAL_DELAY_MS),
            max_delay: Duration::from_millis(DEFAULT_MAX_DELAY_MS),
            multiplier: DEFAULT_MULTIPLIER,
            jitter_factor: DEFAULT_JITTER_FACTOR,
            respect_retry_after: true,
        }
    }
}

impl RetryConfig {
    /// Creates a new retry configuration builder.
    #[must_use]
    pub fn builder() -> RetryConfigBuilder {
        RetryConfigBuilder::default()
    }

    /// Creates a configuration with no retries.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Creates a configuration optimized for aggressive retrying.
    ///
    /// This is useful for critical operations that must succeed.
    #[must_use]
    pub const fn aggressive() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter_factor: 0.25,
            respect_retry_after: true,
        }
    }

    /// Creates a configuration optimized for conservative retrying.
    ///
    /// This is useful for operations where you want to fail fast.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            max_retries: 1,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            multiplier: 1.5,
            jitter_factor: 0.1,
            respect_retry_after: true,
        }
    }
}

/// Builder for creating [`RetryConfig`] instances.
#[derive(Debug, Clone, Default)]
pub struct RetryConfigBuilder {
    max_retries: Option<u32>,
    initial_delay_ms: Option<u64>,
    max_delay_ms: Option<u64>,
    multiplier: Option<f64>,
    jitter_factor: Option<f64>,
    respect_retry_after: Option<bool>,
}

impl RetryConfigBuilder {
    /// Sets the maximum number of retry attempts.
    #[must_use]
    pub const fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Sets the initial delay in milliseconds.
    #[must_use]
    pub const fn initial_delay_ms(mut self, ms: u64) -> Self {
        self.initial_delay_ms = Some(ms);
        self
    }

    /// Sets the initial delay as a Duration.
    ///
    /// If `duration` represents more milliseconds than fit in a `u64`
    /// (over 584 million years), the value saturates to `u64::MAX` rather
    /// than silently truncating.
    #[must_use]
    pub fn initial_delay(mut self, duration: Duration) -> Self {
        self.initial_delay_ms = Some(u64::try_from(duration.as_millis()).unwrap_or(u64::MAX));
        self
    }

    /// Sets the maximum delay in milliseconds.
    #[must_use]
    pub const fn max_delay_ms(mut self, ms: u64) -> Self {
        self.max_delay_ms = Some(ms);
        self
    }

    /// Sets the maximum delay as a Duration.
    ///
    /// If `duration` represents more milliseconds than fit in a `u64`
    /// (over 584 million years), the value saturates to `u64::MAX` rather
    /// than silently truncating.
    #[must_use]
    pub fn max_delay(mut self, duration: Duration) -> Self {
        self.max_delay_ms = Some(u64::try_from(duration.as_millis()).unwrap_or(u64::MAX));
        self
    }

    /// Sets the backoff multiplier.
    #[must_use]
    pub const fn multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = Some(multiplier);
        self
    }

    /// Sets the jitter factor (0.0 to 1.0).
    #[must_use]
    pub const fn jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = Some(factor.clamp(0.0, 1.0));
        self
    }

    /// Sets whether to respect the Retry-After header.
    #[must_use]
    pub const fn respect_retry_after(mut self, respect: bool) -> Self {
        self.respect_retry_after = Some(respect);
        self
    }

    /// Builds the retry configuration.
    #[must_use]
    pub fn build(self) -> RetryConfig {
        RetryConfig {
            max_retries: self.max_retries.unwrap_or(DEFAULT_MAX_RETRIES),
            initial_delay: Duration::from_millis(
                self.initial_delay_ms.unwrap_or(DEFAULT_INITIAL_DELAY_MS),
            ),
            max_delay: Duration::from_millis(self.max_delay_ms.unwrap_or(DEFAULT_MAX_DELAY_MS)),
            multiplier: self.multiplier.unwrap_or(DEFAULT_MULTIPLIER),
            jitter_factor: self.jitter_factor.unwrap_or(DEFAULT_JITTER_FACTOR),
            respect_retry_after: self.respect_retry_after.unwrap_or(true),
        }
    }
}

/// Trait for implementing custom retry policies.
///
/// Implement this trait to define custom retry behavior based on the error
/// type and attempt number.
pub trait RetryPolicy: Send + Sync {
    /// Determines whether to retry and how long to wait.
    ///
    /// # Arguments
    ///
    /// * `error` - The error that occurred
    /// * `attempt` - The current attempt number (1-indexed)
    ///
    /// # Returns
    ///
    /// - `Some(Duration)` - Retry after the specified duration
    /// - `None` - Do not retry
    fn should_retry(&self, error: &Error, attempt: u32) -> Option<Duration>;

    /// Returns the maximum number of retries allowed.
    fn max_retries(&self) -> u32;
}

/// Exponential backoff retry policy with jitter.
///
/// This policy implements exponential backoff where the delay between retries
/// grows exponentially, with random jitter to prevent synchronized retries.
///
/// # Algorithm
///
/// The delay for attempt N is calculated as:
/// ```text
/// base_delay = initial_delay * (multiplier ^ (attempt - 1))
/// capped_delay = min(base_delay, max_delay)
/// jitter = random(-jitter_factor, +jitter_factor) * capped_delay
/// final_delay = capped_delay + jitter
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    config: RetryConfig,
}

impl ExponentialBackoff {
    /// Creates a new exponential backoff policy with the given configuration.
    #[must_use]
    pub const fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Calculates the delay for a given attempt number.
    ///
    /// # Arguments
    ///
    /// * `attempt` - The current attempt number (1-indexed)
    /// * `retry_after` - Optional retry-after duration from the server
    ///
    /// # Returns
    ///
    /// The duration to wait before the next retry attempt.
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        // If server provided retry-after and we should respect it
        if self.config.respect_retry_after {
            if let Some(server_delay) = retry_after {
                // Use server delay, but cap it at max_delay
                return server_delay.min(self.config.max_delay);
            }
        }

        // Calculate exponential backoff.
        //
        // `initial_delay`/`max_delay` are retry-backoff durations (expected
        // to be milliseconds to minutes); converting their millisecond count
        // to `f64` for this approximate exponential/jitter arithmetic cannot
        // lose meaningful precision short of a multi-millennia delay, which
        // is not a value any caller configures.
        #[allow(clippy::cast_precision_loss)]
        let initial_delay_ms = self.config.initial_delay.as_millis() as f64;
        // `attempt` is a retry-attempt counter, realistically well under
        // `i32::MAX`; saturate rather than wrap in the pathological case.
        let exponent = i32::try_from(attempt.saturating_sub(1)).unwrap_or(i32::MAX);
        let base_delay_ms = initial_delay_ms * self.config.multiplier.powi(exponent);

        #[allow(clippy::cast_precision_loss)]
        let max_delay_ms = self.config.max_delay.as_millis() as f64;
        let capped_delay_ms = base_delay_ms.min(max_delay_ms);

        // Add jitter
        let jitter_ms = calculate_jitter(capped_delay_ms, self.config.jitter_factor);
        // `capped_delay_ms` is bounded by `max_delay_ms` above and jitter is
        // a fraction of it, so `final_delay_ms` stays well within `u64`
        // range for any realistic config; the `.max(0.0)` just above rules
        // out sign loss, and truncating the fractional part is intentional
        // (we only need whole-millisecond resolution).
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let final_delay_ms = (capped_delay_ms + jitter_ms).max(0.0) as u64;

        Duration::from_millis(final_delay_ms)
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &RetryConfig {
        &self.config
    }
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self::new(RetryConfig::default())
    }
}

impl RetryPolicy for ExponentialBackoff {
    fn should_retry(&self, error: &Error, attempt: u32) -> Option<Duration> {
        // Check if we've exceeded max retries
        if attempt > self.config.max_retries {
            return None;
        }

        // Check if the error is retryable
        if !error.is_retryable() {
            return None;
        }

        // Get retry-after from error if available
        let retry_after = error.retry_after();

        Some(self.calculate_delay(attempt, retry_after))
    }

    fn max_retries(&self) -> u32 {
        self.config.max_retries
    }
}

/// Calculates jitter for a given delay.
///
/// Jitter helps prevent the "thundering herd" problem by randomizing
/// retry times across multiple clients.
///
/// # Arguments
///
/// * `delay_ms` - The base delay in milliseconds
/// * `jitter_factor` - The jitter factor (0.0 to 1.0)
///
/// # Returns
///
/// A random jitter value between -`jitter_factor` * `delay_ms` and +`jitter_factor` * `delay_ms`
fn calculate_jitter(delay_ms: f64, jitter_factor: f64) -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    // Use a simple hash-based random number generator
    // This avoids pulling in the rand crate as a dependency
    let random_state = RandomState::new();
    let mut hasher = random_state.build_hasher();

    // Hash some entropy sources
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    hasher.write_u64(u64::from(std::process::id()));

    let hash = hasher.finish();
    // `hash` is treated as a source of pseudo-randomness, not an exact
    // value; losing low-order precision converting it (and `u64::MAX`) to
    // `f64` has no effect on the intended use (a roughly-uniform jitter
    // factor in [-1.0, 1.0]).
    #[allow(clippy::cast_precision_loss)]
    let random_factor = (hash as f64 / u64::MAX as f64).mul_add(2.0, -1.0); // Range: -1.0 to 1.0

    random_factor * jitter_factor * delay_ms
}

/// A simple retry policy that retries a fixed number of times with a constant delay.
#[derive(Debug, Clone)]
pub struct ConstantBackoff {
    max_retries: u32,
    delay: Duration,
}

impl ConstantBackoff {
    /// Creates a new constant backoff policy.
    #[must_use]
    pub const fn new(max_retries: u32, delay: Duration) -> Self {
        Self { max_retries, delay }
    }
}

impl RetryPolicy for ConstantBackoff {
    fn should_retry(&self, error: &Error, attempt: u32) -> Option<Duration> {
        if attempt > self.max_retries || !error.is_retryable() {
            return None;
        }
        Some(self.delay)
    }

    fn max_retries(&self) -> u32 {
        self.max_retries
    }
}

/// A retry policy that never retries.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRetry;

impl RetryPolicy for NoRetry {
    fn should_retry(&self, _error: &Error, _attempt: u32) -> Option<Duration> {
        None
    }

    fn max_retries(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{ApiError, ApiErrorType};
    use reqwest::StatusCode;

    fn create_retryable_error() -> Error {
        Error::Api(ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            ApiErrorType::RateLimitError,
            "Rate limited",
        ))
    }

    fn create_non_retryable_error() -> Error {
        Error::Api(ApiError::new(
            StatusCode::BAD_REQUEST,
            ApiErrorType::InvalidRequestError,
            "Invalid request",
        ))
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, DEFAULT_MAX_RETRIES);
        assert_eq!(
            config.initial_delay,
            Duration::from_millis(DEFAULT_INITIAL_DELAY_MS)
        );
        assert_eq!(
            config.max_delay,
            Duration::from_millis(DEFAULT_MAX_DELAY_MS)
        );
        assert!((config.multiplier - DEFAULT_MULTIPLIER).abs() < f64::EPSILON);
    }

    #[test]
    fn test_retry_config_builder() {
        let config = RetryConfig::builder()
            .max_retries(5)
            .initial_delay_ms(1000)
            .max_delay_ms(30000)
            .multiplier(1.5)
            .jitter_factor(0.2)
            .build();

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert!((config.multiplier - 1.5).abs() < f64::EPSILON);
        assert!((config.jitter_factor - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_retry_config_disabled() {
        let config = RetryConfig::disabled();
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_exponential_backoff_should_retry() {
        let policy = ExponentialBackoff::default();
        let error = create_retryable_error();

        // Should retry on first attempt
        let delay = policy.should_retry(&error, 1);
        assert!(delay.is_some());

        // Should retry on second attempt
        let delay = policy.should_retry(&error, 2);
        assert!(delay.is_some());

        // Should not retry after max retries exceeded
        let delay = policy.should_retry(&error, 3);
        assert!(delay.is_none());
    }

    #[test]
    fn test_exponential_backoff_non_retryable() {
        let policy = ExponentialBackoff::default();
        let error = create_non_retryable_error();

        // Should not retry non-retryable errors
        let delay = policy.should_retry(&error, 1);
        assert!(delay.is_none());
    }

    #[test]
    fn test_exponential_backoff_delay_calculation() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter_factor: 0.0, // No jitter for predictable testing
            respect_retry_after: true,
        };
        let policy = ExponentialBackoff::new(config);

        // Attempt 1: 1000ms
        let delay1 = policy.calculate_delay(1, None);
        assert_eq!(delay1, Duration::from_secs(1));

        // Attempt 2: 2000ms (1000 * 2^1)
        let delay2 = policy.calculate_delay(2, None);
        assert_eq!(delay2, Duration::from_secs(2));

        // Attempt 3: 4000ms (1000 * 2^2)
        let delay3 = policy.calculate_delay(3, None);
        assert_eq!(delay3, Duration::from_secs(4));
    }

    #[test]
    fn test_exponential_backoff_max_delay_cap() {
        let config = RetryConfig {
            max_retries: 10,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            multiplier: 2.0,
            jitter_factor: 0.0,
            respect_retry_after: true,
        };
        let policy = ExponentialBackoff::new(config);

        // Later attempts should be capped at max_delay
        let delay = policy.calculate_delay(10, None);
        assert_eq!(delay, Duration::from_secs(5));
    }

    #[test]
    fn test_exponential_backoff_respects_retry_after() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter_factor: 0.0,
            respect_retry_after: true,
        };
        let policy = ExponentialBackoff::new(config);

        // Server says wait 10 seconds
        let delay = policy.calculate_delay(1, Some(Duration::from_secs(10)));
        assert_eq!(delay, Duration::from_secs(10));
    }

    #[test]
    fn test_exponential_backoff_caps_retry_after() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(5),
            multiplier: 2.0,
            jitter_factor: 0.0,
            respect_retry_after: true,
        };
        let policy = ExponentialBackoff::new(config);

        // Server says wait 60 seconds, but max is 5
        let delay = policy.calculate_delay(1, Some(Duration::from_secs(60)));
        assert_eq!(delay, Duration::from_secs(5));
    }

    #[test]
    fn test_constant_backoff() {
        let policy = ConstantBackoff::new(3, Duration::from_secs(1));
        let error = create_retryable_error();

        assert_eq!(policy.should_retry(&error, 1), Some(Duration::from_secs(1)));
        assert_eq!(policy.should_retry(&error, 2), Some(Duration::from_secs(1)));
        assert_eq!(policy.should_retry(&error, 3), Some(Duration::from_secs(1)));
        assert_eq!(policy.should_retry(&error, 4), None);
    }

    #[test]
    fn test_no_retry() {
        let policy = NoRetry;
        let error = create_retryable_error();

        assert_eq!(policy.should_retry(&error, 1), None);
        assert_eq!(policy.max_retries(), 0);
    }

    #[test]
    fn test_jitter_within_bounds() {
        // Test that jitter is within expected bounds
        // Due to randomness, we run multiple iterations
        for _ in 0..100 {
            let jitter = calculate_jitter(1000.0, 0.25);
            assert!((-250.0..=250.0).contains(&jitter));
        }
    }
}
