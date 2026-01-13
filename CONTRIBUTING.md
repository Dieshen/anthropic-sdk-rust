# Contributing to anthropic-sdk-rust

Thank you for your interest in contributing to the Anthropic SDK for Rust! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Cargo (comes with Rust)
- Git

### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/anthropic-sdk-rust.git
   cd anthropic-sdk-rust
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/anthropics/anthropic-sdk-rust.git
   ```
4. Build the project:
   ```bash
   cargo build --all-features
   ```

## Development Workflow

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test additions or fixes

### Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes, following the [code style guidelines](#code-style)

3. Run the test suite:
   ```bash
   cargo test --all-features
   ```

4. Run clippy for linting:
   ```bash
   cargo clippy --all-features -- -D warnings
   ```

5. Format your code:
   ```bash
   cargo fmt
   ```

6. Commit your changes with a descriptive message:
   ```bash
   git commit -m "feat: add support for X feature"
   ```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Formatting, no logic change
- `refactor:` - Code restructuring
- `test:` - Adding or fixing tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add Files API support for beta features
fix: handle empty response in streaming
docs: improve README examples
refactor: simplify error handling in client
```

## Code Style

### General Guidelines

1. **Idiomatic Rust**: Follow Rust conventions and idioms
2. **Documentation**: All public APIs must have rustdoc comments
3. **Error Handling**: Use `Result` types, avoid `unwrap()` in library code
4. **No Unsafe Code**: This project uses `#![forbid(unsafe_code)]`
5. **Testing**: Write tests for new functionality

### Documentation Style

```rust
/// Short description of the item.
///
/// Longer description with more details if needed.
/// Can span multiple lines.
///
/// # Arguments
///
/// * `param` - Description of the parameter
///
/// # Returns
///
/// Description of what is returned.
///
/// # Errors
///
/// Description of when errors occur.
///
/// # Examples
///
/// ```rust
/// use anthropic::SomeType;
///
/// let result = some_function();
/// ```
pub fn some_function() -> Result<()> {
    // ...
}
```

### Testing Style

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_name() {
        // Arrange
        let input = ...;

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

## Testing

### Running Tests

```bash
# Run all tests
cargo test --all-features

# Run tests for a specific crate
cargo test -p anthropic --all-features

# Run a specific test
cargo test test_name --all-features

# Run with output
cargo test --all-features -- --nocapture
```

### Test Categories

1. **Unit Tests**: Test individual functions and types
2. **Integration Tests**: Test API interactions (requires API key)
3. **Doc Tests**: Examples in documentation

### Integration Tests

Integration tests require a valid API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cargo test --all-features -- --ignored
```

## Pull Requests

### Before Submitting

1. Ensure all tests pass: `cargo test --all-features`
2. Run clippy: `cargo clippy --all-features -- -D warnings`
3. Format code: `cargo fmt`
4. Update documentation if needed
5. Add tests for new functionality
6. Update CHANGELOG.md for significant changes

### PR Description

Include:
- Summary of changes
- Motivation for the changes
- Any breaking changes
- Related issues (use `Fixes #123` or `Closes #123`)

### Review Process

1. PRs require at least one approval
2. All CI checks must pass
3. Address review feedback
4. Squash commits before merging (if requested)

## Project Structure

```
anthropic-sdk-rust/
├── anthropic/              # Main SDK crate
│   └── src/
│       ├── lib.rs          # Public exports
│       ├── client.rs       # HTTP client
│       ├── config.rs       # Configuration
│       ├── error.rs        # Error types
│       ├── resources/      # API resources
│       ├── streaming/      # Streaming support
│       ├── types/          # Type definitions
│       └── beta/           # Beta features
│
├── anthropic-bedrock/      # AWS Bedrock integration
│   └── src/
│       ├── lib.rs
│       ├── client.rs
│       ├── auth.rs         # AWS signing
│       ├── error.rs
│       └── eventstream.rs  # EventStream decoder
│
├── anthropic-vertex/       # Google Vertex AI integration
│   └── src/
│       ├── lib.rs
│       ├── client.rs
│       ├── auth.rs         # Google OAuth
│       └── error.rs
│
└── examples/               # Example code
```

## Release Process

Releases are managed by maintainers:

1. Update version in all `Cargo.toml` files
2. Update CHANGELOG.md
3. Create a git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. CI will publish to crates.io

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and general discussion
- **Documentation**: API docs and examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to anthropic-sdk-rust!
