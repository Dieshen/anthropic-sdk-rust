# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-13

### Added

#### Core SDK (`anthropic`)
- **Messages API**: Full support for creating messages with Claude models
  - Streaming and non-streaming responses
  - Tool use (function calling) support
  - System prompts and multi-turn conversations
  - Image and document inputs (Base64, URL, file reference)
  - PDF file support with page-level processing
- **Message Batches API**: Batch processing for high-volume workloads
  - Create, retrieve, list, and cancel batches
  - Stream batch results
  - Auto-pagination for large result sets
- **Models API**: Retrieve model information
  - Get individual model details
  - List all available models with pagination
- **Token Counting API**: Count tokens before making requests
- **Streaming**: Full Server-Sent Events (SSE) streaming support
  - Content block deltas
  - Tool use streaming
  - Thinking blocks streaming
- **Error Handling**: Comprehensive error types with retry guidance
  - Rate limit detection with retry-after headers
  - Automatic retry with exponential backoff
  - Request ID tracking for debugging

#### Beta Features (`beta` feature flag)
- **Extended Thinking**: Support for Claude's thinking/reasoning output
  - Configurable thinking budget
  - Interleaved thinking with responses
- **Web Search Tool**: Server-side web search capability
- **Computer Use Tools**: Bash, Text Editor, and Computer control tools
- **Code Execution Tool**: Sandboxed code execution
- **Files API**: Server-side file storage and management
  - Upload files (text, images, PDFs)
  - Download, list, and delete files
  - File references in messages
- **Skills API**: Reusable functionality packages
  - Create and manage skills
  - Version management

#### AWS Bedrock Integration (`anthropic-bedrock`)
- **BedrockClient**: Full Claude access via AWS Bedrock
  - AWS Signature V4 authentication
  - All Claude models on Bedrock
  - Streaming support
- **EventStream Decoder**: Parse AWS EventStream binary format
  - CRC validation
  - Header parsing
  - Bedrock chunk decoding (base64 payloads)
- **Credential Management**: Multiple authentication methods
  - Environment variables
  - Explicit credentials
  - Session tokens for temporary credentials

#### Google Vertex AI Integration (`anthropic-vertex`)
- **VertexClient**: Full Claude access via Google Cloud
  - OAuth2 authentication
  - Service account and ADC support
  - Regional endpoints
- **Token Management**: Automatic token refresh
- **Streaming**: Full streaming support for Vertex AI

### Technical Details

- **Rust Edition**: 2021
- **MSRV**: 1.75
- **Async Runtime**: Tokio-based with `reqwest` HTTP client
- **No unsafe code**: `#![forbid(unsafe_code)]` enforced
- **Full documentation**: All public APIs documented with examples
- **Comprehensive tests**: 344+ unit tests across all crates

### Dependencies

Core dependencies kept minimal:
- `serde` / `serde_json` for serialization
- `reqwest` for HTTP (with rustls)
- `tokio` for async runtime
- `thiserror` for error handling
- `tracing` for logging

---

## Release Notes

### v0.1.0 - Initial Release

This is the initial release of the Anthropic SDK for Rust. It provides full feature
parity with the official Go and TypeScript SDKs, with idiomatic Rust patterns.

**Highlights:**
- Type-safe API with builder patterns
- Comprehensive streaming support
- AWS Bedrock and Google Vertex AI integrations
- Beta features behind feature flags
- Production-ready with 344+ tests

**Example:**
```rust
use anthropic::{Anthropic, Model};
use anthropic::types::{MessageCreateParams, MessageParam};

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    let client = Anthropic::new()?;

    let response = client.messages().create(
        MessageCreateParams::builder()
            .model(Model::claude_sonnet_4_5_latest())
            .max_tokens(1024)
            .messages(vec![
                MessageParam::user("Hello, Claude!")
            ])
            .build()?
    ).await?;

    println!("{}", response.content_text());
    Ok(())
}
```

[Unreleased]: https://github.com/anthropics/anthropic-sdk-rust/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anthropics/anthropic-sdk-rust/releases/tag/v0.1.0
