# Anthropic SDK for Rust

[![Crates.io](https://img.shields.io/crates/v/anthropic.svg)](https://crates.io/crates/anthropic)
[![Documentation](https://docs.rs/anthropic/badge.svg)](https://docs.rs/anthropic)
[![CI](https://github.com/anthropics/anthropic-sdk-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/anthropics/anthropic-sdk-rust/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official Rust SDK for the Anthropic API, providing access to Claude and other Anthropic models.

## Features

- Full support for the Messages API
- Streaming responses with async iterators
- Tool use (function calling)
- Message batches for bulk processing
- Extended thinking (beta)
- AWS Bedrock integration
- Google Vertex AI integration
- Async-first design with tokio

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
anthropic = "0.1"
tokio = { version = "1", features = ["full"] }
```

For AWS Bedrock support:
```toml
[dependencies]
anthropic-bedrock = "0.1"
```

For Google Vertex AI support:
```toml
[dependencies]
anthropic-vertex = "0.1"
```

## Quick Start

```rust
use anthropic::{Anthropic, Model};
use anthropic::types::{MessageCreateParams, MessageParam};

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    // Client reads ANTHROPIC_API_KEY from environment
    let client = Anthropic::new()?;

    let response = client.messages().create(
        MessageCreateParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Hello, Claude!")],
            1024,
        )
    ).await?;

    println!("{}", response.text());
    Ok(())
}
```

## Streaming

```rust
use anthropic::{Anthropic, Model};
use anthropic::types::{MessageCreateParams, MessageParam};
use anthropic::streaming::{StreamEvent, Delta};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    let client = Anthropic::new()?;

    let mut stream = client.messages().stream(
        MessageCreateParams::new(
            Model::claude_sonnet_4_5_latest(),
            vec![MessageParam::user("Write a haiku about Rust")],
            1024,
        )
    ).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentBlockDelta { delta, .. } => {
                if let Delta::TextDelta { text } = delta {
                    print!("{}", text);
                }
            }
            _ => {}
        }
    }
    Ok(())
}
```

## Tool Use

```rust
use anthropic::types::{Tool, ToolChoice};
use serde_json::json;

let calculator = Tool::new(
    "calculator",
    "Perform arithmetic calculations",
    json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate"
            }
        },
        "required": ["expression"]
    }),
);

let response = client.messages().create(
    MessageCreateParams::new(
        Model::claude_sonnet_4_5_latest(),
        vec![MessageParam::user("What is 42 * 17?")],
        1024,
    )
    .with_tools(vec![calculator])
    .with_tool_choice(ToolChoice::Auto)
).await?;

// Handle tool use blocks in response.content
for block in &response.content {
    if let ContentBlock::ToolUse(tool_use) = block {
        println!("Tool: {}, Input: {}", tool_use.name, tool_use.input);
    }
}
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| [`anthropic`](https://crates.io/crates/anthropic) | Core SDK with Messages, Batches, and Models APIs |
| [`anthropic-bedrock`](https://crates.io/crates/anthropic-bedrock) | AWS Bedrock integration |
| [`anthropic-vertex`](https://crates.io/crates/anthropic-vertex) | Google Vertex AI integration |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `beta` | Enable beta APIs (Files, Skills, extended thinking) |
| `rustls-tls` | Use rustls for TLS (default) |
| `native-tls` | Use native TLS instead of rustls |

## Examples

See the [`examples/`](./examples) directory for more examples:

- [`simple_message.rs`](./examples/simple_message.rs) - Basic message creation
- [`streaming.rs`](./examples/streaming.rs) - Streaming responses
- [`tool_use.rs`](./examples/tool_use.rs) - Function calling with tools
- [`bedrock.rs`](./examples/bedrock.rs) - AWS Bedrock integration
- [`vertex.rs`](./examples/vertex.rs) - Google Vertex AI integration
- [`beta_files.rs`](./examples/beta_files.rs) - Beta Files API

## Documentation

- [API Documentation](https://docs.rs/anthropic)
- [Anthropic API Reference](https://docs.anthropic.com/en/api)
- [CHANGELOG](./CHANGELOG.md)

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.
