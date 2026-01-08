# Anthropic SDK for Rust

The official Rust SDK for the Anthropic API, providing access to Claude and other Anthropic models.

> **Note**: This SDK is currently under development. See [PLAN.md](./PLAN.md) for the implementation roadmap.

## Features

- Full support for the Messages API
- Streaming responses with async iterators
- Tool use (function calling)
- Message batches for bulk processing
- AWS Bedrock integration
- Google Vertex AI integration
- Async-first design with runtime flexibility (tokio/async-std)

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
anthropic = { version = "0.1", features = ["bedrock"] }
```

For Google Vertex AI support:
```toml
[dependencies]
anthropic = { version = "0.1", features = ["vertex"] }
```

## Quick Start

```rust
use anthropic::{Anthropic, Model};
use anthropic::types::MessageCreateParams;

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    // Client reads ANTHROPIC_API_KEY from environment
    let client = Anthropic::new()?;

    let response = client.messages().create(
        MessageCreateParams::builder()
            .model(Model::CLAUDE_SONNET_4_5_LATEST)
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

## Streaming

```rust
use futures::StreamExt;

let mut stream = client.messages().stream(
    MessageCreateParams::builder()
        .model(Model::CLAUDE_SONNET_4_5_LATEST)
        .max_tokens(1024)
        .messages(vec![MessageParam::user("Write a haiku about Rust")])
        .build()?
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
```

## Crate Structure

| Crate               | Description                                      |
| ------------------- | ------------------------------------------------ |
| `anthropic`         | Core SDK with Messages, Batches, and Models APIs |
| `anthropic-bedrock` | AWS Bedrock integration                          |
| `anthropic-vertex`  | Google Vertex AI integration                     |

## Development Status

See [PLAN.md](./PLAN.md) for the detailed implementation plan and current progress.

### Phases

- [ ] Phase 1: Core Infrastructure
- [ ] Phase 2: Core Types
- [ ] Phase 3: Messages API
- [ ] Phase 4: Tool Use
- [ ] Phase 5: Batches API
- [ ] Phase 6: Models API
- [ ] Phase 7: Beta Features
- [ ] Phase 8: Cloud Integrations
- [ ] Phase 9: Polish & Documentation

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Contributing

Contributions are welcome! Please see the development plan in [PLAN.md](./PLAN.md) before starting work.
