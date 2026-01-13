//! Streaming example.
//!
//! This example demonstrates how to use streaming to receive
//! responses from Claude in real-time, token by token.
//!
//! # Running
//!
//! ```bash
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo run --example streaming
//! ```

use anthropic::{Anthropic, Model};
use anthropic::types::{MessageCreateParams, MessageParam};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    // Create a client
    let client = Anthropic::new()?;

    // Create a streaming request
    let mut stream = client.messages().stream(
        MessageCreateParams::builder()
            .model(Model::claude_sonnet_4_5_latest())
            .max_tokens(1024)
            .messages(vec![
                MessageParam::user("Write a short poem about Rust programming.")
            ])
            .build()?
    ).await?;

    println!("Claude is writing...\n");

    // Process streaming events
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => {
                // Print text deltas as they arrive
                if let Some(text) = event.text_delta() {
                    print!("{}", text);
                }
            }
            Err(e) => {
                eprintln!("\nError: {}", e);
                break;
            }
        }
    }

    println!("\n\nStream complete!");

    Ok(())
}
