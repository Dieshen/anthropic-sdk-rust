//! Simple message example.
//!
//! This example demonstrates the basic usage of the Anthropic SDK
//! to send a message to Claude and receive a response.
//!
//! # Running
//!
//! ```bash
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo run --example simple_message
//! ```

use anthropic::{Anthropic, Model};
use anthropic::types::{MessageCreateParams, MessageParam};

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    // Create a client (uses ANTHROPIC_API_KEY environment variable)
    let client = Anthropic::new()?;

    // Create a simple message
    let response = client.messages().create(
        MessageCreateParams::builder()
            .model(Model::claude_sonnet_4_5_latest())
            .max_tokens(1024)
            .messages(vec![
                MessageParam::user("What is the capital of France?")
            ])
            .build()?
    ).await?;

    // Print the response
    println!("Claude says: {}", response.content_text());
    println!("\nUsage:");
    println!("  Input tokens: {}", response.usage.input_tokens);
    println!("  Output tokens: {}", response.usage.output_tokens);

    Ok(())
}
