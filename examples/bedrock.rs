//! AWS Bedrock example.
//!
//! This example demonstrates how to use Claude via AWS Bedrock.
//!
//! # Prerequisites
//!
//! - AWS credentials configured (environment variables or AWS config)
//! - Access to Claude models on Bedrock in your AWS account
//!
//! # Running
//!
//! ```bash
//! export AWS_ACCESS_KEY_ID=AKIA...
//! export AWS_SECRET_ACCESS_KEY=...
//! export AWS_REGION=us-east-1
//! cargo run --example bedrock
//! ```

use anthropic_bedrock::{BedrockClient, CreateMessageRequest, models};

#[tokio::main]
async fn main() -> Result<(), anthropic_bedrock::Error> {
    // Create a Bedrock client (uses AWS credentials from environment)
    let client = BedrockClient::new()?;

    println!("Connected to Bedrock in region: {}", client.region());
    println!("Endpoint: {}\n", client.endpoint());

    // Create a message using Claude on Bedrock
    let response = client.create_message(
        models::CLAUDE_3_5_SONNET_V2,
        CreateMessageRequest::builder(1024)
            .system("You are a helpful assistant running on AWS Bedrock.")
            .user("What are the benefits of running Claude on AWS Bedrock?")
            .build()
    ).await?;

    // Print the response
    println!("Claude says:\n{}", response.text());
    println!("\nUsage:");
    println!("  Input tokens: {}", response.usage.input_tokens);
    println!("  Output tokens: {}", response.usage.output_tokens);

    Ok(())
}
