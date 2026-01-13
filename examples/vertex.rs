//! Google Vertex AI example.
//!
//! This example demonstrates how to use Claude via Google Cloud Vertex AI.
//!
//! # Prerequisites
//!
//! - Google Cloud credentials configured (ADC or service account)
//! - Access to Claude models on Vertex AI in your GCP project
//!
//! # Running
//!
//! ```bash
//! export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
//! # Or use gcloud auth application-default login
//! cargo run --example vertex
//! ```

use anthropic_vertex::{VertexClient, VertexConfig, CreateMessageRequest, models};

#[tokio::main]
async fn main() -> Result<(), anthropic_vertex::Error> {
    // Create a Vertex client
    // You'll need to set your project ID and region
    let config = VertexConfig::builder()
        .project_id("your-gcp-project-id")
        .region("us-east5")  // Claude is available in specific regions
        .build()?;

    let client = VertexClient::with_config(config).await?;

    println!("Connected to Vertex AI");
    println!("Project: {}", client.project_id());
    println!("Region: {}", client.region());
    println!();

    // Create a message using Claude on Vertex AI
    let response = client.create_message(
        models::CLAUDE_3_5_SONNET_V2,
        CreateMessageRequest::builder(1024)
            .system("You are a helpful assistant running on Google Cloud.")
            .user("What are the benefits of running Claude on Google Vertex AI?")
            .build()
    ).await?;

    // Print the response
    println!("Claude says:\n{}", response.text());
    println!("\nUsage:");
    println!("  Input tokens: {}", response.usage.input_tokens);
    println!("  Output tokens: {}", response.usage.output_tokens);

    Ok(())
}
