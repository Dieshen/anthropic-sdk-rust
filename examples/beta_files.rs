//! Beta Files API example.
//!
//! This example demonstrates how to use the beta Files API
//! to upload, list, and manage files on the Anthropic server.
//!
//! # Running
//!
//! ```bash
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo run --example beta_files --features beta
//! ```

use anthropic::Anthropic;
use anthropic::beta::{FileUploadParams, FilesListParams};

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    // Create a client
    let client = Anthropic::new()?;

    // Access the beta Files API
    let files = client.beta().files();

    // Upload a text file
    println!("Uploading a text file...");
    let file = files.upload(FileUploadParams::text(
        "example.txt",
        b"Hello, this is a test file!".to_vec(),
    )).await?;

    println!("Uploaded file:");
    println!("  ID: {}", file.id);
    println!("  Name: {}", file.filename);
    println!("  Size: {} bytes", file.size_bytes);
    println!("  Created: {}", file.created_at);

    // List all files
    println!("\nListing all files...");
    let list_response = files.list(FilesListParams::new().limit(10)).await?;

    println!("Found {} files:", list_response.data.len());
    for f in &list_response.data {
        println!("  - {} ({})", f.filename, f.id);
    }

    // Download the file content
    println!("\nDownloading file content...");
    let content = files.download(&file.id).await?;
    println!("Content: {}", String::from_utf8_lossy(&content));

    // Delete the file
    println!("\nDeleting the file...");
    let deleted = files.delete(&file.id).await?;
    println!("Deleted file: {} ({})", deleted.id, if deleted.deleted { "success" } else { "failed" });

    Ok(())
}
