//! Tool use (function calling) example.
//!
//! This example demonstrates how to define tools that Claude can use
//! to perform actions, such as getting weather information.
//!
//! # Running
//!
//! ```bash
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo run --example tool_use
//! ```

use anthropic::{Anthropic, Model};
use anthropic::types::{
    MessageCreateParams, MessageParam, ToolParam, ToolChoice,
    ContentBlock, ToolResultBlockParam,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anthropic::Error> {
    // Create a client
    let client = Anthropic::new()?;

    // Define a weather tool
    let weather_tool = ToolParam::function(
        "get_weather",
        Some("Get the current weather for a location"),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Tokyo, Japan'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }),
    );

    // First request: Claude decides to use the tool
    println!("Asking Claude about the weather...\n");

    let response = client.messages().create(
        MessageCreateParams::builder()
            .model(Model::claude_sonnet_4_5_latest())
            .max_tokens(1024)
            .tools(vec![weather_tool.clone()])
            .tool_choice(ToolChoice::Auto)
            .messages(vec![
                MessageParam::user("What's the weather like in Paris today?")
            ])
            .build()?
    ).await?;

    // Check if Claude wants to use a tool
    if response.stop_reason == Some(anthropic::types::StopReason::ToolUse) {
        println!("Claude wants to use a tool!");

        // Find the tool use block
        for block in &response.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                println!("Tool: {}", name);
                println!("Input: {}", serde_json::to_string_pretty(input)?);

                // Simulate getting weather data
                let weather_result = json!({
                    "temperature": 22,
                    "unit": "celsius",
                    "conditions": "partly cloudy",
                    "humidity": 65
                });

                // Send the tool result back to Claude
                println!("\nSending tool result back to Claude...\n");

                let final_response = client.messages().create(
                    MessageCreateParams::builder()
                        .model(Model::claude_sonnet_4_5_latest())
                        .max_tokens(1024)
                        .tools(vec![weather_tool.clone()])
                        .messages(vec![
                            MessageParam::user("What's the weather like in Paris today?"),
                            MessageParam::assistant_with_content(response.content.clone()),
                            MessageParam::user_with_content(vec![
                                ToolResultBlockParam::new(
                                    id.clone(),
                                    serde_json::to_string(&weather_result)?,
                                ).into(),
                            ]),
                        ])
                        .build()?
                ).await?;

                println!("Claude says: {}", final_response.content_text());
            }
        }
    } else {
        println!("Claude responded directly: {}", response.content_text());
    }

    Ok(())
}
