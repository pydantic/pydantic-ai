//! OpenAI Chat Completions codec: `pydantic-ai` messages to a request body, and a response body to a `ModelResponse`.
//!
//! Messages use the JSON format `ModelMessagesTypeAdapter` reads and writes, so a history can come straight from Python.

use serde_json::{Map, Value, json};

use super::profile::model_profile;
use super::schema::transform_schema;

pub const REQUEST_PATH: &str = "/chat/completions";

pub fn build_request(
    model: &str,
    instructions: Option<&str>,
    messages: &[Value],
    tools: &[Value],
) -> Result<Value, String> {
    let mut wire_messages = Vec::new();
    if let Some(instructions) = instructions {
        wire_messages.push(json!({"role": "system", "content": instructions}));
    }
    for message in messages {
        match message["kind"].as_str() {
            Some("request") => map_request(message, &mut wire_messages)?,
            Some("response") => wire_messages.push(map_response(message)?),
            kind => return Err(format!("unsupported message kind {kind:?}")),
        }
    }

    let mut body = Map::new();
    body.insert("model".to_owned(), json!(model));
    body.insert("stream".to_owned(), json!(false));
    body.insert("messages".to_owned(), Value::Array(wire_messages));
    if !tools.is_empty() {
        let transformer = model_profile("openai", model)?["json_schema_transformer"].clone();
        let transformer = transformer.as_str().unwrap_or_default();
        let tools = tools
            .iter()
            .map(|tool| map_tool(tool, transformer))
            .collect::<Result<Vec<_>, _>>()?;
        body.insert("tool_choice".to_owned(), json!("auto"));
        body.insert("tools".to_owned(), Value::Array(tools));
    }
    Ok(Value::Object(body))
}

fn map_request(message: &Value, wire_messages: &mut Vec<Value>) -> Result<(), String> {
    for part in parts(message)? {
        let content = &part["content"];
        wire_messages.push(match part["part_kind"].as_str() {
            Some("system-prompt") => json!({"role": "system", "content": content}),
            Some("user-prompt") if content.is_string() => {
                json!({"role": "user", "content": content})
            }
            Some("user-prompt") => return Err("only text user prompts are supported".to_owned()),
            Some("tool-return") => {
                let content = match (part["outcome"].as_str().unwrap_or("success"), content) {
                    ("success", Value::String(text)) => text.clone(),
                    ("success", value) => value.to_string(),
                    ("retried", message) => json!({"error": message}).to_string(),
                    (outcome, _) => {
                        return Err(format!("unsupported tool return outcome {outcome:?}"));
                    }
                };
                json!({"role": "tool", "tool_call_id": part["tool_call_id"], "content": content})
            }
            kind => return Err(format!("unsupported request part kind {kind:?}")),
        });
    }
    Ok(())
}

fn map_response(message: &Value) -> Result<Value, String> {
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    for part in parts(message)? {
        match part["part_kind"].as_str() {
            Some("text") => texts.push(part["content"].as_str().unwrap_or_default()),
            Some("tool-call") => {
                // `args` is a JSON string from OpenAI, and an object when the history came from another provider.
                let arguments = match &part["args"] {
                    Value::String(arguments) => arguments.clone(),
                    Value::Null => "{}".to_owned(),
                    arguments => arguments.to_string(),
                };
                tool_calls.push(json!({
                    "id": part["tool_call_id"],
                    "type": "function",
                    "function": {"name": part["tool_name"], "arguments": arguments},
                }));
            }
            kind => return Err(format!("unsupported response part kind {kind:?}")),
        }
    }
    let content = if texts.is_empty() {
        Value::Null
    } else {
        json!(texts.join("\n\n"))
    };
    let mut wire_message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        wire_message["tool_calls"] = Value::Array(tool_calls);
    }
    Ok(wire_message)
}

fn map_tool(tool: &Value, transformer: &str) -> Result<Value, String> {
    let Some(name) = tool["name"].as_str() else {
        return Err("a tool needs a `name`".to_owned());
    };
    let transformed = transform_schema(
        transformer,
        &tool["parameters_json_schema"],
        tool["strict"].as_bool(),
    )
    .map_err(|error| format!("Tool {name:?}: {error}"))?;
    let mut function = json!({
        "name": name,
        "description": tool["description"].as_str().unwrap_or_default(),
        "parameters": transformed.schema,
    });
    // A tool that does not say is strict when its schema allows it.
    if tool["strict"]
        .as_bool()
        .unwrap_or(transformed.is_strict_compatible)
    {
        function["strict"] = json!(true);
    }
    Ok(json!({"type": "function", "function": function}))
}

/// A `ModelResponse` in the `pydantic-ai` message format.
pub fn parse_response(body: &Value) -> Result<Value, String> {
    let choice = &body["choices"][0];
    let Some(message) = choice["message"].as_object() else {
        return Err("the response has no `choices[0].message`".to_owned());
    };

    let mut response_parts = Vec::new();
    if let Some(content) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
    {
        response_parts.push(json!({"part_kind": "text", "content": content}));
    }
    for tool_call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        response_parts.push(json!({
            "part_kind": "tool-call",
            "tool_name": tool_call["function"]["name"],
            "args": tool_call["function"]["arguments"],
            "tool_call_id": tool_call["id"],
        }));
    }

    let finish_reason = match choice["finish_reason"].as_str() {
        Some("stop") => json!("stop"),
        Some("length") => json!("length"),
        Some("tool_calls" | "function_call") => json!("tool_call"),
        Some("content_filter") => json!("content_filter"),
        _ => Value::Null,
    };
    let tokens = |key: &str| body["usage"][key].as_u64().unwrap_or(0);
    Ok(json!({
        "kind": "response",
        "parts": response_parts,
        "usage": {"input_tokens": tokens("prompt_tokens"), "output_tokens": tokens("completion_tokens")},
        "model_name": body["model"],
        "provider_name": "openai",
        "provider_response_id": body["id"],
        "finish_reason": finish_reason,
    }))
}

fn parts(message: &Value) -> Result<&Vec<Value>, String> {
    message["parts"]
        .as_array()
        .ok_or_else(|| "a message needs a `parts` list".to_owned())
}
