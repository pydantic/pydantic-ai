//! The agent run as a state machine that does no I/O.
//!
//! The host calls `run_step` with the state it got back and an event, and performs the action it returns:
//!
//! | Event | Action |
//! |---|---|
//! | `start` | `send_request` |
//! | `response` | `call_tools` or `done` |
//! | `tool_results` | `send_request` |
//!
//! The core holds nothing between calls: the state goes back to the host as an opaque string. Request and
//! response bodies cross as text too, so a host never re-serializes what the provider reads or wrote.

use serde_json::{Value, json};

use super::openai_chat::{REQUEST_PATH, build_request, parse_response};

const DEFAULT_REQUEST_LIMIT: u64 = 50;
// shortcut: one retry per tool, as `Agent` defaults to; make it part of the `start` event to configure it.
const MAX_TOOL_RETRIES: u64 = 1;

pub fn run_step(state: Option<&str>, event: &Value) -> Result<Value, String> {
    let state = match state {
        Some(state) => {
            Some(serde_json::from_str::<Value>(state).map_err(|e| format!("invalid state: {e}"))?)
        }
        None => None,
    };
    match (event["type"].as_str(), state) {
        (Some("start"), None) => start(event),
        (Some("response"), Some(state)) => handle_response(state, event),
        (Some("tool_results"), Some(state)) => handle_tool_results(state, event),
        (Some("start"), Some(_)) => Err("a `start` event takes no state".to_owned()),
        (Some("response" | "tool_results"), None) => {
            Err("this event needs the state of the run".to_owned())
        }
        (kind, _) => Err(format!("unknown event type {kind:?}")),
    }
}

fn start(event: &Value) -> Result<Value, String> {
    let text = |name: &str| {
        event[name]
            .as_str()
            .ok_or_else(|| format!("`{name}` must be a string"))
    };
    let list = |name: &str| match &event[name] {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => Ok(items.clone()),
        _ => Err(format!("`{name}` must be a list")),
    };
    let mut messages = list("message_history")?;
    messages.push(request_message(
        &event["instructions"],
        vec![json!({"part_kind": "user-prompt", "content": text("user_prompt")?})],
    ));
    send_request(json!({
        "model": text("model")?,
        "instructions": event["instructions"],
        "tools": list("tools")?,
        "request_limit": event["request_limit"].as_u64().unwrap_or(DEFAULT_REQUEST_LIMIT),
        "messages": messages,
        "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0},
        "pending_tool_calls": [],
        "tool_retries": {},
    }))
}

fn send_request(mut state: Value) -> Result<Value, String> {
    let request_limit = state["request_limit"]
        .as_u64()
        .unwrap_or(DEFAULT_REQUEST_LIMIT);
    if count(&state["usage"]["requests"]) >= request_limit {
        return Err(format!(
            "The next request would exceed the request_limit of {request_limit}"
        ));
    }
    let body = build_request(
        state["model"].as_str().unwrap_or_default(),
        state["instructions"].as_str(),
        list(&state["messages"]),
        list(&state["tools"]),
    )?;
    add(&mut state["usage"]["requests"], 1);
    let request = json!({"method": "POST", "path": REQUEST_PATH, "body": body.to_string()});
    Ok(step(
        state,
        json!({"type": "send_request", "request": request}),
    ))
}

fn handle_response(mut state: Value, event: &Value) -> Result<Value, String> {
    if !state["pending_tool_calls"]
        .as_array()
        .is_some_and(Vec::is_empty)
    {
        return Err("the run is waiting for tool results, not for a response".to_owned());
    }
    let status = event["status"].as_u64().unwrap_or(0);
    let body = event["body"]
        .as_str()
        .ok_or("`body` must be the response text")?;
    if status != 200 {
        return Err(format!(
            "status_code: {status}, model_name: {}, body: {body}",
            state["model"]
        ));
    }
    let body = serde_json::from_str(body).map_err(|e| format!("the response is not JSON: {e}"))?;
    let response = parse_response(&body)?;
    add(
        &mut state["usage"]["input_tokens"],
        count(&response["usage"]["input_tokens"]),
    );
    add(
        &mut state["usage"]["output_tokens"],
        count(&response["usage"]["output_tokens"]),
    );

    let parts_of_kind = |kind: &str| {
        list(&response["parts"])
            .iter()
            .filter(move |part| part["part_kind"] == kind)
            .collect::<Vec<_>>()
    };
    let tool_calls: Vec<Value> = parts_of_kind("tool-call")
        .iter()
        .map(|part| json!({"tool_call_id": part["tool_call_id"], "tool_name": part["tool_name"], "args": part["args"]}))
        .collect();
    let texts: Vec<String> = parts_of_kind("text")
        .iter()
        .filter_map(|part| part["content"].as_str())
        .map(str::to_owned)
        .collect();
    push(&mut state["messages"], response);

    let action = if !tool_calls.is_empty() {
        state["pending_tool_calls"] = json!(tool_calls);
        json!({"type": "call_tools", "calls": tool_calls})
    } else if !texts.is_empty() {
        let output = texts.join("\n\n");
        json!({"type": "done", "output": output, "messages": state["messages"], "usage": state["usage"]})
    } else {
        // shortcut: `Agent` asks the model again after an empty response; add that when a run needs it.
        return Err("the model returned neither text nor a tool call".to_owned());
    };
    Ok(step(state, action))
}

fn handle_tool_results(mut state: Value, event: &Value) -> Result<Value, String> {
    let pending = state["pending_tool_calls"].take();
    if list(&pending).is_empty() {
        return Err("the run is not waiting for tool results".to_owned());
    }
    let mut parts = Vec::new();
    for call in list(&pending) {
        let (id, name) = (&call["tool_call_id"], &call["tool_name"]);
        let Some(result) = list(&event["results"])
            .iter()
            .find(|result| &result["tool_call_id"] == id)
        else {
            return Err(format!("no result for tool call {id}"));
        };
        let mut part = json!({"part_kind": "tool-return", "tool_name": name, "tool_call_id": id});
        if let Some(retry) = result.get("retry").filter(|retry| !retry.is_null()) {
            let tool_name = name.as_str().unwrap_or_default();
            let attempt = count(&state["tool_retries"][tool_name]) + 1;
            if attempt > MAX_TOOL_RETRIES {
                return Err(format!(
                    "Tool '{tool_name}' exceeded max retries count of {MAX_TOOL_RETRIES}"
                ));
            }
            state["tool_retries"][tool_name] = json!(attempt);
            part["content"] = retry.clone();
            part["outcome"] = json!("retried");
            part["retry"] = json!({"cause": "model_retry", "attempt": attempt, "max_retries": MAX_TOOL_RETRIES});
        } else {
            part["content"] = result["content"].clone();
        }
        parts.push(part);
    }
    state["pending_tool_calls"] = json!([]);
    let request = request_message(&state["instructions"], parts);
    push(&mut state["messages"], request);
    send_request(state)
}

fn step(state: Value, action: Value) -> Value {
    json!({"state": state.to_string(), "action": action})
}

fn request_message(instructions: &Value, parts: Vec<Value>) -> Value {
    json!({"kind": "request", "parts": parts, "instructions": instructions})
}

fn list(value: &Value) -> &[Value] {
    value.as_array().map_or(&[], Vec::as_slice)
}

fn count(value: &Value) -> u64 {
    value.as_u64().unwrap_or(0)
}

fn add(total: &mut Value, amount: u64) {
    *total = json!(count(total) + amount);
}

fn push(messages: &mut Value, message: Value) {
    if let Value::Array(messages) = messages {
        messages.push(message);
    }
}
