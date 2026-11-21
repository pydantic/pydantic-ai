//! Replays fixtures dumped from the real `pydantic-ai` (`fixtures/generate.py`) through the public entry point.

use serde_json::{Value, json};

fn call(request: Value) -> Value {
    serde_json::from_str(&pydantic_ai_core::call(&request.to_string())).unwrap()
}

fn golden() -> Value {
    serde_json::from_str(include_str!("../fixtures/golden.json")).unwrap()
}

#[test]
fn model_profiles_match_python() {
    let golden = golden();
    let cases = golden["profiles"].as_array().unwrap();
    assert!(cases.len() > 150);
    for case in cases {
        let actual =
            call(json!({"fn": "model_profile", "family": "openai", "model": case["model"]}));
        // Compared as strings, so key order has to match too.
        assert_eq!(
            actual["ok"].to_string(),
            case["expected"].to_string(),
            "model {}",
            case["model"]
        );
    }
}

#[test]
fn transformed_schemas_match_python() {
    let golden = golden();
    let cases = golden["schemas"].as_array().unwrap();
    assert!(cases.len() > 400);
    let mut failures = Vec::new();
    for case in cases {
        let actual = call(json!({
            "fn": "transform_schema",
            "transformer": case["transformer"],
            "schema": case["schema"],
            "strict": case["strict"],
        }));
        let expected = &case["expected"];
        let matches = match expected.get("error") {
            Some(error) => &actual["error"] == error,
            None => actual["ok"].to_string() == expected.to_string(),
        };
        if !matches {
            failures.push(format!(
                "{} {} strict={}\n  expected: {expected}\n  actual:   {actual}",
                case["name"], case["transformer"], case["strict"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn malformed_requests_are_errors() {
    for request in [
        json!({"fn": "model_profile", "family": "openai"}),
        json!({"fn": "transform_schema", "transformer": "openai", "schema": {}, "strict": "true"}),
        json!({"fn": "transform_schema", "schema": {}}),
        json!({"fn": "nope"}),
    ] {
        assert!(call(request.clone())["error"].is_string(), "{request}");
    }
}

fn run_step(state: &Value, event: Value) -> Value {
    call(json!({"fn": "run_step", "state": state, "event": event}))
}

fn scenario(name: &str) -> Value {
    let runs: Value = serde_json::from_str(include_str!("../fixtures/runs.json")).unwrap();
    let scenarios = runs["scenarios"].as_array().unwrap();
    scenarios
        .iter()
        .find(|scenario| scenario["name"] == name)
        .unwrap()
        .clone()
}

/// Drives the core the way a host does, with the tool results and responses the Python run saw.
#[test]
fn agent_runs_match_python() {
    for name in ["tool_call", "tool_retry", "text_only", "message_history"] {
        let scenario = scenario(name);
        let mut out = run_step(&Value::Null, scenario["start"].clone());
        for step in scenario["steps"].as_array().unwrap() {
            let action = &out["ok"]["action"];
            assert_eq!(action["type"], "send_request", "{name}: {out}");
            assert_eq!(action["request"]["path"], "/chat/completions");
            // Compared as strings, so the request body has the key order `pydantic-ai` sends.
            assert_eq!(
                action["request"]["body"],
                step["request"].to_string(),
                "{name}"
            );

            out = run_step(
                &out["ok"]["state"],
                json!({"type": "response", "status": 200, "body": step["response"].to_string()}),
            );
            if let Some(results) = step.get("tool_results") {
                let expected = json!({"type": "call_tools", "calls": step["tool_calls"]});
                assert_eq!(out["ok"]["action"], expected, "{name}: {out}");
                out = run_step(
                    &out["ok"]["state"],
                    json!({"type": "tool_results", "results": results}),
                );
            }
        }
        let action = &out["ok"]["action"];
        assert_eq!(action["type"], "done", "{name}: {out}");
        assert_eq!(action["output"], scenario["output"], "{name}");
        assert_eq!(action["usage"], scenario["usage"], "{name}");
        let messages = action["messages"].as_array().unwrap();
        let new_messages = scenario["new_messages"].as_array().unwrap();
        assert_eq!(
            &messages[messages.len() - new_messages.len()..],
            new_messages.as_slice(),
            "{name}"
        );
    }
}

#[test]
fn run_limits_are_errors() {
    let scenario = scenario("tool_call");
    let step = &scenario["steps"][0];
    let response = json!({"type": "response", "status": 200, "body": step["response"].to_string()});
    let tool_call_id = &step["tool_calls"][0]["tool_call_id"];
    let retry =
        json!({"type": "tool_results", "results": [{"tool_call_id": tool_call_id, "retry": "no"}]});

    let mut start = scenario["start"].clone();
    start["request_limit"] = json!(1);
    let out = run_step(&Value::Null, start);
    let out = run_step(&out["ok"]["state"], response.clone());
    let out = run_step(
        &out["ok"]["state"],
        json!({"type": "tool_results", "results": step["tool_results"]}),
    );
    assert_eq!(
        out["error"],
        "The next request would exceed the request_limit of 1"
    );

    let out = run_step(&Value::Null, scenario["start"].clone());
    let out = run_step(&out["ok"]["state"], response.clone());
    let out = run_step(&out["ok"]["state"], retry.clone());
    let out = run_step(&out["ok"]["state"], response.clone());
    let out = run_step(&out["ok"]["state"], retry);
    assert_eq!(
        out["error"],
        "Tool 'get_temperature' exceeded max retries count of 1"
    );

    let out = run_step(&Value::Null, scenario["start"].clone());
    let failed = json!({"type": "response", "status": 429, "body": "slow down"});
    assert!(
        run_step(&out["ok"]["state"], failed)["error"]
            .as_str()
            .unwrap()
            .starts_with("status_code: 429")
    );
    assert!(
        run_step(
            &out["ok"]["state"],
            json!({"type": "tool_results", "results": []})
        )["error"]
            .is_string()
    );
}
