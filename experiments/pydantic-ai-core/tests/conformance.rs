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
