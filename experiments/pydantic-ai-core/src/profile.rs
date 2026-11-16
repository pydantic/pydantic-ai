//! Port of `pydantic_ai.profiles.openai.openai_model_profile`.

use serde_json::{Value, json};

pub fn model_profile(family: &str, model_name: &str) -> Result<Value, String> {
    match family {
        "openai" => Ok(openai_model_profile(model_name)),
        _ => Err(format!("unknown profile family {family:?}")),
    }
}

struct Reasoning {
    enabled_by_default: bool,
    can_be_disabled: bool,
    supports_mode: bool,
    supports_context: bool,
    efforts: Option<&'static [&'static str]>,
}

const fn reasoning(
    enabled_by_default: bool,
    can_be_disabled: bool,
    supports_mode: bool,
    supports_context: bool,
    efforts: Option<&'static [&'static str]>,
) -> Reasoning {
    Reasoning {
        enabled_by_default,
        can_be_disabled,
        supports_mode,
        supports_context,
        efforts,
    }
}

const NO_REASONING: Reasoning = reasoning(false, false, false, false, None);
const PRO_EFFORTS: Option<&[&str]> = Some(&["medium", "high", "xhigh"]);
const CHAT_EFFORTS: Option<&[&str]> = Some(&["medium"]);
const O_SERIES_EFFORTS: Option<&[&str]> = Some(&["low", "medium", "high"]);
const GPT_6_EFFORTS: Option<&[&str]> = Some(&["low", "medium", "high", "xhigh", "max"]);
const GPT_6_PREFIXES: [&str; 5] = [
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-6.1-sol",
    "gpt-6.1-terra",
];

/// The first matching prefix wins, so a specific prefix sits above the broader one it would also match.
const REASONING_BY_PREFIX: [(&str, Reasoning); 25] = [
    (
        "gpt-6-astra",
        reasoning(true, false, true, true, GPT_6_EFFORTS),
    ),
    (
        "gpt-6.1-sol",
        reasoning(true, false, true, true, GPT_6_EFFORTS),
    ),
    ("gpt-6.1-terra", reasoning(true, true, true, true, None)),
    ("gpt-6-sol", reasoning(true, true, true, true, None)),
    ("gpt-6-luna", reasoning(true, true, true, true, None)),
    ("gpt-5.6", reasoning(true, true, true, true, None)),
    (
        "gpt-5.5-pro",
        reasoning(true, false, false, true, PRO_EFFORTS),
    ),
    ("gpt-5.5", reasoning(true, true, false, true, None)),
    (
        "gpt-5.4-pro",
        reasoning(true, false, false, true, PRO_EFFORTS),
    ),
    ("gpt-5.4", reasoning(false, true, false, true, None)),
    (
        "gpt-5.3-chat",
        reasoning(true, false, false, false, CHAT_EFFORTS),
    ),
    ("gpt-5.3", reasoning(false, true, false, false, None)),
    (
        "gpt-5.2-pro",
        reasoning(true, false, false, false, PRO_EFFORTS),
    ),
    (
        "gpt-5.2-chat",
        reasoning(true, false, false, false, CHAT_EFFORTS),
    ),
    ("gpt-5.2", reasoning(false, true, false, false, None)),
    ("gpt-5.1-codex", reasoning(true, false, false, false, None)),
    (
        "gpt-5.1-chat",
        reasoning(true, false, false, false, CHAT_EFFORTS),
    ),
    ("gpt-5.1", reasoning(false, true, false, false, None)),
    ("gpt-5-chat", NO_REASONING),
    (
        "gpt-5-pro",
        reasoning(true, false, false, false, Some(&["high"])),
    ),
    ("gpt-5-codex", reasoning(true, false, false, false, None)),
    (
        "gpt-5",
        reasoning(
            true,
            false,
            false,
            false,
            Some(&["minimal", "low", "medium", "high"]),
        ),
    ),
    ("o1", reasoning(true, false, false, false, O_SERIES_EFFORTS)),
    ("o3", reasoning(true, false, false, false, O_SERIES_EFFORTS)),
    ("o4", reasoning(true, false, false, false, O_SERIES_EFFORTS)),
];

fn openai_model_profile(model_name: &str) -> Value {
    let starts_with =
        |prefixes: &[&str]| prefixes.iter().any(|prefix| model_name.starts_with(prefix));
    let is_gpt_6 = starts_with(&GPT_6_PREFIXES);
    let is_gpt_5_6_or_later = model_name.starts_with("gpt-5.6") || is_gpt_6;
    let is_gpt_oss = model_name.starts_with("gpt-oss");

    let reasoning = REASONING_BY_PREFIX
        .iter()
        .find(|(prefix, _)| model_name.starts_with(prefix))
        .map_or(&NO_REASONING, |(_, reasoning)| reasoning);
    let supports_thinking = reasoning.enabled_by_default || reasoning.can_be_disabled;

    let mut native_tools = vec![
        "code_execution",
        "file_search",
        "image_generation",
        "mcp_server",
        "web_search",
    ];
    if (starts_with(&["gpt-5.4", "gpt-5.5"]) || is_gpt_5_6_or_later)
        && !model_name.starts_with("gpt-5.4-nano")
    {
        native_tools.push("tool_search");
    }
    native_tools.sort_unstable();

    let mut profile = json!({
        "json_schema_transformer": if is_gpt_oss { "gpt_oss" } else { "openai" },
        "supports_json_schema_output": true,
        "supports_json_object_output": true,
        "supports_image_output": model_name.starts_with("gpt-5")
            || is_gpt_6
            || ["o3", "4.1", "4o"].iter().any(|part| model_name.contains(part)),
        "supports_inline_system_prompts": true,
        "supports_thinking": supports_thinking,
        "thinking_always_enabled": reasoning.enabled_by_default && !reasoning.can_be_disabled,
        "openai_system_prompt_role": model_name.starts_with("o1-mini").then_some("user"),
        "openai_chat_supports_web_search": model_name.contains("-search-preview"),
        "openai_supports_encrypted_reasoning_content": supports_thinking,
        "openai_supports_reasoning": supports_thinking,
        "thinking_enabled_by_default": reasoning.enabled_by_default,
        "openai_supports_reasoning_effort_none": reasoning.can_be_disabled,
        "openai_responses_supports_reasoning_mode": reasoning.supports_mode,
        "openai_responses_supports_reasoning_context": reasoning.supports_context,
        "openai_supports_phase": starts_with(&["gpt-5.3-codex", "gpt-5.4", "gpt-5.5"]) || is_gpt_5_6_or_later,
        "openai_supports_prompt_cache_breakpoints": is_gpt_5_6_or_later,
        "openai_supports_minimal_reasoning_effort": match reasoning.efforts {
            Some(efforts) => efforts.contains(&"minimal"),
            None => !is_gpt_5_6_or_later,
        },
        "supported_native_tools": native_tools,
    });
    if let Some(efforts) = reasoning.efforts {
        let mut efforts = efforts.to_vec();
        efforts.sort_unstable();
        profile["openai_reasoning_efforts"] = json!(efforts);
    }
    if is_gpt_oss {
        profile["default_framing_announcement"] = json!("always");
    }
    profile
}
