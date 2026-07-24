// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

fn write_file_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "write_file".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            })),
        },
    }
}

fn call(arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: "call_test".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "write_file".to_string(),
            arguments: arguments.to_string(),
        },
    }
}

#[test]
fn poolside_parser_does_not_reinject_tool_prompt() {
    let prompt =
        PoolsideV1Parser.system_prompt(&[write_file_tool()], &ToolChoice::Mode("auto".to_string()));

    assert!(prompt.is_empty());
}

#[test]
fn missing_write_path_is_not_executable_after_backfill() {
    let mut calls = vec![call(serde_json::json!({"content": "hello"}))];
    let tools = [write_file_tool()];
    backfill_required_params(&mut calls, &tools);
    let validated = validate_tool_calls(calls, &tools);

    assert!(validated.valid.is_empty());
    assert_eq!(validated.errors.len(), 1);
    assert!(validated.errors[0].contains("path"));
}

#[test]
fn empty_write_path_is_not_executable() {
    let validated = validate_tool_calls(
        vec![call(serde_json::json!({
            "path": "   ",
            "content": "hello"
        }))],
        &[write_file_tool()],
    );

    assert!(validated.valid.is_empty());
    assert_eq!(validated.errors.len(), 1);
    assert!(validated.errors[0].contains("path"));
}

#[test]
fn empty_content_remains_schema_valid() {
    let validated = validate_tool_calls(
        vec![call(serde_json::json!({
            "path": "/tmp/a",
            "content": ""
        }))],
        &[write_file_tool()],
    );

    assert_eq!(validated.valid.len(), 1);
    assert!(validated.errors.is_empty());
}

/// Regression: a poolside_v1 `write` whose `<arg_value>` file content contains a
/// foreign name marker (`"name"` — routine in JSON/source) must NOT make the
/// STREAMING detector emit a spurious empty-args tool call. Before the fix,
/// `extract_streaming_name` scanned the whole buffered body (including the arg
/// value), matched the stray `"name"`, seeded a `ToolCallStart{arguments:""}`,
/// then dropped the real args at `</tool_call>` — leaving the client with
/// `arguments:""` and Poolside `pool`'s `json.Unmarshal("")` failing with
/// "unexpected end of JSON input". The `<arg_key>` guard keeps poolside on its
/// whole-body parse so the complete, valid args are emitted.
#[test]
fn poolside_stream_large_arg_value_with_name_marker_yields_valid_args() {
    // file content deliberately contains `"name"` (Hermes marker) + `<function`
    // (Qwen marker) + `call:` (Gemma marker) — all three false-match triggers —
    // AND UTF-8/emoji, which real file writes routinely include.
    let content = r#"{"name": "srv 🚀"}
async fn call: <function=x> // 日本語 émojis 🔥🎉
// padding to make the value large ················· café"#;
    let body = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>main.rs</arg_value>\
         <arg_key>contents</arg_key><arg_value>{content}</arg_value></tool_call>"
    );

    // Feed in small chunks to exercise the incremental name-extraction seam.
    // Split on CHAR boundaries (never mid-UTF-8) — `process` takes `&str`, so a
    // partial multibyte char could never reach the detector in production either
    // (the detokenizer buffers incomplete UTF-8 upstream).
    let mut det = StreamingToolDetector::new();
    let mut outputs = Vec::new();
    let chars: Vec<char> = body.chars().collect();
    for chunk in chars.chunks(5) {
        let s: String = chunk.iter().collect();
        outputs.extend(det.process(&s));
    }

    // No spurious start under a bogus (non-"write") name.
    for o in &outputs {
        if let DetectorOutput::ToolCallStart { name, .. } = o {
            assert_eq!(name, "write", "spurious tool-call start under bogus name {name:?}");
        }
    }

    // The emitted args must be valid JSON carrying the real path + contents.
    let args = args_from_outputs(&outputs);
    let v: serde_json::Value =
        serde_json::from_str(&args).unwrap_or_else(|e| panic!("args not valid JSON: {e}; args={args:?}"));
    assert_eq!(v.get("path").and_then(|x| x.as_str()), Some("main.rs"));
    assert_eq!(v.get("contents").and_then(|x| x.as_str()), Some(content));
}

/// Local copy of the streaming_frag helper (that module is a sibling test file).
fn args_from_outputs(outputs: &[DetectorOutput]) -> String {
    let mut frags = String::new();
    for o in outputs {
        if let DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
            frags.push_str(fragment);
        }
    }
    if !frags.is_empty() {
        return frags;
    }
    for o in outputs {
        if let DetectorOutput::ToolCallDelta { args, .. } = o {
            return args.clone();
        }
        if let DetectorOutput::ToolCall(call, _) = o {
            return call.function.arguments.clone();
        }
    }
    panic!("no args emitted (neither fragments, delta, nor complete call)");
}
