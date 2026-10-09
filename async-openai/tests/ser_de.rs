use async_openai::types::chat::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionStreamOptions, ChatCompletionToolChoiceOption, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, ToolChoiceAllowedMode,
};

#[test]
fn chat_types_serde() {
    let request: CreateChatCompletionRequest = CreateChatCompletionRequestArgs::default()
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("your are a calculator")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("what is the result of 1+1")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();
    // serialize the request
    let serialized = serde_json::to_string(&request).unwrap();
    // deserialize the request
    let deserialized: CreateChatCompletionRequest = serde_json::from_str(&serialized).unwrap();
    assert_eq!(request, deserialized);
}

#[test]
fn stream_options_none_fields_not_serialized() {
    // When include_obfuscation is None, it should not appear in the serialized JSON.
    // This is important for OpenAI-compatible providers (like NVIDIA NIM) that reject unknown fields.
    let stream_options = ChatCompletionStreamOptions {
        include_usage: Some(true),
        include_obfuscation: None,
    };

    let serialized = serde_json::to_string(&stream_options).unwrap();

    // Verify include_usage is present
    assert!(serialized.contains("include_usage"));
    // Verify include_obfuscation is NOT present (not even as null)
    assert!(
        !serialized.contains("include_obfuscation"),
        "include_obfuscation should not be serialized when None, but got: {}",
        serialized
    );

    // Test when both are None
    let stream_options_empty = ChatCompletionStreamOptions {
        include_usage: None,
        include_obfuscation: None,
    };

    let serialized_empty = serde_json::to_string(&stream_options_empty).unwrap();
    assert_eq!(serialized_empty, "{}");

    // Test roundtrip deserialization
    let deserialized: ChatCompletionStreamOptions = serde_json::from_str(&serialized).unwrap();
    assert_eq!(stream_options, deserialized);
}

/// `tool_choice` of type `allowed_tools` as the OpenAI API documents it
/// (`ChatCompletionAllowedToolsChoice` in `openapi.documented.yml`): `allowed_tools` is one
/// object holding `mode` and `tools`, not an array of them.
#[test]
fn allowed_tools_tool_choice_takes_the_documented_object_shape() {
    let documented = serde_json::json!({
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "required",
            "tools": [{ "type": "function", "function": { "name": "get_weather" } }]
        }
    });

    let choice: ChatCompletionToolChoiceOption = serde_json::from_value(documented.clone())
        .expect("the documented allowed_tools shape deserializes");
    let ChatCompletionToolChoiceOption::AllowedTools(allowed) = &choice else {
        panic!("deserialized as {choice:?}");
    };
    assert_eq!(allowed.allowed_tools.mode, ToolChoiceAllowedMode::Required);
    assert_eq!(
        allowed.allowed_tools.tools,
        vec![serde_json::json!({ "type": "function", "function": { "name": "get_weather" } })]
    );
    assert_eq!(
        serde_json::to_value(&choice).expect("serializes"),
        documented,
        "the choice serializes back to the documented shape"
    );

    let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "gpt-4o-mini",
        "messages": [{ "role": "user", "content": "What is the weather in Paris?" }],
        "tool_choice": documented
    }))
    .expect("a request carrying the documented allowed_tools shape deserializes");
    assert_eq!(request.tool_choice, Some(choice));
}
