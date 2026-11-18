#![cfg(all(feature = "cpu", feature = "ready"))]

mod common;
#[path = "common/grpc_frontend_fixture.rs"]
mod fixture;

use serde_json::json;
use vllm_router_rs::backend::EngineFrontend;
use vllm_router_rs::protocols::spec::CompletionRequest;

#[tokio::test]
async fn grpc_prefix_selection_uses_the_ids_actually_dispatched() {
    use axum::body::to_bytes;
    use std::sync::Arc;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::kv_index::{
        CacheKey, HashMode, KvBlockIndexer, ResidencyOwner, RoutableTarget, StorageTier,
    };
    use vllm_router_rs::policies::KvAwarePolicy;
    use vllm_router_rs::protocols::spec::ChatCompletionRequest;
    use vllm_router_rs::routers::http::router::Router;
    let model = fixture::model_fixture();
    let warm = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let cold = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![warm.grpc_url.clone(), cold.grpc_url.clone()], &context)
        .await
        .unwrap();
    let warm_worker = context.worker_registry.get_by_url(&warm.grpc_url).unwrap();
    warm_worker.increment_load();
    let frontend = EngineFrontend::new();
    let policy = context.policy_registry.get_default_policy();
    let policy = policy.as_any().downcast_ref::<KvAwarePolicy>().unwrap();
    let bodies = [
        json!({"messages":[{"role":"user","content":"hello prefix"}]}),
        json!({"messages":[{"role":"user","content":[{"type":"text","text":"hello prefix"}]}]}),
        json!({"messages":[{"role":"user","content":"call f"},{"role":"assistant","content":"calling","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"done"}]}),
        json!({"messages":[{"role":"user","content":"a longer chat to truncate"}],"truncate_prompt_tokens":6}),
    ];
    for mut body in bodies.into_iter().chain(
        [
            json!({"prompt":[1,2,3,4,5,6,7,8],"truncate_prompt_tokens":6}),
            json!({"prompt": "hello prefix"}),
            json!({"prompt": "a longer text to truncate", "truncate_prompt_tokens": 6}),
        ]
        .into_iter(),
    ) {
        body["model"] = json!(model.path().to_str().unwrap());
        let completion = body.get("prompt").is_some();
        let expected = if completion {
            let request: CompletionRequest = serde_json::from_value(body.clone()).unwrap();
            let mut prepared = frontend.prepare_completions(&request).await.unwrap();
            assert_eq!(prepared.len(), 1);
            prepared.pop().unwrap().tokenized.token_ids
        } else {
            frontend
                .prepare(serde_json::from_value(body.clone()).unwrap())
                .await
                .unwrap()
                .tokenized
                .token_ids
        };
        if body.get("truncate_prompt_tokens").is_some() {
            assert_eq!(expected.len(), 6);
        }
        let index = Arc::new(KvBlockIndexer::new());
        for (url, name) in [(&warm.grpc_url, "warm"), (&cold.grpc_url, "cold")] {
            policy.register_target_with_cache_identity(
                url,
                RoutableTarget {
                    instance_id: Arc::from(name),
                    dp_rank: 0,
                },
                CacheKey {
                    model: Arc::from("model"),
                    hash_mode: HashMode::Sha256,
                    block_size: 2,
                },
                index.clone(),
            );
        }
        let hashes = KvAwarePolicy::local_hashes(&expected[..(expected.len() - 1) / 2 * 2], 2);
        let blocks: Vec<_> = hashes
            .into_iter()
            .enumerate()
            .map(|(i, hash)| (Arc::from(format!("block-{i}")), hash))
            .collect();
        index.store(
            0,
            ResidencyOwner::Worker {
                source: Arc::from("warm"),
                dp_rank: 0,
                incarnation: 0,
            },
            StorageTier::Device,
            None,
            &blocks,
        );
        let before = warm.captured().len();
        let response = if completion {
            let request: CompletionRequest = serde_json::from_value(body).unwrap();
            router
                .route_typed_request(None, &request, "/v1/completions", None)
                .await
        } else {
            let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            router
                .route_typed_request(None, &request, "/v1/chat/completions", None)
                .await
        };
        assert!(response.status().is_success());
        to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert_eq!(warm.captured().len(), before + 1);
        assert!(cold.captured().is_empty());
        assert_eq!(warm.captured().last().unwrap().token_ids, expected.as_ref());
        assert_eq!(warm_worker.load(), 1);
    }
}

#[tokio::test]
async fn grpc_completion_text_prompt_is_encoded_with_special_tokens_by_default() {
    let model = fixture::model_fixture();
    let frontend = EngineFrontend::new();
    let mut request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
    }))
    .unwrap();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 1);
    let ids = &prepared[0].tokenized.token_ids;
    assert!(!ids.is_empty());
    // The fixture adds a BOS post-processor, so the default encode starts
    // with <s> (290).
    assert_eq!(ids[0], 290);

    request
        .other
        .insert("add_special_tokens".to_string(), json!(false));
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    let ids = &prepared[0].tokenized.token_ids;
    assert!(!ids.is_empty());
    assert_ne!(ids[0], 290);
}

#[tokio::test]
async fn grpc_completion_batch_prepare_returns_one_request_per_prompt() {
    // Lowering is the fan-out unit Stage 3 will dispatch: a 2-prompt batch
    // prepares to 2 requests, in client order, each encoded separately.
    let model = fixture::model_fixture();
    let frontend = EngineFrontend::new();
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": ["first", "second"],
    }))
    .unwrap();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 2);
    assert!(!prepared[0].tokenized.token_ids.is_empty());
    assert!(!prepared[1].tokenized.token_ids.is_empty());
    assert_ne!(
        prepared[0].tokenized.token_ids,
        prepared[1].tokenized.token_ids
    );

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": [[1], [2, 3]],
    }))
    .unwrap();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 2);
    assert_eq!(prepared[0].tokenized.token_ids.as_ref(), &[1]);
    assert_eq!(prepared[1].tokenized.token_ids.as_ref(), &[2, 3]);
}

#[tokio::test]
async fn grpc_completion_batched_prompts_are_rejected_before_tokenize() {
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    // Batched with a model: 400, and nothing is dispatched.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": ["first", "second"],
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(
        String::from_utf8_lossy(&body).contains("batched prompts"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(worker.captured().is_empty());

    // Batched without a model: still 400, not 503. The batch check runs
    // before tokenizer resolution, so a batch is never tokenized first —
    // the old order reached resolution here and answered 503.
    let request: CompletionRequest =
        serde_json::from_value(json!({"prompt": ["first", "second"]})).unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(
        String::from_utf8_lossy(&body).contains("batched prompts"),
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(worker.captured().is_empty());
}

#[tokio::test]
async fn grpc_completion_logprobs_echo_round_trip() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::backend::pb::candidate_token_info::TokenInfo;
    use vllm_router_rs::backend::pb::{candidate_tokens, CandidateTokenInfo};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    // Two streamed frames: ids [7, 8] then [9], each with logprobs, plus
    // one candidate token (id 10) at the first output position.
    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![
                    CandidateTokenInfo {
                        tokens: vec![TokenInfo {
                            id: 10,
                            logprob: -1.0,
                            rank: 2,
                        }],
                    },
                    CandidateTokenInfo { tokens: vec![] },
                ],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![CandidateTokenInfo { tokens: vec![] }],
            },
        ]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    // Echo: the prompt text leads the generated text.
    assert!(
        choice["text"].as_str().unwrap().starts_with("hello prefix"),
        "{choice}"
    );
    assert_eq!(choice["finish_reason"], "stop");
    let logprobs = &choice["logprobs"];
    let tokens = logprobs["tokens"].as_array().unwrap();
    let token_logprobs = logprobs["token_logprobs"].as_array().unwrap();
    let text_offset = logprobs["text_offset"].as_array().unwrap();
    let top_logprobs = logprobs["top_logprobs"].as_array().unwrap();
    // Prompt entries (first null, rest -0.1 from the mock) then 3 output
    // entries [-0.5, -0.6, -0.7]; the prompt length is derived, not assumed.
    assert_eq!(tokens.len(), token_logprobs.len());
    assert_eq!(tokens.len(), text_offset.len());
    assert_eq!(tokens.len(), top_logprobs.len());
    assert!(token_logprobs[0].is_null());
    assert_eq!(
        &token_logprobs[token_logprobs.len() - 3..],
        &[json!(-0.5), json!(-0.6), json!(-0.7)]
    );
    assert!(token_logprobs[1..token_logprobs.len() - 3]
        .iter()
        .all(|value| *value == json!(-0.1)));
    // Offsets start at 0 and the first output token starts where the prompt
    // text ends (12 chars for "hello prefix").
    assert_eq!(text_offset[0], 0);
    assert_eq!(text_offset[token_logprobs.len() - 3], 12);
    for window in text_offset.windows(2) {
        assert!(window[0].as_u64().unwrap() <= window[1].as_u64().unwrap());
    }
    // The sampled token joins the candidate map at the first output position.
    let top0 = top_logprobs[token_logprobs.len() - 3].as_object().unwrap();
    assert_eq!(top0.len(), 2);
    let mut values: Vec<f64> = top0.values().map(|v| v.as_f64().unwrap()).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![-1.0, -0.5]);
    assert_eq!(json["usage"]["completion_tokens"], 3);

    // The wire request asked for exactly what the client asked for.
    let captured = worker.captured();
    assert_eq!(captured.len(), 1);
    let options = captured[0].response_options.as_ref().unwrap();
    assert!(options.output_logprobs);
    assert!(options.prompt_logprobs);
    assert!(matches!(
        options
            .output_candidates
            .as_ref()
            .and_then(|c| c.select.clone()),
        Some(candidate_tokens::Select::TopN(2))
    ));
    assert!(matches!(
        options
            .prompt_candidates
            .as_ref()
            .and_then(|c| c.select.clone()),
        Some(candidate_tokens::Select::TopN(2))
    ));
}

#[tokio::test]
async fn grpc_completion_echo_with_max_tokens_zero_returns_the_prompt_only() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::MockVllmRs;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        emit_prompt_logprobs: true,
        zero_max_tokens_returns_empty: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    // Echo + max_tokens: 0: the answer is the prompt, the one token the
    // engine had to generate is dropped.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "max_tokens": 0,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    assert_eq!(choice["text"], "hello prefix");
    assert_eq!(choice["finish_reason"], "stop");
    let token_logprobs = choice["logprobs"]["token_logprobs"].as_array().unwrap();
    assert!(token_logprobs[0].is_null());
    assert!(token_logprobs[1..]
        .iter()
        .all(|value| *value == json!(-0.1)));
    // The engine was asked for exactly one token.
    assert_eq!(worker.captured().last().unwrap().max_new_tokens, 1);

    // Without echo the same request answers with empty text.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "max_tokens": 0,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["choices"][0]["text"], "");
    assert!(json["choices"][0]["logprobs"].is_null());
    // Without `echo` the engine is asked for zero tokens, and the reply
    // reports zero completion tokens, matching vLLM HTTP.
    assert_eq!(worker.captured().last().unwrap().max_new_tokens, 0);
    assert_eq!(json["usage"]["completion_tokens"], 0);
}

#[tokio::test]
async fn grpc_completion_logprobs_echo_streams_per_chunk() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![],
            },
        ]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "stream": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(!chunks.is_empty(), "{text}");
    // The first chunk leads with the echoed prompt and prompt logprobs
    // (first position null).
    let first = &chunks[0]["choices"][0];
    assert!(
        first["text"].as_str().unwrap().starts_with("hello prefix"),
        "{first}"
    );
    assert!(first["logprobs"]["token_logprobs"][0].is_null());
    // A later chunk carries the first output token's logprob.
    assert!(chunks.iter().any(|chunk| {
        chunk["choices"][0]["logprobs"]["token_logprobs"]
            .as_array()
            .is_some_and(|values| values.iter().any(|v| *v == json!(-0.5)))
    }));
}

#[tokio::test]
async fn grpc_completion_late_prompt_info_streams_in_a_later_chunk() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    // PromptInfo arrives after the first output frame, as a
    // non-conforming engine might send it.
    let state = MockVllmRs {
        frames: Some(vec![
            ("".to_string(), vec![7, 8]),
            ("".to_string(), vec![9]),
        ]),
        frame_logprobs: Some(vec![
            FrameLogprobs {
                logprobs: vec![-0.5, -0.6],
                ranks: vec![1, 1],
                candidates: vec![],
            },
            FrameLogprobs {
                logprobs: vec![-0.7],
                ranks: vec![1],
                candidates: vec![],
            },
        ]),
        emit_prompt_logprobs: true,
        prompt_info_after_first_frame: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "logprobs": 2,
        "echo": true,
        "stream": true,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    let chunks: Vec<serde_json::Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert!(!chunks.is_empty(), "{text}");
    // The first chunk carries only the first frame's output logprobs:
    // PromptInfo had not arrived yet.
    let first = &chunks[0]["choices"][0];
    assert_eq!(
        first["logprobs"]["token_logprobs"],
        json!([-0.5, -0.6]),
        "{first}"
    );
    // A later chunk merges the prompt logprobs in front of its output
    // logprobs (first prompt position null, rest -0.1) instead of
    // losing them.
    let merged = chunks
        .iter()
        .find(|chunk| {
            chunk["choices"][0]["logprobs"]["token_logprobs"]
                .as_array()
                .is_some_and(|values| values.first().is_some_and(|value| value.is_null()))
        })
        .expect("prompt logprobs merged into a later chunk");
    let values = merged["choices"][0]["logprobs"]["token_logprobs"]
        .as_array()
        .unwrap();
    assert_eq!(values[values.len() - 1], json!(-0.7));
    assert!(values[1..values.len() - 1]
        .iter()
        .all(|value| *value == json!(-0.1)));
}

#[tokio::test]
async fn grpc_completion_echo_with_truncated_text_prompt_stays_aligned() {
    use axum::body::to_bytes;
    use common::mock_vllm_rs::{FrameLogprobs, MockVllmRs};
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let state = MockVllmRs {
        frame_logprobs: Some(vec![FrameLogprobs {
            logprobs: vec![-0.5, -0.6, -0.7],
            ranks: vec![1, 1, 1],
            candidates: vec![],
        }]),
        emit_prompt_logprobs: true,
        ..Default::default()
    };
    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn_with_state(state).await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "a longer text to truncate",
        "truncate_prompt_tokens": 6,
        "logprobs": 2,
        "echo": true,
    }))
    .unwrap();
    // The frontend's echo text is decoded from the truncated ids, not the
    // client's full string.
    let frontend = EngineFrontend::new();
    let prepared = frontend.prepare_completions(&request).await.unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].tokenized.token_ids.len(), 6);
    let echo_text = prepared[0]
        .completion
        .as_ref()
        .and_then(|options| options.echo_text.clone())
        .unwrap();
    assert_ne!(echo_text, "a longer text to truncate");

    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &json["choices"][0];
    // The reply echoes the truncated text, then the generated text (the
    // router decodes the mock's [7, 8, 9] itself; the mock's reply_text is
    // not used on this path).
    let text = choice["text"].as_str().unwrap();
    assert!(text.starts_with(&echo_text), "{text} vs {echo_text}");
    assert!(text.len() > echo_text.len());
    // Prompt logprobs cover exactly the 6 dispatched ids, then the 3
    // output entries; the first output token starts where the echoed text
    // ends.
    let logprobs = &choice["logprobs"];
    let tokens = logprobs["tokens"].as_array().unwrap();
    let token_logprobs = logprobs["token_logprobs"].as_array().unwrap();
    let text_offset = logprobs["text_offset"].as_array().unwrap();
    assert_eq!(tokens.len(), 9);
    assert_eq!(token_logprobs.len(), 9);
    assert_eq!(text_offset.len(), 9);
    assert!(token_logprobs[0].is_null());
    assert!(token_logprobs[1..6]
        .iter()
        .all(|value| *value == json!(-0.1)));
    assert_eq!(
        &token_logprobs[6..],
        &[json!(-0.5), json!(-0.6), json!(-0.7)]
    );
    assert_eq!(text_offset[6], echo_text.chars().count() as u64);
    // The worker received exactly the truncated ids.
    assert_eq!(worker.captured().last().unwrap().token_ids.len(), 6);
}

#[tokio::test]
async fn grpc_completion_text_prompt_bos_default_end_to_end() {
    use axum::body::to_bytes;
    use vllm_router_rs::config::{PolicyConfig, RouterConfig};
    use vllm_router_rs::routers::http::router::Router;

    let model = fixture::model_fixture();
    let worker = common::mock_vllm_rs::MockVllmRsServer::spawn().await;
    let config = RouterConfig {
        policy: PolicyConfig::KvAware {
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
            static_kv_event_sources: Default::default(),
        },
        disable_circuit_breaker: true,
        ..Default::default()
    };
    let context = common::create_test_context(config);
    let router = Router::new(vec![worker.grpc_url.clone()], &context)
        .await
        .unwrap();

    // Default (add_special_tokens omitted → true): the ids the worker
    // receives start with the fixture BOS token (290).
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_eq!(worker.captured().last().unwrap().token_ids[0], 290);

    // Explicit false: no BOS on the dispatched ids.
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": model.path().to_str().unwrap(),
        "prompt": "hello prefix",
        "add_special_tokens": false,
    }))
    .unwrap();
    let response = router
        .route_typed_request(None, &request, "/v1/completions", None)
        .await;
    assert!(response.status().is_success());
    to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert_ne!(worker.captured().last().unwrap().token_ids[0], 290);
}
