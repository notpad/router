use crate::core::Worker;
use crate::policies::{AffinityKey, AffinityRecord, RequestHeaders};
use axum::body::Body;
use futures_util::StreamExt;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

/// Format a full error chain for debugging (walks source() recursively).
/// Produces output like: "outer error caused by: middle error caused by: root cause"
pub fn error_chain(err: &dyn std::error::Error) -> String {
    let mut chain = vec![err.to_string()];
    let mut source = err.source();
    while let Some(s) = source {
        chain.push(s.to_string());
        source = s.source();
    }
    chain.join(" caused by: ")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PdEndpoint {
    pub http_url: String,
    pub dp_rank: Option<usize>,
}

impl PdEndpoint {
    pub(super) fn for_worker(worker: &dyn Worker) -> Self {
        Self {
            http_url: worker.base_url().to_owned(),
            dp_rank: worker.dp_rank(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Stage {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PdLoadSample {
    pub prefill_requests: usize,
    pub active_prefill_tokens: u64,
    pub unknown_prefill_requests: usize,
    pub pending_decode_requests: usize,
    pub active_decode_requests: usize,
}

impl PdLoadSample {
    fn publish(self, stage: Stage, endpoint: &PdEndpoint) {
        let rank = endpoint
            .dp_rank
            .map_or_else(|| "aggregate".to_owned(), |rank| rank.to_string());
        let record = |name: &'static str, value: f64| {
            metrics::gauge!(name, "worker" => endpoint.http_url.clone(), "rank" => rank.clone())
                .set(value);
        };
        match stage {
            Stage::Prefill => {
                record(
                    "vllm_router_pd_prefill_occupancy",
                    self.prefill_requests as f64,
                );
                record(
                    "vllm_router_pd_active_prefill_tokens",
                    self.active_prefill_tokens as f64,
                );
                record(
                    "vllm_router_pd_unknown_prefill_requests",
                    self.unknown_prefill_requests as f64,
                );
            }
            Stage::Decode => {
                record(
                    "vllm_router_pd_pending_decode_requests",
                    self.pending_decode_requests as f64,
                );
                record(
                    "vllm_router_pd_active_decode_requests",
                    self.active_decode_requests as f64,
                );
            }
        }
    }

    pub(super) fn occupancy(self, stage: Stage) -> usize {
        match stage {
            Stage::Prefill => self.prefill_requests,
            Stage::Decode => self.pending_decode_requests + self.active_decode_requests,
        }
    }

    fn idle(self) -> bool {
        self.prefill_requests == 0
            && self.active_prefill_tokens == 0
            && self.unknown_prefill_requests == 0
            && self.pending_decode_requests == 0
            && self.active_decode_requests == 0
    }
}

pub(super) type Loads = HashMap<(Stage, PdEndpoint), PdLoadSample>;

/// Backend-effective tokens prepared outside the admission transaction.
#[derive(Debug, Clone)]
pub struct PdPrefillInput {
    pub prompts: Option<Arc<[Vec<u32>]>>,
    /// Per-rank placement snapshot.
    /// Hashing and index queries are completed before the admission lock.
    pub work_by_rank: HashMap<Option<usize>, crate::policies::PrefillWorkEstimate>,
    pub allow_credit: bool,
    pub fallback_reason: Option<&'static str>,
}

/// Token inputs are keyed by the P base URL: each backend renders its own
/// effective input, so different tokenizer/template configurations never mix.
pub struct PdRequestContext<'a> {
    pub prefill_inputs: Option<&'a HashMap<String, PdPrefillInput>>,
    pub model_id: Option<&'a str>,
    pub legacy_policy_input: Option<&'a str>,
    pub(crate) prefill_cache: Option<&'a crate::policies::PreparedCacheSelection>,
    pub(crate) decode_cache: Option<&'a crate::policies::PreparedCacheSelection>,
    pub headers: Option<&'a RequestHeaders>,
    /// Stable session identity for the optional kv_aware decode binding.
    /// Resolved at request entry, only when the affinity switch is on;
    /// `None` selects per request on both stages.
    pub affinity_key: Option<AffinityKey>,
    /// Borrowed LoRA name from the request; used for Decode binding only with an eligible affinity key.
    pub lora_name: Option<&'a str>,
    /// Request cache salt scoping the binding digest.
    pub cache_salt: Option<&'a str>,
    pub route: &'a str,
}

#[derive(Debug, Clone)]
pub struct StageDecision {
    pub fallback_reason: Option<&'static str>,
    pub credited_prefix_tokens: Option<u64>,
    pub worker_index: usize,
    pub endpoint: PdEndpoint,
    pub score: Option<f64>,
    /// Immutable token work charged to this attempt after KV credit.
    pub new_prefill_tokens: Option<u64>,
    /// How session affinity shaped this stage's choice: bind / sticky_hit /
    /// no_session / capacity_bypass, set only while the switch is on.
    pub affinity_outcome: Option<&'static str>,
    /// Held binding reference for this stage's decision; the admission
    /// attempt releases it on cleanup.
    pub affinity: Option<Arc<AffinityRecord>>,
}

#[derive(Debug, Clone)]
pub struct PdSelection {
    pub prefill: StageDecision,
    pub decode: StageDecision,
}

#[derive(Debug, Default)]
pub struct PdLoadManager {
    loads: Arc<Mutex<Loads>>,
}

impl PdLoadManager {
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn snapshot(&self) -> Loads {
        self.loads.lock().clone()
    }

    /// The router supplies selection; this manager owns only admission and
    /// lifecycle state. No policy, worker registry, or connector lives here.
    ///
    /// Both stages are charged here. Callers that dispatch Decode only after
    /// Prefill completes use [`Self::admit_deferred`] instead.
    pub(super) fn admit(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
    ) -> Option<(PdSelection, PdAttempt)> {
        self.admit_inner(select, false)
    }

    /// Admit the Prefill stage only and leave the Decode commitment open.
    ///
    /// The supplied selection still carries a Decode decision, but it is
    /// provisional: it takes no affinity binding and nothing is charged for it
    /// until [`Self::commit_decode`] runs. A provisional pick must therefore
    /// never be dispatched.
    pub(super) fn admit_deferred(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
    ) -> Option<(PdSelection, PdAttempt)> {
        self.admit_inner(select, true)
    }

    /// Admit N units atomically under a single loads lock.
    ///
    /// Stage 3 fan-out: one client request is N = P x n units, and each unit
    /// goes through a complete admission. Calling `admit` N times would
    /// release the lock between units (other requests interleave) and leave
    /// the first N-1 holding slots when the Nth fails. Here the selector runs
    /// N times against the post-charge load, so units spread across pairs,
    /// and any failure rolls every charge back before returning `None`.
    ///
    /// Attempt handles are built after the lock is released: dropping one
    /// re-enters the loads lock, which would deadlock under the held lock.
    pub(super) fn admit_many(
        &self,
        mut select: impl FnMut(&Loads) -> Option<PdSelection>,
        n: usize,
        defer_decode: bool,
    ) -> Option<Vec<(PdSelection, PdAttempt)>> {
        let now = std::time::Instant::now();
        let charged: Vec<(PdSelection, ChargedUnit)> = {
            let mut loads = self.loads.lock();
            let mut charged = Vec::with_capacity(n);
            for _ in 0..n {
                let Some(selection) = select(&loads) else {
                    for (_, unit) in charged.iter().rev() {
                        uncharge_selection(&mut loads, unit);
                    }
                    return None;
                };
                let unit = charge_selection(&mut loads, &selection, defer_decode, now);
                charged.push((selection, unit));
            }
            charged
        };
        Some(
            charged
                .into_iter()
                .map(|(selection, unit)| {
                    let attempt = PdAttempt(Arc::new(AttemptInner {
                        admitted_at: now,
                        loads: self.loads.clone(),
                        prefill: unit.prefill.clone(),
                        prefill_worker: Mutex::new(None),
                        decode_worker: Mutex::new(None),
                        prefill_tokens: unit.prefill_tokens,
                        state: Mutex::new(AttemptState {
                            prefill: true,
                            decode: unit.decode_state,
                            decode_endpoint: unit.decode_endpoint,
                            affinity: unit.affinity,
                            pending_since: unit.pending_since,
                        }),
                    }));
                    (selection, attempt)
                })
                .collect(),
        )
    }

    fn admit_inner(
        &self,
        select: impl FnOnce(&Loads) -> Option<PdSelection>,
        defer_decode: bool,
    ) -> Option<(PdSelection, PdAttempt)> {
        let now = std::time::Instant::now();
        let (selection, unit) = {
            let mut loads = self.loads.lock();
            let selection = select(&loads)?;
            let unit = charge_selection(&mut loads, &selection, defer_decode, now);
            (selection, unit)
        };
        let attempt = PdAttempt(Arc::new(AttemptInner {
            admitted_at: now,
            loads: self.loads.clone(),
            prefill: unit.prefill.clone(),
            prefill_worker: Mutex::new(None),
            decode_worker: Mutex::new(None),
            prefill_tokens: unit.prefill_tokens,
            state: Mutex::new(AttemptState {
                prefill: true,
                decode: unit.decode_state,
                decode_endpoint: unit.decode_endpoint,
                affinity: unit.affinity,
                pending_since: unit.pending_since,
            }),
        }));
        Some((selection, attempt))
    }

    /// Charge and bind the Decode stage chosen at dispatch time.
    ///
    /// The attempt state lock serializes this against the release paths: a
    /// concurrent drop either finishes first, so this finds a non-`Unbound`
    /// state and charges nothing, or it waits and then releases what was
    /// charged. Returns `None` when the attempt already left `Unbound` or the
    /// selector refused every candidate.
    pub(super) fn commit_decode(
        &self,
        attempt: &PdAttempt,
        select: impl FnOnce(&Loads) -> Option<StageDecision>,
    ) -> Option<StageDecision> {
        let inner = &attempt.0;
        let mut state = inner.state.lock();
        if state.decode != DecodeState::Unbound {
            return None;
        }
        let mut loads = self.loads.lock();
        let decision = select(&loads)?;
        let d = loads
            .entry((Stage::Decode, decision.endpoint.clone()))
            .or_default();
        d.pending_decode_requests += 1;
        d.publish(Stage::Decode, &decision.endpoint);
        state.decode_endpoint = Some(decision.endpoint.clone());
        state.affinity = decision.affinity.clone();
        state.pending_since = Some(std::time::Instant::now());
        state.decode = DecodeState::Pending;
        metrics::histogram!("vllm_router_pd_decode_selection_gap_seconds")
            .record(inner.admitted_at.elapsed().as_secs_f64());
        Some(decision)
    }

    /// Charge a P/D pair the caller already bound.
    ///
    /// Program scheduling picks the decode engine by affinity before dispatch,
    /// so it cannot go through policy selection, but its occupancy must stay
    /// visible to native admission and to `/metrics`. The prefill token count is
    /// unknown here, so the attempt is charged as an unknown-size prefill.
    pub(super) fn admit_bound(
        &self,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
    ) -> PdAttempt {
        let prefill = PdEndpoint::for_worker(prefill_worker.as_ref());
        let decode = PdEndpoint::for_worker(decode_worker.as_ref());
        {
            let mut loads = self.loads.lock();
            let p = loads.entry((Stage::Prefill, prefill.clone())).or_default();
            p.prefill_requests += 1;
            p.unknown_prefill_requests += 1;
            p.publish(Stage::Prefill, &prefill);
            let d = loads.entry((Stage::Decode, decode.clone())).or_default();
            d.pending_decode_requests += 1;
            d.publish(Stage::Decode, &decode);
        }
        let attempt = PdAttempt(Arc::new(AttemptInner {
            admitted_at: std::time::Instant::now(),
            loads: self.loads.clone(),
            prefill,
            prefill_worker: Mutex::new(None),
            decode_worker: Mutex::new(None),
            prefill_tokens: None,
            state: Mutex::new(AttemptState {
                prefill: true,
                decode: DecodeState::Pending,
                decode_endpoint: Some(decode),
                affinity: None,
                pending_since: Some(std::time::Instant::now()),
            }),
        }));
        attempt.attach_workers(prefill_worker, decode_worker);
        attempt
    }
}

/// One unit's charge, recorded under the loads lock so its attempt handle
/// can be built after the lock is released.
struct ChargedUnit {
    prefill: PdEndpoint,
    prefill_tokens: Option<u64>,
    decode_endpoint: Option<PdEndpoint>,
    affinity: Option<Arc<AffinityRecord>>,
    decode_state: DecodeState,
    pending_since: Option<std::time::Instant>,
}

/// Charge one selection against `loads`. Neither stage is charged until the
/// selection has succeeded; the caller owns rollback on failure.
fn charge_selection(
    loads: &mut Loads,
    selection: &PdSelection,
    defer_decode: bool,
    now: std::time::Instant,
) -> ChargedUnit {
    let prefill = &selection.prefill;
    let decode = &selection.decode;
    let p = loads
        .entry((Stage::Prefill, prefill.endpoint.clone()))
        .or_default();
    p.prefill_requests += 1;
    if let Some(tokens) = prefill.new_prefill_tokens {
        p.active_prefill_tokens += tokens;
    } else {
        p.unknown_prefill_requests += 1;
    }
    p.publish(Stage::Prefill, &prefill.endpoint);
    let (decode_endpoint, affinity, decode_state, pending_since) = if defer_decode {
        (None, None, DecodeState::Unbound, None)
    } else {
        let d = loads
            .entry((Stage::Decode, decode.endpoint.clone()))
            .or_default();
        d.pending_decode_requests += 1;
        d.publish(Stage::Decode, &decode.endpoint);
        (
            Some(decode.endpoint.clone()),
            decode.affinity.clone(),
            DecodeState::Pending,
            Some(now),
        )
    };
    ChargedUnit {
        prefill: prefill.endpoint.clone(),
        prefill_tokens: prefill.new_prefill_tokens,
        decode_endpoint,
        affinity,
        decode_state,
        pending_since,
    }
}

/// Reverse `charge_selection`: subtract the charge and drop idle entries.
/// Mirrors the arithmetic in `AttemptInner::finish_prefill` /
/// `finish_decode`, but without an attempt handle (no worker counters: those
/// are only incremented at attach time, after admission succeeds).
fn uncharge_selection(loads: &mut Loads, charged: &ChargedUnit) {
    let prefill_key = (Stage::Prefill, charged.prefill.clone());
    if let Some(load) = loads.get_mut(&prefill_key) {
        load.prefill_requests -= 1;
        if let Some(tokens) = charged.prefill_tokens {
            load.active_prefill_tokens -= tokens;
        } else {
            load.unknown_prefill_requests -= 1;
        }
        load.publish(Stage::Prefill, &charged.prefill);
        if load.idle() {
            loads.remove(&prefill_key);
        }
    }
    if let Some(endpoint) = &charged.decode_endpoint {
        let decode_key = (Stage::Decode, endpoint.clone());
        if let Some(load) = loads.get_mut(&decode_key) {
            load.pending_decode_requests -= 1;
            load.publish(Stage::Decode, endpoint);
            if load.idle() {
                loads.remove(&decode_key);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeState {
    /// Admitted without a decode commitment; nothing charged.
    Unbound,
    Pending,
    Active,
    Finished,
}

#[derive(Debug)]
struct AttemptState {
    prefill: bool,
    decode: DecodeState,
    /// Committed decode endpoint; `None` while the commitment is deferred.
    decode_endpoint: Option<PdEndpoint>,
    /// Decode binding held alive until this attempt finishes.
    affinity: Option<Arc<AffinityRecord>>,
    /// When the decode slot was charged, for the pending histogram.
    pending_since: Option<std::time::Instant>,
}

struct AttemptInner {
    admitted_at: std::time::Instant,
    loads: Arc<Mutex<Loads>>,
    prefill: PdEndpoint,
    prefill_worker: Mutex<Option<Arc<dyn Worker>>>,
    decode_worker: Mutex<Option<Arc<dyn Worker>>>,
    prefill_tokens: Option<u64>,
    /// Guards every stage transition and the decode fields below, so a commit
    /// and a release cannot interleave halfway through.
    state: Mutex<AttemptState>,
}

impl AttemptInner {
    fn finish_prefill(&self) {
        let mut state = self.state.lock();
        if !state.prefill {
            return;
        }
        let mut loads = self.loads.lock();
        let key = (Stage::Prefill, self.prefill.clone());
        if let Some(load) = loads.get_mut(&key) {
            load.prefill_requests -= 1;
            if let Some(tokens) = self.prefill_tokens {
                load.active_prefill_tokens -= tokens;
            } else {
                load.unknown_prefill_requests -= 1;
            }
            load.publish(Stage::Prefill, &self.prefill);
            if load.idle() {
                loads.remove(&key);
            }
        }
        if let Some(worker) = self.prefill_worker.lock().as_ref() {
            worker.decrement_load();
        }
        state.prefill = false;
    }

    fn begin_decode(&self) {
        let mut state = self.state.lock();
        if state.decode != DecodeState::Pending {
            return;
        }
        let Some(endpoint) = state.decode_endpoint.clone() else {
            return;
        };
        let mut loads = self.loads.lock();
        let load = loads
            .get_mut(&(Stage::Decode, endpoint.clone()))
            .expect("admitted decode");
        load.pending_decode_requests -= 1;
        load.active_decode_requests += 1;
        load.publish(Stage::Decode, &endpoint);
        if let Some(since) = state.pending_since {
            metrics::histogram!("vllm_router_pd_decode_pending_seconds")
                .record(since.elapsed().as_secs_f64());
        }
        state.decode = DecodeState::Active;
    }

    fn finish_decode(&self) {
        let mut state = self.state.lock();
        match state.decode {
            DecodeState::Finished => return,
            // Nothing was charged, and closing the state here is what makes a
            // later commit fail instead of leaking a charge.
            DecodeState::Unbound => {
                state.decode = DecodeState::Finished;
                return;
            }
            DecodeState::Pending | DecodeState::Active => {}
        }
        let Some(endpoint) = state.decode_endpoint.clone() else {
            state.decode = DecodeState::Finished;
            return;
        };
        let mut loads = self.loads.lock();
        let key = (Stage::Decode, endpoint.clone());
        if let Some(load) = loads.get_mut(&key) {
            match state.decode {
                DecodeState::Pending => load.pending_decode_requests -= 1,
                DecodeState::Active => load.active_decode_requests -= 1,
                DecodeState::Unbound | DecodeState::Finished => unreachable!(),
            }
            load.publish(Stage::Decode, &endpoint);
            if load.idle() {
                loads.remove(&key);
            }
        }
        if let Some(worker) = self.decode_worker.lock().as_ref() {
            worker.decrement_load();
        }
        state.pending_since = None;
        state.decode = DecodeState::Finished;
    }
}

impl Drop for AttemptInner {
    fn drop(&mut self) {
        self.finish_prefill();
        self.finish_decode();
        // Restart the binding's idle clock after this attempt finishes.
        if let Some(record) = self.state.lock().affinity.take() {
            record.release_ref();
        }
    }
}

#[derive(Clone)]
pub struct PdAttempt(Arc<AttemptInner>);

impl std::fmt::Debug for AttemptInner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AttemptInner")
            .field("prefill", &self.prefill)
            .field("state", &self.state)
            .finish()
    }
}

impl std::fmt::Debug for PdAttempt {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, formatter)
    }
}

/// Body ownership releases D even when a producer still holds another clone
/// of the attempt handle. The last attempt clone separately cleans up P.
struct DecodeBodyGuard(PdAttempt);

impl Drop for DecodeBodyGuard {
    fn drop(&mut self) {
        self.0.finish_decode();
    }
}

impl PdAttempt {
    pub(super) fn attach_workers(
        &self,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
    ) {
        self.attach_prefill_worker(prefill_worker);
        self.attach_decode_worker(decode_worker);
    }

    pub(super) fn attach_prefill_worker(&self, prefill_worker: Arc<dyn Worker>) {
        prefill_worker.increment_load();
        *self.0.prefill_worker.lock() = Some(prefill_worker);
    }

    /// Register the decode worker once it is known. Deferred commitments learn
    /// it at dispatch time instead of at admission.
    pub(super) fn attach_decode_worker(&self, decode_worker: Arc<dyn Worker>) {
        decode_worker.increment_load();
        *self.0.decode_worker.lock() = Some(decode_worker);
    }

    /// True until the decode stage is charged, i.e. for a deferred commitment.
    pub(super) fn decode_pending_commit(&self) -> bool {
        self.0.state.lock().decode == DecodeState::Unbound
    }

    /// Committed decode endpoint, or `None` while the commitment is deferred.
    #[cfg(test)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(super) fn decode_endpoint(&self) -> Option<PdEndpoint> {
        self.0.state.lock().decode_endpoint.clone()
    }
    pub fn finish_prefill(&self) {
        self.0.finish_prefill();
    }
    pub fn begin_decode(&self) {
        self.0.begin_decode();
    }
    pub fn finish_decode(&self) {
        self.0.finish_decode();
    }
    pub fn release(&self) {
        self.finish_prefill();
        self.finish_decode();
    }

    pub fn track_response(self, response: axum::response::Response) -> axum::response::Response {
        if self.0.state.lock().decode != DecodeState::Active {
            self.release();
            return response;
        }
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, self.track_body(body))
    }

    /// Own occupancy through EOF, stream failure, or early client body drop.
    /// Capturing the guard in unfold's state also covers an unpolled body.
    pub fn track_body(self, body: Body) -> Body {
        let stream = futures_util::stream::unfold(
            (body.into_data_stream(), DecodeBodyGuard(self)),
            |(mut stream, attempt)| async move {
                match stream.next().await {
                    Some(Ok(bytes)) => Some((Ok(bytes), (stream, attempt))),
                    Some(Err(error)) => {
                        attempt.0.finish_decode();
                        Some((Err(error), (stream, attempt)))
                    }
                    None => {
                        attempt.0.finish_decode();
                        None
                    }
                }
            },
        );
        Body::from_stream(stream)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::fmt;

    // Simple custom error for testing error chains
    #[derive(Debug)]
    struct TestError {
        msg: String,
        source: Option<Box<dyn std::error::Error>>,
    }

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.msg)
        }
    }

    impl std::error::Error for TestError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.source.as_deref()
        }
    }

    #[test]
    fn test_error_chain_single_error() {
        let err = TestError {
            msg: "something broke".into(),
            source: None,
        };
        assert_eq!(error_chain(&err), "something broke");
    }

    #[test]
    fn test_error_chain_nested_errors() {
        let inner = TestError {
            msg: "root cause".into(),
            source: None,
        };
        let outer = TestError {
            msg: "outer error".into(),
            source: Some(Box::new(inner)),
        };
        assert_eq!(error_chain(&outer), "outer error caused by: root cause");
    }

    #[test]
    fn test_error_chain_triple_nested() {
        let root = TestError {
            msg: "connection reset".into(),
            source: None,
        };
        let middle = TestError {
            msg: "HTTP send failed".into(),
            source: Some(Box::new(root)),
        };
        let top = TestError {
            msg: "prefill request failed".into(),
            source: Some(Box::new(middle)),
        };
        assert_eq!(
            error_chain(&top),
            "prefill request failed caused by: HTTP send failed caused by: connection reset"
        );
    }

    fn test_endpoint(url: &str) -> PdEndpoint {
        PdEndpoint {
            http_url: url.to_string(),
            dp_rank: None,
        }
    }

    fn test_decision(
        endpoint: PdEndpoint,
        worker_index: usize,
        tokens: Option<u64>,
    ) -> StageDecision {
        StageDecision {
            fallback_reason: None,
            credited_prefix_tokens: None,
            worker_index,
            endpoint,
            score: None,
            new_prefill_tokens: tokens,
            affinity_outcome: None,
            affinity: None,
        }
    }

    fn test_selection(tokens: u64, call: usize) -> PdSelection {
        // Alternate endpoints so the batch spreads across pairs.
        let (prefill_url, decode_url) = if call % 2 == 0 {
            ("http://p0", "http://d0")
        } else {
            ("http://p1", "http://d1")
        };
        PdSelection {
            prefill: test_decision(test_endpoint(prefill_url), 0, Some(tokens)),
            decode: test_decision(test_endpoint(decode_url), 0, None),
        }
    }

    #[test]
    fn admit_many_charges_all_units() {
        let manager = PdLoadManager::default();
        let calls = std::cell::Cell::new(0usize);
        let admitted = manager.admit_many(
            |_| {
                let call = calls.get();
                calls.set(call + 1);
                Some(test_selection(10 * (call as u64 + 1), call))
            },
            3,
            false,
        );
        let admitted = admitted.expect("all three units admit");
        assert_eq!(admitted.len(), 3);
        // p0/d0 take units 0 and 2 (10 + 30 tokens), p1/d1 take unit 1 (20).
        let loads = manager.snapshot();
        let p0 = loads
            .get(&(Stage::Prefill, test_endpoint("http://p0")))
            .unwrap();
        assert_eq!(p0.prefill_requests, 2);
        assert_eq!(p0.active_prefill_tokens, 40);
        let p1 = loads
            .get(&(Stage::Prefill, test_endpoint("http://p1")))
            .unwrap();
        assert_eq!(p1.prefill_requests, 1);
        assert_eq!(p1.active_prefill_tokens, 20);
        let d0 = loads
            .get(&(Stage::Decode, test_endpoint("http://d0")))
            .unwrap();
        assert_eq!(d0.pending_decode_requests, 2);
        let d1 = loads
            .get(&(Stage::Decode, test_endpoint("http://d1")))
            .unwrap();
        assert_eq!(d1.pending_decode_requests, 1);
        assert_eq!(
            admitted[0].1.decode_endpoint(),
            Some(test_endpoint("http://d0"))
        );
        assert_eq!(
            admitted[1].1.decode_endpoint(),
            Some(test_endpoint("http://d1"))
        );
    }

    #[test]
    fn admit_many_rolls_back_on_failure() {
        let manager = PdLoadManager::default();
        let calls = std::cell::Cell::new(0usize);
        let admitted = manager.admit_many(
            |_| {
                let call = calls.get();
                calls.set(call + 1);
                if call >= 2 {
                    None
                } else {
                    Some(test_selection(10, call))
                }
            },
            3,
            false,
        );
        assert!(admitted.is_none());
        assert!(manager.snapshot().is_empty());
    }

    #[test]
    fn admit_many_deferred_charges_prefill_only() {
        let manager = PdLoadManager::default();
        let calls = std::cell::Cell::new(0usize);
        let admitted = manager
            .admit_many(
                |_| {
                    let call = calls.get();
                    calls.set(call + 1);
                    Some(test_selection(7, call))
                },
                2,
                true,
            )
            .expect("deferred units admit");
        assert_eq!(admitted.len(), 2);
        assert!(admitted
            .iter()
            .all(|(_, attempt)| attempt.decode_pending_commit()));
        assert!(admitted
            .iter()
            .all(|(_, attempt)| attempt.decode_endpoint().is_none()));
        let loads = manager.snapshot();
        assert_eq!(loads.len(), 2);
        assert!(loads.contains_key(&(Stage::Prefill, test_endpoint("http://p0"))));
        assert!(loads.contains_key(&(Stage::Prefill, test_endpoint("http://p1"))));
    }

    #[test]
    fn admit_many_drop_releases_charges() {
        let manager = PdLoadManager::default();
        let admitted = manager
            .admit_many(|_| Some(test_selection(5, 0)), 2, false)
            .expect("units admit");
        assert_eq!(admitted.len(), 2);
        drop(admitted);
        assert!(manager.snapshot().is_empty());
    }
}
