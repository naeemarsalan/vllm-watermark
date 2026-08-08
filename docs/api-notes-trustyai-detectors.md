# TrustyAI/FMS Guardrails detector contract + orchestrator wiring

**Mostly STATIC** (source/spec reading, not execution against a live
detector or orchestrator process — no detector/orchestrator binary was run
in this task). Registry/API lookups (GitHub API, quay.io registry API) are
tagged `OFFICIAL-SRC`. The RHOAI/docs.redhat.com cross-check (§4) could
**not** be fetched directly in this environment (Akamai `403` on every
attempt — see §4) and is `CORROBORATED` via `WebSearch`-extracted text
only, not raw-fetched and independently re-verified character-for-character.
Fetched 2026-08-08. Every field name below is quoted from a cited file/line,
never from memory.

This is the single source of truth for Tasks B3/C3. Do not restate a
schema field from memory — cite this doc's section or re-fetch.

---

## 0. Pinned sources

| Repo | Ref used | Resolved commit | How pinned |
|---|---|---|---|
| `trustyai-explainability/guardrails-detectors` | `main` (HEAD) | `747a4d3ef6f7d384b73f929a0162228ad56d98de` (2026-06-22) | `gh api repos/.../commits/main` |
| `foundation-model-stack/fms-guardrails-orchestrator` | tag `0.18.3` (latest release, 2026-01-15) | `6d2cec987223335adcc3803f884dae7a4aa59492` | `gh api repos/.../git/refs/tags/0.18.3` |

**Why `main` for the detectors repo, not a release tag:** its latest
*release* is `v0.3.0` (2025-05-22), but `main` has 13+ months of unreleased
commits (latest push 2026-06-22) including the `huggingface` detector,
`llm_judge` detector, and the current `common/scheme.py` models — none of
which exist at `v0.3.0`. Using the stale tag would describe a contract that
predates most of what RHOAI ships today. Flagging this explicitly: **this
repo has no current tagged release that matches its `main` branch**
(`OFFICIAL-SRC`, `gh api repos/trustyai-explainability/guardrails-detectors/tags`).

---

## 1. Detector contract: `POST /api/v1/text/contents`

### 1.1 Request/response models (pydantic)

Source: `detectors/common/scheme.py`
(https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/common/scheme.py)

```python
class ContentAnalysisHttpRequest(BaseModel):
    contents: List[str] = Field(
        min_length=1,
        title="Contents",
        description="Field allowing users to provide list of texts for analysis. ...",
    )
    detector_params: Optional[Dict] = Field(
        description="Optional detector parameters, used on a per-detector basis"
    )

class ContentAnalysisResponse(BaseModel):
    start: int = Field(example=14)
    end: int = Field(example=26)
    text: str = Field(example="abc@def.com")
    detection: str = Field(default="detection", example="Net.EmailAddress")
    detection_type: str = Field(example="pii")
    score: float = Field(example=0.8)
    evidences: Optional[List[EvidenceObj]] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ContentsAnalysisResponse(RootModel):
    root: List[List[ContentAnalysisResponse]]

class Error(BaseModel):
    code: int
    message: str
```

Notes:
- `detector_params` is typed `Optional[Dict]` but has **no `default=`** in
  its `Field(...)` call. `fastapi==0.136.3` (pinned in
  `detectors/pyproject.toml`) pulls Pydantic v2, where `Optional[X]` does
  **not** implicitly default to `None` the way it did in Pydantic v1 — a
  field is only optional-to-omit if it has an explicit default. Read
  strictly, `detector_params` is a **required key** in the JSON body (its
  *value* may be `null`/`{}`). Every example in this repo's own docs sends
  it explicitly as `{}` or a populated dict — never omits it. (`STATIC` —
  not executed against the real server to confirm 422-on-omission.)
- Response is **list-of-lists**, not objects keyed by detector: the outer
  list has exactly `len(contents)` elements, in the same order; each inner
  list holds zero or more detections for that one content string. An empty
  inner list `[]` means "no detection" (safe), not an error.
- `Error{code, message}` — note the field name is **`message`**, not
  `details` (contrast with the orchestrator's own error shape in §2.6,
  which uses `details`).

### 1.2 Two live implementations of the same route

There is no single canonical handler — each detector *kind* in this repo
ships its own FastAPI app that mounts the same path.

**a) `huggingface` detector** — `detectors/huggingface/app.py`
(https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/detectors/huggingface/app.py):

```python
@app.post(
    "/api/v1/text/contents",
    response_model=ContentsAnalysisResponse,
    responses={404: {"model": Error, ...}, 422: {"model": Error, ...}},
)
async def detector_unary_handler(request: ContentAnalysisHttpRequest):
    detectors: List[Detector] = list(app.get_all_detectors().values())
    if not len(detectors) or not detectors[0]:
        raise RuntimeError("Detector is not initialized")
    result = await run_in_threadpool(detectors[0].run, request)
    return ContentsAnalysisResponse(root=result)
```

One container = one loaded model (`MODEL_DIR` env var), so it ignores
`detector-id`/routing entirely and always runs the single registered
detector. `detector.py`'s `run()` dispatches per model architecture
(`GraniteForCausalLM` → risk-name loop over 7 fixed risk categories;
`*ForSequenceClassification` → per-label threshold check;
`*ForTokenClassification` → per-token span extraction), each producing
`ContentAnalysisResponse` objects with `start=0, end=len(text)` for the
sequence-classifier/causal-lm cases, or real char offsets for token
classification.

Verbatim example from `docs/hf_examples.md`
(https://github.com/trustyai-explainability/guardrails-detectors/blob/747a4d3ef6f7d384b73f929a0162228ad56d98de/docs/hf_examples.md):

```bash
curl -X POST http://localhost:8000/api/v1/text/contents \
  -H 'accept: application/json' \
  -H 'detector-id: hap' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": ["You dotard, I really hate this stuff", "I simply love this stuff"],
    "detector_params": {}
  }'
```
```json
[
  [
    {
      "start": 0, "end": 36,
      "text": "You dotard, I really hate this stuff",
      "detection": "single_label_classification",
      "detection_type": "LABEL_1",
      "score": 0.9634233713150024,
      "evidences": []
    }
  ],
  []
]
```

Per-request `detector_params` keys this implementation actually reads
(`detector.py::_resolve_params`, `STATIC`): `threshold` (float 0–1),
`label_thresholds` (dict label→float), `safe_labels` (list of str/int),
`max_length` (positive int, clamped to model capacity) — all optional,
env-var (`THRESHOLD`, `LABEL_THRESHOLDS`, `SAFE_LABELS`, `MAX_LENGTH`)
defaults apply when absent from the request.

**b) `builtIn` detector** — `detectors/built_in/app.py`:

```python
@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
def detect_content(request: ContentAnalysisHttpRequest, raw_request: Request):
    headers = dict(raw_request.headers)
    detections = []
    for content in request.contents:
        message_detections = []
        for detector_kind in request.detector_params:
            detector_registry = app.get_all_detectors().get(detector_kind)
            if detector_registry is None:
                raise HTTPException(status_code=400, detail=f"Detector {detector_kind} not found")
            ...
            message_detections += detector_registry.handle_request(content, request.detector_params, headers)
        detections.append(message_detections)
    return ContentsAnalysisResponse(root=detections)
```

Here `detector_params`' **top-level keys** (`regex`, `file_type`, custom
names) select which built-in sub-detector runs — `detector-id` is
irrelevant to routing in this implementation too. Example
(`docs/builtin_examples.md`):

```bash
curl -X POST http://localhost:8080/api/v1/text/contents \
  -H "Content-Type: application/json" \
  -d '{"contents": ["Hi my email is abc@def.com", "..."], "detector_params": {"regex": ["email", "us-phone-number"]}}'
```

**Conclusion for our design:** in *both* reference implementations,
`detector-id` (header) is accepted/documented but **not read by the
Python handler code** — it's present because the orchestrator always
sends it (§2.5) and because the OpenAPI contract (§1.3) marks it
`required`, but nothing in this repo's server code inspects
`request.headers["detector-id"]`. Scheme selection has to come from
`detector_params` (or from which container/route is hit), not from that
header, if we want to follow this repo's own pattern. (`STATIC`.)

### 1.3 Headers, per the orchestrator's copy of this API's OpenAPI spec

The detectors repo itself ships no OpenAPI YAML; the orchestrator repo
vendors one describing the same contract:
`docs/api/openapi_detector_api.yaml`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/docs/api/openapi_detector_api.yaml):

```yaml
/api/v1/text/contents:
  post:
    parameters:
      - name: detector-id
        in: header
        required: true
        schema: { type: string, title: Detector-Id }
        example: dummy-en-pii-v1
    responses:
      "200": {...}
      "404": { schema: {$ref: '#/components/schemas/Error'} }
      "422": { schema: {$ref: '#/components/schemas/Error'} }
```

So the *contract* says `detector-id` is `required: true` on this endpoint
(and `required: false` with a default on `/api/v1/text/chat`) — but as
shown in §1.2, neither shipped Python implementation actually enforces or
reads it. Treat "required" here as a client-side (orchestrator) promise,
not something the reference detector servers validate.
`Content-Type: application/json` is required by both implementations
(FastAPI's default JSON body parsing).

### 1.4 Error conventions

`detectors/common/app.py`'s `DetectorBaseAPI` installs two exception
handlers app-wide:

- Pydantic `RequestValidationError` → `422` with body
  `{"code": 422, "message": "Missing required parameters: [...]"}` (for
  `type == "missing"`), or `"Parameters with invalid type: [...]"`, or a
  generic `"Invalid parameters: [...]"` fallback.
- `StarletteHTTPException` → passthrough with `{"code": <status>, "message": <detail>}`.

Handler-level `HTTPException`s seen in the code: `400` ("Detector `{kind}`
not found" — built-in detector, unknown `detector_params` key), `500`
("Detection error, check detector logs" — `BaseDetectorRegistry.throw_internal_detector_error`,
used by regex/file-type/custom registries) and a bare `500` with no detail
(huggingface `detector_unary_handler`'s unhandled exceptions bubble up as
FastAPI's default 500).

`GET /health` (added centrally by `DetectorBaseAPI.__init__`, both
implementations inherit it) returns the **plain string** `"ok"`, not a
JSON object — `async def health(): return "ok"`.

### 1.5 Chunking is not a detector-service concern

Nothing in `detectors/common/`, `detectors/huggingface/`, or
`detectors/built_in/` has a notion of "chunk" or "chunker" at all — a
detector server just runs its detection over whatever strings appear in
`contents`. Whether those strings are per-sentence fragments or one
whole document is decided entirely on the orchestrator side before the
call is made (§3). This confirms whole-document (non-chunked) detection
is possible and needs zero support from the detector implementation.

---

## 2. Orchestrator: config schema, endpoint(s), health, image

### 2.1 `config.yaml` schema (Rust source, ground truth)

Source: `src/config.rs`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/config.rs)

```rust
pub struct ServiceConfig {
    pub hostname: String,
    pub port: Option<u16>,
    pub path_prefix: Option<String>,      // see §2.3
    pub request_timeout: Option<u64>,
    pub tls: Option<Tls>,
    pub grpc_dns_probe_interval: Option<u64>,
    pub resolution_strategy: Option<String>,
    pub resolution_strategy_timeout: Option<u64>,
    pub max_retries: Option<usize>,
    pub http2_keep_alive_interval: Option<u64>,
    pub keep_alive_timeout: Option<u64>,
    #[serde(default, deserialize_with = "from_env")]
    pub api_token: Option<String>,        // resolved from an env-var NAME
}

pub struct DetectorConfig {
    pub service: ServiceConfig,
    pub health_service: Option<ServiceConfig>,
    pub chunker_id: String,               // REQUIRED — no #[serde(default)]
    pub default_threshold: f64,           // REQUIRED
    #[serde(rename = "type")]
    pub r#type: DetectorType,             // REQUIRED (text_contents/text_generation/text_chat/text_context_doc)
}

pub struct OrchestratorConfig {
    pub generation: Option<GenerationConfig>,
    #[serde(alias = "chat_generation")] #[serde(alias = "chat_completions")]
    pub openai: Option<OpenAiConfig>,
    pub chunkers: Option<HashMap<String, ChunkerConfig>>,
    pub detectors: HashMap<String, DetectorConfig>,   // required, non-empty (else Error::NoDetectorsConfigured)
    pub tls: Option<HashMap<String, TlsConfig>>,
    #[serde(default)] pub passthrough_headers: HashSet<String>,
    #[serde(default)] pub rewrite_forwarded_access_header: bool,
    #[serde(default = "default_detector_concurrent_requests")]  // 5
    pub detector_concurrent_requests: usize,
    #[serde(default = "default_chunker_concurrent_requests")]   // 5
    pub chunker_concurrent_requests: usize,
}
```

Verbatim registered-detector example from the shipped `config/config.yaml`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/config/config.yaml):

```yaml
detectors:
    hap-en:
        type: text_contents
        service:
            hostname: localhost
            port: 8080
            tls: detector
        health_service:
            hostname: localhost
            port: 8081
        chunker_id: en_regex
        default_threshold: 0.5
```

### 2.2 `chunker_id: whole_doc_chunker` — chunker is config-mandatory, but "no chunking" is a built-in option

`chunker_id: String` has no `Option<>`/`#[serde(default)]`, so every
`DetectorConfig` entry in `config.yaml` **must** name a chunker. But
`"whole_doc_chunker"` is a magic constant — `DEFAULT_CHUNKER_ID` in
`src/clients/chunker.rs` line 46 — that requires **no corresponding
entry** under the top-level `chunkers:` map (which is itself `Option<>`
and can be omitted entirely). `tests/test_config.yaml`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/tests/test_config.yaml)
demonstrates this directly:

```yaml
detectors:
  angle_brackets_detector_whole_doc:
    type: text_contents
    service: { hostname: localhost }
    chunker_id: whole_doc_chunker
    default_threshold: 0.5
```

So: **a chunker is mandatory at the config-schema level, but whole-document
(unchunked) detection is fully supported via the built-in
`whole_doc_chunker` sentinel — no chunker service needs to be deployed.**
Some client-facing endpoints additionally *reject* detectors configured
with `whole_doc_chunker` (`src/orchestrator/common/utils.rs::validate_detectors`,
`supports_whole_doc_chunker: bool` parameter) — the standalone
`/api/v2/text/detection/content` handler passes `true` for this flag
(`text_content_detection.rs`), so whole-doc detectors **are** allowed on
the endpoint we care about.

### 2.3 Per-detector URL path prefix — confirms the fallback design works

`src/clients.rs` `create_http_client()`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/clients.rs#L200-L221):

```rust
let mut base_url = Url::parse(&format!("{}://{}", protocol, &service_config.hostname))...;
base_url.set_port(Some(port))...;
if let Some(prefix) = &service_config.path_prefix {
    let trimmed = prefix.trim_matches('/');
    if !trimmed.is_empty() {
        base_url.set_path(&format!("/{}", trimmed));
    }
}
```

So `ServiceConfig.path_prefix` (Option<String>, §2.1) is real and wired
in — a detector entry configured with
`service: {hostname: h, port: p, path_prefix: "/kgw"}` sends its
`/api/v1/text/contents` request to `http://h:p/kgw/api/v1/text/contents`.
**This confirms the fallback design (two registered detector entries with
different URL paths pointing at the same backing service, e.g.
`.../kgw/api/v1/text/contents` vs `.../synthid/api/v1/text/contents`) is
directly supported by the orchestrator config schema, no workaround
needed.** `path_prefix` is recent: it shipped in release `0.18.3`
(the exact tag pinned in §0) per that release's changelog —
"Adding path prefixes so that llm-d deployments are compatible", PR #523
by @m-misiura
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/releases/tag/0.18.3,
`gh api repos/.../releases/tags/0.18.3`). **Any orchestrator build older
than 0.18.3 will not have this field** — worth pinning the deployed
version if we rely on it.

### 2.4 Standalone detection-only endpoint: `POST /api/v2/text/detection/content`

Route registration, `src/server/routes.rs`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/server/routes.rs):

```rust
.route("/api/v2/text/detection/content", post(detection_content))
```

handler → `models::TextContentDetectionHttpRequest` → validated →
`TextContentDetectionTask` → `Orchestrator::handle()` →
`TextContentDetectionResult`.

Request/response models, `src/models.rs`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/models.rs):

```rust
#[serde(deny_unknown_fields)]
pub struct TextContentDetectionHttpRequest {
    pub content: String,                              // required, non-empty (else ValidationError)
    pub detectors: HashMap<String, DetectorParams>,    // required, non-empty
}

pub struct TextContentDetectionResult {
    pub detections: Vec<ContentAnalysisResponse>,      // flat, sorted by start, merged across all requested detectors
}
```

`DetectorParams` is `BTreeMap<String, serde_json::Value>` — arbitrary
JSON-valued keys, matching the detector-side `detector_params: Optional[Dict]`.

Same shape as the orchestrator's hand-written OpenAPI doc
(`docs/api/orchestrator_openapi_0_1_0.yaml`, component names differ —
`DetectionContentRequest`/`DetectionContentResponse` there vs
`TextContentDetectionHttpRequest`/`TextContentDetectionResult` in the Rust
source; the **path and JSON shape agree**, treat the Rust struct as ground
truth if they ever diverge, since the OpenAPI YAML is hand-maintained, not
generated):

```yaml
DetectionContentRequest:
  properties:
    detectors: {type: object, default: {}, example: {hap-v1-model-en: {}}}
    content: {type: string, example: "my text here"}
  required: [detectors, content]
  additionalProperties: false
DetectionContentResponse:
  properties:
    detections: {type: array, items: {$ref: '#/components/schemas/DetectionContentResponseObject'}}
  required: [detections]
```

Verbatim JSON example request/response for this endpoint family — the
Rust source shows the shape but not a full worked example; the closest
verbatim example available (same envelope: `content`, `detectors` map)
is from `docs/api/orchestrator_openapi_0_1_0.yaml`'s
`stream-content` sibling endpoint:

```json
{"detectors": {"hap-v1-model-en": {}}, "content": "my text here"}
```

`ContentAnalysisResponse` — the element type of `detections[]` in the
response — is defined in `src/clients/detector.rs`, and is the
orchestrator's own struct (not a straight passthrough of the detector's
identically-named pydantic model in §1.1 — same field names, one field
added):

```rust
pub struct ContentAnalysisResponse {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub detection: String,
    pub detection_type: String,
    pub detector_id: Option<String>,   // stamped in by the orchestrator, not sent by the detector (see §2.5)
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")] pub evidence: Option<Vec<EvidenceObj>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")] pub metadata: Metadata,
}
```

Other `/api/v2/...` routes registered alongside it (context, not this
task's focus): `/api/v2/text/detection/stream-content` (ND-JSON
streaming, same `{content, detectors}` shape per line),
`/api/v2/text/detection/chat`, `/api/v2/text/detection/context`,
`/api/v2/text/detection/generated`, `/api/v2/text/generation-detection`,
and — only if the config has a top-level `openai:` block —
`/api/v2/chat/completions-detection` and `/api/v2/text/completions-detection`.
Legacy v1: `/api/v1/task/classification-with-text-generation` (+
server-streaming variant).

### 2.5 `detector_params` forwarding — CONFIRMED forwarded, one carve-out

This directly resolves the open question in the task: **the orchestrator
does forward client-supplied `detector_params` to the detector**, with a
single reserved-key exception.

`src/orchestrator/common/tasks.rs::text_contents_detections()`
(https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/src/orchestrator/common/tasks.rs#L185-L235):

```rust
let inputs = detectors.iter().map(|(detector_id, params)| {
    let config = ctx.config.detector(detector_id)...;
    let chunks = chunk_map.get(&config.chunker_id)...;
    Ok((detector_id.clone(), params.clone(), chunks))
})...;
// ... per detector, concurrently:
let default_threshold = ctx.config.detector(&detector_id).unwrap().default_threshold;
let threshold = params.pop_threshold().unwrap_or(default_threshold);   // <-- MUTATES params, removes "threshold" key
let detections = detect_text_contents(client, headers, detector_id.clone(), params, chunks.clone(), true).await?
    .into_iter().filter(|d| d.score >= threshold).collect();
```

`params.pop_threshold()` (`src/models.rs`) removes exactly the key
`"threshold"` from the client's per-detector `DetectorParams` map — for
the orchestrator's own score-filtering against `default_threshold` — and
only that key. Everything else survives into `detect_text_contents()` →
`src/orchestrator/common/client.rs`:

```rust
pub async fn detect_text_contents(client, headers, detector_id, params, chunks, apply_chunk_offset) -> ... {
    let contents = chunks.iter().map(|c| c.text.clone()).collect();
    let request = ContentAnalysisRequest::new(contents, params);   // params (minus "threshold") sent verbatim
    let response = client.text_contents(&detector_id, request, headers).await?;
    ...
}
```

`ContentAnalysisRequest` (`src/clients/detector.rs`) is exactly
`{contents: Vec<String>, detector_params: DetectorParams}` — serialized
straight onto the wire as the detector's own request body from §1.1.

**Practical consequence for our design:** a client can send
`"detectors": {"kgw-detector": {"scheme": "kgw"}}` (or any custom key) to
`/api/v2/text/detection/content`, and `{"scheme": "kgw"}` arrives at our
detector's `detector_params` unmodified. The only reserved word is
literally `"threshold"` — pick anything else for scheme selection. This
means the fallback (two registered detector entries with different
`path_prefix`es, §2.3) is available but not *required* — per-request
`detector_params` forwarding alone is sufficient for scheme selection,
confirmed by source, not by assumption.

### 2.6 Headers sent to the detector, and error-response shapes

`src/clients/detector.rs::DetectorClient::post()`:

```rust
const DETECTOR_ID_HEADER_NAME: &str = "detector-id";
const MODEL_HEADER_NAME: &str = "x-model-name";
...
headers.append(DETECTOR_ID_HEADER_NAME, model_id.parse().unwrap());
headers.append(CONTENT_TYPE, JSON_CONTENT_TYPE);
headers.append(MODEL_HEADER_NAME, model_id.parse().unwrap());
```

`model_id` here is the detector's *configured ID* (the YAML key under
`detectors:`, e.g. `hap-en`), not anything client-supplied. So the
orchestrator **always** sends `detector-id` and `x-model-name` set to the
config key, on every call — corroborates §1.3's claim that `detector-id`
is a promise from the client side, even though the reference detector
implementations ignore it. Beyond these two plus `Content-Type`, only
headers whose lowercase name is in the top-level
`passthrough_headers: HashSet<String>` (default **empty** — nothing is
forwarded unless explicitly allow-listed) are copied from the original
inbound client request (`src/server/routes.rs::filter_headers()`), with
an optional `rewrite_forwarded_access_header` flag that turns an inbound
`X-Forwarded-Access-Token` into `Authorization: Bearer <token>`.

Error mapping to the orchestrator's own client-facing response,
`src/server/errors.rs::From<orchestrator::Error> for Error`:

```rust
pub struct Error {                       // orchestrator's OWN error shape — note "details", not "message"
    #[serde(with = "http_serde::status_code")] pub code: StatusCode,
    pub details: String,
}
```
```rust
match value {
    DetectorNotFound(_) | ChunkerNotFound(_) => 404,
    DetectorRequestFailed{error,..} | ChunkerRequestFailed{..} | ... => match error.status_code() {
        400 | 422 | 404 | 503 => passthrough that code, details = value.to_string(),
        _ => 500, details = "unexpected error occurred while processing request",  // masks e.g. a raw detector 500
    },
    JsonError(_) | Validation(_) => 422,
    _ => 500 (generic),
}
```

So: a **downstream detector 500 does not leak** into the orchestrator's
client response — it's collapsed to a generic masked 500. Only detector
400/422/404/503 pass through with their original body text. Compare the
detector's own error field name (`message`, §1.1/1.4) against the
orchestrator's (`details`) — **do not assume the two error envelopes are
interchangeable** when building client-side error handling.

### 2.7 Health / readiness

`src/server/routes.rs::health_router()`:

```rust
.route("/health", get(health))
.route("/info", get(info))
```

- `GET /health` — **liveness only**, always `200`, body
  `{"<CARGO_PKG_NAME>": "<CARGO_PKG_VERSION>"}`. Does **not** probe any
  dependency. `Cargo.toml` pins `name = "fms-guardrails-orchestr8"`,
  `version = "0.18.3"` at the ref in §0
  (https://github.com/foundation-model-stack/fms-guardrails-orchestrator/blob/6d2cec987223335adcc3803f884dae7a4aa59492/Cargo.toml)
  — i.e. `GET /health` on this build literally returns
  `{"fms-guardrails-orchestr8": "0.18.3"}`. This also explains the
  "orchestr8" name in the task prompt: it's the internal crate/binary
  name, distinct from the GitHub repo name
  `fms-guardrails-orchestrator`.
- `GET /info?probe=<bool>` — **readiness**, probes (or returns cached
  results for) every configured client (`generation`, `openai`,
  `chunkers`, `detectors`), returns `200` if the orchestrator successfully
  probed (not necessarily all-healthy — read the body) or `503` if the
  probe itself failed. Response: `{"services": {"<name>": HealthCheckResult}}`
  where `HealthCheckResult{status: "HEALTHY"|"UNHEALTHY"|"UNKNOWN", code?: <int, omitted if 2xx>, reason?: <string>}`
  (`src/health.rs`).
- Each detector's own health is checked via its `health_service:` config
  block (falls back to the primary `service:` block if omitted) —
  concretely a `GET /health` against that detector pod, which per §1.4
  returns the bare string `"ok"` in this repo's reference implementations.

### 2.8 Container image

The orchestrator repo itself has no image-publish GitHub Action (only
`Dockerfile.amd64`/`.ppc64le`/`.s390x` — build recipes, no CI push
workflow); images are built/published by a separate Red Hat/ODH Konflux
pipeline. Two relevant quay.io repos found, both queried live via the
**public** quay.io registry API (`GET https://quay.io/api/v1/repository/<ns>/<name>?includeTags=true`,
no auth required, `OFFICIAL-SRC`, fetched 2026-08-08):

**Upstream ODH-branded build** — `quay.io/opendatahub/ta-guardrails-orchestrator`:

| Tag | Pushed | Manifest digest |
|---|---|---|
| `odh-3.4.2.git` | 2026-04-06T21:17:48Z | `sha256:147cc8586a2a9455a896dd51c28e8fc4d60d8aa416131d3e0d1aff0e97f52083` |
| `odh-3.4-ea2.git` | 2026-03-09T18:25:11Z | `sha256:b8b4dcd27933bea64c2779262aca8d29ca503d8bab17286fcf505c187db36ecd` |
| `odh-stable` | 2026-01-20T09:20:07Z | `sha256:757f3c3e05fe0c40bcfa8d29d0511ddcaff640a38c4ee507696cf126009381b6` |

`odh-3.4.2.git` is the newest by push date and its tag name lines up with
the RHOAI 3.4.x train this repo targets (AGENTS.md). **`odh-stable` is
older** than `odh-3.4.2.git` despite the name — don't assume `-stable`
means "latest".

**Red-Hat-productized downstream image** (the one referenced in RHOAI's
own disconnected-install image lists, e.g.
`red-hat-data-services/rhoai-disconnected-install-helper`) —
`quay.io/modh/odh-fms-guardrails-orchestrator-rhel9`:

| Tag | Pushed | Manifest digest |
|---|---|---|
| `rhoai-2.22` | 2025-11-17T14:06:xxZ | `sha256:ec0172d86ff676f3433c2b51e51379cc878042622fcf86581f31bf6ee4ed95ed` |

This repo's newest tag is `rhoai-2.22` (Nov 2025) — it has **no entry for
any `3.x` RHOAI release**. Cross-checked: `rhoai-3.4.md` in the same
`rhoai-disconnected-install-helper` repo (fetched clean via `gh api`, not
blocked — GitHub, not `docs.redhat.com`) lists "Additional images" for
RHOAI 3.4 and contains **zero** references to
`fms-guardrails-orchestrator`, `guardrails-detectors`, or
`odh-fms-guardrails-orchestrator-rhel9`. This is consistent with (but does
not by itself prove) fact `C11`'s claim that RHOAI 3.4 downgrades FMS
Guardrails — it's also consistent with the productized-image pipeline
having simply moved to a different registry path for 3.x that this
particular helper repo doesn't track. **Flag as `OPEN`**: which exact
`quay.io`/`registry.redhat.io` image the RHOAI 3.4 `GuardrailsOrchestrator`
CR actually pulls was not directly confirmed (the CR references images
indirectly through operator-managed defaults, not a user-visible
`image:` field in the CRs seen in §4).

---

## 3. Chunker requirement — summary

Answered fully in §2.2: **`chunker_id` is a required field in every
`DetectorConfig` entry, but the sentinel value `whole_doc_chunker`
(`DEFAULT_CHUNKER_ID`, `src/clients/chunker.rs:46`) needs no chunker
service and no `chunkers:` config entry — it's the orchestrator's built-in
"treat the whole document as one chunk" behavior.** Whole-document
detection is fully possible; per-sentence chunking is opt-in, not
mandatory.

---

## 4. RHOAI 3.4 docs cross-check — **partial, network-limited**

**Direct fetch of `docs.redhat.com` was blocked in this environment on
every attempt** — `WebFetch` and `curl` both got Akamai edge `403`s
(`Access Denied`, reference IDs logged, e.g.
`18.21847b5c.1786162915.8a279140`) against:
- `.../3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-guardrails_safety`
- `.../3.4/html-single/enabling_ai_safety_with_guardrails/index`
- `.../3.4/pdf/enabling_ai_safety_with_guardrails/....pdf`

This is the same URL cited by `docs/facts.md` fact `C11` as
`OFFICIAL-SRC` "fetched 2026-08-08" — I could **not** independently
reproduce that fetch in this session/environment to re-verify its
verbatim quote. Recording this as a gap rather than silently reusing the
prior claim: **the "legacy"/deprecation wording in `C11` should be
re-verified by whoever can reach `docs.redhat.com` from an unblocked
network path** before it's used to justify a design decision.

What follows is `CORROBORATED` only — extracted via `WebSearch` (which
appears able to read the PDF/HTML server-side even though direct
`WebFetch`/`curl` cannot), not a raw fetch I could grep myself. Treat
verbatim-looking quotes below with appropriate caution; only the
`trustyai.org` material (successfully `WebFetch`-fetched directly, not
blocked) is `OFFICIAL-SRC`.

### 4.1 Chapter reordering across RHOAI versions (`CORROBORATED`)

Search results show the guardrails guide's chapter structure changed
release over release:
- RHOAI 3.0–3.3: "Chapter 1. Enabling AI safety with Guardrails" (generic/FMS-first).
- RHOAI 3.4: **"Chapter 1. Enabling AI safety with NeMo Guardrails"**
  (`https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-nemo-guardrails_nemo-guardrails`)
  — NeMo Guardrails is now Chapter 1.
- RHOAI 3.5: **"Chapter 3. Using FMS Guardrails for AI safety"**
  (`https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/enabling_ai_safety_with_guardrails/using-guardrails-for-ai-safety_safety`)
  — FMS Guardrails demoted to chapter 3, "Using" rather than "Enabling"
  in the title.

This is a real, observable trend (FMS Guardrails moving from primary to
secondary billing across 3.0→3.5) but I did not obtain the exact
"deprecated"/"legacy" sentence verbatim in this session.

### 4.2 Release-note status (`CORROBORATED`, needs re-verification)

Per `WebSearch` extraction of the RHOAI 3.4 release notes
(`https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/release_notes/new-features-and-enhancements_relnotes`):
NeMo Guardrails (Tech Preview since 3.3) is reported as **fully supported**
in 3.4, while the "Guardrails Orchestrator from TrustyAI with Llama Stack"
(i.e. the `GuardrailsOrchestrator` CR, §4.3) is reported as still
**Technology Preview** in 3.4. If accurate, this is a more precise and
more useful fact than `C11`'s blanket "legacy" framing: it isn't that FMS
Guardrails is being removed, it's that the CR-based orchestrator path is
not yet GA while NeMo Guardrails is. **This needs a direct, re-verified
fetch of the actual release-notes page before being relied on** — flag as
`OPEN` pending that.

### 4.3 `GuardrailsOrchestrator` CR — RHOAI-specific delta (`OFFICIAL-SRC`, `trustyai.org`, fetched clean)

Fetched directly (not blocked) from the TrustyAI project's own docs,
https://trustyai.org/docs/main/gorch-tutorial — this is the upstream
project's documentation of the exact CR that RHOAI's operator ships, so
treat it as authoritative for the CR shape even though it isn't
`docs.redhat.com` itself:

```yaml
apiVersion: trustyai.opendatahub.io/v1alpha1
kind: GuardrailsOrchestrator
metadata:
  name: gorch-sample
spec:
  orchestratorConfig: "fms-orchestr8-config-nlp"   # name of a ConfigMap holding config.yaml (§2.1 schema, verbatim)
  enableBuiltInDetectors: True                     # injects the `builtIn` detector (§1.2b) as a sidecar
  enableGuardrailsGateway: True                     # injects a SEPARATE component, the "Guardrails Gateway"
  guardrailsGatewayConfig: "fms-orchestr8-config-gateway"
  replicas: 1
```

Other documented spec fields: `otelExporter` (`protocol`,
`otlpEndpoint`/`otlpMetricsEndpoint`, `otlpExport`), `logLevel`,
`tlsSecrets`, `customDetectorsConfig`, and `autoConfig`
(`inferenceServiceToGuardrail`, `detectorServiceLabelToMatch`,
`enableBuiltInDetectors`, `enableGuardrailsGateway`, `replicas` — an
auto-discovery mode that finds a labeled KServe `InferenceService` +
detector `Service`s and generates the `config.yaml` `ConfigMap` for you).

**The `orchestratorConfig` field points at a `ConfigMap` whose `config.yaml`
uses exactly the schema documented in §2.1** — the CR does not reinvent
the detector-registration format, it's a Kubernetes-native way to author
and mount the same file. Example `ConfigMap`, same page:

```yaml
kind: ConfigMap
apiVersion: v1
metadata: { name: fms-orchestr8-config-nlp }
data:
  config.yaml: |
    generation:
      service: { hostname: llm-predictor.guardrails-test.svc.cluster.local, port: 8033 }
    detectors:
      hap:
        service: { hostname: http:/detector-host/api/v1/text/contents, port: 8000 }
        chunker_id: whole_doc_chunker
        default_threshold: 0.5
```

**Important — do not conflate the "Guardrails Gateway" with the
orchestrator's own `/api/v2/...` API (§2.4).** The Gateway
(`enableGuardrailsGateway`) is a *different* reverse-proxy component with
its own `config.yaml` shape (`detectors: [{name, detector_params}]` +
`routes: [{name, detectors}]`) and OpenAI-shaped client-facing endpoints,
e.g.:

```bash
curl "https://$GORCH_ROUTE_HTTP/pii/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [{"role": "user", "content": "..."}]}'
```

vs. the orchestrator's own detection endpoints, exercised directly in the
same tutorial:

```bash
curl -X POST "https://$GORCH_ROUTE_HTTP/api/v2/chat/completions-detection" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llm",
    "messages": [{"content": "You dotard, I really hate this stuff", "role": "user"}],
    "detectors": {"input": {"hap": {}}, "output": {"hap": {}}}
  }'
```
```json
{
  "id": "086980692dc1431f9c32cd56ba607067",
  "detections": {
    "input": [{"message_index": 0, "results": [{
      "start": 0, "end": 36, "detection": "sequence_classifier",
      "detection_type": "sequence_classification", "detector_id": "hap",
      "score": 0.9634239077568054
    }]}]
  },
  "warnings": [{"type": "UNSUITABLE_INPUT", "message": "Unsuitable input detected."}]
}
```

This example is `/api/v2/chat/completions-detection`, not
`/api/v2/text/detection/content` (that specific endpoint's live example
wasn't present on this page) — but it confirms the request/response
envelope (`detectors: {"<id>": {<params>}}`, response
`detections.{input,output}[].results[]` with the same field names as
§2.4's `ContentAnalysisResponse`) is exactly what the Rust source
predicts, with no RHOAI-specific field renames spotted. **No wire-format
fork was found between the plain upstream orchestrator and the
RHOAI/TrustyAI-operator-deployed one — the delta is entirely in
deployment mechanism (CR + ConfigMap vs. hand-run binary + file), not in
the request/response contract.**

---

## Sources (exact, as fetched 2026-08-08)

- `https://api.github.com/repos/trustyai-explainability/guardrails-detectors`
  (+ `/commits/main`, `/tags`, `/git/trees/747a4d3ef6f7d384b73f929a0162228ad56d98de?recursive=1`) via `gh api`
- Raw file contents via `gh api repos/trustyai-explainability/guardrails-detectors/contents/<path>?ref=747a4d3ef6f7d384b73f929a0162228ad56d98de -H "Accept: application/vnd.github.raw"`:
  `detectors/common/scheme.py`, `detectors/common/app.py`,
  `detectors/built_in/app.py`, `detectors/built_in/base_detector_registry.py`,
  `detectors/huggingface/app.py`, `detectors/huggingface/detector.py`,
  `README.md`, `docs/hf_examples.md`, `docs/builtin_examples.md`
- `https://api.github.com/repos/foundation-model-stack/fms-guardrails-orchestrator`
  (+ `/releases`, `/git/refs/tags/0.18.3`, `/releases/tags/0.18.3`,
  `/git/trees/6d2cec987223335adcc3803f884dae7a4aa59492?recursive=1`,
  `/commits/main`) via `gh api`
- Raw file contents via `gh api .../contents/<path>?ref=6d2cec987223335adcc3803f884dae7a4aa59492 -H "Accept: application/vnd.github.raw"`:
  `config/config.yaml`, `tests/test_config.yaml`, `src/config.rs`,
  `src/clients.rs`, `src/clients/chunker.rs`, `src/clients/detector.rs`,
  `src/models.rs`, `src/server/routes.rs`, `src/server/errors.rs`,
  `src/orchestrator/handlers/text_content_detection.rs`,
  `src/orchestrator/common.rs`, `src/orchestrator/common/utils.rs`,
  `src/orchestrator/common/client.rs`, `src/orchestrator/common/tasks.rs`,
  `src/orchestrator/errors.rs`, `src/health.rs`, `Cargo.toml`,
  `docs/api/openapi_detector_api.yaml`, `docs/api/orchestrator_openapi_0_1_0.yaml`
- `https://quay.io/api/v1/repository/opendatahub/ta-guardrails-orchestrator?includeTags=true`
  and `https://quay.io/api/v1/repository/modh/odh-fms-guardrails-orchestrator-rhel9?includeTags=true`
  (public quay.io registry API, no auth) via `curl`
- `https://api.github.com/search/code?q=fms-guardrails-orchestrator+quay.io` and
  `gh api repos/red-hat-data-services/rhoai-disconnected-install-helper/contents/rhoai-3.4.md`
  via `gh api` / `gh search code`
- `https://trustyai.org/docs/main/gorch-tutorial` — fetched directly via `WebFetch`, succeeded (not Akamai-blocked)
- `docs.redhat.com` (RHOAI 3.4 guardrails guide, html/html-single/pdf variants) — **attempted, blocked (`403`)**,
  see §4 for what was recoverable via `WebSearch` instead and what remains `OPEN`
- This repo (read-only, `STATIC`): `AGENTS.md`, `docs/facts.md` (fact `C11`), `docs/implementation.md`
