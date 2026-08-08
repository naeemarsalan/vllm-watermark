# NeMo Guardrails extension surface + RHOAI 3.4 fit (Phase 3 research)

**Mixed STATIC / OFFICIAL-SRC / CORROBORATED** (see per-claim tags; conventions
match [`docs/facts.md`](facts.md)). Source/spec reading of the upstream
`NVIDIA-NeMo/Guardrails` repo (`gh api`, direct raw-file fetch) is
`OFFICIAL-SRC`. The RHOAI 3.4 product-docs cross-check hit the same
`docs.redhat.com` Akamai block already recorded in
[`docs/api-notes-trustyai-detectors.md`](api-notes-trustyai-detectors.md) §4
— everything about RHOAI's own prose (support-tier wording, chapter
numbering) is `CORROBORATED` via `WebSearch` extraction, not a raw fetch I
could grep myself. The RHOAI-specific `NemoGuardrails` **custom resource
schema**, by contrast, **was** fetched directly and cleanly from
`opendatahub-io/architecture-context` on GitHub (not `docs.redhat.com`), so
that part is `OFFICIAL-SRC`. Fetched 2026-08-08. No code in this repo was
changed or executed for this task — this is a pure research note answering
`docs/facts.md` C11/D5 and `docs/implementation.md` Phase 3's directive to
"re-check the current RHOAI lifecycle and extension points" before building
a platform adapter.

Every field name and code sample below is copied verbatim from a cited
file/line — none is invented. Where I could not verify something (RHOAI's
exact deployed image, whether a custom `actions.py` file survives the RHOAI
dashboard's ConfigMap flow, etc.), it is flagged `OPEN`, not asserted.

---

## 0. Pinned sources

| Repo | Ref used | Resolved commit / tag | Note |
|---|---|---|---|
| `NVIDIA-NeMo/Guardrails` | `develop` (HEAD) | `f5900d1e9e61513b6bc189e1f58b9e50e94fbd9a` (2026-08-07) | `gh api repos/NVIDIA-NeMo/Guardrails/commits/develop`. **The repo moved**: `gh api repos/NVIDIA/NeMo-Guardrails` 302-redirects and its JSON body reports `"full_name":"NVIDIA-NeMo/Guardrails"` — old links (`github.com/NVIDIA/NeMo-Guardrails`) still resolve but the canonical org is now `NVIDIA-NeMo`. |
| `NVIDIA-NeMo/Guardrails` | tag | `v0.23.0` (2026-07-01) | latest tagged release; `gh api repos/NVIDIA-NeMo/Guardrails/releases` — `develop` is ~5 weeks ahead of it |
| `opendatahub-io/architecture-context` | `main` | — | `architecture/rhoai-3.4/contracts/schemas/trustyai-service-operator/nemoguardrails.v1alpha1.json` — the RHOAI 3.4 `NemoGuardrails` CRD's OpenAPI schema, exported from the actual operator build |
| `opendatahub-io/odh-dashboard` | `main` | — | `packages/gen-ai/bff/...` — the RHOAI dashboard's own Go client code for provisioning a `NemoGuardrails` CR, useful as a second, independent confirmation of the CR shape and of what RHOAI's UI actually wires by default |

License: every `NVIDIA-NeMo/Guardrails` file fetched below carries
`SPDX-License-Identifier: Apache-2.0` in its header, and the repo's
`LICENSE.md` is the Apache-2.0 text verbatim — note the GitHub API's
auto-detected `license.spdx_id` field reports `NOASSERTION` for this repo
(probably a detector quirk from having three license files:
`LICENSE.md`, `LICENSE-Apache-2.0.txt`, `LICENCES-3rd-party`); the
per-file SPDX headers are the ground truth, not that field.

---

## 1. The mechanism: custom Python actions, not a fixed detector protocol

**Direct answer to the framing question:** there is **no first-class
"external detector server" interface** in NeMo Guardrails comparable to
FMS's fixed `POST /api/v1/text/contents` contract
([`docs/api-notes-trustyai-detectors.md`](api-notes-trustyai-detectors.md)
§1). The pattern is: write a Python **action** (`@action`-decorated
async function) that does whatever HTTP call it wants, register it, and
reference it from a Colang **flow** that is wired into `rails.input.flows`
or `rails.output.flows` in `config.yml`. Every third-party integration in
the `nemoguardrails/library/` tree (jailbreak detection, Prompt Security,
AutoAlign, Pangea, ActiveFence, Trend Micro, CrowdStrike AIDR, …) is built
this exact way — an `actions.py` that calls the vendor's own HTTP API, no
special SDK or fixed schema imposed by NeMo Guardrails itself. `STATIC`,
confirmed by reading multiple independent library integrations, not just
one.

### 1.1 The `@action` decorator and `RailOutcome`

Source: `docs/configure-rails/actions/creating-actions.mdx`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/docs/configure-rails/actions/creating-actions.mdx)
and `nemoguardrails/actions/rail_outcome.py`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/actions/rail_outcome.py).

```python
from nemoguardrails.actions import action

@action()
async def my_custom_action():
    """A simple custom action."""
    return "result"
```

An action that is meant to decide whether to block content returns a
`RailOutcome` (`nemoguardrails/actions/rail_outcome.py`):

```python
class RailDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"

@dataclass(frozen=True, slots=True)
class RailOutcome:
    decision: RailDecision
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    transforms: tuple[TransformSpec, ...] = ()

    @classmethod
    def allow(cls, *, reason=None, metadata=None) -> "RailOutcome": ...
    @classmethod
    def block(cls, *, reason=None, metadata=None) -> "RailOutcome": ...
    @classmethod
    def transform(cls, rewrites, *, reason=None, metadata=None) -> "RailOutcome": ...
```

`metadata` is exactly where scheme-specific evidence (our `z_score`,
`p_value`, `key_id`) would go — the docstring says as much: "neutral
evidence the decision is based on (policy violations, categories, scores,
or backend-specific details)."

### 1.2 A real external-HTTP-call action, verbatim (closest analog to our detector)

`nemoguardrails/library/jailbreak_detection/actions.py`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/library/jailbreak_detection/actions.py)
and its companion `request.py`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/library/jailbreak_detection/request.py)
show the full external-call shape used throughout the library:

```python
@action()
async def jailbreak_detection_heuristics(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    jailbreak_config = llm_task_manager.config.rails.config.jailbreak_detection
    jailbreak_api_url = jailbreak_config.server_endpoint
    prompt = context.get("user_message")
    ...
    jailbreak = await jailbreak_detection_heuristics_request(
        prompt, jailbreak_api_url, lp_threshold, ps_ppl_threshold, http_client=http_client,
    )
    ...
    return RailOutcome.block() if jailbreak else RailOutcome.allow()
```

```python
async def jailbreak_detection_heuristics_request(
    prompt: str,
    api_url: str = "http://localhost:1337/heuristics",
    lp_threshold: Optional[float] = None,
    ps_ppl_threshold: Optional[float] = None,
    http_client: Optional[HTTPClient] = None,
):
    payload = {"prompt": prompt, "lp_threshold": lp_threshold, "ps_ppl_threshold": ps_ppl_threshold}
    response = await http_call(http_client, "POST", api_url, json=payload, raise_for_status=False)
    if response.status_code != 200:
        log.error(f"Jailbreak check API request failed with status {response.status_code}")
        return None
    result = response.json()
    return result["jailbreak"]
```

This is a plain `POST <configured-url>` with a JSON body and JSON response
— any URL, any port, any auth header the action author adds. `http_call`
(`nemoguardrails/http/`) is NeMo's own thin async HTTP wrapper (timeout,
retry, instrumentation), not a protocol constraint.

`nemoguardrails/library/prompt_security/actions.py`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/library/prompt_security/actions.py)
is an even closer analog because it checks **bot output**, not just user
input, and maps a vendor JSON response into `RailOutcome`:

```python
@action(is_system_action=True)
async def protect_text(
    user_prompt: Optional[str] = None,
    bot_response: Optional[str] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    ...
    if bot_response:
        return _protect_text_outcome(
            await ps_protect_api_async(ps_protect_url, ps_app_id, None, None, bot_response, http_client=http_client),
            TransformTarget.BOT_MESSAGE,
        )
```

`_protect_text_outcome` reads a vendor-specific `{"result": {"action":
"block"|"modify"|"log"}}` shape and returns `RailOutcome.block()` /
`.transform()` / `.allow()` accordingly — i.e. "call an arbitrary external
service, parse its JSON, decide" is the established idiom, not something
we would be inventing.

### 1.3 `context` gives output-rail actions the generated text

`docs/configure-rails/actions/action-parameters.mdx`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/docs/configure-rails/actions/action-parameters.mdx),
"Common Context Variables" table:

| Variable | Description | Availability |
|---|---|---|
| `last_user_message` | The most recent user message | Always available after user input |
| `bot_message` | The current bot message (in output rails) | Available in output rails |

So an **output-rail** action (the kind we'd use — checking generated text
for a watermark, analogous to the FMS `text_contents` detector applied to
the model's output) reads `context.get("bot_message")` to get exactly the
text to POST to our detector. `STATIC`.

### 1.4 Registration — file-based is the default, and it is exactly the mechanism our action needs

`docs/configure-rails/actions/registering-actions.mdx`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/docs/configure-rails/actions/registering-actions.mdx):
actions dropped in a config folder's `actions.py` (or an `actions/`
package) are **auto-registered on config load** — no separate
registration step, no server restart wiring beyond deploying the file.
Alternatives exist (`LLMRails.register_action()` programmatic API,
LangChain-tool registration, a `config.py` `init()` hook) but file-based
`actions.py` is the one that matches "ship a ConfigMap" deployment, which
is also how RHOAI's own `NemoGuardrails` CR works (§4).

### 1.5 `actions_server_url` is NOT a generic detector protocol — don't conflate it

The one thing that could be mistaken for "a first-class external detector
interface" is the optional **actions server**
(`nemoguardrails/actions_server/actions_server.py`,
https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/actions_server/actions_server.py).
Its entire contract is:

```python
class RequestBody(BaseModel):
    action_name: str = ""
    action_parameters: Dict = Field(default={})

class ResponseBody(BaseModel):
    status: str = "success"  # success / failed
    result: Optional[str]

@app.post("/v1/actions/run", response_model=ResponseBody)
async def run_action(body: RequestBody):
    result, status = await app.action_dispatcher.execute_action(body.action_name, body.action_parameters)
    return {"status": status, "result": result}
```

This is a generic RPC for running **NeMo's own already-registered Python
actions** out-of-process for horizontal scaling
(`ActionDispatcher(load_all_actions=True)` loads the same `actions.py`
files as the in-process case) — it is not a fixed wire contract that any
external detector service can implement to "plug in." A detector wanting
to be called this way would still need to exist as a `@action`-decorated
Python function loaded by *this* process. It is not the integration
surface we want; §2's plain custom action calling our HTTP endpoint is.
`STATIC`.

---

## 2. `config.yml` wiring — verbatim worked example (input-rail case) + generalization to output

`examples/configs/jailbreak_detection/config.yml`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/examples/configs/jailbreak_detection/config.yml),
fetched whole:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

rails:
  config:
    jailbreak_detection:
      server_endpoint: "http://localhost:1337/heuristics"
      lp_threshold: 89.79
      ps_ppl_threshold: 1845.65
      embedding: "Snowflake/snowflake-arctic-embed-m-long"

  input:
    flows:
      - jailbreak detection heuristics
      - jailbreak detection model
```

The `rails.config.jailbreak_detection` block is itself a typed pydantic
model registered by the library extension
(`nemoguardrails/library/jailbreak_detection/rail_config.py`,
https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/library/jailbreak_detection/rail_config.py):

```python
class JailbreakDetectionConfig(RailConfigBaseModel):
    server_endpoint: Optional[str] = Field(default=None, description="The endpoint for the jailbreak detection heuristics/model container.")
    ...
    @model_validator(mode="after")
    def validate_urls(self) -> "JailbreakDetectionConfig":
        if self.server_endpoint and not self.server_endpoint.startswith(("http://", "https://")):
            raise ValueError(...)
        return self
```

i.e. "point this rail at an arbitrary `http(s)://` endpoint" is a
first-class, precedented pattern for `rails.config.<name>` blocks — we are
not inventing new config-schema conventions if our own action reads its
endpoint URL the same way (e.g. from an env var, or from a
`rails.config.watermark_detection.server_endpoint`-shaped block if we
register one — the latter requires writing our own
`RailConfigBaseModel`/`build_config_spec()` per `rail_config.py`'s pattern,
which is optional; a plain env var is simpler and equally valid since
nothing requires a rail to register custom config fields).

For **output** rails the shape is identical, only the `rails:` key
changes — `examples/configs/self_check_thinking/config.yml`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/examples/configs/self_check_thinking/config.yml):

```yaml
rails:
  output:
    flows:
      - self check output
```

and the Colang flow referenced there is a plain named `flow` block
(`nemoguardrails/library/jailbreak_detection/flows.co`,
https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/library/jailbreak_detection/flows.co):

```text
flow jailbreak detection heuristics
  $response = await JailbreakDetectionHeuristicsAction
  $is_jailbreak = $response.is_blocked
  if $is_jailbreak
    bot refuse to respond
    abort
```

---

## 3. `POST /v1/checks` — a standalone, generation-free check endpoint (the NeMo analog of FMS's `/api/v2/text/detection/content`)

Source: `nemoguardrails/server/api.py`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/server/api.py)
and `nemoguardrails/server/schemas/openai.py`
(https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/server/schemas/openai.py).
(Note: `WebSearch`-extracted RHOAI release-note prose calls this
"`/v1/guardrails/checks`" — the actual route registered in source is
`/v1/checks`; treat the fetched source as ground truth over the
paraphrase, per this doc's own citation discipline.)

```python
@app.post("/v1/checks", response_model=GuardrailCheckResponse, response_model_exclude_none=True)
async def guardrail_check(body: GuardrailCheckRequest, request: Request):
    ...
    result = await llm_rails.check_async(messages=messages)
    return GuardrailCheckResponse(status=_map_rail_status(result.status), content=result.content, rail=result.rail)
```

```python
class GuardrailCheckRequest(OpenAIChatCompletionRequest):
    """Request body for the /v1/checks endpoint."""
    guardrails: GuardrailsDataInput = Field(default_factory=GuardrailsDataInput)

class GuardrailCheckResponse(BaseModel):
    status: str          # "passed" | "modified" | "blocked" -- RailStatus values
    content: str          # content after rails processing
    rail: Optional[str]   # name of the blocking rail, if any
```

Critically, `LLMRails.check_async` (`nemoguardrails/rails/llm/llmrails.py`,
https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/nemoguardrails/rails/llm/llmrails.py#L1617)
picks which rails to run **from the message roles you send it, with no LLM
call in between**:

```python
async def check_async(self, messages: List[dict], rail_types: Optional[List[RailType]] = None) -> RailsResult:
    """
    - Only user messages: runs input rails
    - Only assistant messages: runs output rails
    - Both user and assistant messages: runs both input and output rails
    ...
    """
```

So `POST /v1/checks` with a body containing a single
`{"role": "assistant", "content": "<candidate watermarked text>"}` message
runs **only the output rails** — including a custom watermark-check action
wired via `rails.output.flows` — against that text, with no model
generation involved. This is functionally the NeMo equivalent of FMS's
detection-only `/api/v2/text/detection/content`
([`docs/api-notes-trustyai-detectors.md`](api-notes-trustyai-detectors.md)
§2.4): a way to submit arbitrary text for a rails verdict without running a
full chat turn. `STATIC` (read, not executed against a live server in this
task).

---

## 4. No watermark-detection rail exists anywhere in the ecosystem — confirmed by absence

Searched, all negative:

- `grep -i watermark` over the full `develop` file-tree path listing of
  `NVIDIA-NeMo/Guardrails` (2,601 paths, including every file under
  `nemoguardrails/library/`) — **zero matches**. `OFFICIAL-SRC` (exhaustive
  path listing via `gh api .../git/trees/develop?recursive=1`).
- `gh search code "watermark" --repo NVIDIA-NeMo/Guardrails` — **zero
  matches** (content search, not just path names). `OFFICIAL-SRC`.
- `gh search code "watermark" --owner NVIDIA-NeMo` (whole org, all repos)
  — all hits are unrelated: RL checkpoint/replay-buffer "watermark"
  terminology, and image-generation prompt templates instructing a model
  **not** to render visual watermarks in synthetic images. None concern
  text-content watermark detection. `OFFICIAL-SRC`.
- `WebSearch` "`NeMo Guardrails` watermark detection rail 2026" — results
  describe the five rail types (input/dialog/retrieval/execution/output)
  and the existing library (content safety, jailbreak, PII, topic control)
  with no watermark mention; explicitly: "the search results did not
  contain specific information about watermark detection as a feature
  within NeMo Guardrails." `CORROBORATED` (absence).
- `WebSearch` "NVIDIA NIM microservice text watermark detection AI Act" —
  the only NVIDIA↔watermarking connection found is **video**: NVIDIA's
  Cosmos NIM microservice partners with Google to apply **SynthID for
  video**, not text. No text-watermark-detection NIM found. `CORROBORATED`
  (absence).

Net: this repo's finding (B19 in `docs/facts.md`: "No other inference
stack ships text watermarking") extends cleanly to the guardrails layer —
**no upstream or NVIDIA-shipped watermark-detection rail exists to
integrate with; a watermark-check action for NeMo Guardrails would have to
be authored by us**, exactly as the direct `/v1/watermark/detect` endpoint
and the FMS-contract routes in `detector/app.py` already are.

---

## 5. What RHOAI 3.4 actually ships for NeMo Guardrails

### 5.1 Support tier (`CORROBORATED` — `docs.redhat.com` blocked, see below)

`WebSearch` extraction of RHOAI 3.4 release notes and the guardrails guide
(same Akamai-blocked host as
[`docs/api-notes-trustyai-detectors.md`](api-notes-trustyai-detectors.md)
§4 — every direct `WebFetch` attempt against
`docs.redhat.com/.../3.4/html/enabling_ai_safety_with_guardrails/...`
returned HTTP 403 in this session too, confirming that finding is not
session-specific):

> NeMo Guardrails, introduced in Red Hat OpenShift AI 3.3 as a Technology
> Preview, is fully supported with RHOAI 3.4.

and the chapter is now **Chapter 1** of the guardrails guide ("Enabling AI
safety with NeMo Guardrails"), matching
`docs/api-notes-trustyai-detectors.md` §4.1's independently-found chapter
reordering (FMS Guardrails demoted across 3.0→3.5). This is *not* a raw
fetch I verified character-for-character — flag the exact wording as
`CORROBORATED`, re-verify from an unblocked network path before treating
it as a compliance-grade quote (same caveat as C11).

### 5.2 The `NemoGuardrails` custom resource — fetched cleanly, ground truth (`OFFICIAL-SRC`)

Not from `docs.redhat.com` — from `opendatahub-io/architecture-context`'s
exported operator CRD schema,
`architecture/rhoai-3.4/contracts/schemas/trustyai-service-operator/nemoguardrails.v1alpha1.json`
(https://github.com/opendatahub-io/architecture-context/blob/main/architecture/rhoai-3.4/contracts/schemas/trustyai-service-operator/nemoguardrails.v1alpha1.json).
`apiVersion: trustyai.opendatahub.io/v1alpha1`, `kind: NemoGuardrails` —
same API group as the legacy `GuardrailsOrchestrator` CR
([`docs/api-notes-trustyai-detectors.md`](api-notes-trustyai-detectors.md)
§4.3), confirmed independently by
`opendatahub-io/odh-dashboard`'s Go constants
(`packages/gen-ai/bff/internal/constants/guardrails.go`,
https://github.com/opendatahub-io/odh-dashboard/blob/main/packages/gen-ai/bff/internal/constants/guardrails.go):
`NemoGuardrailsAPIVersion = "trustyai.opendatahub.io/v1alpha1"`,
`NemoGuardrailsKind = "NemoGuardrails"`.

`spec` schema (only `nemoConfigs` is required):

```
spec:
  nemoConfigs:      # REQUIRED — list of {name, configMaps: [string], default: bool}
                     # "NemoConfig should be the names of the configmaps containing
                     #  NeMO server configuration files. All files in NemoConfigs
                     #  will be mounted to /app/config/$Name"
  replicas: int      # default 1
  env: [EnvVar]       # standard k8s env (value / valueFrom secret|configMap|field ref)
  caBundleConfig: {...}  # custom CA trust
```

**This is a thin, generic wrapper: the CR's job is "mount these ConfigMaps
at `/app/config/$Name` and run the NeMo Guardrails server against them."**
It does not encode any NeMo-specific policy about what those files may
contain — no field restricts `nemoConfigs[].configMaps` to
`config.yml`-only ConfigMaps. Whatever the upstream NeMo Guardrails server
accepts in a config directory (§1–§2: `config.yml`, `actions.py` or an
`actions/` package, `*.co` Colang flow files, `prompts.yml`) is, by this
schema, whatever RHOAI runs. **This was not executed** — I did not deploy
a `NemoGuardrails` CR with a ConfigMap containing a custom `actions.py` and
confirm the mount + auto-registration works end-to-end on an actual RHOAI
3.4 cluster. Flag explicitly as `OPEN`, to be closed by Phase 3 execution
on the cluster, not assumed from the schema alone.

### 5.3 What RHOAI's own dashboard wires by default — narrower than the CR schema allows

`opendatahub-io/odh-dashboard`'s provisioning code
(`packages/gen-ai/bff/internal/integrations/kubernetes/nemo_guardrails.go`,
https://github.com/opendatahub-io/odh-dashboard/blob/main/packages/gen-ai/bff/internal/integrations/kubernetes/nemo_guardrails.go)
shows exactly what the RHOAI **UI's** one-click "initialize guardrails"
flow creates — a placeholder `ConfigMap` with:

```yaml
config.yaml: |
  models:
    - type: main
      engine: openai
      model: placeholder
      api_key_env_var: OPENAI_API_KEY
      parameters:
        base_url: "http://placeholder.invalid/v1"
  rails:
    input:
      flows:
        - self check input
    output:
      flows:
        - self check output
prompts.yml: |
  ... (self_check_input / self_check_output LLM prompts) ...
rails.co: "# Using built-in self-check rails\n"
```

i.e. the dashboard's default only wires the **LLM-based** `self check
input`/`self check output` rails (an LLM call judging the text against a
policy prompt, not a custom Python action) — it never generates or mounts
an `actions.py`. **This is the dashboard UI's default, not a limit imposed
by the CR schema (§5.2)** — nothing here stops an admin from authoring
their own `ConfigMap` with a custom `actions.py`+`flows.co`+`config.yml`
(the exact §1–§2 shape) and pointing a `NemoGuardrails` CR's `nemoConfigs`
at it directly (via `oc apply`, bypassing the dashboard). But that admin
path is what §5.2 flags `OPEN` — the dashboard-driven path visibly does
**not** support custom detector actions today, only the two built-in
self-check flows.

---

## 6. Bottom line — integration paths for the watermark detector

**(a) FMS/TrustyAI Guardrails Orchestrator — legacy but shipped; current
Phase 3 target.** Already fully specified and STATIC-verified in
[`docs/api-notes-trustyai-detectors.md`](api-notes-trustyai-detectors.md):
our `detector/app.py` already implements
`POST /api/v1/text/contents` (+ scheme-forced aliases) exactly to this
contract. `docs/facts.md` C5/C11: RHOAI still ships this path (Technology
Preview per §4.2 of that doc, not removed), it is just no longer the
headlined chapter. No new work needed for this path beyond what
`docs/implementation.md` Phase 3 already calls for (deploy + execute
against a live orchestrator).

**(b) A NeMo Guardrails custom action wrapping our direct
`/v1/watermark/detect` endpoint.** This is the path this document
establishes as viable *by source review* (not yet executed). Concrete,
non-invented sketch, following §1–§2's verified patterns exactly:

`config.yml` (output rail — checks the model's generated text, the
watermark-relevant direction):

```yaml
rails:
  output:
    flows:
      - watermark check
```

`actions.py` (auto-registered per §1.4; field/parameter names —
`context`, `http_client`, `RailOutcome`, `http_call` signature — are all
cited in §1.1–§1.3, not invented; `VLLM_WATERMARK_DETECTOR_URL` and the
JSON shape are OUR OWN `detector/app.py` contract, not NeMo's):

```python
import os
from typing import Optional
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import HTTPClient, http_call

@action(is_system_action=True)
async def watermark_check(
    context: Optional[dict] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs,
) -> RailOutcome:
    bot_response = (context or {}).get("bot_message")
    if not bot_response:
        return RailOutcome.allow()
    url = os.environ["VLLM_WATERMARK_DETECTOR_URL"]  # e.g. http://vllm-watermark-detector:8000/v1/watermark/detect
    resp = await http_call(http_client, "POST", url, json={"text": bot_response}, raise_for_status=False)
    if resp.status_code != 200:
        return RailOutcome.allow()  # fail open, matching the library's own convention (§1.2)
    result = resp.json()
    return RailOutcome.allow(metadata={
        "verdict": result["verdict"], "z_score": result["z_score"],
        "p_value": result["p_value"], "scheme": result["scheme"],
    })
```

`flows.co`:

```text
flow watermark check
  $result = await WatermarkCheckAction
```

Design note (`OPEN`, ours to decide, not NeMo's): the sketch above always
`allow()`s and only attaches `metadata` — a watermark-absence verdict is
evidence for a compliance record, not necessarily grounds to block a
response, unlike jailbreak/content-safety rails whose whole point is to
block. Whether/how `RailOutcome.metadata` from an output rail gets
surfaced to a caller of `/v1/checks` (§3) or `/v1/chat/completions` needs
checking against `GuardrailCheckResponse`'s fields (§3: only `status`,
`content`, `rail` are modeled — no generic metadata passthrough was seen
in `GuardrailCheckResponse`) — if metadata isn't surfaced end-to-end,
recording the verdict would need to happen action-side (e.g. the action
itself calls back to our own logging, the same zero-content-retention
logging `detector/app.py` already does) rather than relying on NeMo to
relay it to the API caller. Flag this specific gap as `OPEN` — not
executed or confirmed in this task.

**(c) `POST /v1/checks` as the standalone call site (§3).** Once (b) is
wired as an output-rail flow, `POST /v1/checks` with
`{"messages": [{"role": "assistant", "content": "<text>"}], "guardrails": {"config_ids": [...]}}`
against the NeMo Guardrails server RHOAI's `NemoGuardrails` CR stands up
is the generation-free detection call — the NeMo equivalent of calling
FMS's `/api/v2/text/detection/content` directly. Not executed.

**(d) What remains genuinely open, not to be presented as resolved:**
- §5.2: whether an admin-authored `ConfigMap` containing a custom
  `actions.py` actually mounts and auto-registers correctly under RHOAI
  3.4's `NemoGuardrails` operator, end-to-end on the cluster.
- §5.1: the exact "legacy"/support-tier wording for both FMS and NeMo
  Guardrails in RHOAI 3.4, unverified past `WebSearch` extraction because
  `docs.redhat.com` is unreachable from this environment (same block as
  `docs/api-notes-trustyai-detectors.md` §4 and fact C11).
- (b)'s metadata-passthrough question above.
- Whether RHOAI exposes `/v1/checks` (vs. only the dashboard's own
  chat-completions-shaped UI flow) to a namespace's other workloads/
  network policy — not investigated in this task.

None of these block Phase 3 from proceeding on path (a) (already the
documented current target) while (b)/(c) are validated by execution in
parallel, per `docs/implementation.md` Phase 3's acceptance criteria
("If no supported/current extension exists, accept only after the direct
detector service is executed and the platform-integration gap is recorded
precisely as `OPEN`").

---

## Sources (exact, as fetched 2026-08-08)

- `https://api.github.com/repos/NVIDIA/NeMo-Guardrails` (redirect →
  `NVIDIA-NeMo/Guardrails`), `/commits/develop`, `/releases`,
  `/git/trees/develop?recursive=1` via `gh api`
- Raw file contents via
  `gh api repos/NVIDIA-NeMo/Guardrails/contents/<path>?ref=develop -H "Accept: application/vnd.github.raw"`:
  `LICENSE.md`,
  `docs/configure-rails/actions/creating-actions.mdx`,
  `docs/configure-rails/actions/registering-actions.mdx`,
  `docs/configure-rails/actions/action-parameters.mdx`,
  `docs/about/rail-types.mdx`,
  `nemoguardrails/actions/rail_outcome.py`,
  `nemoguardrails/library/jailbreak_detection/actions.py`,
  `nemoguardrails/library/jailbreak_detection/request.py`,
  `nemoguardrails/library/jailbreak_detection/flows.co`,
  `nemoguardrails/library/jailbreak_detection/rail_config.py`,
  `nemoguardrails/library/prompt_security/actions.py`,
  `nemoguardrails/actions_server/actions_server.py`,
  `nemoguardrails/server/api.py`,
  `nemoguardrails/server/schemas/openai.py`,
  `examples/configs/jailbreak_detection/config.yml`,
  `examples/configs/jailbreak_detection/README.md`,
  `examples/configs/self_check_thinking/config.yml`,
  `examples/configs/prompt_security/config.yml`
- `gh search code "watermark" --repo NVIDIA-NeMo/Guardrails`, `--owner
  NVIDIA-NeMo` via `gh search code`
- `https://api.github.com/repos/opendatahub-io/architecture-context/contents/architecture/rhoai-3.4/contracts/schemas/trustyai-service-operator/nemoguardrails.v1alpha1.json`
  via `gh api`
- `https://api.github.com/repos/opendatahub-io/odh-dashboard/contents/packages/gen-ai/bff/internal/constants/guardrails.go`
  and `.../internal/integrations/kubernetes/nemo_guardrails.go` via `gh api`
- `WebFetch` attempt against
  `docs.redhat.com/.../3.4/html/enabling_ai_safety_with_guardrails/enabling-ai-safety-with-nemo-guardrails_nemo-guardrails`
  — **blocked, HTTP 403** (same Akamai block as
  `docs/api-notes-trustyai-detectors.md` §4)
- `WebSearch`: `"RHOAI 3.4 \"NeMo Guardrails\" OpenShift AI documentation config.yml custom action"`,
  `"RHOAI \"NeMoGuardrails\" custom resource ConfigMap config.yml opendatahub trustyai"`,
  `"\"NeMo Guardrails\" watermark detection rail 2026"`,
  `"NVIDIA NIM microservice text watermark detection AI Act"`
- This repo (read-only, `STATIC`): `AGENTS.md`, `docs/facts.md` (facts
  C5/C11, B19), `docs/implementation.md` (Phase 3),
  `docs/api-notes-trustyai-detectors.md`, `detector/app.py`
