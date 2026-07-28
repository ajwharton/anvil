# Anvil — post-training design note

**Project:** [Anvil](https://github.com/ajwharton/anvil) (this repo)  
**Status:** Design SSOT  
**Outcome:** Forge **sovereign domain experts** from base models — small four-verb API, own hardware (lab GPU → dual DGX Spark → Jetson edge), **vision** first-class, live observability, **agent-operable** control plane (HTTP/MCP; user-supplied brain).  
**Date:** 2026-07-16 (de-brand 2026-07-17; agent dual-focus 2026-07-18; expert workflow 2026-07-26)

Inspiration / non-affiliation one-liner: see `README.md` (only public place we name prior product art).  
Product thesis: [`product.md`](product.md) · roadmap ladder: [`roadmap.md`](roadmap.md) (Expert-v0/v1/v2) · agent split: [`agentic-control.md`](agentic-control.md) · operator brief: [`agent-context.md`](agent-context.md).

Related personal lab context (private): dual-Spark serve, starwatch agents, Jetson robot — not required to use Anvil.

---

## 1. Product shape

### 1.0 Primary workflow (domain expert forge)

Stable product loop — independent of whether the substrate is today’s transformers
or a later algorithm behind the same verbs:

```text
  domain data (you)          base model (you)
         │                          │
         ▼                          ▼
  place: CAS / JSONL / trajectories + recipe plan + gates
         │
         ▼
  train: four verbs + named losses (SFT / preference / GRPO / …)
         │
         ▼
  observe: metrics.jsonl + probes  ──►  classify healthy|noisy|cliff|broken
         │                                        │
         │         ┌── stop / export expert ◄─────┤
         │         ├── patch knobs / resume       │
         └────────►└── switch method / next stage ┘
                      (audited; human policy)
```

| Step | Anvil owns | You own |
|------|------------|---------|
| Place | Convert helpers, media store, Example/Trajectory schemas | Corpus, licenses, rewards |
| Plan | Recipes, architecture gates, knobs | Choice of base + pattern (or agent under policy) |
| Train | Backends, LoRA, train/sample consistency | GPUs, data iteration |
| Observe | `RunMetricsWriter`, `/observe`, SSE, tripwires | Judgment of “enough” (or agent) |
| Act | Pause/patch/export/queue APIs + MCP | Policy, force limits, spend |
| Export | PEFT/path adapters | Where the expert runs next |

**Profiles** (same workflow): robotics/edge VLM · org self-host large base · solo lab.  
**Prioritization:** Expert-v0/v1/v2 + **P.Recipes** in [`roadmap.md`](roadmap.md) — not “fill every kit checkbox.”

### 1.0b Recipe book (control plane, not docs)

Recipes are **first-class control objects**, not README snippets:

```text
  shipped atlas  ─┐
  personal book  ─┼─► plan_recipe / suggest ─► train + observe
                  │                              │
                  └◄──── promote-from-run ───────┘
```

| Object | Role |
|--------|------|
| **Recipe** | Pattern + knobs + gates + stop/probe defaults; binds to **family/shape**, optional instance pin |
| **Meta-recipe** | Graph of recipes + cliff edges + calibration vs production mode |
| **Personal book** | Operator-local library (sovereign); versioned; org-shareable without Anvil hosting |

Design rules: same SSOT for UI and agents; card/gates still block impossible combos;
calibration overshoot may update experience only when explicitly promoted.
Detail: [`recipes.md`](recipes.md).

### 1.1 Layering

| Piece | Role |
|-------|------|
| **Client SDK** | Thin. You write loops on a laptop/CPU box—or an agent does, via the same contract. |
| **Control / observe APIs** | HTTP + SSE SSOT for web UI **and** agents (`anvil-web`, `AnvilControlClient`). |
| **MCP + harness** | Anvil-owned tools and optional loop; **user brings** the agent model (see agentic-control.md). |
| **Train / sample workers** | Warm bases + LoRA adapters. Scheduling, multi-node, fault tolerance as backends grow. |
| **Recipes / meta-recipes / book** | Shipped atlas + **personal recipe book**; stage graphs and cliff→next policies; family/shape binding (see [`recipes.md`](recipes.md)). |
| **Renderers** | Chat/tool/reasoning/multimodal rendering so train and sample share one message↔token contract. |

### 1.2 The primitive surface

```text
ServiceClient
  └─ create_lora_training_client(base_model, rank=…)
        ├─ forward_backward(batch, loss_fn=…)   # grads w.r.t. LoRA
        ├─ optim_step(AdamParams | …)           # apply optimizer
        ├─ save_state / load_state              # train+optim checkpoint
        ├─ save_weights_and_get_sampling_client()
        └─ SamplingClient.sample(prompt, params)
  └─ export / download checkpoint archive (LoRA / weights)
```

**Design intent:** not a black-box `train()`. You own **data, loss, reward, environment**; Anvil owns a stable contract across **distributed PEFT training and colocated sampling**.

Why LoRA is load-bearing (not just “efficient FT”):

1. **Shared base pools** — many adapters on the same warm base process.  
2. **Small artifacts** (tens–hundreds of MB) — download, version, A/B, edge deploy.  
3. **RL-friendly** — LoRA can match full FT for many post-training loads when set up correctly.  
4. **`sample` sees current adapter** without shipping full MoE weights per step.

### 1.3 In-scope recipe families

- **SFT** chat, **math RL** (verifiable), **code RL** (sandbox), **preference** (DPO + multi-stage RLHF), **distillation**, **tool/RAG RL**, **multi-agent**, **audio**, **VLM image classification**.  
- **Eval** harness alongside train.  
- **Weight export** for local serve.  
- **RL debugger** — live metrics, inference probes during training, advantage/IS cliffs; optional J-Lens residual schema (spike parked — not required for the product path).

### 1.4 Explicit non-goals (early)

- Not a hosted multi-tenant cloud by default.  
- Not full-weight fine-tune by default.  
- Not “run any model” without a supported worker path.  
- Jetson is edge inference / data plane first — export is the bridge, not remote train of 30B+ VLMs.

---

## 2. Why a product-shaped API (not another kit)

Open RL stacks already exist (**TRL**, **OpenRLHF**, **veRL**, **Axolotl**, **Unsloth**, etc.). They are powerful **kits**. Anvil targets a **product-shaped API**:

| Kit (today) | Anvil (goal) |
|-------------|----------------|
| “Run this training script on these GPUs” | “Call four verbs; backend may be Spark, cloud, or Jetson sim” |
| Train/serve often different stacks | **Same render + adapter ID** for sample and train |
| Infra is your problem | **Pluggable backends** with one client contract |
| Text-only recipes | **Modality-neutral** batches (text, image, video frames, later audio) |
| Post-hoc eval only | **Observe while training** (metrics + live probes) |

Democratizing RL = **lower the systems tax** while **keeping algorithm control**.

### 2.1 Knobs are cheap; architecture patterns are the product

There are only so many training knobs (`rank`, `lr`, `loss_fn`, freeze masks, batch/seq, export format). Exposing them in a UI is not hard and is not differentiation.

The hard part—and where Anvil should spend design energy—is **deriving defensible recipes from model shape + job intent**:

```text
ModelShape     dense_lm | dense_vlm | edge_student | moe_lm
JobPattern     sft_chat | vlm_sft | vlm_classifier | rl_verifiable | preference_dpo | robot_offline
        └──────────► RecipePlan (knobs + freeze policy + export hint + rationale + cautions)
```

Examples of judgments the stack should encode (see `anvil/recipes/`):

- **Qwen2.5-VL-3B** → `edge_student`: smaller rank, freeze vision encoder, export bias toward ONNX/TRT/Jetson.
- **Dense VLM SFT** → LoRA language + mm projector; do **not** open the vision encoder until data proves need.
- **On-policy RL** → lower LR, more steps, `importance_sampling` / PPO family; sample must see current adapter.
- **Robot offline** → same multimodal message schema as lab; never raw-actuate from sample.

The four verbs remain the **runtime** contract. Recipes are the **intelligence** layer above them. The web UI is recipe-first; knobs are an expert escape hatch.

**Model cards are the architecture oracle.** Prefer `config.json` + HF card (`architectures`, `model_type`, `vision_config`, param count, `pipeline_tag`) over name heuristics. Public RL/SFT research (TRL VLM LoRA, GRPO, LoRA-for-post-train, four-verb product analyses) supplies *pattern shape*; we still supply data and run fine-tunes on lab/edge hardware. See `anvil/recipes/` (`inspect_base_model`, `run_sft` / `run_vlm_sft` / `run_grpo`).

**Bounded catalog + gates.** 15 product recipes (`anvil/recipes/catalog.py`) span dense LM, MoE, lab VLM, edge student. Each recipe declares recommended / stretch / blocked shapes plus size and rank bounds. Users may `force=True` past a block, but the boundary stays explicit in the plan’s `gate` field.

---

## 3. Name and layering

**Name:** **Anvil**.

```text
┌─────────────────────────────────────────────────────────┐
│  Client (laptop / agent / trainer process)              │
│  anvil.ServiceClient → TrainingClient / SamplingClient  │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP/gRPC + object store for media
┌───────────────────────────▼─────────────────────────────┐
│  Control plane                                          │
│  sessions · adapter registry · job queue · audit · UI   │
└───────┬─────────────────────────────┬───────────────────┘
        │                             │
┌───────▼──────────┐         ┌────────▼───────────────────┐
│ Train workers    │         │ Sample workers             │
│ LoRA PEFT train  │◄───────►│ vLLM / SGLang / HF gen     │
│ (Spark, multi-GPU│  shared │ same base+adapter snapshot │
│  optional Ray)   │ adapter │                            │
└──────────────────┘         └────────────┬───────────────┘
                                          │ export GGUF/ONNX/TRT
                                 ┌────────▼────────┐
                                 │ Edge (Jetson)   │
                                 │ vision + policy │
                                 └─────────────────┘
```

**Invariant:** one **adapter ID** + one **renderer** + one **base model ID** across train, sample, and export.

---

## 4. API contract (v0 — deliberately small)

### 4.1 Core verbs

Typed surface: `anvil/client/` + `anvil/protocol/`.  
`ServiceClient(endpoint="fake://")` exercises the full loop in-process (no GPU).  
`local://` runs hand-rolled torch+PEFT verbs. HTTP: `anvil serve` + `RemoteBackend`. Sample split: `VLLMSampleBackend`.

```python
import anvil
from anvil import ServiceClient, Datum, ModelInput, AdamParams, SamplingParams

svc = anvil.ServiceClient(endpoint="fake://")  # later: http://… lab control plane

tc = svc.create_lora_training_client(
    base_model="Qwen/Qwen3.5-4B",   # or vision tower + LM
    rank=32,
    modalities=["text", "image"],   # capability flags
)

# Train step — Datum = model_input + loss_fn_inputs
datum = Datum(
    model_input=ModelInput.from_ints(input_tokens),
    loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
)
fut = tc.forward_backward([datum], loss_fn="cross_entropy")
print(fut.result().loss)
tc.optim_step(AdamParams(learning_rate=1e-4)).result()
tc.save_state("step-100")

# Online RL
sc = tc.save_weights_and_get_sampling_client(name="step-100")
out = sc.sample(
    prompt=ModelInput.from_ints(prompt_tokens),
    sampling_params=SamplingParams(max_tokens=512, temperature=0.7),
).result()
# score out → build Datum with logprobs/advantages → forward_backward("importance_sampling")

# Export
result = tc.export_adapter("./out-adapter", format="peft")  # HF LoRA dir
```

### 4.2 Loss surface (hardest design choice)

The **server** runs `forward_backward` with a known `loss_fn` (e.g. cross_entropy) and accumulates LoRA grads. For RL you typically:

1. `sample` completions  
2. Compute reward **client-side** (or in a sandbox worker)  
3. `forward_backward` with a policy-gradient / PPO-clip / GRPO-style loss that needs **logprobs of sampled tokens** under current (and often old) policy  

**Anvil v0 should support:**

| Loss family | Server needs | Client supplies |
|-------------|--------------|-----------------|
| SFT CE | tokens + labels | batch |
| DPO / IPO | preferred/rejected pairs | pairs |
| On-policy RL | logprobs under θ, advantages | trajectories + advantages |
| Custom | **registered** loss plugins in worker | config + tensors |

Avoid “upload arbitrary Python to execute on GPU” in v0 (security nightmare). Prefer **named losses** + **tensors/logprobs** the client already computed where needed.

### 4.3 Futures / pipelining

Non-blocking handles so you can pipeline `forward_backward` while the next batch prepares. Local backend implements a true async GPU queue (`VerbQueue`); single-GPU backends still return futures for API compatibility.

### 4.4 RL observability (Phase 2.5)

While training runs, continuously:

- Append per-step **metrics** (reward mean/std, within-group reward std / advantage-collapse tripwire, IS mean_ratio, loss).  
- Sample a fixed **probe set** from the live policy every K steps — eyes catch reward hacking and the rollover into negative marginal returns.  
- Stream both into `anvil-web` (`/observe/{run_id}`).  
- **J-Lens / latent-space** residual readouts (optional): schema + API live (`jlens.jsonl` + `GET /api/observe/{run_id}/jlens`); forge spike CLI. **Product path does not depend on J-Lens** — spike parked after 7B mixed re-fit still failed order gate (2026-07-19; `docs/spikes/jlens-math.md`).

---

## 5. Vision as first-class (not a bolt-on)

Anvil should not be “LLM trainer + optional images later.”

### 5.1 Unified example schema

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "Is the grasp reachable?"},
      {"type": "image", "ref": "s3://…/frame_0042.jpg"}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "…"}]}
  ],
  "meta": {"env": "robot-sim", "reward": null}
}
```

- **Media store:** content-addressed blobs (local dir, MinIO, S3). Batch only carries **refs + crops/timestamps**, not multi-MB base64 in every grad step if avoidable.  
- **Renderer:** expands refs → model-specific image tokens (Qwen-VL, InternVL, patch schemes, etc.).  
- **Train worker:** same renderer as sample worker (critical for RL).

### 5.2 Vision-specific training modes

| Mode | Use case | Notes |
|------|----------|--------|
| **VLM SFT** | Instruction following with images | Standard CE on text tokens; image encoder may freeze or LoRA |
| **VLM classifier / rubric** | “Is this a good grasp?” | Classifier head or graded text |
| **Vision RL** | Navigate / pick / UI agent | Reward from sim or human; frames as observations |
| **Distill** | Big VLM → small edge VLM | On-policy / off-policy teacher |
| **Encoder PEFT** | Only ViT/adaptor LoRA | Keeps LM frozen for robotics latency |

### 5.3 Freezing policy (explicit knobs)

```text
lora_targets = {
  "language": true,
  "vision_encoder": false | true,
  "mm_projector": true,
}
```

Default for robotics: **LoRA language + projector**, freeze heavy vision encoder until data proves need — better for Jetson export size and stability.

---

## 6. Backend matrix (own hardware story)

### 6.1 Dual DGX Spark (forge + hammer)

Already in play for **serve**. For **Anvil train**:

| Backend | Fit |
|---------|-----|
| **A. Single-node LoRA train** on forge (or hammer) | v0 — simple, honest |
| **B. Role split** | Sample on head (vLLM), train on peer — clean separation |
| **C. TP=2 train** | Only when base is huge; higher systems cost |

Recommendation: **v0 = A + optional B**. Don’t require fabric TP for the first open-source win.

Reuse dual-Spark lessons: NVMe models (not USB), explicit host IPs, long NCCL timeouts, worker-first only when multi-node train needs it.

### 6.2 Jetson (edge vision robot)

Jetson is **not** a train worker for 30B+ VLMs. It is:

1. **Inference edge** for exported small VLMs / policies  
2. **Data plane** — capture, label, upload trajectories to Anvil  
3. **On-device micro-RL** later (tiny policy heads, RL-on-the-edge research)

```text
Robot (Jetson)                     Lab (Spark / Anvil)
  cameras → preprocess               store frames (refs)
  run exported ONNX/TRT/GGUF  ◄──── export adapter + optional distill
  log (obs, action, reward)   ─────► offline RL / SFT batches
```

**Design rules for Jetson:**

- Export targets: **ONNX / TensorRT / GGUF** as appropriate.  
- Prefer **small dense VLMs** or **distilled students**, not full MoE bases.  
- Keep **same message schema** as lab so sim→real doesn’t rewrite rewards.  
- Optional: Anvil `SamplingClient` with `backend=jetson://…` for remote sample (power/thermal constrained).

### 6.3 Laptop / CPU client

Always supported: control plane client only — author loops on CPU, train on lab GPUs.

---

## 7. Architecture modules (implementation sketch)

```text
anvil/
  client/           # ServiceClient, TrainingClient, SamplingClient, futures, remote
  protocol/         # Datum, ModelInput, AdamParams, multimodal Message, serde
  render/           # ToyTextRenderer; HFChatRenderer
  media/            # LocalMediaStore (cas://sha256/…)
  control/          # Session, AdapterRegistry, gate-override audit
  observe/          # RunMetricsWriter — metrics.jsonl + probes.jsonl
  workers/
    sample.py       # VLLMSampleBackend + LoRA hot-swap
  serve/            # HTTP four-verb transport
  web/              # anvil-web control plane + /observe
  losses/           # named registry: ce, is, ppo, dpo, …
  export/           # format tags peft/gguf/onnx/trt
  backends/
    fake.py         # in-process golden tests
    local.py        # torch+PEFT hand-rolled verbs
  recipes/          # architecture → pattern → plan; GRPO/SFT helpers
recipes/            # sl_loop and operator scripts
```

**Bootstrap on existing OSS** rather than rewrite kernels:

- Train: **PEFT + torch** (hand-rolled verbs; no HF Trainer swallowing the contract).  
- Sample: **vLLM** (already on Sparks) + HF generate for Tier-0 probes.  
- Don’t reimplement FlashAttention; wrap.

The product is the **contract + control plane + media/render consistency + RL debugger**, not a new CUDA stack.

---

## 8. Phased roadmap

See **`docs/roadmap.md`** for live exit criteria. Summary:

| Phase | Focus |
|-------|--------|
| **0** | Spec, stubs, fake backend, web shell |
| **1** | Local Anvil (text, single GPU) — **done** |
| **2** | Sample/train split, GRPO/IS/PPO, futures — **done** |
| **2.5** | RL observability (metrics, probes, adapter sync, J-Lens readouts) — **current** |
| **3** | Vision first-class |
| **4** | Robot / Jetson edge loop |
| **5** | Multi-tenant lab (optional) |

---

## 9. Non-goals (v0–v1)

- Full-parameter fine-tune of 100B+ MoEs on two Sparks.  
- Bit-identical reimplementation of any proprietary scheduler.  
- Arbitrary remote code execution for custom losses.  
- Replacing day-to-day dual-Spark **serve** of large bases (Anvil is train/adapt/export).  
- Claiming bases before weights exist — design for **any HF VLM/LLM** first.

---

## 10. Relationship to the concrete lab stack

| Asset | Role in Anvil world |
|-------|---------------------|
| **forge + hammer** | Train + sample backends; fabric for heavy models later |
| **DS4 dual-Spark serve** | Production inference; can host **merged/adapter** checkpoints from Anvil |
| **Bench tools** | Pre/post train eval |
| **starwatch** | Real tool-use environment + rewards (align, safety, success) |
| **Jetson (incoming)** | Edge vision policy + data collection |
| **mia-rl Ship** | Stay separate — Anvil is **Lab** infrastructure, not Phi-4 coach Ship |

---

## 11. Open questions (decide before heavy code)

1. **Language:** Python-first client (must); control plane Go vs Python?  
2. **Transport:** HTTP+JSON v0 vs gRPC for media-heavy batches?  
3. **Adapter hot-swap:** How fast must sample see new LoRA (every step vs every N)?  
4. **MoE LoRA:** Expert-specific adapters — need research for large MoE bases.  
5. **Safety:** Sandbox for code RL rewards; robot actions never raw from sample without supervisor.  
6. **License:** Apache-2.0 (locked); keep branding as Anvil only.  

---

## 12. Suggested first experiments

1. **Anvil SFT** small dense LoRA on forge, 100 steps, export, serve.  
2. **Anvil GRPO** on a 10-problem math set with exact-match reward + live probes.  
3. **Vision smoke:** LoRA on **`Qwen/Qwen2.5-VL-3B-Instruct`** (see `docs/models.md`) — “describe grasp frame” on a folder of images.  
4. **Jetson dry-run:** export that student (or AWQ/TRT); measure FPS/power on device (when hardware arrives).  

---

## 13. Bottom line

Anvil democratizes RL by shipping a **tiny, stable verb set** + **LoRA multi-tenancy** + **train/sample consistency** + **observe-while-training**, not by inventing new optimizers.

1. **Four verbs and LoRA-first economics.**  
2. **Pluggable backends** — dual Spark now, Jetson on the edge path.  
3. **Vision from day one** in the data model and renderer.  
4. **Export to real serve** (vLLM today, TensorRT/ONNX on Jetson tomorrow).  
5. **RL debugger** — metrics, probes, eventual latent monitors — so negative returns are visible *during* the run.

Until larger bases land: Qwen/Phi/VLM small models prove the shape.

---

## References

- Ben Anderson, “Anatomy of a Modern Finetuning API” — https://benanderson.work/blog/anatomy-of-finetuning-api/  
- TRL GRPO — https://huggingface.co/docs/trl/grpo_trainer  
- PEFT LoRA — https://huggingface.co/docs/peft/conceptual_guides/lora  
- HF VLM fine-tune cookbook — https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl  
- Live exit criteria: `docs/roadmap.md`
