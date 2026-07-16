# Open-source “Tinker-shaped” post-training — design note

**Project:** [Anvil](https://github.com/ajwharton/anvil) (this repo)  
**Status:** Design SSOT · Phase 0 scaffold  
**Outcome:** Democratized RL+SFT with a Tinker-shaped API, own hardware (lab GPU → dual DGX Spark → Jetson edge), **vision** first-class.  
**Date:** 2026-07-16  

Working name **Anvil** = open tool. **Tinker** = Thinking Machines’ hosted API (inspiration only; no affiliation).

Related personal lab context (private): dual-Spark serve, starwatch agents, Jetson robot — not required to use Anvil.

---

## 1. What Tinker actually is (from public docs + cookbook)

Sources: [Tinker docs](https://tinker-docs.thinkingmachines.ai/), [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), [product page](https://thinkingmachines.ai/tinker/), analyses of the API shape.

### 1.1 Product split

| Piece | Role |
|-------|------|
| **`tinker` SDK** | Thin client. You write loops on a laptop/CPU box. |
| **Cloud workers** | Warm base-model pools + LoRA adapters. Scheduling, multi-node, fault tolerance. |
| **`tinker-cookbook`** | Recipes + abstractions (SFT, GRPO/PPO-style RL, DPO/RLHF, distillation, tools, multi-agent, **audio**, **VLM classifier**). |
| **`tml-renderers`** | Chat/tool/reasoning/multimodal rendering so train and sample share one message↔token contract. |

### 1.2 The primitive surface (the thing to copy)

```text
ServiceClient
  └─ create_lora_training_client(base_model, rank=…)
        ├─ forward_backward(batch, loss_fn=…)   # grads w.r.t. LoRA
        ├─ optim_step(AdamParams | …)           # apply optimizer
        ├─ save_state / load_state              # train+optim checkpoint
        ├─ save_weights_and_get_sampling_client()
        └─ SamplingClient.sample(prompt, params)
  └─ RestClient → download checkpoint archive (LoRA / weights export)
```

**Design intent:** not a black-box `train()`. You own **data, loss, reward, environment**; they own **distributed PEFT training and colocated sampling**.

Why LoRA is load-bearing (not just “efficient FT”):

1. **Shared base pools** — many users’ adapters on the same warm base process.  
2. **Small artifacts** (tens–hundreds of MB) — download, version, A/B, edge deploy.  
3. **RL-friendly** — claim (and TML research) that LoRA can match full FT for many post-training loads when set up correctly.  
4. **`sample` sees current adapter** without shipping full MoE weights per step.

### 1.3 What the cookbook proves is in scope

- **SFT** chat, **math RL** (verifiable), **code RL** (sandbox), **preference** (DPO + multi-stage RLHF), **distillation**, **tool/RAG RL**, **multi-agent**, **audio** (Inkling), **VLM image classification**.  
- **Eval** harness alongside train.  
- **Weight export** for local serve.

### 1.4 What Tinker is *not*

- Not open-source cloud.  
- Not full-weight fine-tune by default.  
- Not “run any model you download” without their worker support.  
- Not an edge runtime (Jetson/robot) — export is the bridge.

---

## 2. Why open-source a Tinker *shape*

Open RL stacks already exist (**TRL**, **OpenRLHF**, **veRL**, **Axolotl**, **Unsloth**, etc.). They are powerful **kits**. Tinker is a **product-shaped API**:

| Kit (today) | Tinker-shaped OSS (goal) |
|-------------|---------------------------|
| “Run this training script on these GPUs” | “Call four verbs; backend may be Spark, cloud, or Jetson sim” |
| Train/serve often different stacks | **Same render + adapter ID** for sample and train |
| Infra is your problem | **Pluggable backends** with one client contract |
| Text-only recipes | **Modality-neutral** batches (text, image, video frames, later audio) |

Democratizing RL = **lower the systems tax** while **keeping algorithm control**. Copy the *API philosophy*, not the proprietary cluster.

---

## 3. Working name and layering

**Working name:** **Anvil** (working title — “tinker on an anvil you own”). Rename freely.

```text
┌─────────────────────────────────────────────────────────┐
│  Client (laptop / Grok / starwatch trainer process)     │
│  anvil.ServiceClient → TrainingClient / SamplingClient  │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP/gRPC + object store for media
┌───────────────────────────▼─────────────────────────────┐
│  Control plane                                          │
│  sessions · adapter registry · job queue · auth         │
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

### 4.1 Core verbs (mirror Tinker)

```python
# Pseudocode — not implemented
svc = anvil.ServiceClient(endpoint="http://forge:7600")  # or local://

tc = svc.create_lora_training_client(
    base_model="Qwen/Qwen3.5-4B",   # or vision tower + LM
    rank=32,
    modalities=["text", "image"],   # capability flags
)

# Train step
fut = tc.forward_backward(
    batch=Batch(...),               # token ids OR multimodal items
    loss="cross_entropy",           # or custom loss_fn id / pickled? see §5
)
tc.optim_step(AdamParams(lr=1e-4))
tc.save_state("step-100")

# Online RL
sc = tc.save_weights_and_get_sampling_client()
out = sc.sample(prompt=Messages(...), max_tokens=512, temperature=0.7)
# score out → build batch → forward_backward with RL loss

# Export
path = tc.export_adapter(format="peft")   # HF LoRA
path = tc.export_merged(format="gguf")    # optional pipeline
```

### 4.2 Loss surface (hardest design choice)

Tinker lets the **server** run `forward_backward` with a known `loss_fn` (e.g. cross_entropy) and accumulate LoRA grads. For RL you typically:

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

Match Tinker: non-blocking handles so you can pipeline `forward_backward` while the next batch prepares. Local backend can implement as true async GPU queue; single-GPU backend can still return futures for API compatibility.

---

## 5. Vision as first-class (not a bolt-on)

Tinker already claims **text and vision** and ships a **VLM classifier** recipe. Anvil should not be “LLM trainer + optional images later.”

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
- **Renderer:** expands refs → model-specific image tokens (Qwen-VL, InternVL, Inkling-style patches, etc.).  
- **Train worker:** same renderer as sample worker (critical for RL).

### 5.2 Vision-specific training modes

| Mode | Use case | Notes |
|------|----------|--------|
| **VLM SFT** | Instruction following with images | Standard CE on text tokens; image encoder may freeze or LoRA |
| **VLM classifier / rubric** | “Is this a good grasp?” | Cookbook-like classifier head or graded text |
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

Already in play for **serve** (DS4 TP=2 fabric). For **Anvil train**:

| Backend | Fit |
|---------|-----|
| **A. Single-node LoRA train** on forge (or hammer) | v0 — simple, honest |
| **B. Role split** | Sample on head (vLLM), train on peer — Tinker-like separation |
| **C. TP=2 train** | Only when base is huge; higher systems cost |

Recommendation: **v0 = A + optional B**. Don’t require fabric TP for the first open-source win.

Reuse lessons from DS4 dual-Spark: NVMe models (not USB), explicit host IPs, long NCCL timeouts, worker-first only when multi-node train needs it.

### 6.2 Jetson (edge vision robot) — arriving hardware

Jetson is **not** a Tinker train worker for 30B+ VLMs. It is:

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

- Export targets: **ONNX / TensorRT / GGUF / MLX-less** as appropriate.  
- Prefer **small dense VLMs** or **distilled students**, not full MoE Inkling.  
- Keep **same message schema** as lab so sim→real doesn’t rewrite rewards.  
- Optional: Anvil `SamplingClient` with `backend=jetson://…` for remote sample (power/thermal constrained).

### 6.3 Laptop / CPU client

Always supported: control plane client only. Matches Tinker’s “author on CPU.”

---

## 7. Architecture modules (implementation sketch)

```text
anvil/
  client/           # ServiceClient, TrainingClient, SamplingClient, futures
  protocol/         # protobuf/OpenAPI: Batch, LossSpec, SampleRequest
  render/           # text + multimodal renderers (pluggable)
  media/            # content-addressed blob store
  control/          # session, adapter registry, auth (local first)
  workers/
    train_peft.py   # HF PEFT / Unsloth / custom
    sample_vllm.py  # vLLM OpenAI-compatible or native
  losses/           # ce, dpo, grpo, … registered
  export/           # peft, merge, gguf, onnx
  recipes/          # sl_loop, rl_loop, vlm_classifier, robot_offline_rl
  backends/
    local_gpu.yaml
    dual_spark.yaml
    jetson_edge.yaml
```

**Bootstrap on existing OSS** rather than rewrite kernels:

- Train: **PEFT + TRL** or **OpenRLHF/veRL** behind the same verbs.  
- Sample: **vLLM** (already on Sparks).  
- Don’t reimplement FlashAttention; wrap.

The product is the **contract + control plane + media/render consistency**, not a new CUDA stack.

---

## 8. Phased roadmap

### Phase 0 — Spec & compatibility (1–2 weeks, design/code stubs)

- Freeze OpenAPI for four verbs + export.  
- Golden tests: SFT loop on 0.5B–4B text model on single GPU.  
- Document loss plugin ABI.  

### Phase 1 — Local Anvil on forge (text)

- `anvil serve --backend local` on forge.  
- SFT + simple GRPO on verifiable math (cookbook parity).  
- Export LoRA → load into existing vLLM container.  
- Client from Mac.  

### Phase 2 — Sample/train split + dual-Spark optional

- Dedicated sample worker (vLLM) with hot-swap LoRA.  
- Async futures, basic queue.  
- Starwatch or tool-use RL toy with real tools.  

### Phase 3 — Vision

- Media store + VLM renderer.  
- VLM SFT and classifier recipe.  
- Freeze-encoder defaults.  

### Phase 4 — Robot / Jetson loop

- Offline trajectory format from robot logs.  
- Distill pipeline: lab teacher → Jetson student.  
- Edge sample backend + thermal/power constraints documented.  

### Phase 5 — Multi-user / “mini-SaaS” (optional)

- Auth, multi-adapter isolation, warm base pools — only if people run shared lab hardware.  
- This is where Tinker’s cloud moat is; OSS can stay single-tenant longer.

---

## 9. Non-goals (v0–v1)

- Full-parameter fine-tune of 100B+ MoEs on two Sparks.  
- Bit-identical reimplementation of Tinker’s proprietary scheduler.  
- Arbitrary remote code execution for custom losses.  
- Replacing DS4 day-to-day serve (keep serve path; Anvil is train/adapt).  
- Claiming Inkling-Small support before weights exist — design for **any HF VLM/LLM** first.

---

## 10. Relationship to your concrete stack

| Asset | Role in Anvil world |
|-------|---------------------|
| **forge + hammer** | Train + sample backends; fabric for heavy models later |
| **DS4 dual-Spark serve** | Production inference; can host **merged/adapter** checkpoints from Anvil |
| **Bench tools** (`/mnt/data/tools`) | Pre/post train eval (lm-eval, compare scripts) |
| **starwatch** | Real tool-use environment + rewards (align, safety, success) |
| **Jetson (incoming)** | Edge vision policy + data collection |
| **Tinker (hosted)** | Optional parallel track for Inkling / huge bases until Small is local |
| **mia-rl Ship** | Stay separate — Anvil is **Lab** infrastructure, not Phi-4 coach Ship |

---

## 11. Open questions (decide before heavy code)

1. **Language:** Python-first client (must); control plane Go vs Python?  
2. **Transport:** HTTP+JSON v0 vs gRPC for media-heavy batches?  
3. **Adapter hot-swap:** How fast must sample see new LoRA (every step vs every N)?  
4. **MoE LoRA:** Expert-specific adapters — need research for Inkling-Small.  
5. **Safety:** Sandbox for code RL rewards; robot actions never raw from sample without supervisor.  
6. **Name / license:** Apache-2.0 vs MIT; avoid “Tinker” trademark.  

---

## 12. Suggested first experiments (when coding starts)

1. **Anvil SFT** Qwen3.5-4B LoRA on forge, 100 steps, export, serve beside DS4.  
2. **Anvil GRPO** on a 10-problem math set with exact-match reward.  
3. **Vision smoke:** small VLM LoRA “describe grasp frame” on a folder of images.  
4. **Jetson dry-run:** export student model; measure FPS/power on device (when hardware arrives).  

---

## 13. Bottom line

Tinker democratizes RL by selling a **tiny, stable verb set** + **LoRA multi-tenancy** + **train/sample consistency**, not by inventing new optimizers.

An open-source Anvil should:

1. **Copy the four verbs and LoRA-first economics.**  
2. **Pluggable backends** — dual Spark now, Jetson on the edge path.  
3. **Vision from day one** in the data model and renderer.  
4. **Export to real serve** (vLLM today, TensorRT/ONNX on Jetson tomorrow).  
5. **Sit in Lab**, leverage cookbook ideas, stay compatible enough that recipes port.

When **Inkling-Small** weights land: candidate base for local Anvil + optional hosted Tinker A/B. Until then: Qwen/Phi/VLM small models prove the shape.

---

## References

- https://tinker-docs.thinkingmachines.ai/  
- https://thinkingmachines.ai/tinker/  
- https://github.com/thinking-machines-lab/tinker-cookbook  
- https://thinkingmachines.ai/news/introducing-inkling/  
- Ben Anderson, “Anatomy of a Modern Finetuning API” (Tinker primitive analysis)  
- Local: `aiops/docs/ds4-dual-spark.md`, forge bench tools under `/mnt/data/tools`
