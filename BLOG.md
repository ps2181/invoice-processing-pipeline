# Invoice Processing Pipeline — Multi-Agent RL Environment for Financial Fraud Detection

**Meta PyTorch OpenEnv Hackathon Grand Finale | April 25–26, 2026**
**Team: Pritam Satpathy + Gnana Nawin T**

---

## The Problem

Invoice fraud costs businesses an estimated 5% of annual revenue. Finance teams manually process thousands of invoices every month — extracting vendor names, dates, line items, totals — and checking them against purchase orders for discrepancies. The work is slow (hours per batch), error-prone (typos, OCR noise, format chaos), and gameable (phantom vendors, price gouging, duplicate submissions).

We built an RL training environment that teaches LLMs to do this automatically — and improves itself when it discovers its own blind spots.

---

## What We Built

An OpenEnv-compatible environment deployed on HuggingFace Spaces:
**[https://ps2181-invoice-processing-pipeline.hf.space](https://ps2181-invoice-processing-pipeline.hf.space)**

---

## Architecture: 5-Agent System

```
          ┌─────────────────────────────────────────────────────────┐
          │              ADVERSARIAL REWARD (dashed)                │
          │                                                         │
          ▼                                                         │
  ┌───────────────┐                                                 │
  │   Generator    │◄───── Regulator biases fraud type ◄────┐       │
  │ Creates fraud  │                                        │       │
  └───────┬───────┘                                        │       │
          │ Raw invoice text                                │       │
          ▼                                                 │       │
  ┌───────────────┐                                        │       │
  │   Extractor    │                                        │       │
  │ Text → JSON    │                                        │       │
  └───────┬───────┘                                        │       │
          │ Structured data                          ┌─────┴─────┐ │
          ▼                                          │ Regulator  │ │
  ┌───────────────┐                                  │ Cross-     │ │
  │    Auditor     │────── decision history ────────►│ episode    │ │
  │ Fraud detect   │                                  │ meta-agent │ │
  └───────┬───────┘                                  └───────────┘ │
          │ Verdict + flags                                         │
          ▼                                                         │
  ┌───────────────┐                                                 │
  │   Approver     │────────────────────────────────────────────────┘
  │ Approve/reject │
  └───────┬───────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  4 Independent Reward Signals        │
  │  Format · Field · Math · Completeness│
  └──────────────────────────────────────┘
```

| Agent | Role | Reward Signal |
|-------|------|---------------|
| **Generator** | Creates clean or fraudulent invoices | Rewarded when fraud slips past Auditor (adversarial self-play) |
| **Extractor** | Reads raw invoice text → structured JSON | 4 independent signals: format, field accuracy, math consistency, completeness |
| **Auditor** | Reviews extraction, flags fraud | +0.99 correct detection, +0.90 clean clearance, 0.01 for miss/false positive |
| **Approver** | Final approve/reject/escalate decision | +0.95 correct decision |
| **Regulator** | Monitors Auditor blind spots across episodes | Precision + recall of blind spot predictions |

---

## The Key Innovation: The Regulator

The Regulator is a cross-episode meta-agent — it watches the Auditor's decision history over 30 episodes and identifies systematic failure patterns:

```
AUDITOR PERFORMANCE TRACKER (last 30 episodes)

Fraud Type            Detection Rate
─────────────────────────────────────
phantom_vendor        31%   ⚠ BLIND SPOT
price_gouging         74%   ✓ OK
math_fraud            81%   ✓ OK
duplicate_submission  62%   ✓ OK

False Positive Rate:  12%   ✓ OK

REGULATOR VERDICT: Recommend retraining on phantom_vendor
```

When the Regulator detects a blind spot, the Generator automatically starts producing more of that fraud type — closing the self-improvement loop without human intervention.

This directly addresses **Theme #1 (Fleet AI Scalable Oversight)** and **Theme #4 (Self-Improvement)**.

---

## 7 Tasks (Progressive Difficulty)

| Task | Difficulty | What the Agent Does |
|------|-----------|---------------------|
| `easy` | Easy | Extract fields from a single clean invoice |
| `medium` | Medium | Clean + normalise a batch of messy invoices (typos, date chaos, currency symbols) |
| `hard` | Hard | Extract + reconcile against purchase orders, flag discrepancies |
| `expert` | Expert | Fraud audit: classify phantom_vendor / price_gouging / math_fraud / duplicate_submission |
| `adversarial` | Hard | Extract from OCR-corrupted invoice with SUBTOTAL trap and FX noise lines |
| `negotiate` | Medium | Ask clarification questions then submit extraction (bonus for ≤2 questions) |
| `supply_chain` | Expert | Detect quantity shortfalls, price spikes, phantom deliveries in delivery records |

---

## Design Decisions

### 4 Independent Reward Functions (Anti-Hacking)

Per the hackathon guide: *"use multiple independent reward functions — if you only have one, it is easier for the model to hack it."*

```python
format_reward()       # Are all 5 required JSON keys present?       weight: 0.10
field_reward()        # Do vendor/date/currency/total match?         weight: 0.40
math_reward()         # Does qty × unit_price = amount for all items? weight: 0.25
completeness_reward() # Are all line items present (recall)?          weight: 0.25
```

During training we observed the model maximising `math_reward` (0.97) and `completeness_reward` (1.0) while `field_reward` stayed at 0.00 — the model learned to output arithmetic-consistent JSON while hallucinating values. Our independent signals made this reward hacking immediately visible, confirming the design choice.

### Adversarial Self-Play

The Generator is rewarded when its fraud evades the Auditor:
- Fraud undetected, Approver approves → Generator reward: **0.85**
- Auditor missed but Approver caught → Generator reward: **0.60**
- Auditor caught it → Generator reward: **0.10**

This creates evolutionary pressure: the Generator evolves harder-to-detect fraud, forcing the Auditor to improve.

### Dynamic Difficulty

The environment tracks recent agent scores per task (rolling window of 10 episodes) and adjusts generation parameters:
- Agent scoring ≥ 0.85 → harder parameters (more invoices, more OCR noise, more discrepancies)
- Agent scoring < 0.60 → easier parameters
- In between → standard

### All Rewards Clamped to (0.01, 0.99)

Avoids `log(0)` in policy gradient and prevents the model from getting stuck at boundaries.

---

## Tech Stack

```
Environment:  FastAPI + OpenEnv-core + Pydantic
Deployment:   HuggingFace Spaces (Docker, port 7860)
UI:           Gradio (mounted at /web)
Training:     TRL GRPOTrainer + Unsloth (Qwen2.5-1.5B-Instruct, 4-bit QLoRA)
Model:        unsloth/Qwen2.5-1.5B-Instruct  r=16 LoRA
Reward:       4 local signals + live /grader endpoint on HF Space
```

---

## Training Setup

GRPO (Group Relative Policy Optimization) with:
- `num_generations = 4` — 4 completions per prompt, compared within group
- `max_steps = 200`
- `learning_rate = 5e-6`
- Live `/grader` endpoint on HF Space as environment verifier

The training loop:
```
Colab samples episode → HF Space /reset → gets live invoice
Model generates JSON extraction
HF Space /grader scores it against ground truth
GRPO updates model toward higher-scoring completions
```

---

## What Worked (Achievements)

### 1. Reward Hacking Detection — Caught at Step 10

The independent reward signals caught a classic reward hacking pattern immediately. The model maximised math and completeness while hallucinating field values. Without 4 independent signals, this would have been invisible behind a rising aggregate reward.

| Step | Total Reward | Env Score | Format | Math |
|------|-------------|-----------|--------|------|
| 10   | 2.361       | 0.113     | 0.900  | 0.347 |
| 20   | 2.595       | 0.282     | 0.900  | 0.413 |
| 30   | 2.657       | 0.304     | 0.950  | 0.403 |

Environment score rose **0.113 → 0.304 in 30 steps** — a 169% improvement in correct invoice extraction as scored by the live environment grader.

### 2. Live Environment as Verifier

Training Colab directly calls `/grader` on the deployed HF Space — the environment IS the reward function. No separate reward model. Deterministic and reproducible.

### 3. Regulator Concept Validated

The cross-episode tracking logic works: the Regulator correctly identifies `phantom_vendor` as the Auditor's weakest category and triggers Generator bias toward that fraud type. No other OpenEnv environment we've seen implements a cross-episode meta-agent.

### 4. Full 7-Task Ladder Deployed

All 7 tasks are live on the HF Space with independent graders, schemas, and difficulty calibration. The progressive structure directly supports curriculum learning.

### 5. Clean OpenEnv API Compliance

Standard `reset()` / `step()` / `state()` interface, WebSocket support, Swagger docs at `/docs`, Gradio UI at `/web`. Drop-in compatible with any OpenEnv training script.

---

## Where We're Having Problems (Honest Assessment)

### 1. Field Reward Plateau

The `field_reward` (vendor name, date, currency, total accuracy) remains the hardest signal for the 1.5B model to crack. Even at step 30, the environment score is 0.304 — meaning the model still hallucinates field values despite correct structure and math. We suspect this is a model capacity issue: Qwen2.5-1.5B may not have enough parameters to learn extraction patterns from raw OCR text in 200 steps.

**Potential fix:** Switching to Qwen2.5-7B or adding a light SFT warmup phase with 50–100 correct extraction examples before RL.

### 2. Multi-Agent Coordination Not Yet Trained End-to-End

The 5-agent architecture is designed and the environment supports it, but we haven't yet run the full adversarial training loop (Generator vs Auditor) end-to-end with GRPO. Currently, the Extractor is trained in isolation. The Regulator logic runs as environment-side code, not as a trainable agent.

**Potential fix:** Implementing a two-phase training loop — Phase 1: train Extractor on easy/medium, Phase 2: train Auditor against Generator with Regulator feedback.

### 3. Compute Constraints

4-bit QLoRA on a Colab T4 limits batch sizes and generation counts. With `num_generations=4`, each step is slow enough that we couldn't push past ~50 steps in the available time. The reward curves are trending upward but haven't converged.

**Potential fix:** Onsite compute credits (HF GPU Spaces) should allow `num_generations=8` and 500+ steps.

### 4. OCR Noise Robustness

The `adversarial` task (trap-resistant extraction with SUBTOTAL/FX noise lines) works as an environment, but the model hasn't been trained on it yet. Early inference tests show the model consistently falls for fake SUBTOTAL lines.

**Potential fix:** Adding adversarial examples to the curriculum after the model achieves ≥0.60 on `medium`.

---

## What Makes This Novel

1. **Regulator agent** — no other OpenEnv environment has a cross-episode meta-agent that monitors another agent for systematic cognitive blind spots

2. **Closed self-improvement loop** — Regulator detects blind spot → Generator biases fraud generation toward that type → Auditor forced to improve → no human intervention required

3. **Adversarial Generator arms race** — Generator rewarded for evading Auditor creates evolutionary pressure on fraud detection

4. **Live environment as verifier** — training Colab directly calls `/grader` on deployed HF Space — the environment IS the reward function

5. **4 independent reward signals** — made reward hacking immediately visible during training (detected it at step 10)

---

## Theme Alignment

| Theme | Alignment |
|-------|-----------|
| **#1 Multi-Agent** | 5 agents with conflicting incentives (Generator vs Auditor) |
| **#1 Sub: Fleet AI Oversight** (bonus) | Regulator monitors Auditor cross-episode |
| **#3.1 Professional Tasks** | Invoice processing = core enterprise workflow |
| **#3.1 Sub: Scaler AI Labs** (bonus) | Multi-agent RL for enterprise financial workflows |
| **#4 Self-Improvement** | Generator adapts based on Regulator blind spot findings |

---

## Links

- **Live Environment:** [https://ps2181-invoice-processing-pipeline.hf.space](https://ps2181-invoice-processing-pipeline.hf.space)
- **Gradio UI:** [https://ps2181-invoice-processing-pipeline.hf.space/web](https://ps2181-invoice-processing-pipeline.hf.space/web)
- **API Docs:** [https://ps2181-invoice-processing-pipeline.hf.space/docs](https://ps2181-invoice-processing-pipeline.hf.space/docs)
- **GitHub:** [https://github.com/ps2181/invoice-processing-pipeline](https://github.com/ps2181/invoice-processing-pipeline)

---

## Team

**Pritam Satpathy** + **Gnana Nawin T**
Meta PyTorch OpenEnv Hackathon Grand Finale
Scaler School of Technology, Bangalore — April 25–26, 2026