<div align="center">

# When the System Learns to Pressure-Test Itself

**How we built a 5-agent adversarial RL environment that detects invoice fraud —**  
**and automatically gets harder when it finds its own blind spots.**

<br/>
*Meta PyTorch OpenEnv Hackathon · Grand Finale · April 25–26, 2026*  
*Pritam Satpathy & Gnana Nawin T · Scaler School of Technology, Bangalore*

</div>

---

## The Problem Nobody Talks About

Invoice fraud is boring to talk about and devastating in practice.

It costs businesses an estimated **5% of annual revenue**, and it doesn't announce itself — it hides in purchase order line items, disguised as rounding errors, vendor name typos, and suspiciously round numbers that only look wrong if you already know what to look for.

Finance teams today catch it manually. They compare thousands of invoices against purchase orders, cross-reference vendor registries, and flag anything that smells off. It's slow, it's error-prone, and critically — **it doesn't improve**. A human who misses phantom vendor fraud on Monday is statistically likely to miss it again on Friday.

We asked a different question:

> *What if you could build an LLM system that not only detects fraud, but gets better at detecting the exact fraud types it's currently failing on — automatically, without any human retraining the loop?*

That's what we built.

---

## The Core Idea: Make the System Pressure-Test Itself

Most multi-agent RL setups have agents that operate independently within a single episode. Ours doesn't.

We added a **cross-episode Regulator** — an agent that watches the Auditor across 30 rolling episodes, tracks which fraud types it's systematically missing, and quietly biases the Generator to produce more of those exact scenarios.

No human decides *"let's train more on phantom vendors."* The Regulator notices the detection rate for phantom vendors is at `31%` and trending downward, raises the alarm, and tells the Generator to send more phantom vendor invoices. **The loop closes itself.**

<div align="center">
<img width="1710" height="326" alt="image" src="https://github.com/user-attachments/assets/319654c3-aa24-47e8-9716-734d4e902168" />
</div>


The Auditor sees more of exactly what it's failing on. The Generator gets rewarded for finding those gaps. The Regulator earns points for predicting blind spots *before* they go critical. Every agent has skin in the game.

---

## Five Agents, One Closed Loop

<div align="center">

| Agent | Role | Reward Signal |
|:---:|:---|:---|
| **Generator** | Creates clean or fraudulent invoices, biased by Regulator's blind-spot weights | `+0.85` evades both · `+0.60` evades Auditor · `+0.10` caught |
| **Extractor** | Raw invoice text → structured JSON | format `0.10` · field accuracy `0.40` · math `0.25` · completeness `0.25` |
| **Auditor** | Fraud classification with fraud type + confidence score | `+0.99` correct type · `+0.90` clean cleared · `+0.01` miss or FP |
| **Approver** | Final approve / escalate / reject, gated by confidence | `≥0.80` → reject · `0.50–0.80` → escalate · `<0.50` → approve |
| **Regulator** | Cross-episode meta-agent, 30-episode rolling window | precision `0.35` + recall `0.35` + no over-flagging `0.15` + early warning `0.15` |

</div>

The **Regulator** is the part that makes this genuinely different. Most RL environments treat each episode as independent. The Regulator sits outside that — accumulating detection rates, computing trend slopes over 5-episode windows, and warning of *emerging* blind spots before they go critical. It's proactive oversight, not reactive retraining.

---

## Seven Tasks, One Curriculum

<div align="center">

| # | Task | What the Agent Faces | Difficulty |
|:---:|:---|:---|:---:|
| 1 | `easy` | Single clean invoice — extract 5 fields | Easy |
| 2 | `medium` | Batch with date chaos, vendor typos, currency noise | Medium |
| 3 | `hard` | Extraction + PO reconciliation — flag overcharges, missing items | Hard |
| 4 | `expert` | Full fraud audit across all four fraud types | Expert |
| 5 | `adversarial` | OCR corruption, SUBTOTAL traps, fake TAX/FX noise lines | Expert |
| 6 | `negotiate` | Ask clarifying questions first (bonus for ≤2), then extract | Medium |
| 7 | `supply_chain` | Detect quantity shortfalls, price spikes, phantom deliveries | Expert |

</div>

The difficulty also adjusts **dynamically** based on the agent's rolling score. Score above `0.85`? The next batch gets heavier OCR corruption, more PO discrepancies, deeper adversarial traps. Drop below `0.60`? It eases off. The agent is always working at its productive edge.

---

## The Part Where We Caught Our Own Reward Hacking

This was the most interesting moment in the project.

At training step 10, we had:

```
math_consistency:   0.97  
completeness:       1.00  
field_accuracy:     0.00  :(  ← hallucinating every actual value
```

The model had figured out that it could score well by outputting JSON that was *arithmetically correct* — quantities times unit prices summed to the totals perfectly — while **hallucinating every actual value**. Vendor name: made up. Date: made up. Currency: made up. All internally consistent. All completely wrong.

This is reward hacking. A single aggregated reward would have happily reported high performance and called it a day.

Our four **independent** reward signals made the failure immediately visible. We could see exactly which signal the model had learned to game and which it was ignoring.

> **That's the entire argument for independent reward functions: not just diversity, but diagnosability.**

We adjusted training emphasis. By step 30, field accuracy had climbed from `0.00` to `0.30+` while math consistency stayed stable.

<div align="center">

| Step | Total Reward | Env Score | Format | Math Consistency |
|:---:|:---:|:---:|:---:|:---:|
| 10 | 2.361 | 0.113 | 0.900 | 0.347 |
| 20 | 2.595 | 0.282 | 0.900 | 0.413 |
| 30 | 2.657 | **0.304** | **0.950** | 0.403 |

**Environment score: `0.113 → 0.304` in 30 steps — a 169% improvement in live-graded extraction accuracy.**

</div>

---

## The Reward Architecture

### 🔍 Extractor — 4 Independent Signals

```python
reward_format(extracted)             # weight 0.10 — all 5 required JSON keys present?
reward_field_accuracy(extracted, gt) # weight 0.40 — vendor / date / currency / total match?
reward_math_consistency(extracted)   # weight 0.25 — qty × unit_price = amount per line?
reward_completeness(extracted, gt)   # weight 0.25 — all expected line items present?

# All clamped to (0.01, 0.99) — no log(0), no gradient collapse at boundaries
```

### Auditor — Precision-Weighted

<div align="center">

| Outcome | Reward | Why |
|:---|:---:|:---|
| Correct fraud type detected | **0.99** | Rewards precise classification, not just flagging |
| Clean invoice correctly approved | **0.90** | Keeps false-positive rate honest |
| Compound fraud — one of two types caught | **0.65** | Partial credit prevents discouragement on hard cases |
| Fraud flagged but wrong type | **0.50** | Penalises sloppiness while crediting intent |
| Miss or false positive | **0.01** | Near-zero punishes both failure modes symmetrically |

</div>

### Regulator — Cross-Episode

```
Total = Precision(0.35) + Recall(0.35) + No-over-flagging(0.15) + Early-warning-bonus(0.15)
```

The early-warning bonus rewards the Regulator for predicting emerging blind spots *before* detection rates cross the critical threshold — proactive oversight, not reactive alarm.

---

## Building With OpenEnv

The environment is a FastAPI app deployed on HuggingFace Spaces, exposing the standard OpenEnv interface. The training Colab connects directly to the live Space — `/grader` *is* the reward function. There's no separate scoring script. **The environment and the verifier are the same thing.**

```bash
# Start an episode
POST /reset  {"task_id": "expert"}

# Submit an extraction or audit result
POST /step   {"episode_id": "...", "extracted_data": {...}}

# Check Regulator state anytime
GET  /regulator/report       # detection rates, blind spots, generator bias weights
GET  /regulator/forecast     # trend slopes, emerging blind spots with early warnings
GET  /regulator/calibration  # overconfidence / underconfidence per fraud type
```

Training uses **GRPO via TRL** with **Unsloth-optimised 4-bit QLoRA** on `Qwen2.5-1.5B-Instruct` — three separate LoRA adapters for Extractor, Auditor, and Generator, each trained on their own reward signal.

```
Colab → /reset  (fresh synthetic invoice from live environment)
      → model generates JSON extraction
      → /grader  scores against ground truth
      → GRPO updates weights toward higher-reward completions
      → repeat 200 steps
```

---

## What We Learned

**Reward design is product design.** Every reward function is a specification for the behaviour you actually want. Getting the Auditor reward right — where catching the *right* fraud type earns `0.99` but the *wrong* type earns `0.50` and missing entirely earns `0.01` — took more thinking than most of the engineering.

**Multiple reward signals are diagnostics, not just incentives.** We didn't add four signals to the Extractor because the theory said to. We added them because we wanted to *see* where the model was failing. They paid off immediately at step 10.

**Cross-episode agents change what's possible.** The Regulator couldn't exist in a single-episode design. Most RL environments treat each episode as independent. Giving one agent access to the history of another creates a fundamentally different kind of oversight — one that looks less like evaluation and more like a genuine colleague watching your back.

---

## Try It

<div align="center">

| Resource | Link |
|:---|:---|
| **Live Environment** | [ps2181-invoice-processing-pipeline.hf.space](https://ps2181-invoice-processing-pipeline.hf.space) |
| **Gradio Demo UI** | [/web](https://ps2181-invoice-processing-pipeline.hf.space/web) |
| **API Docs** | [/docs](https://ps2181-invoice-processing-pipeline.hf.space/docs) |
| **Training Colab** | [Open notebook](https://colab.research.google.com/drive/1C1_3giNt-NmbzKNFJr5_L1fms3L8LfmB) |
| **GitHub** | [invoice-processing-pipeline](https://github.com/ps2181/invoice-processing-pipeline) |
| **Extractor Model** | [ps2181/extractor-lora-qwen2.5-1.5b](https://huggingface.co/ps2181/extractor-lora-qwen2.5-1.5b) |
| **Auditor Model** | [ps2181/auditor-lora-qwen2.5-1.5b](https://huggingface.co/ps2181/auditor-lora-qwen2.5-1.5b) |
| **Generator Model** | [ps2181/generator-lora-qwen2.5-1.5b](https://huggingface.co/ps2181/generator-lora-qwen2.5-1.5b) |
|  **Demo Video** | [video](https://youtu.be/QSB4UOLvaC8?si=SGnIwsfTW4JGsU3e) |

</div>

---

<div align="center">

*Built for the Meta PyTorch OpenEnv Hackathon 2026.*  
*Theme alignment: Multi-Agent Interactions (#1) · Fleet AI Scalable Oversight (#1 bonus) · Professional Tasks (#3.1) · Self-Improvement (#4)*

<br/>

**Pritam Satpathy & Gnana Nawin T · Scaler School of Technology · Bangalore**

</div>
