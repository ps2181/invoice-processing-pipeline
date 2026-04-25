---
title: Invoice Processing Pipeline
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - multi-agent
  - grpo
  - rlhf
  - fraud-detection
  - invoice
---

# 🧾 Invoice Processing Pipeline — Self-Improving Multi-Agent Fraud Detection

> **Meta PyTorch OpenEnv Hackathon** · Team: Pritam Satpathy & Gnana Nawin T

A **5-agent adversarial system** that continuously improves its own fraud detection through a closed reinforcement learning loop — built on the OpenEnv framework with GRPO-trained Qwen2.5 LoRA agents.

---

## What Makes This Different

Most multi-agent systems are static pipelines. Ours **gets harder for itself over time**:

1. **Regulator** watches the Auditor across episodes, identifies fraud types it keeps missing (blind spots), and uses **predictive trend analysis** to warn before a blind spot becomes critical
2. **Generator** receives bias weights from the Regulator and creates invoices skewed toward those blind spot fraud types — adversarially exploiting the Auditor's weakness
3. **Extractor** parses the invoice into structured JSON with 4 independent reward signals
4. **Auditor** flags fraud with confidence scores — its mistakes feed back into the Regulator
5. **Approver** makes the final approve / escalate / reject decision

Every episode the loop closes: the Regulator updates the Generator's weights, making the next batch harder in exactly the ways the Auditor failed. **The system pressure-tests its own weakest point.**

---

## Three Novel Features

| Feature | What it does |
|---------|-------------|
| **Predictive Regulator** | Computes trend slope over 5-episode windows — warns of *emerging* blind spots before they become critical, not just after |
| **Compound Fraud** | Invoices can carry two fraud signals simultaneously (e.g. phantom vendor + price gouging). Partial credit for catching one; full reward for both |
| **Confidence Calibration** | Tracks (confidence, correct?) pairs per fraud type. Detects *overconfident misses* — the Auditor saying "90% sure, approved" on a fraudulent invoice — the most dangerous failure mode |

---

## Trained LoRA Agents

All three agents trained with GRPO on live environment data:

| Agent | Model | HF Hub |
|-------|-------|--------|
| Extractor | Qwen2.5-1.5B-Instruct + LoRA r=16 | [ps2181/extractor-lora-qwen2.5-1.5b](https://huggingface.co/ps2181/extractor-lora-qwen2.5-1.5b) |
| Auditor   | Qwen2.5-1.5B-Instruct + LoRA r=16 | [ps2181/auditor-lora-qwen2.5-1.5b](https://huggingface.co/ps2181/auditor-lora-qwen2.5-1.5b) |
| Generator | Qwen2.5-1.5B-Instruct + LoRA r=16 | [ps2181/generator-lora-qwen2.5-1.5b](https://huggingface.co/ps2181/generator-lora-qwen2.5-1.5b) |

**Training setup:** 4-bit QLoRA, r=16, Unsloth + TRL GRPOTrainer, live HF Space as reward verifier

---

## Reward Signals

### Extractor (4 independent signals)
| Signal | Max | What it measures |
|--------|-----|-----------------|
| Format | 0.10 | Required fields present |
| Field accuracy | 0.40 | Vendor / date / currency / total correct |
| Math consistency | 0.25 | qty × unit_price = amount, sum = total |
| Completeness | 0.25 | All line items captured |

### Auditor
| Outcome | Reward |
|---------|--------|
| Correct fraud type detected | 0.99 |
| Clean invoice correctly approved | 0.90 |
| Compound fraud — one type caught | 0.65 |
| Fraud detected, wrong type | 0.50 |
| Miss or false positive | 0.01 |

### Generator (adversarial)
| Outcome | Reward |
|---------|--------|
| Evades both Auditor and Approver | 0.85 |
| Evades Auditor, Approver catches | 0.60 |
| Auditor catches it | 0.10 |

### Regulator
Precision (0.35) + Recall (0.35) + No over-flagging (0.15) + Early warning bonus (0.15)

---

## API Endpoints

### Core OpenEnv
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode (`{"task_id": "easy\|medium\|hard\|expert\|adversarial\|negotiate\|supply_chain"}`) |
| `/step`  | POST | Submit extracted data, get reward + feedback |
| `/grader`| POST | Score without modifying state |
| `/state` | GET  | Episode metadata |
| `/health`| GET  | Health check |
| `/ws`    | WS   | WebSocket interface |

### Multi-Agent
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/multi/reset`   | POST | Start 5-agent episode, Generator biased by Regulator |
| `/multi/extract` | POST | Score Extractor output (4 signals) |
| `/multi/audit`   | POST | Score Auditor output, update tracker |
| `/multi/approve` | POST | Run Approver, compute Generator reward |

### Regulator
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/regulator/report`      | GET  | Detection rates, blind spots, weights |
| `/regulator/forecast`    | GET  | Predictive trend analysis |
| `/regulator/calibration` | GET  | Confidence calibration per fraud type |
| `/regulator/predict`     | POST | Score Regulator blind spot predictions |

---

## Quick Start

```bash
# Health check
curl https://ps2181-invoice-processing-pipeline.hf.space/health

# Start an episode
curl -X POST https://ps2181-invoice-processing-pipeline.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "easy"}'

# Start a multi-agent episode
curl -X POST https://ps2181-invoice-processing-pipeline.hf.space/multi/reset

# Get Regulator report
curl https://ps2181-invoice-processing-pipeline.hf.space/regulator/report
```

---

## Fraud Types

| Type | Description |
|------|-------------|
| `phantom_vendor` | Vendor not in the Approved Vendor Registry |
| `price_gouging` | Unit price > 150% of market max |
| `math_fraud` | Invoice total ≠ sum of line items |
| `duplicate_submission` | Same invoice_id or vendor+date+total already seen |
| `compound_fraud` | Two fraud signals in one invoice |

---

## Links

- **Live Demo**: [ps2181-invoice-processing-pipeline.hf.space/web](https://ps2181-invoice-processing-pipeline.hf.space/web)
- **API Docs**: [ps2181-invoice-processing-pipeline.hf.space/docs](https://ps2181-invoice-processing-pipeline.hf.space/docs)
- **GitHub**: [github.com/ps2181/invoice-processing-pipeline](https://github.com/ps2181/invoice-processing-pipeline)
- **OpenEnv**: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
