---
title: Invoice Processing Pipeline
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Invoice Processing Pipeline — OpenEnv Environment

An OpenEnv environment where an AI agent learns to **extract**, **clean**, and **reconcile** invoice data — a task that mirrors real-world accounts-payable workflows affecting every business.

The agent receives raw invoice text (simulating OCR output or messy CSV imports), processes it into structured data, and receives graded scores (0.0–1.0) with detailed feedback at every step.

---

## Motivation

Invoice processing is one of the most common, tedious, and error-prone tasks in business operations. Finance teams spend countless hours:

- **Extracting** vendor names, dates, line items, and totals from unstructured documents
- **Cleaning** inconsistent formats (dates, currencies, vendor name variations)
- **Reconciling** invoices against purchase orders to catch overcharges, missing items, and billing errors

This environment provides a controlled, reproducible setting to train and evaluate AI agents on these tasks, with clear partial-credit signals that make it suitable for RL training.

---

## Project Structure

```
invoice_processing_pipeline/
├── models.py              Pydantic models: InvoiceAction, InvoiceObservation, InvoiceState
├── client.py              Python client (sync + async) for training code
├── inference.py           LLM baseline agent (OpenAI-compatible)
├── server/
│   ├── __init__.py
│   ├── environment.py     Core logic: invoice generation, graders, reward computation
│   └── app.py             FastAPI server with /reset, /step, /state endpoints
├── openenv.yaml           OpenEnv metadata
├── Dockerfile             Container build
├── requirements.txt       Python dependencies
├── pyproject.toml         Package configuration
└── README.md              This file
```

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy` | Easy | Extract structured fields from a **single, clean** invoice |
| `medium` | Medium | Clean and normalise a **batch of messy** invoices (3–5 invoices) |
| `hard` | Hard | Extract, clean, AND **reconcile against purchase orders** with discrepancy detection |

### Easy: Single Invoice Extraction

The agent receives a well-formatted invoice with clear structure. It must extract: vendor name, date, currency, total, and all line items with descriptions, quantities, unit prices, and amounts.

### Medium: Batch Invoice Cleaning

The agent receives 3–5 invoices with realistic messiness:
- **Date format chaos**: `01/15/2024`, `15-01-2024`, `January 15, 2024`, `15.01.2024`
- **Vendor name typos**: `"Acme Crp"`, `"GloablTech Solutions"`, `"Prmie Office Supplies"`
- **Mixed currency formats**: `$`, `€`, `£` symbols instead of `USD`, `EUR`, `GBP` codes
- **String/number mixing**: amounts like `"$149.99"` instead of `149.99`
- **Math errors**: `qty × unit_price ≠ amount` in some line items

### Hard: Invoice-PO Reconciliation

The agent receives messy invoices PLUS purchase orders and must:
1. Clean all invoice data (same as medium)
2. Compare each invoice against its corresponding PO
3. Flag discrepancies: overcharges, extra items, and missing items

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `raw_text` | string | Raw invoice text (OCR-style or batch format) |
| `task_id` | string | `easy`, `medium`, or `hard` |
| `difficulty` | string | Same as `task_id` |
| `task_description` | string | What the agent should do |
| `attempt_number` | int | Current attempt (0 = just reset) |
| `max_attempts` | int | Maximum allowed attempts (5) |
| `feedback` | string | Detailed grader feedback from last attempt |
| `hint` | string | Appears after 2+ failed attempts |
| `reference_data` | string | Purchase order data (hard task only) |

---

## Action Space

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `extracted_data` | JSON object | Yes | Structured invoice data (format depends on task) |
| `explanation` | string | No | Agent reasoning (optional) |

### Expected `extracted_data` format by task:

**Easy:**
```json
{
    "vendor": "Acme Corp",
    "date": "2024-06-15",
    "currency": "USD",
    "total": 1249.95,
    "line_items": [
        {"description": "Laptop Computer", "qty": 1, "unit_price": 1099.99, "amount": 1099.99},
        {"description": "Wireless Mouse", "qty": 5, "unit_price": 29.99, "amount": 149.95}
    ]
}
```

**Medium:**
```json
{
    "invoices": [
        {"vendor": "...", "date": "YYYY-MM-DD", "currency": "USD", "total": 0.0, "line_items": [...]}
    ]
}
```

**Hard:**
```json
{
    "invoices": [...],
    "discrepancies": [
        {"invoice_idx": 0, "type": "overcharge", "item_description": "Laptop Computer", "detail": "Invoice price 1199.99 vs PO price 1099.99"}
    ]
}
```

---

## Reward Function

Rewards are provided at **every step** (not just terminal), giving agents a rich training signal.

### Easy Task Scoring (0.0–1.0)

| Component | Weight | Condition |
|-----------|--------|-----------|
| Vendor name | 0.15 | Exact match (case-insensitive) |
| Date | 0.10 | Exact match (YYYY-MM-DD) |
| Currency | 0.05 | Exact match (3-letter code) |
| Total | 0.20 | Within ±0.01 |
| Line items | 0.50 | Per-item matching on description, qty, unit_price, amount |

### Medium Task Scoring

Average of per-invoice scores using the Easy grading rubric across the full batch.

### Hard Task Scoring

| Component | Weight |
|-----------|--------|
| Extraction + Cleaning | 60% (same as Medium grading) |
| Discrepancy Detection | 40% (precision + recall of flagged discrepancies) |

### Attempt Penalty

If all 5 attempts are exhausted without reaching 95% score, a **0.85× multiplier** is applied to the final reward.

---

## Setup and Usage

### Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd invoice_processing_pipeline

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Test with curl
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'
```

### Docker

```bash
docker build -t invoice-env .
docker run -p 7860:7860 invoice-env
```

### Running the Baseline

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:7860

python inference.py
```

### Python Client

```python
from client import InvoiceEnvClient

with InvoiceEnvClient("http://localhost:7860") as env:
    result = env.reset(task_id="easy")
    print(result["observation"]["raw_text"])

    result = env.step({
        "vendor": "Acme Corp",
        "date": "2024-06-15",
        "currency": "USD",
        "total": 1249.95,
        "line_items": [...]
    })
    print(f"Score: {result['reward']}")
    print(f"Feedback: {result['observation']['feedback']}")
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode (`{"task_id": "easy\|medium\|hard"}`) |
| `/step` | POST | Submit extracted data, get reward + feedback |
| `/state` | GET | Get current episode metadata |
| `/tasks` | GET | List all tasks with schemas |
| `/grader` | POST | Score a submission without modifying state |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger API docs |

---

## Baseline Scores

| Agent | Easy | Medium | Hard | Average |
|-------|------|--------|------|---------|
| Oracle (ground truth) | 1.00 | 1.00 | 1.00 | 1.00 |
| Qwen2.5-72B-Instruct | ~0.90 | ~0.65 | ~0.45 | ~0.67 |
| Random (empty JSON) | 0.00 | 0.00 | 0.00 | 0.00 |

*Scores are approximate and may vary due to random invoice generation.*

---

## Design Decisions

- **Synthetic data generation**: Every episode creates fresh invoices, preventing memorisation and ensuring reproducibility via random seeds.
- **Partial credit at every step**: The grader scores each component independently (vendor, date, line items, etc.), giving agents fine-grained reward signal.
- **Progressive difficulty**: Easy tests pure extraction, Medium adds data quality issues, Hard adds cross-document reasoning.
- **Realistic noise**: Vendor typos, date format variations, and currency symbol mixing are modelled after actual OCR and data entry errors.
- **Attempt-based penalty**: Encourages agents to get it right early rather than brute-forcing over many attempts.

---

## Links

- OpenEnv GitHub: https://github.com/meta-pytorch/OpenEnv
- Hugging Face Environment Hub: https://huggingface.co/openenv