"""
Invoice Processing Pipeline — Core Environment

Seven tasks:
  easy         — Extract structured fields from a single, clean invoice.
  medium       — Clean & normalise a batch of messy invoices.
  hard         — Extract, clean, AND reconcile against purchase orders.
  expert       — Fraud audit: classify phantom_vendor, price_gouging,
                 duplicate_submission, math_fraud.
  adversarial  — Extract from an invoice with OCR corruption, a misleading
                 SUBTOTAL trap, fabricated TAX/ADJUSTMENT lines, and a
                 foreign-currency FX noise line.
  negotiate    — Ask clarification questions then submit extraction; bonus
                 for solving with ≤2 questions.
  supply_chain — Identify anomalies in delivery records: quantity_shortfall,
                 price_spike, unauthorized_substitution, phantom_delivery.

Each episode generates fresh synthetic data so the agent cannot memorize.
Dynamic difficulty: generation parameters adapt based on recent agent scores.
"""

from __future__ import annotations

import collections
import copy
import json
import random
import re
import string
import threading
import uuid
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models import InvoiceAction, InvoiceObservation, InvoiceState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VENDORS = [
    "Acme Corp", "GlobalTech Solutions", "Prime Office Supplies",
    "DataStream Inc", "CloudNine Services", "Metro Logistics",
    "Pinnacle Electronics", "Summit Consulting", "Vertex Manufacturing",
    "Horizon Digital", "NexGen Software", "BluePeak Analytics",
]

ITEMS = [
    ("Laptop Computer", 899.99, 1299.99),
    ("Wireless Mouse", 19.99, 49.99),
    ("USB-C Hub", 29.99, 79.99),
    ("Monitor Stand", 39.99, 89.99),
    ("Keyboard", 49.99, 149.99),
    ("Webcam HD", 59.99, 129.99),
    ("Desk Lamp", 24.99, 69.99),
    ("Notebook Pack", 9.99, 29.99),
    ("Printer Paper (Ream)", 7.99, 14.99),
    ("Whiteboard Markers (Set)", 5.99, 12.99),
    ("External SSD 1TB", 79.99, 149.99),
    ("Headset", 39.99, 99.99),
    ("Cable Management Kit", 14.99, 34.99),
    ("Ergonomic Chair", 299.99, 599.99),
    ("Standing Desk Converter", 199.99, 399.99),
]

CURRENCIES = ["USD", "EUR", "GBP"]
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}

# Expert task: vendors not in the approved registry
PHANTOM_VENDORS = [
    "QuickSupply Hub", "TechVendor Direct", "GlobalOffice Ltd",
    "FastInvoice Inc", "EasyBill Solutions", "RapidProcure Co",
    "BudgetSource LLC", "OfficeLink International",
]

# Expert task: market price ceilings derived from ITEMS ranges
MARKET_PRICE_MAX = {name: hi for name, _lo, hi in [
    ("Laptop Computer", 899.99, 1299.99),
    ("Wireless Mouse", 19.99, 49.99),
    ("USB-C Hub", 29.99, 79.99),
    ("Monitor Stand", 39.99, 89.99),
    ("Keyboard", 49.99, 149.99),
    ("Webcam HD", 59.99, 129.99),
    ("Desk Lamp", 24.99, 69.99),
    ("Notebook Pack", 9.99, 29.99),
    ("Printer Paper (Ream)", 7.99, 14.99),
    ("Whiteboard Markers (Set)", 5.99, 12.99),
    ("External SSD 1TB", 79.99, 149.99),
    ("Headset", 39.99, 99.99),
    ("Cable Management Kit", 14.99, 34.99),
    ("Ergonomic Chair", 299.99, 599.99),
    ("Standing Desk Converter", 199.99, 399.99),
]}
PRICE_GOUGE_FACTOR = 1.5  # unit_price > 1.5x market max = price gouging

# Adversarial task: OCR character corruption map (digit → visually similar letter)
OCR_CORRUPTION_MAP: Dict[str, str] = {
    "0": "O", "1": "l", "5": "S", "6": "b", "8": "B", "2": "Z", "9": "q"
}

# Adversarial task: approximate FX rates (not live — used only as a distractor)
FX_RATES: Dict[str, float] = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79}

# Supply chain: item catalogue (item_name, unit_price_low, unit_price_high)
SC_ITEMS = [
    ("Steel Bolts (Box)", 12.00, 20.00),
    ("Aluminum Sheet 1m x 2m", 45.00, 70.00),
    ("Industrial Lubricant 5L", 30.00, 50.00),
    ("Safety Helmet", 15.00, 25.00),
    ("Hydraulic Pump", 200.00, 350.00),
    ("Copper Cable 100m", 80.00, 120.00),
    ("PVC Pipe 3m", 8.00, 15.00),
    ("Circuit Breaker 20A", 25.00, 40.00),
    ("Welding Rod Pack", 18.00, 28.00),
    ("Air Filter Unit", 55.00, 90.00),
]

SC_SUPPLIERS = [
    "IronWorks Ltd", "MetalForge Co", "SafetyFirst Supply",
    "ElectroParts Inc", "PipelineSource", "FastenersWorld",
]

SC_ANOMALY_TYPES = [
    "quantity_shortfall", "price_spike",
    "unauthorized_substitution", "phantom_delivery",
]

# ---------------------------------------------------------------------------
# Dynamic difficulty: class-level performance tracker (shared across sessions)
# ---------------------------------------------------------------------------

_PERF_HISTORY: Dict[str, collections.deque] = {
    task: collections.deque(maxlen=10)
    for task in ["easy", "medium", "hard", "expert", "adversarial", "negotiate", "supply_chain"]
}
_PERF_LOCK = threading.Lock()


def _get_dynamic_params(task_id: str) -> Dict[str, Any]:
    """Return generation parameters adjusted by recent agent performance."""
    with _PERF_LOCK:
        history = list(_PERF_HISTORY.get(task_id, []))

    avg = sum(history) / len(history) if history else 0.72  # mid-range default

    if avg >= 0.85:  # agent is doing well → harder
        return {
            "n_invoices": (4, 6),
            "ocr_intensity": 0.55,
            "n_discrepancies": (3, 5),
            "n_deliveries": (6, 7),
            "n_anomalies": 3,
        }
    elif avg < 0.60:  # agent is struggling → easier
        return {
            "n_invoices": (2, 3),
            "ocr_intensity": 0.15,
            "n_discrepancies": (1, 2),
            "n_deliveries": (4, 5),
            "n_anomalies": 2,
        }
    else:
        return {
            "n_invoices": (3, 5),
            "ocr_intensity": 0.35,
            "n_discrepancies": (2, 3),
            "n_deliveries": (4, 7),
            "n_anomalies": random.randint(2, 3),
        }


def _record_score(task_id: str, score: float) -> None:
    """Push a score into the class-level performance history."""
    with _PERF_LOCK:
        if task_id in _PERF_HISTORY:
            _PERF_HISTORY[task_id].append(score)


# ---------------------------------------------------------------------------
# Shared generators
# ---------------------------------------------------------------------------

def _rand_date(start_year: int = 2024, end_year: int = 2025) -> date:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _format_date_clean(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _format_date_messy(d: date) -> str:
    """Return a randomly-chosen messy date format."""
    formats = [
        "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %b %Y",
        "%m-%d-%y", "%d.%m.%Y", "%Y/%m/%d",
    ]
    return d.strftime(random.choice(formats))


def _typo_vendor(name: str) -> str:
    """Introduce a subtle typo into a vendor name."""
    strategies = ["swap", "drop", "double", "case"]
    strat = random.choice(strategies)
    idx = random.randint(1, max(1, len(name) - 2))
    if strat == "swap" and idx < len(name) - 1:
        return name[:idx] + name[idx + 1] + name[idx] + name[idx + 2:]
    elif strat == "drop":
        return name[:idx] + name[idx + 1:]
    elif strat == "double":
        return name[:idx] + name[idx] + name[idx:]
    else:
        return name[:idx] + name[idx].swapcase() + name[idx + 1:]


def _corrupt_ocr(text: str, intensity: float = 0.3) -> str:
    """Randomly replace digits with visually similar letters per OCR_CORRUPTION_MAP."""
    result = []
    for ch in text:
        if ch in OCR_CORRUPTION_MAP and random.random() < intensity:
            result.append(OCR_CORRUPTION_MAP[ch])
        else:
            result.append(ch)
    return "".join(result)


def _generate_line_items(n: int) -> List[Dict[str, Any]]:
    chosen = random.sample(ITEMS, min(n, len(ITEMS)))
    items = []
    for desc, lo, hi in chosen:
        qty = random.randint(1, 20)
        unit_price = round(random.uniform(lo, hi), 2)
        amount = round(qty * unit_price, 2)
        items.append({
            "description": desc,
            "qty": qty,
            "unit_price": unit_price,
            "amount": amount,
        })
    return items


def _generate_invoice(vendor: str | None = None, currency: str | None = None) -> Dict[str, Any]:
    vendor = vendor or random.choice(VENDORS)
    currency = currency or random.choice(CURRENCIES)
    inv_date = _rand_date()
    line_items = _generate_line_items(random.randint(2, 6))
    total = round(sum(it["amount"] for it in line_items), 2)
    return {
        "invoice_id": f"INV-{random.randint(10000, 99999)}",
        "vendor": vendor,
        "date": _format_date_clean(inv_date),
        "currency": currency,
        "total": total,
        "line_items": line_items,
    }


# ===================================================================
# TASK: EASY — single invoice extraction
# ===================================================================

def _render_clean_invoice(inv: Dict[str, Any]) -> str:
    """Render a single invoice as semi-structured text (OCR-style)."""
    sym = CURRENCY_SYMBOLS.get(inv["currency"], "$")
    lines = [
        f"INVOICE",
        f"-------",
        f"Invoice #: {inv['invoice_id']}",
        f"Vendor: {inv['vendor']}",
        f"Date: {inv['date']}",
        f"Currency: {inv['currency']}",
        f"",
        f"Items:",
        f"{'Description':<30} {'Qty':>5} {'Unit Price':>12} {'Amount':>12}",
        f"{'-'*30} {'-'*5} {'-'*12} {'-'*12}",
    ]
    for it in inv["line_items"]:
        lines.append(
            f"{it['description']:<30} {it['qty']:>5} {sym}{it['unit_price']:>10.2f} {sym}{it['amount']:>10.2f}"
        )
    lines.append(f"{'':>30} {'':>5} {'TOTAL':>12} {sym}{inv['total']:>10.2f}")
    return "\n".join(lines)


def _grade_easy_with_breakdown(
    submitted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Tuple[float, str, Dict[str, Any]]:
    """Grade single-invoice extraction. Returns (score, feedback, reward_breakdown)."""
    score = 0.0
    feedback_parts = []
    breakdown: Dict[str, Any] = {}

    # Vendor (0.15)
    sub_vendor = submitted.get("vendor", "").strip()
    v_correct = sub_vendor.lower() == ground_truth["vendor"].lower()
    v_score = 0.15 if v_correct else 0.0
    score += v_score
    breakdown["vendor"] = {"score": v_score, "max": 0.15, "status": "correct" if v_correct else "wrong"}
    feedback_parts.append(
        "Vendor: correct" if v_correct
        else f"Vendor: wrong (expected '{ground_truth['vendor']}', got '{sub_vendor}')"
    )

    # Date (0.10)
    sub_date = submitted.get("date", "").strip()
    d_correct = sub_date == ground_truth["date"]
    d_score = 0.10 if d_correct else 0.0
    score += d_score
    breakdown["date"] = {"score": d_score, "max": 0.10, "status": "correct" if d_correct else "wrong"}
    feedback_parts.append(
        "Date: correct" if d_correct
        else f"Date: wrong (expected '{ground_truth['date']}', got '{sub_date}')"
    )

    # Currency (0.05)
    sub_cur = submitted.get("currency", "").strip().upper()
    c_correct = sub_cur == ground_truth["currency"]
    c_score = 0.05 if c_correct else 0.0
    score += c_score
    breakdown["currency"] = {"score": c_score, "max": 0.05, "status": "correct" if c_correct else "wrong"}
    feedback_parts.append(
        "Currency: correct" if c_correct
        else f"Currency: wrong (expected '{ground_truth['currency']}', got '{sub_cur}')"
    )

    # Total (0.20)
    try:
        sub_total = float(submitted.get("total", 0))
        t_correct = abs(sub_total - ground_truth["total"]) < 0.01
        t_score = 0.20 if t_correct else 0.0
        score += t_score
        breakdown["total"] = {"score": t_score, "max": 0.20, "status": "correct" if t_correct else "wrong"}
        feedback_parts.append(
            "Total: correct" if t_correct
            else f"Total: wrong (expected {ground_truth['total']}, got {sub_total})"
        )
    except (ValueError, TypeError):
        breakdown["total"] = {"score": 0.0, "max": 0.20, "status": "wrong"}
        feedback_parts.append("Total: could not parse")

    # Line items (0.50)
    sub_items = submitted.get("line_items", [])
    gt_items = ground_truth["line_items"]
    if not isinstance(sub_items, list):
        li_score = 0.0
        li_status = "wrong"
        feedback_parts.append("Line items: not a list")
    else:
        li_frac = _grade_line_items(sub_items, gt_items)
        li_score = round(li_frac * 0.50, 4)
        score += li_score
        li_status = "correct" if li_frac >= 0.99 else ("partial" if li_frac > 0.0 else "wrong")
        feedback_parts.append(
            f"Line items: {li_frac:.0%} match ({len(sub_items)} submitted, {len(gt_items)} expected)"
        )
    breakdown["line_items"] = {"score": li_score, "max": 0.50, "status": li_status}

    return round(min(score, 1.0), 4), "; ".join(feedback_parts), breakdown


def _grade_easy(submitted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Thin wrapper — drops breakdown for backward compatibility."""
    score, feedback, _ = _grade_easy_with_breakdown(submitted, ground_truth)
    return score, feedback


def _grade_line_items(submitted: List[Dict], expected: List[Dict]) -> float:
    """Compare line items, return fraction matched (0-1)."""
    if not expected:
        return 1.0 if not submitted else 0.0

    matched = 0
    used = set()
    for gt in expected:
        best = -1
        best_score = 0.0
        for i, sub in enumerate(submitted):
            if i in used:
                continue
            s = _item_similarity(sub, gt)
            if s > best_score:
                best_score = s
                best = i
        if best >= 0 and best_score > 0.3:
            matched += best_score
            used.add(best)

    return matched / len(expected)


def _item_similarity(sub: Dict, gt: Dict) -> float:
    """Score a single line item match (0-1)."""
    s = 0.0
    sd = sub.get("description", "").lower().strip()
    gd = gt["description"].lower().strip()
    if sd == gd:
        s += 0.25
    elif sd in gd or gd in sd:
        s += 0.15

    try:
        if int(sub.get("qty", -1)) == gt["qty"]:
            s += 0.25
    except (ValueError, TypeError):
        pass

    try:
        if abs(float(sub.get("unit_price", -1)) - gt["unit_price"]) < 0.01:
            s += 0.25
    except (ValueError, TypeError):
        pass

    try:
        if abs(float(sub.get("amount", -1)) - gt["amount"]) < 0.01:
            s += 0.25
    except (ValueError, TypeError):
        pass

    return s


# ===================================================================
# TASK: MEDIUM — batch cleaning & normalisation
# ===================================================================

def _make_messy_invoice(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Take a clean invoice dict and introduce messiness."""
    messy = copy.deepcopy(inv)

    d = date.fromisoformat(inv["date"])
    messy["date"] = _format_date_messy(d)

    if random.random() < 0.5:
        messy["vendor"] = _typo_vendor(inv["vendor"])

    sym = CURRENCY_SYMBOLS.get(inv["currency"], "$")
    if random.random() < 0.4:
        messy["currency"] = sym
    if random.random() < 0.3:
        messy["total"] = f"{sym}{inv['total']}"

    for it in messy["line_items"]:
        if random.random() < 0.3:
            it["amount"] = f"{sym}{it['amount']}"
        if random.random() < 0.2:
            it["unit_price"] = f"{sym}{it['unit_price']}"
        if random.random() < 0.15:
            it["amount"] = round(it["qty"] * float(str(it["unit_price"]).replace(sym, "")) + random.uniform(0.5, 5.0), 2)

    return messy


def _render_messy_batch(invoices: List[Dict[str, Any]]) -> str:
    """Render a batch of messy invoices as CSV-ish text."""
    lines = ["=== INVOICE BATCH (requires cleaning) ===", ""]
    for i, inv in enumerate(invoices):
        lines.append(f"--- Invoice {i+1} ---")
        lines.append(f"Vendor: {inv['vendor']}")
        lines.append(f"Date: {inv['date']}")
        lines.append(f"Currency: {inv.get('currency', 'N/A')}")
        lines.append(f"Total: {inv.get('total', 'N/A')}")
        lines.append("Items:")
        for it in inv["line_items"]:
            lines.append(f"  - {it['description']} | qty: {it.get('qty','?')} | price: {it.get('unit_price','?')} | amount: {it.get('amount','?')}")
        lines.append("")
    return "\n".join(lines)


def _grade_medium(submitted: Dict[str, Any], ground_truths: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Grade batch cleaning. submitted should have 'invoices' key."""
    sub_invoices = submitted.get("invoices", [])
    if not isinstance(sub_invoices, list):
        return 0.0, "Expected 'invoices' key with a list of cleaned invoices."

    n_expected = len(ground_truths)
    total_score = 0.0
    feedback_parts = []

    for idx, gt in enumerate(ground_truths):
        if idx < len(sub_invoices):
            s, fb = _grade_easy(sub_invoices[idx], gt)
            total_score += s
            feedback_parts.append(f"Invoice {idx+1}: {s:.2f} ({fb})")
        else:
            feedback_parts.append(f"Invoice {idx+1}: missing")

    if len(sub_invoices) > n_expected:
        feedback_parts.append(f"Extra invoices submitted: {len(sub_invoices) - n_expected}")

    avg = total_score / n_expected if n_expected > 0 else 0.0
    return round(min(avg, 1.0), 4), "; ".join(feedback_parts)


# ===================================================================
# TASK: HARD — extraction + cleaning + reconciliation against POs
# ===================================================================

def _generate_purchase_order(inv: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict]]:
    """Generate a PO that mostly matches the invoice but may differ."""
    po = copy.deepcopy(inv)
    po["po_id"] = f"PO-{random.randint(10000, 99999)}"

    discrepancies = []

    if random.random() < 0.6 and po["line_items"]:
        idx = random.randint(0, len(po["line_items"]) - 1)
        original_price = po["line_items"][idx]["unit_price"]
        overcharge = round(original_price * random.uniform(1.05, 1.25), 2)
        discrepancies.append({
            "type": "overcharge",
            "item_description": po["line_items"][idx]["description"],
            "po_price": original_price,
            "invoice_price": overcharge,
        })
        inv["line_items"][idx]["unit_price"] = overcharge
        inv["line_items"][idx]["amount"] = round(inv["line_items"][idx]["qty"] * overcharge, 2)

    if random.random() < 0.4:
        extra = _generate_line_items(1)[0]
        inv["line_items"].append(extra)
        discrepancies.append({
            "type": "extra_item",
            "item_description": extra["description"],
            "detail": "Item on invoice but not on purchase order",
        })

    if random.random() < 0.3 and len(po["line_items"]) > 2:
        removed = po["line_items"].pop(random.randint(0, len(po["line_items"]) - 1))
        discrepancies.append({
            "type": "missing_item",
            "item_description": removed["description"],
            "detail": "Item on purchase order but not on invoice",
        })

    inv["total"] = round(sum(it["amount"] for it in inv["line_items"]), 2)
    po["total"] = round(sum(it["amount"] for it in po["line_items"]), 2)

    return po, discrepancies


def _render_po(po: Dict[str, Any]) -> str:
    """Render purchase order text."""
    lines = [
        f"PURCHASE ORDER: {po['po_id']}",
        f"Vendor: {po['vendor']}",
        f"Date: {po['date']}",
        f"Currency: {po['currency']}",
        f"",
        "Ordered Items:",
    ]
    sym = CURRENCY_SYMBOLS.get(po["currency"], "$")
    for it in po["line_items"]:
        lines.append(f"  - {it['description']} x{it['qty']} @ {sym}{it['unit_price']:.2f} = {sym}{it['amount']:.2f}")
    lines.append(f"PO Total: {sym}{po['total']:.2f}")
    return "\n".join(lines)


def _grade_hard(submitted: Dict[str, Any], ground_truths: List[Dict[str, Any]],
                expected_discrepancies: List[List[Dict]]) -> Tuple[float, str]:
    """Grade extraction + cleaning + reconciliation."""
    extraction_score, extraction_fb = _grade_medium(submitted, ground_truths)

    sub_discrepancies = submitted.get("discrepancies", [])
    if not isinstance(sub_discrepancies, list):
        disc_score = 0.0
        disc_fb = "No discrepancies list submitted"
    else:
        all_expected = []
        for disc_list in expected_discrepancies:
            all_expected.extend(disc_list)

        if not all_expected:
            disc_score = 1.0 if not sub_discrepancies else 0.5
            disc_fb = "No discrepancies expected"
        else:
            matched = 0
            for exp in all_expected:
                for sub in sub_discrepancies:
                    if _discrepancy_match(sub, exp):
                        matched += 1
                        break
            precision = matched / len(sub_discrepancies) if sub_discrepancies else 0.0
            recall = matched / len(all_expected) if all_expected else 1.0
            disc_score = (precision + recall) / 2
            disc_fb = f"Discrepancies: {matched}/{len(all_expected)} found, precision={precision:.2f}, recall={recall:.2f}"

    total = extraction_score * 0.60 + disc_score * 0.40
    feedback = f"Extraction: {extraction_score:.2f}; {disc_fb}"
    return round(min(total, 1.0), 4), feedback


def _discrepancy_match(submitted: Dict, expected: Dict) -> bool:
    sub_type = submitted.get("type", "").lower().strip()
    exp_type = expected.get("type", "").lower().strip()
    if sub_type != exp_type:
        return False
    sub_desc = submitted.get("item_description", "").lower().strip()
    exp_desc = expected.get("item_description", "").lower().strip()
    if sub_desc and exp_desc:
        if sub_desc == exp_desc or sub_desc in exp_desc or exp_desc in sub_desc:
            return True
    return False


# ===================================================================
# TASK: EXPERT — invoice fraud audit
# ===================================================================

def _generate_expert_batch() -> Tuple[List[Dict], List[Dict], List[Dict], str]:
    n_invoices = random.randint(4, 6)
    n_fraudulent = random.randint(2, 3)

    all_indices = list(range(n_invoices))
    random.shuffle(all_indices)
    fraud_indices = set(all_indices[:n_fraudulent])

    fraud_types = random.sample(
        ["phantom_vendor", "price_gouging", "duplicate_submission", "math_fraud"],
        min(n_fraudulent, 4)
    )
    fraud_type_map = {idx: fraud_types[i] for i, idx in enumerate(list(fraud_indices))}

    invoices: List[Dict] = []
    ground_truth: List[Dict] = []
    invoice_history: List[Dict] = []

    for _ in range(3):
        h = _generate_invoice()
        invoice_history.append(h)

    for i in range(n_invoices):
        inv = _generate_invoice()

        if i in fraud_indices:
            ftype = fraud_type_map[i]

            if ftype == "phantom_vendor":
                inv["vendor"] = random.choice(PHANTOM_VENDORS)

            elif ftype == "price_gouging":
                item = random.choice(inv["line_items"])
                market_max = MARKET_PRICE_MAX.get(item["description"], item["unit_price"])
                item["unit_price"] = round(market_max * random.uniform(1.6, 2.2), 2)
                item["amount"] = round(item["qty"] * item["unit_price"], 2)
                inv["total"] = round(sum(it["amount"] for it in inv["line_items"]), 2)

            elif ftype == "duplicate_submission":
                original = random.choice(invoice_history)
                inv = copy.deepcopy(original)

            elif ftype == "math_fraud":
                real_total = round(sum(it["amount"] for it in inv["line_items"]), 2)
                inv["total"] = round(real_total * random.uniform(1.08, 1.18), 2)

            ground_truth.append({
                "invoice_id": inv["invoice_id"],
                "verdict": "flagged",
                "fraud_type": ftype,
            })
        else:
            invoice_history.append(inv)
            ground_truth.append({
                "invoice_id": inv["invoice_id"],
                "verdict": "approved",
                "fraud_type": None,
            })

        invoices.append(inv)

    reference_text = _render_expert_reference(invoice_history)
    return invoices, ground_truth, invoice_history, reference_text


def _render_expert_batch(invoices: List[Dict]) -> str:
    lines = ["=== INVOICE AUDIT BATCH ===", ""]
    for i, inv in enumerate(invoices):
        sym = CURRENCY_SYMBOLS.get(inv["currency"], "$")
        lines.append(f"--- Invoice {i+1} (ID: {inv['invoice_id']}) ---")
        lines.append(f"Vendor: {inv['vendor']}")
        lines.append(f"Date: {inv['date']}")
        lines.append(f"Currency: {inv['currency']}")
        lines.append(f"Total: {sym}{inv['total']:.2f}")
        lines.append("Line Items:")
        for it in inv["line_items"]:
            lines.append(
                f"  - {it['description']} | qty: {it['qty']} | "
                f"unit_price: {sym}{it['unit_price']:.2f} | amount: {sym}{it['amount']:.2f}"
            )
        lines.append("")
    return "\n".join(lines)


def _render_expert_reference(invoice_history: List[Dict]) -> str:
    lines = ["=== REFERENCE DATA ===", ""]

    lines.append("-- Approved Vendor Registry --")
    for v in sorted(VENDORS):
        lines.append(f"  {v}")
    lines.append("")

    lines.append("-- Market Price Catalog (maximum unit prices) --")
    for item_name, max_price in sorted(MARKET_PRICE_MAX.items()):
        lines.append(f"  {item_name}: max ${max_price:.2f}")
    lines.append("")

    lines.append("-- Recent Invoice History (previously approved invoices) --")
    for h in invoice_history:
        sym = CURRENCY_SYMBOLS.get(h["currency"], "$")
        lines.append(
            f"  {h['invoice_id']} | {h['vendor']} | {h['date']} | "
            f"{h['currency']} | Total: {sym}{h['total']:.2f}"
        )
    lines.append("")

    return "\n".join(lines)


def _grade_expert(submitted: Dict[str, Any], ground_truth: List[Dict]) -> Tuple[float, str]:
    audit_results = submitted.get("audit_results", [])
    if not isinstance(audit_results, list) or not audit_results:
        return 0.0, "Expected 'audit_results' key with a list of audit objects."

    sub_map = {}
    for r in audit_results:
        if isinstance(r, dict) and "invoice_id" in r:
            sub_map[r["invoice_id"]] = r

    n = len(ground_truth)
    total_score = 0.0
    feedback_parts = []

    for gt in ground_truth:
        inv_id = gt["invoice_id"]
        sub = sub_map.get(inv_id)

        if sub is None:
            feedback_parts.append(f"{inv_id}: missing from submission")
            continue

        sub_verdict = sub.get("verdict", "").lower().strip()
        gt_verdict = gt["verdict"]

        if gt_verdict == "approved":
            if sub_verdict == "approved":
                total_score += 1.0
                feedback_parts.append(f"{inv_id}: correct (legitimate invoice approved)")
            else:
                feedback_parts.append(f"{inv_id}: false positive (legitimate invoice wrongly flagged as {sub.get('fraud_type', '?')})")
        else:
            if sub_verdict != "flagged":
                feedback_parts.append(f"{inv_id}: missed fraud (expected {gt['fraud_type']}, approved instead)")
            else:
                sub_ftype = sub.get("fraud_type", "").lower().strip()
                if sub_ftype == gt["fraud_type"]:
                    total_score += 1.0
                    feedback_parts.append(f"{inv_id}: correct fraud detected ({gt['fraud_type']})")
                else:
                    total_score += 0.5
                    feedback_parts.append(
                        f"{inv_id}: flagged but wrong type "
                        f"(expected '{gt['fraud_type']}', got '{sub_ftype}')"
                    )

    score = total_score / n if n > 0 else 0.0
    return round(min(score, 1.0), 4), "; ".join(feedback_parts)


# ===================================================================
# TASK: ADVERSARIAL — OCR corruption + misleading SUBTOTAL + FX noise
# ===================================================================

def _generate_adversarial_invoice(ocr_intensity: float = 0.35) -> Tuple[Dict, str]:
    """
    Generate a single invoice with three adversarial layers:
      1. OCR character corruption in label/vendor strings (not in numbers).
      2. A fake SUBTOTAL line (60-80% of true total) + fabricated TAX and
         ADJUSTMENT lines. The TOTAL line remains correct.
      3. A foreign-currency FX equivalent reference line (distractor only).
    Returns (clean_ground_truth, adversarial_rendered_text).
    """
    inv = _generate_invoice()
    true_total = inv["total"]
    currency = inv["currency"]
    sym = CURRENCY_SYMBOLS[currency]

    # Trap 2: misleading SUBTOTAL block
    fake_subtotal = round(true_total * random.uniform(0.60, 0.80), 2)
    fake_tax = round(true_total * random.uniform(0.05, 0.12), 2)
    fake_adjustment = round(true_total - fake_subtotal - fake_tax, 2)

    # Trap 3: FX noise
    other_currencies = [c for c in CURRENCIES if c != currency]
    fx_currency = random.choice(other_currencies)
    fx_rate = FX_RATES[fx_currency] / FX_RATES[currency]
    fx_equivalent = round(true_total * fx_rate, 2)
    fx_sym = CURRENCY_SYMBOLS[fx_currency]

    # Build text — OCR corruption on labels/strings only, NOT on number format strings
    lines = [
        _corrupt_ocr("INVOICE", ocr_intensity),
        _corrupt_ocr("-------", ocr_intensity),
        f"Invoice #: {inv['invoice_id']}",
        f"Vendor: {_corrupt_ocr(inv['vendor'], ocr_intensity * 0.5)}",
        f"Date: {inv['date']}",
        f"Currency: {inv['currency']}",
        "",
        _corrupt_ocr("Items:", ocr_intensity),
        f"{'Description':<30} {'Qty':>5} {'Unit Price':>12} {'Amount':>12}",
        f"{'-'*30} {'-'*5} {'-'*12} {'-'*12}",
    ]
    for it in inv["line_items"]:
        lines.append(
            f"{it['description']:<30} {it['qty']:>5} {sym}{it['unit_price']:>10.2f} {sym}{it['amount']:>10.2f}"
        )
    lines.append("")
    lines.append(f"{'':>30} {'':>5} {'SUBTOTAL':>12} {sym}{fake_subtotal:>10.2f}   <- SUBTOTAL")
    lines.append(f"{'':>30} {'':>5} {'TAX':>12} {sym}{fake_tax:>10.2f}")
    lines.append(f"{'':>30} {'':>5} {'ADJUSTMENT':>12} {sym}{fake_adjustment:>10.2f}")
    lines.append(f"{'':>30} {'':>5} {'TOTAL':>12} {sym}{true_total:>10.2f}")
    lines.append("")
    lines.append(
        f"[FX Reference] Approximate equivalent: {fx_sym}{fx_equivalent:.2f} {fx_currency} "
        f"(rate {fx_rate:.4f})"
    )

    return inv, "\n".join(lines)


def _grade_adversarial(
    submitted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Tuple[float, str, Dict[str, Any]]:
    """Grade adversarial extraction — same schema as easy."""
    return _grade_easy_with_breakdown(submitted, ground_truth)


# ===================================================================
# TASK: NEGOTIATE — clarification-then-extract with bonus
# ===================================================================

def _answer_clarification(question: str, inv: Dict[str, Any]) -> str:
    """Return a scripted answer to an agent clarification question."""
    q = question.lower()
    if "vendor" in q or "supplier" in q or "company" in q:
        return f"The vendor name is '{inv['vendor']}'."
    if "date" in q or "when" in q or "issued" in q:
        return f"The invoice date is {inv['date']} (ISO 8601: YYYY-MM-DD)."
    if "currency" in q or "denomination" in q:
        return f"The currency is {inv['currency']}."
    if "total" in q or "amount due" in q or "sum" in q:
        return (
            "The TOTAL line on the invoice is the authoritative figure. "
            "Ignore SUBTOTAL and any reference equivalent lines."
        )
    if "line item" in q or "items" in q or "description" in q:
        descs = ", ".join(it["description"] for it in inv["line_items"])
        return f"The invoice contains the following line items: {descs}."
    return (
        "I can confirm: focus on the TOTAL line, not SUBTOTAL. "
        "All line item amounts are qty × unit_price. "
        "FX reference lines are for display only."
    )


def _grade_negotiate(
    submitted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    clarification_count: int,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Grade a negotiate submission. Same schema as easy.
    Bonus: +15% for ≤2 clarifications, +10% for ≤4, when base_score >= 0.70.
    """
    base_score, feedback, breakdown = _grade_easy_with_breakdown(submitted, ground_truth)

    bonus = 0.0
    if base_score >= 0.70:
        if clarification_count <= 2:
            bonus = round(base_score * 0.15, 4)
        elif clarification_count <= 4:
            bonus = round(base_score * 0.10, 4)

    final = round(min(base_score + bonus, 1.0), 4)
    bonus_note = (
        f"; Bonus: +{bonus:.4f} ({clarification_count} clarification(s) used)"
        if bonus > 0 else ""
    )
    return final, feedback + bonus_note, breakdown


# ===================================================================
# TASK: SUPPLY CHAIN — delivery anomaly detection
# ===================================================================

def _generate_delivery_records(
    n_deliveries: int = None,
    n_anomalies: int = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate n_deliveries delivery records with n_anomalies injected.
    Returns (records, expected_anomalies).
    """
    if n_deliveries is None:
        n_deliveries = random.randint(4, 7)
    if n_anomalies is None:
        n_anomalies = random.randint(2, 3)
    n_anomalies = min(n_anomalies, n_deliveries)

    anomaly_slots = random.sample(range(n_deliveries), n_anomalies)
    anomaly_type_pool = random.sample(SC_ANOMALY_TYPES, min(n_anomalies, len(SC_ANOMALY_TYPES)))
    anomaly_type_map = {
        slot: anomaly_type_pool[i % len(anomaly_type_pool)]
        for i, slot in enumerate(anomaly_slots)
    }

    records: List[Dict] = []
    expected_anomalies: List[Dict] = []

    # Valid PO IDs (phantom_delivery uses a non-existent one)
    valid_po_ids = [f"PO-{random.randint(10000, 99999)}" for _ in range(n_deliveries - 1)]

    for i in range(n_deliveries):
        item_name, lo, hi = random.choice(SC_ITEMS)
        supplier = random.choice(SC_SUPPLIERS)
        qty_ordered = random.randint(5, 50)
        price_ordered = round(random.uniform(lo, hi), 2)

        qty_delivered = qty_ordered
        price_delivered = price_ordered
        item_delivered = item_name
        po_id = random.choice(valid_po_ids) if valid_po_ids else f"PO-{random.randint(10000, 99999)}"

        anomaly_type = anomaly_type_map.get(i)
        detail = ""

        if anomaly_type == "quantity_shortfall":
            qty_delivered = max(1, int(qty_ordered * random.uniform(0.40, 0.84)))
            detail = (
                f"Delivered {qty_delivered} of {qty_ordered} ordered "
                f"({qty_delivered/qty_ordered:.0%} fulfillment)."
            )
        elif anomaly_type == "price_spike":
            price_delivered = round(price_ordered * random.uniform(1.26, 1.75), 2)
            detail = (
                f"Delivered price {price_delivered:.2f} exceeds ordered price "
                f"{price_ordered:.2f} by {(price_delivered/price_ordered - 1):.0%}."
            )
        elif anomaly_type == "unauthorized_substitution":
            other_items = [it for it in SC_ITEMS if it[0] != item_name]
            alt_name, alt_lo, alt_hi = random.choice(other_items)
            item_delivered = alt_name
            price_delivered = round(random.uniform(alt_lo, alt_hi), 2)
            detail = (
                f"Ordered '{item_name}' but received '{item_delivered}' — "
                f"unauthorized substitution."
            )
        elif anomaly_type == "phantom_delivery":
            po_id = f"PO-PHANTOM-{random.randint(1000, 9999)}"
            detail = (
                f"Delivery references PO {po_id} which does not exist "
                f"in the approved purchase order registry."
            )

        delivery_id = f"DLV-{random.randint(10000, 99999)}"
        record = {
            "delivery_id": delivery_id,
            "po_id": po_id,
            "supplier": supplier,
            "item_ordered": item_name,
            "item_delivered": item_delivered,
            "qty_ordered": qty_ordered,
            "qty_delivered": qty_delivered,
            "price_ordered": price_ordered,
            "price_delivered": price_delivered,
        }
        records.append(record)

        if anomaly_type:
            expected_anomalies.append({
                "delivery_id": delivery_id,
                "anomaly_type": anomaly_type,
                "detail": detail,
            })

    return records, expected_anomalies


def _render_delivery_records(records: List[Dict]) -> str:
    lines = ["=== SUPPLY CHAIN DELIVERY RECORDS ===", ""]
    header = (
        f"{'Delivery ID':<16} {'PO ID':<18} {'Supplier':<22} "
        f"{'Item Ordered':<28} {'Item Delivered':<28} "
        f"{'Qty Ord':>8} {'Qty Del':>8} {'Price Ord':>10} {'Price Del':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in records:
        lines.append(
            f"{r['delivery_id']:<16} {r['po_id']:<18} {r['supplier']:<22} "
            f"{r['item_ordered']:<28} {r['item_delivered']:<28} "
            f"{r['qty_ordered']:>8} {r['qty_delivered']:>8} "
            f"{r['price_ordered']:>10.2f} {r['price_delivered']:>10.2f}"
        )
    lines.append("")
    return "\n".join(lines)


def _grade_supply_chain(
    submitted: Dict[str, Any], expected_anomalies: List[Dict]
) -> Tuple[float, str]:
    """
    Grade supply chain anomaly detection.
    submitted: {"anomalies": [{"delivery_id": str, "anomaly_type": str, "detail": str}]}
    Scoring: F1-like (precision + recall) / 2.
    Partial credit: correct delivery_id only = 0.5 of a match.
    """
    sub_list = submitted.get("anomalies", [])
    if not isinstance(sub_list, list):
        return 0.0, "Expected 'anomalies' key with a list."

    if not expected_anomalies:
        if not sub_list:
            return 1.0, "No anomalies expected; none submitted."
        return 0.5, f"No anomalies expected but {len(sub_list)} submitted (false positives)."

    matched_full = 0
    matched_partial = 0
    used_sub: set = set()

    for exp in expected_anomalies:
        best_match = 0.0
        best_i = -1
        for i, sub in enumerate(sub_list):
            if i in used_sub:
                continue
            id_match = sub.get("delivery_id", "").strip() == exp["delivery_id"]
            type_match = sub.get("anomaly_type", "").strip() == exp["anomaly_type"]
            if id_match and type_match:
                best_match = 1.0
                best_i = i
                break
            elif id_match and best_match < 0.5:
                best_match = 0.5
                best_i = i
        if best_i >= 0:
            used_sub.add(best_i)
            if best_match == 1.0:
                matched_full += 1
            else:
                matched_partial += 1

    effective_matched = matched_full + matched_partial * 0.5
    precision = effective_matched / len(sub_list) if sub_list else 0.0
    recall = effective_matched / len(expected_anomalies)
    score = (precision + recall) / 2.0

    feedback = (
        f"Anomalies: {matched_full} exact + {matched_partial} partial of "
        f"{len(expected_anomalies)} expected; precision={precision:.2f}, recall={recall:.2f}"
    )
    return round(min(score, 1.0), 4), feedback


# ===================================================================
# Environment
# ===================================================================

class InvoiceEnvironment:
    """Core invoice processing environment — 7 tasks with dynamic difficulty."""

    TASKS = {
        "easy": {
            "description": (
                "Extract structured data from a single invoice. "
                "Return a JSON object with keys: vendor, date (YYYY-MM-DD), "
                "currency (3-letter code), total (number), "
                "line_items (list of {description, qty, unit_price, amount})."
            ),
            "max_attempts": 5,
        },
        "medium": {
            "description": (
                "Clean and normalise a batch of messy invoices. "
                "Fix date formats to YYYY-MM-DD, correct vendor name typos, "
                "standardise currency to 3-letter codes, ensure amounts are numbers, "
                "and verify line item math (qty * unit_price = amount). "
                "Return {invoices: [cleaned invoice objects]}."
            ),
            "max_attempts": 5,
        },
        "hard": {
            "description": (
                "Extract and clean invoice data, then reconcile against purchase orders. "
                "Identify discrepancies: overcharges (invoice price > PO price), "
                "extra items (on invoice but not PO), missing items (on PO but not invoice). "
                "Return {invoices: [cleaned], discrepancies: [{invoice_idx, type, item_description, detail}]}."
            ),
            "max_attempts": 5,
        },
        "expert": {
            "description": (
                "Fraud audit: review a batch of invoices and identify which are fraudulent. "
                "Use the reference data (approved vendor registry, market price catalog, invoice history) "
                "to detect: phantom_vendor (vendor not in approved registry), "
                "price_gouging (unit price > 150% of market max), "
                "duplicate_submission (invoice already in history with same ID or vendor+date+total), "
                "math_fraud (invoice total exceeds sum of line items). "
                "Return {audit_results: [{invoice_id, verdict ('approved'|'flagged'), "
                "fraud_type (null or one of the above), evidence (brief explanation)}]}."
            ),
            "max_attempts": 5,
        },
        "adversarial": {
            "description": (
                "Extract structured data from a deliberately adversarial invoice. "
                "The text contains OCR character corruption (e.g. 0→O, 1→l, 5→S), "
                "a misleading SUBTOTAL line (lower than the true total), "
                "fake TAX/ADJUSTMENT lines, and a foreign-currency FX reference. "
                "The TOTAL line is always correct. Verify: sum of line items = total. "
                "Return the same JSON as the easy task: "
                "{vendor, date, currency, total, line_items}."
            ),
            "max_attempts": 5,
        },
        "negotiate": {
            "description": (
                "The invoice has deliberately ambiguous fields. "
                "You may ask clarification questions by submitting {'question': 'your question'}. "
                "When ready, submit the full extraction: "
                "{vendor, date, currency, total, line_items}. "
                "You earn a 10-15% bonus for solving correctly with ≤2 questions."
            ),
            "max_attempts": 7,
        },
        "supply_chain": {
            "description": (
                "Analyse supply chain delivery records. Identify anomalies: "
                "'quantity_shortfall' (< 85% of ordered qty delivered), "
                "'price_spike' (delivered price > 125% ordered price), "
                "'unauthorized_substitution' (different item delivered than ordered), "
                "'phantom_delivery' (no matching PO). "
                "Return {'anomalies': [{'delivery_id': str, 'anomaly_type': str, 'detail': str}]}."
            ),
            "max_attempts": 5,
        },
    }

    def __init__(self):
        self._state = InvoiceState()
        self._ground_truth: Any = None
        self._raw_text: str = ""
        self._reference_data: str = ""
        self._messy_invoices: List[Dict] = []
        self._expected_discrepancies: List[List[Dict]] = []
        self._expert_ground_truth: List[Dict] = []
        self._supply_chain_records: List[Dict] = []
        self._expected_sc_anomalies: List[Dict] = []
        self._ocr_intensity: float = 0.35

    def reset(self, task_id: str = "easy") -> Tuple[InvoiceObservation, float, bool, Dict]:
        """Reset the environment for a new episode."""
        if task_id not in self.TASKS:
            task_id = "easy"

        self._state = InvoiceState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            done=False,
            last_reward=0.0,
            best_reward=0.0,
            rewards=[],
            conversation_history=[],
            clarification_count=0,
        )

        self._reference_data = ""
        self._expected_discrepancies = []

        if task_id == "easy":
            inv = _generate_invoice()
            self._ground_truth = inv
            self._raw_text = _render_clean_invoice(inv)

        elif task_id == "medium":
            params = _get_dynamic_params("medium")
            n = random.randint(*params["n_invoices"])
            clean_invoices = [_generate_invoice() for _ in range(n)]
            self._ground_truth = clean_invoices
            messy = [_make_messy_invoice(copy.deepcopy(inv)) for inv in clean_invoices]
            self._messy_invoices = messy
            self._raw_text = _render_messy_batch(messy)

        elif task_id == "hard":
            params = _get_dynamic_params("hard")
            n = random.randint(*params["n_invoices"])
            clean_invoices = [_generate_invoice() for _ in range(n)]
            self._expected_discrepancies = []
            po_texts = []

            for inv in clean_invoices:
                po, discs = _generate_purchase_order(inv)
                self._expected_discrepancies.append(discs)
                po_texts.append(_render_po(po))

            self._ground_truth = clean_invoices
            messy = [_make_messy_invoice(copy.deepcopy(inv)) for inv in clean_invoices]
            self._raw_text = _render_messy_batch(messy)
            self._reference_data = "\n\n".join(po_texts)

        elif task_id == "expert":
            invoices, gt, _history, reference_text = _generate_expert_batch()
            self._ground_truth = invoices
            self._expert_ground_truth = gt
            self._raw_text = _render_expert_batch(invoices)
            self._reference_data = reference_text

        elif task_id == "adversarial":
            params = _get_dynamic_params("adversarial")
            self._ocr_intensity = params["ocr_intensity"]
            inv, adv_text = _generate_adversarial_invoice(ocr_intensity=self._ocr_intensity)
            self._ground_truth = inv
            self._raw_text = adv_text

        elif task_id == "negotiate":
            inv = _generate_invoice()
            messy_inv = _make_messy_invoice(copy.deepcopy(inv))
            self._ground_truth = inv
            self._raw_text = _render_messy_batch([messy_inv])

        elif task_id == "supply_chain":
            params = _get_dynamic_params("supply_chain")
            n_del = random.randint(*params["n_deliveries"])
            n_ano = params["n_anomalies"]
            records, expected_anomalies = _generate_delivery_records(n_del, n_ano)
            self._supply_chain_records = records
            self._expected_sc_anomalies = expected_anomalies
            self._raw_text = _render_delivery_records(records)

        task_info = self.TASKS[task_id]
        obs = InvoiceObservation(
            raw_text=self._raw_text,
            task_id=task_id,
            difficulty=task_id,
            task_description=task_info["description"],
            attempt_number=0,
            max_attempts=task_info["max_attempts"],
            feedback="",
            hint="",
            reference_data=self._reference_data,
            reward_breakdown=None,
            conversation_history=[],
        )
        return obs, 0.0, False, {"episode_id": self._state.episode_id}

    def step(self, action: InvoiceAction) -> Tuple[InvoiceObservation, float, bool, Dict]:
        """Process one agent action."""
        self._state.step_count += 1
        task_id = self._state.task_id
        task_info = self.TASKS[task_id]
        attempt = self._state.step_count

        score = 0.0
        feedback = ""
        reward_breakdown = None

        # --- negotiate: check if agent is asking a clarification question ---
        if task_id == "negotiate" and "question" in action.extracted_data and len(action.extracted_data) == 1:
            question = str(action.extracted_data["question"])
            answer = _answer_clarification(question, self._ground_truth)
            self._state.conversation_history.append({"role": "agent", "content": question})
            self._state.conversation_history.append({"role": "env", "content": answer})
            self._state.clarification_count += 1

            obs = InvoiceObservation(
                raw_text=self._raw_text,
                task_id=task_id,
                difficulty=task_id,
                task_description=task_info["description"],
                attempt_number=attempt,
                max_attempts=task_info["max_attempts"],
                feedback=f"[ENV] {answer}",
                hint="",
                reference_data=self._reference_data,
                reward_breakdown=None,
                conversation_history=list(self._state.conversation_history),
            )
            self._state.last_reward = 0.0
            self._state.rewards.append(0.0)
            return obs, 0.0, False, {
                "episode_id": self._state.episode_id,
                "best_reward": self._state.best_reward,
                "clarification_answered": True,
            }

        # --- Normal grading path ---
        if task_id == "easy":
            score, feedback, reward_breakdown = _grade_easy_with_breakdown(
                action.extracted_data, self._ground_truth
            )
        elif task_id == "medium":
            score, feedback = _grade_medium(action.extracted_data, self._ground_truth)
        elif task_id == "hard":
            score, feedback = _grade_hard(
                action.extracted_data, self._ground_truth, self._expected_discrepancies
            )
        elif task_id == "expert":
            score, feedback = _grade_expert(action.extracted_data, self._expert_ground_truth)
        elif task_id == "adversarial":
            score, feedback, reward_breakdown = _grade_adversarial(
                action.extracted_data, self._ground_truth
            )
        elif task_id == "negotiate":
            score, feedback, reward_breakdown = _grade_negotiate(
                action.extracted_data,
                self._ground_truth,
                self._state.clarification_count,
            )
        elif task_id == "supply_chain":
            score, feedback = _grade_supply_chain(
                action.extracted_data, self._expected_sc_anomalies
            )

        # Record for dynamic difficulty
        _record_score(task_id, score)

        self._state.best_reward = max(self._state.best_reward, score)
        self._state.last_reward = score
        self._state.rewards.append(score)

        done = score >= 0.95 or attempt >= task_info["max_attempts"]
        self._state.done = done

        reward = score
        if done and attempt >= task_info["max_attempts"] and score < 0.95:
            reward = score * 0.85

        # Hint after 2 failed attempts
        hint = ""
        if attempt >= 2 and score < 0.7:
            hints = {
                "easy": "Make sure dates are YYYY-MM-DD, amounts are numbers, and all line items are included.",
                "medium": "Check for vendor name typos, mixed date formats, and currency symbols mixed into amounts.",
                "hard": "Compare each invoice line item against the PO. Look for price differences and items present in one but not the other.",
                "expert": (
                    "Check each invoice against all 4 fraud types: "
                    "(1) Is the vendor in the Approved Vendor Registry? "
                    "(2) Do any unit prices exceed 150% of the market max? "
                    "(3) Does this invoice_id or vendor+date+total appear in the invoice history? "
                    "(4) Does the invoice total equal the sum of line item amounts?"
                ),
                "adversarial": (
                    "Ignore the SUBTOTAL, TAX, ADJUSTMENT lines — they are traps. "
                    "The TOTAL line is correct. Sum the line items yourself to verify. "
                    "OCR corruption only affects labels, not numeric values."
                ),
                "negotiate": (
                    "Submit {'question': 'your question'} to ask for clarification. "
                    "You earn a bonus for solving with ≤2 questions. "
                    "When ready, submit the full extraction."
                ),
                "supply_chain": (
                    "Check for: qty_delivered < 85% of qty_ordered (quantity_shortfall); "
                    "price_delivered > 125% of price_ordered (price_spike); "
                    "item_delivered != item_ordered (unauthorized_substitution); "
                    "PO ID starting with PO-PHANTOM- (phantom_delivery)."
                ),
            }
            hint = hints.get(task_id, "")

        obs = InvoiceObservation(
            raw_text=self._raw_text,
            task_id=task_id,
            difficulty=task_id,
            task_description=task_info["description"],
            attempt_number=attempt,
            max_attempts=task_info["max_attempts"],
            feedback=feedback,
            hint=hint,
            reference_data=self._reference_data,
            reward_breakdown=reward_breakdown,
            conversation_history=list(self._state.conversation_history),
        )

        return obs, round(reward, 4), done, {
            "episode_id": self._state.episode_id,
            "best_reward": self._state.best_reward,
        }

    @property
    def state(self) -> InvoiceState:
        return self._state
