"""
Invoice Processing Pipeline — Core Environment

Four tasks:
  easy   — Extract structured fields from a single, relatively clean invoice.
  medium — Clean & normalise a batch of messy invoices (date formats, vendor
           name typos, currency symbols, duplicate detection).
  hard   — Extract, clean, AND reconcile against purchase orders; flag
           mismatches, overcharges, and missing items.
  expert — Fraud audit: identify fraudulent invoices in a batch and classify
           the fraud type (phantom_vendor, price_gouging, duplicate_submission,
           math_fraud) using an approved vendor registry, market price catalog,
           and invoice history.

Each episode generates fresh synthetic data so the agent cannot memorize.
"""

from __future__ import annotations

import copy
import json
import random
import re
import string
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
# Used to detect price_gouging (invoice unit_price > PRICE_GOUGE_FACTOR * market_max)
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


def _grade_easy(submitted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """Grade single-invoice extraction. Returns (score, feedback)."""
    score = 0.0
    feedback_parts = []

    # Vendor (0.15)
    sub_vendor = submitted.get("vendor", "").strip()
    if sub_vendor.lower() == ground_truth["vendor"].lower():
        score += 0.15
        feedback_parts.append("Vendor: correct")
    else:
        feedback_parts.append(f"Vendor: wrong (expected '{ground_truth['vendor']}', got '{sub_vendor}')")

    # Date (0.10)
    sub_date = submitted.get("date", "").strip()
    if sub_date == ground_truth["date"]:
        score += 0.10
        feedback_parts.append("Date: correct")
    else:
        feedback_parts.append(f"Date: wrong (expected '{ground_truth['date']}', got '{sub_date}')")

    # Currency (0.05)
    sub_cur = submitted.get("currency", "").strip().upper()
    if sub_cur == ground_truth["currency"]:
        score += 0.05
        feedback_parts.append("Currency: correct")
    else:
        feedback_parts.append(f"Currency: wrong (expected '{ground_truth['currency']}', got '{sub_cur}')")

    # Total (0.20)
    try:
        sub_total = float(submitted.get("total", 0))
        if abs(sub_total - ground_truth["total"]) < 0.01:
            score += 0.20
            feedback_parts.append("Total: correct")
        else:
            feedback_parts.append(f"Total: wrong (expected {ground_truth['total']}, got {sub_total})")
    except (ValueError, TypeError):
        feedback_parts.append("Total: could not parse")

    # Line items (0.50)
    sub_items = submitted.get("line_items", [])
    gt_items = ground_truth["line_items"]
    if not isinstance(sub_items, list):
        feedback_parts.append("Line items: not a list")
    else:
        item_score = _grade_line_items(sub_items, gt_items)
        score += item_score * 0.50
        feedback_parts.append(f"Line items: {item_score:.0%} match ({len(sub_items)} submitted, {len(gt_items)} expected)")

    return round(min(score, 1.0), 4), "; ".join(feedback_parts)


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
    # description
    sd = sub.get("description", "").lower().strip()
    gd = gt["description"].lower().strip()
    if sd == gd:
        s += 0.25
    elif sd in gd or gd in sd:
        s += 0.15

    # qty
    try:
        if int(sub.get("qty", -1)) == gt["qty"]:
            s += 0.25
    except (ValueError, TypeError):
        pass

    # unit_price
    try:
        if abs(float(sub.get("unit_price", -1)) - gt["unit_price"]) < 0.01:
            s += 0.25
    except (ValueError, TypeError):
        pass

    # amount
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

    # Messy date
    d = date.fromisoformat(inv["date"])
    messy["date"] = _format_date_messy(d)

    # Possibly typo the vendor
    if random.random() < 0.5:
        messy["vendor"] = _typo_vendor(inv["vendor"])

    # Mix currency symbol into amounts (remove currency field sometimes)
    sym = CURRENCY_SYMBOLS.get(inv["currency"], "$")
    if random.random() < 0.4:
        messy["currency"] = sym  # symbol instead of code
    if random.random() < 0.3:
        messy["total"] = f"{sym}{inv['total']}"  # string instead of number

    # Mess up some line item amounts
    for it in messy["line_items"]:
        if random.random() < 0.3:
            it["amount"] = f"{sym}{it['amount']}"
        if random.random() < 0.2:
            it["unit_price"] = f"{sym}{it['unit_price']}"
        if random.random() < 0.15:
            # Wrong amount (qty * unit_price ≠ amount)
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
    if len(sub_invoices) != n_expected:
        # Partial credit still possible
        pass

    total_score = 0.0
    feedback_parts = []

    for idx, gt in enumerate(ground_truths):
        if idx < len(sub_invoices):
            s, fb = _grade_easy(sub_invoices[idx], gt)
            total_score += s
            feedback_parts.append(f"Invoice {idx+1}: {s:.2f} ({fb})")
        else:
            feedback_parts.append(f"Invoice {idx+1}: missing")

    # Penalise extra invoices
    if len(sub_invoices) > n_expected:
        feedback_parts.append(f"Extra invoices submitted: {len(sub_invoices) - n_expected}")

    avg = total_score / n_expected if n_expected > 0 else 0.0
    return round(min(avg, 1.0), 4), "; ".join(feedback_parts)


# ===================================================================
# TASK: HARD — extraction + cleaning + reconciliation against POs
# ===================================================================

def _generate_purchase_order(inv: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a PO that mostly matches the invoice but may differ."""
    po = copy.deepcopy(inv)
    po["po_id"] = f"PO-{random.randint(10000, 99999)}"

    discrepancies = []

    # Possibly change a price (overcharge)
    if random.random() < 0.6 and po["line_items"]:
        idx = random.randint(0, len(po["line_items"]) - 1)
        original_price = po["line_items"][idx]["unit_price"]
        # PO has the CORRECT price; invoice will be higher (overcharge)
        overcharge = round(original_price * random.uniform(1.05, 1.25), 2)
        discrepancies.append({
            "type": "overcharge",
            "item_description": po["line_items"][idx]["description"],
            "po_price": original_price,
            "invoice_price": overcharge,
        })
        # We'll modify the invoice later
        inv["line_items"][idx]["unit_price"] = overcharge
        inv["line_items"][idx]["amount"] = round(inv["line_items"][idx]["qty"] * overcharge, 2)

    # Possibly add an extra item to invoice (not in PO)
    if random.random() < 0.4:
        extra = _generate_line_items(1)[0]
        inv["line_items"].append(extra)
        discrepancies.append({
            "type": "extra_item",
            "item_description": extra["description"],
            "detail": "Item on invoice but not on purchase order",
        })

    # Possibly remove an item from invoice (missing from invoice)
    if random.random() < 0.3 and len(po["line_items"]) > 2:
        removed = po["line_items"].pop(random.randint(0, len(po["line_items"]) - 1))
        discrepancies.append({
            "type": "missing_item",
            "item_description": removed["description"],
            "detail": "Item on purchase order but not on invoice",
        })

    # Recalculate totals
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
    # Extraction/cleaning portion (60%)
    extraction_score, extraction_fb = _grade_medium(submitted, ground_truths)

    # Discrepancy detection portion (40%)
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
            disc_score = (precision + recall) / 2  # F1-like
            disc_fb = f"Discrepancies: {matched}/{len(all_expected)} found, precision={precision:.2f}, recall={recall:.2f}"

    total = extraction_score * 0.60 + disc_score * 0.40
    feedback = f"Extraction: {extraction_score:.2f}; {disc_fb}"
    return round(min(total, 1.0), 4), feedback


def _discrepancy_match(submitted: Dict, expected: Dict) -> bool:
    """Check if a submitted discrepancy matches an expected one."""
    # Type must match
    sub_type = submitted.get("type", "").lower().strip()
    exp_type = expected.get("type", "").lower().strip()
    if sub_type != exp_type:
        return False

    # Item description should roughly match
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
    """
    Generate 4-6 invoices with 2-3 fraudulent ones.
    Returns: (invoices, ground_truth, invoice_history, reference_text)
      - invoices: raw invoice dicts shown to the agent
      - ground_truth: [{invoice_id, verdict, fraud_type}]
      - invoice_history: prior legitimate invoices (for duplicate detection)
      - reference_text: approved vendors + market prices + history (reference_data)
    """
    n_invoices = random.randint(4, 6)
    n_fraudulent = random.randint(2, 3)

    # Shuffle so fraud isn't always at the end
    all_indices = list(range(n_invoices))
    random.shuffle(all_indices)
    fraud_indices = set(all_indices[:n_fraudulent])

    # Assign distinct fraud types (no repeats if possible)
    fraud_types = random.sample(
        ["phantom_vendor", "price_gouging", "duplicate_submission", "math_fraud"],
        min(n_fraudulent, 4)
    )
    fraud_type_map = {idx: fraud_types[i] for i, idx in enumerate(list(fraud_indices))}

    invoices: List[Dict] = []
    ground_truth: List[Dict] = []
    invoice_history: List[Dict] = []  # legitimate invoices submitted previously

    # Pre-generate 3 history invoices so duplicate_submission always has material
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
                # Inflate one item's unit_price to 160-220% of market max
                item = random.choice(inv["line_items"])
                market_max = MARKET_PRICE_MAX.get(item["description"], item["unit_price"])
                item["unit_price"] = round(market_max * random.uniform(1.6, 2.2), 2)
                item["amount"] = round(item["qty"] * item["unit_price"], 2)
                inv["total"] = round(sum(it["amount"] for it in inv["line_items"]), 2)

            elif ftype == "duplicate_submission":
                # Reuse a history invoice (same invoice_id & total)
                original = random.choice(invoice_history)
                inv = copy.deepcopy(original)
                # Keep same invoice_id so the duplicate is detectable

            elif ftype == "math_fraud":
                # Inflate total by 8-18% beyond the real sum
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
    """Render invoices for fraud audit as readable text."""
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
    """Render reference data: approved vendor registry, market prices, invoice history."""
    lines = ["=== REFERENCE DATA ===", ""]

    # Approved Vendor Registry
    lines.append("-- Approved Vendor Registry --")
    for v in sorted(VENDORS):
        lines.append(f"  {v}")
    lines.append("")

    # Market Price Catalog
    lines.append("-- Market Price Catalog (maximum unit prices) --")
    for item_name, max_price in sorted(MARKET_PRICE_MAX.items()):
        lines.append(f"  {item_name}: max ${max_price:.2f}")
    lines.append("")

    # Recent Invoice History
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
    """
    Grade fraud audit results.
    submitted: {"audit_results": [{invoice_id, verdict, fraud_type, evidence}]}
    Scoring per invoice:
      - approved invoice, correctly approved:       1.0
      - approved invoice, wrongly flagged:          0.0  (false positive)
      - fraudulent invoice, correctly flagged+type: 1.0
      - fraudulent invoice, flagged but wrong type: 0.5
      - fraudulent invoice, wrongly approved:       0.0  (missed fraud)
    """
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

        else:  # gt_verdict == "flagged"
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
# Environment
# ===================================================================

class InvoiceEnvironment:
    """Core invoice processing environment."""

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
    }

    def __init__(self):
        self._state = InvoiceState()
        self._ground_truth: Any = None
        self._raw_text: str = ""
        self._reference_data: str = ""
        self._messy_invoices: List[Dict] = []
        self._expected_discrepancies: List[List[Dict]] = []
        self._expert_ground_truth: List[Dict] = []

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
        )

        self._reference_data = ""
        self._expected_discrepancies = []

        if task_id == "easy":
            inv = _generate_invoice()
            self._ground_truth = inv
            self._raw_text = _render_clean_invoice(inv)

        elif task_id == "medium":
            n = random.randint(3, 5)
            clean_invoices = [_generate_invoice() for _ in range(n)]
            self._ground_truth = clean_invoices
            messy = [_make_messy_invoice(copy.deepcopy(inv)) for inv in clean_invoices]
            self._messy_invoices = messy
            self._raw_text = _render_messy_batch(messy)

        elif task_id == "hard":
            n = random.randint(2, 4)
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
        )
        return obs, 0.0, False, {"episode_id": self._state.episode_id}

    def step(self, action: InvoiceAction) -> Tuple[InvoiceObservation, float, bool, Dict]:
        """Process one agent action."""
        self._state.step_count += 1
        task_id = self._state.task_id
        task_info = self.TASKS[task_id]
        attempt = self._state.step_count

        # Grade
        if task_id == "easy":
            score, feedback = _grade_easy(action.extracted_data, self._ground_truth)
        elif task_id == "medium":
            score, feedback = _grade_medium(action.extracted_data, self._ground_truth)
        elif task_id == "hard":
            score, feedback = _grade_hard(
                action.extracted_data, self._ground_truth, self._expected_discrepancies
            )
        else:  # expert
            score, feedback = _grade_expert(action.extracted_data, self._expert_ground_truth)

        # Track best
        self._state.best_reward = max(self._state.best_reward, score)
        self._state.last_reward = score
        self._state.rewards.append(score)

        # Done conditions
        done = score >= 0.95 or attempt >= task_info["max_attempts"]
        self._state.done = done

        # Attempt penalty for using all attempts
        reward = score
        if done and attempt >= task_info["max_attempts"] and score < 0.95:
            reward = score * 0.85  # penalty

        # Hint after 2 failed attempts
        hint = ""
        if attempt >= 2 and score < 0.7:
            if task_id == "easy":
                hint = "Make sure dates are YYYY-MM-DD, amounts are numbers, and all line items are included."
            elif task_id == "medium":
                hint = "Check for vendor name typos, mixed date formats, and currency symbols mixed into amounts."
            elif task_id == "hard":
                hint = "Compare each invoice line item against the PO. Look for price differences and items present in one but not the other."
            else:  # expert
                hint = (
                    "Check each invoice against all 4 fraud types: "
                    "(1) Is the vendor in the Approved Vendor Registry? "
                    "(2) Do any unit prices exceed 150% of the market max? "
                    "(3) Does this invoice_id or vendor+date+total appear in the invoice history? "
                    "(4) Does the invoice total equal the sum of line item amounts?"
                )

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
        )

        return obs, round(reward, 4), done, {
            "episode_id": self._state.episode_id,
            "best_reward": self._state.best_reward,
        }

    @property
    def state(self) -> InvoiceState:
        return self._state