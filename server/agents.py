"""
Lazy-loading inference wrappers for the 3 trained LoRA agents.
Each model is loaded on first call and cached for the session.
Falls back gracefully to None if GPU unavailable or models fail to load.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Optional

# HF Hub model IDs
EXTRACTOR_HUB = "ps2181/extractor-lora-qwen2.5-1.5b"
AUDITOR_HUB   = "ps2181/auditor-lora-qwen2.5-1.5b"
GENERATOR_HUB = "ps2181/generator-lora-qwen2.5-1.5b"
BASE_MODEL    = "unsloth/Qwen2.5-1.5B-Instruct"

EXTRACTOR_SYSTEM = (
    "Extract invoice fields. Return JSON only: "
    "{vendor, date (YYYY-MM-DD), currency (USD/EUR/GBP), total (float), "
    "line_items [{description, qty, unit_price, amount}]}"
)

AUDITOR_SYSTEM = (
    "You are an invoice fraud auditor. Review each invoice and output a JSON array.\n"
    "For each invoice output: "
    "{\"invoice_id\": \"INV-XXXXX\", \"verdict\": \"approved\" or \"flagged\", "
    "\"fraud_type\": null or one of [\"phantom_vendor\",\"price_gouging\","
    "\"math_fraud\",\"duplicate_submission\"], \"confidence\": 0.0-1.0}\n"
    "Output ONLY valid JSON: {\"audit_results\": [...]}"
)

GENERATOR_SYSTEM = (
    "You are an invoice generator. Create a realistic fraudulent invoice as JSON.\n"
    "Output ONLY valid JSON with keys: vendor, date, currency, total, line_items, invoice_id."
)

_lock = threading.Lock()
# Maps name -> (model, tokenizer, device) | None
_cache: dict = {}
_load_errors: dict = {}


def _load(hub_id: str):
    """Load base model + LoRA adapter. Returns (model, tokenizer, device) or None."""
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        has_gpu = torch.cuda.is_available()
        device = "cuda" if has_gpu else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        if has_gpu:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="cpu",
                torch_dtype="auto",
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(base, hub_id)
        model.eval()
        return model, tokenizer, device

    except Exception as exc:
        print(f"[agents] Could not load {hub_id}: {exc}")
        return None


def _get(name: str, hub_id: str):
    with _lock:
        if name not in _cache:
            print(f"[agents] Loading {name}…")
            result = _load(hub_id)
            _cache[name] = result
            if result is None:
                _load_errors[name] = f"load failed for {hub_id}"
        return _cache[name]


def _generate(model, tokenizer, device: str, system: str, user: str, max_new_tokens: int = 400) -> str:
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if device == "cuda":
        inputs = inputs.to("cuda")
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def _parse(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

def run_extractor(raw_text: str, ref_data: str = "") -> tuple[dict, bool]:
    """
    Run trained Extractor LoRA on raw invoice text.
    Returns (extracted_dict, used_model).
    used_model=False means model unavailable, caller should use rule-based fallback.
    """
    result = _get("extractor", EXTRACTOR_HUB)
    if result is None:
        return {}, False
    model, tokenizer, device = result
    user = (f"REF:\n{ref_data[:200]}\n\nINVOICE:\n{raw_text[:600]}"
            if ref_data else raw_text[:600])
    text = _generate(model, tokenizer, device, EXTRACTOR_SYSTEM, user)
    parsed = _parse(text)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    return (parsed if isinstance(parsed, dict) else {}), True


def run_auditor(raw_text: str, ref_data: str, n_invoices: int) -> tuple[list, bool]:
    """
    Run trained Auditor LoRA on invoice batch text.
    Returns (audit_results_list, used_model).
    """
    result = _get("auditor", AUDITOR_HUB)
    if result is None:
        return [], False
    model, tokenizer, device = result
    user = (
        f"INVOICE BATCH:\n{raw_text[:800]}\n\n"
        f"REFERENCE DATA:\n{ref_data[:400]}\n\n"
        'Audit all invoices. Output: {"audit_results": [...]}'
    )
    text = _generate(model, tokenizer, device, AUDITOR_SYSTEM, user)
    parsed = _parse(text)
    if isinstance(parsed, dict):
        results = parsed.get("audit_results", [])
    elif isinstance(parsed, list):
        results = parsed
    else:
        results = []
    return results, True


def run_generator(fraud_type: str, blind_spots: list | None = None) -> tuple[dict, bool]:
    """
    Run trained Generator LoRA to create a fraudulent invoice.
    Returns (invoice_dict, used_model).
    """
    result = _get("generator", GENERATOR_HUB)
    if result is None:
        return {}, False
    model, tokenizer, device = result
    ctx = f"Regulator blind spots: {blind_spots}" if blind_spots else ""
    user = (
        f"Generate a realistic invoice with {fraud_type} fraud. {ctx}\n"
        "Output JSON only: {{vendor, date, currency, total, line_items, invoice_id}}"
    )
    text = _generate(model, tokenizer, device, GENERATOR_SYSTEM, user)
    parsed = _parse(text)
    return (parsed if isinstance(parsed, dict) else {}), True


def models_status() -> dict[str, str]:
    """Return load status for all 3 agents."""
    names = ["extractor", "auditor", "generator"]
    status = {}
    for n in names:
        if n in _cache:
            status[n] = "loaded ✅" if _cache[n] is not None else f"failed ❌ ({_load_errors.get(n,'')})"
        else:
            status[n] = "not loaded yet"
    return status
