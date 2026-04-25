"""
Multi-Agent Environment for Invoice Processing Pipeline
=======================================================

5 agents with distinct reward signals:

  Generator  — creates clean or fraudulent invoices (adversarial self-play).
               Biases fraud type toward Regulator-detected blind spots.

  Extractor  — extracts structured JSON from raw invoice text.
               4 independent reward signals: format, field_accuracy, math, completeness.

  Auditor    — classifies each invoice as approved/flagged with fraud type.
               +0.99 correct detection, +0.90 clean clearance, +0.01 miss / false positive.

  Approver   — final approve/reject/escalate decision (rule-based threshold).

  Regulator  — cross-episode meta-agent. Monitors Auditor over 30-episode window.
               Detects systematic blind spots. Feeds back to Generator.
               Reward: precision + recall of blind spot predictions.

HTTP endpoints (added to app.py):
  POST /multi/reset              Start a new multi-agent episode
  POST /multi/extract            Score an Extractor submission
  POST /multi/audit              Score an Auditor submission + record to tracker
  POST /multi/approve            Rule-based Approver decision
  GET  /multi/state/{episode_id} Episode state
  GET  /regulator/report         Current Regulator tracker state
"""

from __future__ import annotations

import collections
import copy
import random
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAUD_TYPES = ["phantom_vendor", "price_gouging", "math_fraud", "duplicate_submission"]
TRACKER_WINDOW = 30           # episodes in rolling window
BLIND_SPOT_THRESHOLD = 0.50   # detection rate below this = blind spot


# ---------------------------------------------------------------------------
# AuditorPerformanceTracker — cross-episode singleton
# ---------------------------------------------------------------------------

class AuditorPerformanceTracker:
    """
    Thread-safe singleton that tracks Auditor detection rates over the last
    TRACKER_WINDOW episodes.  The Regulator reads this to identify blind spots;
    the Generator reads generator_weights() to bias fraud generation.
    """

    _instance: Optional["AuditorPerformanceTracker"] = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "AuditorPerformanceTracker":
        with cls._class_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._initialise()
                cls._instance = obj
        return cls._instance

    def _initialise(self) -> None:
        self._fraud_history: Dict[str, collections.deque] = {
            ft: collections.deque(maxlen=TRACKER_WINDOW) for ft in FRAUD_TYPES
        }
        self._fp_history: collections.deque = collections.deque(maxlen=TRACKER_WINDOW)
        self._total_audits: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write path

    def record_audit(
        self,
        true_fraud_type: Optional[str],
        predicted_verdict: str,
        predicted_fraud_type: Optional[str],
    ) -> None:
        """
        Record one invoice audit result into the rolling window.
        true_fraud_type=None means the invoice was clean (used for FP tracking).
        """
        with self._lock:
            self._total_audits += 1
            if true_fraud_type is None:
                self._fp_history.append(predicted_verdict == "flagged")
            elif true_fraud_type in self._fraud_history:
                detected = (
                    predicted_verdict == "flagged"
                    and predicted_fraud_type == true_fraud_type
                )
                self._fraud_history[true_fraud_type].append(detected)

    # ------------------------------------------------------------------
    # Read path

    def detection_rates(self) -> Dict[str, Optional[float]]:
        with self._lock:
            return {
                ft: (sum(h) / len(h) if h else None)
                for ft, h in self._fraud_history.items()
            }

    def false_positive_rate(self) -> Optional[float]:
        with self._lock:
            return sum(self._fp_history) / len(self._fp_history) if self._fp_history else None

    def blind_spots(self, threshold: float = BLIND_SPOT_THRESHOLD) -> List[str]:
        """Return fraud types where detection rate < threshold (and have data)."""
        rates = self.detection_rates()
        return [ft for ft, rate in rates.items() if rate is not None and rate < threshold]

    def generator_weights(self) -> Dict[str, float]:
        """
        Sampling weights for fraud type generation.
        Blind spots share 60% weight; healthy types share 40%.
        Falls back to uniform if no blind spots detected.
        """
        spots = self.blind_spots()
        if not spots:
            w = 1.0 / len(FRAUD_TYPES)
            return {ft: round(w, 4) for ft in FRAUD_TYPES}

        n_blind = len(spots)
        n_healthy = len(FRAUD_TYPES) - n_blind
        blind_w = 0.60 / n_blind
        healthy_w = (0.40 / n_healthy) if n_healthy > 0 else 0.0

        return {
            ft: round(blind_w if ft in spots else healthy_w, 4)
            for ft in FRAUD_TYPES
        }

    def report(self) -> Dict[str, Any]:
        rates = self.detection_rates()
        spots = self.blind_spots()
        fp = self.false_positive_rate()
        weights = self.generator_weights()

        formatted_rates = {}
        for ft in FRAUD_TYPES:
            r = rates[ft]
            status = "no data"
            if r is not None:
                if r < BLIND_SPOT_THRESHOLD:
                    status = f"{r:.0%}  ⚠ BLIND SPOT"
                else:
                    status = f"{r:.0%}  ✓ OK"
            formatted_rates[ft] = status

        fp_str = f"{fp:.0%}  ✓ OK" if fp is not None else "no data"

        return {
            "total_audits_recorded": self._total_audits,
            "window": TRACKER_WINDOW,
            "detection_rates": formatted_rates,
            "false_positive_rate": fp_str,
            "blind_spots": spots,
            "generator_weights": weights,
            "verdict": (
                f"Recommend retraining on: {', '.join(spots)}"
                if spots
                else "Auditor performance OK across all fraud types"
            ),
        }

    def reset_for_demo(self) -> None:
        """Seed tracker with realistic demo data (for hackathon demo only)."""
        with self._lock:
            self._initialise()
            # Simulate 20 episodes: phantom_vendor weak (31%), others decent
            for _ in range(13):
                self._fraud_history["phantom_vendor"].append(False)
            for _ in range(6):
                self._fraud_history["phantom_vendor"].append(True)
            for _ in range(18):
                self._fraud_history["price_gouging"].append(True)
            for _ in range(6):
                self._fraud_history["price_gouging"].append(False)
            for _ in range(17):
                self._fraud_history["math_fraud"].append(True)
            for _ in range(4):
                self._fraud_history["math_fraud"].append(False)
            for _ in range(15):
                self._fraud_history["duplicate_submission"].append(True)
            for _ in range(7):
                self._fraud_history["duplicate_submission"].append(False)
            for _ in range(2):
                self._fp_history.append(True)
            for _ in range(16):
                self._fp_history.append(False)
            self._total_audits = 20


# Global singleton — imported by app.py
tracker = AuditorPerformanceTracker()


# ---------------------------------------------------------------------------
# 4 Independent Extractor reward functions
# ---------------------------------------------------------------------------

def reward_format(extracted: Dict[str, Any]) -> float:
    """Weight 0.10 — are all 5 required JSON keys present?"""
    required = {"vendor", "date", "currency", "total", "line_items"}
    present = required.intersection(extracted.keys())
    return round(len(present) / len(required) * 0.10, 4)


def reward_field_accuracy(extracted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Weight 0.40 — do vendor/date/currency/total match ground truth?"""
    score = 0.0
    if extracted.get("vendor", "").lower().strip() == ground_truth.get("vendor", "").lower():
        score += 0.10
    if extracted.get("date", "").strip() == ground_truth.get("date", ""):
        score += 0.10
    if extracted.get("currency", "").upper().strip() == ground_truth.get("currency", ""):
        score += 0.05
    try:
        if abs(float(extracted.get("total", 0)) - float(ground_truth.get("total", -1))) < 0.01:
            score += 0.15
    except (ValueError, TypeError):
        pass
    return round(min(score, 0.40), 4)


def reward_math_consistency(extracted: Dict[str, Any]) -> float:
    """Weight 0.25 — does qty × unit_price = amount for all line items?"""
    items = extracted.get("line_items", [])
    if not isinstance(items, list) or not items:
        return 0.01
    correct = 0
    for item in items:
        try:
            qty = float(item.get("qty", 0))
            up = float(item.get("unit_price", 0))
            amt = float(item.get("amount", -1))
            if abs(qty * up - amt) < 0.02:
                correct += 1
        except (ValueError, TypeError):
            pass
    frac = correct / len(items)
    return round(max(0.01, min(frac * 0.25, 0.25)), 4)


def reward_completeness(extracted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Weight 0.25 — recall: how many expected line items are present?"""
    sub_items = extracted.get("line_items", [])
    gt_items = ground_truth.get("line_items", [])
    if not gt_items:
        return 0.25 if not sub_items else 0.01
    if not isinstance(sub_items, list) or not sub_items:
        return 0.01
    matched = 0
    for gt in gt_items:
        gt_desc = gt.get("description", "").lower()
        for sub in sub_items:
            if gt_desc in sub.get("description", "").lower():
                matched += 1
                break
    frac = matched / len(gt_items)
    return round(max(0.01, min(frac * 0.25, 0.25)), 4)


def combined_extractor_reward(
    extracted: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute all 4 signals. Returns (total_reward, breakdown_dict)."""
    f = reward_format(extracted)
    fa = reward_field_accuracy(extracted, ground_truth)
    m = reward_math_consistency(extracted)
    c = reward_completeness(extracted, ground_truth)
    total = round(max(0.01, min(f + fa + m + c, 0.99)), 4)
    return total, {
        "format": f,
        "field_accuracy": fa,
        "math_consistency": m,
        "completeness": c,
    }


# ---------------------------------------------------------------------------
# Auditor reward
# ---------------------------------------------------------------------------

def compute_auditor_reward(
    predicted_verdict: str,
    predicted_fraud_type: Optional[str],
    true_verdict: str,
    true_fraud_type: Optional[str],
) -> Tuple[float, str]:
    """
    +0.99 correct fraud detection (right verdict + right type)
    +0.90 correct clean clearance
    +0.50 flagged fraud but wrong type
    +0.01 miss or false positive
    """
    is_fraud = true_verdict == "flagged"
    pred_flagged = predicted_verdict == "flagged"

    if is_fraud and pred_flagged:
        if predicted_fraud_type == true_fraud_type:
            return 0.99, f"Correct: {true_fraud_type} detected"
        return 0.50, f"Flagged but wrong type (expected {true_fraud_type}, got {predicted_fraud_type})"
    elif not is_fraud and not pred_flagged:
        return 0.90, "Correct: clean invoice approved"
    elif not is_fraud and pred_flagged:
        return 0.01, f"False positive: clean invoice flagged as {predicted_fraud_type}"
    else:
        return 0.01, f"Missed fraud: {true_fraud_type} not detected"


# ---------------------------------------------------------------------------
# Generator reward (adversarial self-play)
# ---------------------------------------------------------------------------

def compute_generator_reward(auditor_detected: bool, approver_approved: bool) -> float:
    """
    Fraud fully undetected + approved  → 0.85
    Auditor missed but Approver caught → 0.60
    Auditor caught it                  → 0.10
    """
    if not auditor_detected and approver_approved:
        return 0.85
    elif not auditor_detected and not approver_approved:
        return 0.60
    return 0.10


# ---------------------------------------------------------------------------
# Regulator reward
# ---------------------------------------------------------------------------

def compute_regulator_reward(
    predicted_blind_spots: List[str],
    actual_blind_spots: List[str],
) -> Tuple[float, str]:
    """Precision (0.40) + recall (0.40) + no-over-flag bonus (0.20)."""
    if not actual_blind_spots and not predicted_blind_spots:
        return 0.99, "Correctly predicted no blind spots"
    if not actual_blind_spots:
        return 0.01, "False alarm: predicted blind spots when none exist"
    if not predicted_blind_spots:
        return 0.01, "Missed all blind spots"

    correct = set(predicted_blind_spots) & set(actual_blind_spots)
    prec = len(correct) / len(predicted_blind_spots)
    rec = len(correct) / len(actual_blind_spots)
    no_over_flag = 1.0 if prec >= 0.5 else 0.0
    score = round(max(0.01, min(0.40 * prec + 0.40 * rec + 0.20 * no_over_flag, 0.99)), 4)
    return score, f"Blind spot prediction: precision={prec:.2f}, recall={rec:.2f}"


# ---------------------------------------------------------------------------
# Approver (rule-based)
# ---------------------------------------------------------------------------

def approver_decision(
    auditor_verdict: str,
    auditor_confidence: float,
    auditor_fraud_type: Optional[str],
) -> Dict[str, Any]:
    """
    Simple rule-based Approver.
    HIGH confidence flag  → reject
    MEDIUM confidence flag → escalate
    LOW confidence flag   → escalate
    Approved              → approve
    """
    if auditor_verdict != "flagged":
        return {"decision": "approve", "reason": "Auditor cleared invoice"}

    if auditor_confidence >= 0.80:
        return {
            "decision": "reject",
            "reason": f"High-confidence {auditor_fraud_type} fraud detected ({auditor_confidence:.0%})",
        }
    elif auditor_confidence >= 0.50:
        return {
            "decision": "escalate",
            "reason": f"Medium-confidence {auditor_fraud_type} flag — needs human review",
        }
    else:
        return {
            "decision": "escalate",
            "reason": f"Low-confidence flag on {auditor_fraud_type} — needs human review",
        }


# ---------------------------------------------------------------------------
# Biased invoice generator (uses tracker weights)
# ---------------------------------------------------------------------------

def _generate_expert_batch_biased(
    fraud_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict], List[Dict], str]:
    """
    Generate an expert fraud audit batch with fraud type sampling biased
    by the Regulator's generator_weights().

    Returns (invoices, ground_truth_list, reference_text).
    Reuses generation helpers from environment.py.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from server.environment import (
        _generate_invoice, _render_expert_batch, _render_expert_reference,
        PHANTOM_VENDORS, MARKET_PRICE_MAX, VENDORS,
    )

    if fraud_weights is None:
        fraud_weights = tracker.generator_weights()

    n_invoices = random.randint(4, 6)
    n_fraudulent = random.randint(2, 3)

    all_indices = list(range(n_invoices))
    random.shuffle(all_indices)
    fraud_indices = set(all_indices[:n_fraudulent])

    # Weighted fraud type selection
    types_pool = list(fraud_weights.keys())
    weights_pool = [fraud_weights[ft] for ft in types_pool]
    chosen_fraud_types = random.choices(types_pool, weights=weights_pool, k=n_fraudulent)
    fraud_type_map = {idx: chosen_fraud_types[i] for i, idx in enumerate(list(fraud_indices))}

    invoices: List[Dict] = []
    ground_truth: List[Dict] = []
    invoice_history: List[Dict] = []

    for _ in range(3):
        invoice_history.append(_generate_invoice())

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
                inv = copy.deepcopy(random.choice(invoice_history))

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
    raw_text = _render_expert_batch(invoices)
    return invoices, ground_truth, raw_text, reference_text


# ---------------------------------------------------------------------------
# MultiAgentEpisode data class
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentEpisode:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    invoices: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    reference_data: str = ""
    fraud_weights_used: Dict[str, float] = field(default_factory=dict)

    # Extractor stage
    extractor_result: Optional[Dict[str, Any]] = None
    extractor_reward: float = 0.0
    extractor_breakdown: Dict[str, float] = field(default_factory=dict)

    # Auditor stage
    auditor_results: List[Dict[str, Any]] = field(default_factory=list)
    auditor_rewards: List[float] = field(default_factory=list)
    mean_auditor_reward: float = 0.0

    # Approver stage
    approver_results: List[Dict[str, Any]] = field(default_factory=list)

    # Generator reward (computed after full pipeline)
    generator_rewards: List[float] = field(default_factory=list)
    mean_generator_reward: float = 0.0

    done: bool = False


# ---------------------------------------------------------------------------
# Session registry for multi-agent episodes
# ---------------------------------------------------------------------------

_MAX_MULTI_SESSIONS = 100
_multi_sessions: "collections.OrderedDict[str, MultiAgentEpisode]" = collections.OrderedDict()
_multi_lock = threading.Lock()


def create_episode() -> MultiAgentEpisode:
    """Create a new multi-agent episode with Regulator-biased Generator."""
    weights = tracker.generator_weights()
    invoices, ground_truth, raw_text, reference_data = _generate_expert_batch_biased(weights)

    ep = MultiAgentEpisode(
        invoices=invoices,
        ground_truth=ground_truth,
        raw_text=raw_text,
        reference_data=reference_data,
        fraud_weights_used=weights,
    )

    with _multi_lock:
        _multi_sessions[ep.episode_id] = ep
        while len(_multi_sessions) > _MAX_MULTI_SESSIONS:
            _multi_sessions.popitem(last=False)

    return ep


def get_episode(episode_id: str) -> Optional[MultiAgentEpisode]:
    with _multi_lock:
        return _multi_sessions.get(episode_id)


# ---------------------------------------------------------------------------
# Stage handlers (called by HTTP endpoints)
# ---------------------------------------------------------------------------

def handle_extract(
    episode_id: str,
    extracted_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Score Extractor output against the first invoice ground truth.
    Returns reward + breakdown.
    """
    ep = get_episode(episode_id)
    if ep is None:
        return {"error": "Episode not found. Call /multi/reset first."}

    # Use first clean invoice as reference for extraction grading
    # (the expert task expects audit, but extraction is graded on the first invoice)
    gt = ep.invoices[0] if ep.invoices else {}
    total, breakdown = combined_extractor_reward(extracted_data, gt)

    ep.extractor_result = extracted_data
    ep.extractor_reward = total
    ep.extractor_breakdown = breakdown

    return {
        "episode_id": episode_id,
        "reward": total,
        "breakdown": breakdown,
        "feedback": (
            f"Extractor: format={breakdown['format']:.2f}, "
            f"field={breakdown['field_accuracy']:.2f}, "
            f"math={breakdown['math_consistency']:.2f}, "
            f"completeness={breakdown['completeness']:.2f}"
        ),
    }


def handle_audit(
    episode_id: str,
    audit_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score Auditor output. Records results to AuditorPerformanceTracker.
    audit_results: [{"invoice_id": str, "verdict": str, "fraud_type": str|None, "confidence": float}]
    """
    ep = get_episode(episode_id)
    if ep is None:
        return {"error": "Episode not found. Call /multi/reset first."}

    gt_map = {gt["invoice_id"]: gt for gt in ep.ground_truth}
    rewards = []
    feedbacks = []
    approver_inputs = []

    for result in audit_results:
        inv_id = result.get("invoice_id", "")
        pred_verdict = result.get("verdict", "approved").lower()
        pred_ftype = result.get("fraud_type")
        confidence = float(result.get("confidence", 0.5))

        gt = gt_map.get(inv_id)
        if gt is None:
            feedbacks.append(f"{inv_id}: not found in episode")
            continue

        true_verdict = gt["verdict"]
        true_ftype = gt["fraud_type"]

        reward, fb = compute_auditor_reward(pred_verdict, pred_ftype, true_verdict, true_ftype)
        rewards.append(reward)
        feedbacks.append(f"{inv_id}: {fb}")

        # Record to global tracker
        tracker.record_audit(true_ftype, pred_verdict, pred_ftype)

        approver_inputs.append({
            "invoice_id": inv_id,
            "auditor_verdict": pred_verdict,
            "auditor_confidence": confidence,
            "auditor_fraud_type": pred_ftype,
        })

    mean_reward = round(sum(rewards) / len(rewards), 4) if rewards else 0.01
    ep.auditor_results = audit_results
    ep.auditor_rewards = rewards
    ep.mean_auditor_reward = mean_reward
    ep.approver_results = approver_inputs  # stage input ready

    return {
        "episode_id": episode_id,
        "mean_reward": mean_reward,
        "per_invoice_rewards": dict(zip([r.get("invoice_id", i) for i, r in enumerate(audit_results)], rewards)),
        "feedback": "; ".join(feedbacks),
        "tracker_report": tracker.report(),
    }


def handle_approve(episode_id: str) -> Dict[str, Any]:
    """
    Run rule-based Approver on Auditor results. Computes Generator reward.
    """
    ep = get_episode(episode_id)
    if ep is None:
        return {"error": "Episode not found"}
    if not ep.approver_results:
        return {"error": "Run /multi/audit before /multi/approve"}

    decisions = []
    gen_rewards = []
    gt_map = {gt["invoice_id"]: gt for gt in ep.ground_truth}

    for inp in ep.approver_results:
        inv_id = inp["invoice_id"]
        decision = approver_decision(
            inp["auditor_verdict"],
            inp["auditor_confidence"],
            inp["auditor_fraud_type"],
        )
        decisions.append({"invoice_id": inv_id, **decision})

        # Generator reward for fraud invoices
        gt = gt_map.get(inv_id, {})
        if gt.get("verdict") == "flagged":
            auditor_detected = inp["auditor_verdict"] == "flagged"
            approver_approved = decision["decision"] == "approve"
            gen_rewards.append(compute_generator_reward(auditor_detected, approver_approved))

    mean_gen = round(sum(gen_rewards) / len(gen_rewards), 4) if gen_rewards else 0.0
    ep.generator_rewards = gen_rewards
    ep.mean_generator_reward = mean_gen
    ep.done = True

    return {
        "episode_id": episode_id,
        "decisions": decisions,
        "generator_reward": mean_gen,
        "feedback": (
            f"Approver processed {len(decisions)} invoices. "
            f"Generator adversarial reward: {mean_gen:.3f}"
        ),
    }
