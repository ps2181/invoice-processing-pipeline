# Round 2 Problem Statement

## Title

Autonomous Accounts Payable Control Tower

## Theme Alignment

This environment intentionally covers more than one Round 2 theme:

- Multi-Agent Interactions
- Long-Horizon Planning and Instruction Following
- World Modeling across professional tasks
- Self-Improving agent systems

## Core Problem Statement

Design an OpenEnv environment in which one or more AI agents operate as an accounts-payable control tower for a mid-sized company. The agent must transform noisy financial documents into reliable, auditable decisions across invoice extraction, normalization, reconciliation, fraud review, clarification, and downstream supply-chain checks.

The real-world challenge is not simple OCR or single-step parsing. A strong agent must maintain working memory across turns, choose when to ask for clarification, compare multiple sources of truth, detect subtle inconsistencies, and maximize business-safe outcomes under limited attempts.

In practical terms, the environment simulates the day-to-day work of finance operations teams, procurement reviewers, and audit analysts who must:

- extract structured fields from raw invoices
- normalize inconsistent or corrupted records
- reconcile invoices with purchase orders
- flag fraud and policy violations
- investigate ambiguous cases through clarification
- reason about upstream delivery anomalies that affect billing decisions

## Environment

The environment is a realistic finance-operations workspace exposed through the standard OpenEnv `reset()`, `step()`, and `state()` API.

At each episode reset, the environment generates a fresh synthetic business scenario containing some combination of:

- raw invoice text rendered as clean OCR, messy OCR, or adversarial OCR
- purchase order reference data
- approved vendor registry and historical invoice records
- delivery logs and supply-chain records
- ambiguous fields that require clarification turns

The environment models several constraints found in real operations:

- incomplete or corrupted documents
- multiple representations of the same entity
- conflicting sources of truth
- deceptive lines such as fake tax or subtotal entries
- fraud patterns that require policy grounding, not just extraction
- limited action budget with step penalties and attempt caps

This makes the world state richer than a single-document benchmark and aligns well with OpenEnv’s focus on realistic agent behavior.

## Agent Capabilities

The intended agent stack can be either a single frontier model or a coordinated set of specialists. A multi-agent version is especially compelling for this theme.

Recommended agent roles:

- Extractor Agent: converts raw text into structured invoice objects
- Reconciliation Agent: compares invoices against purchase orders and flags mismatches
- Audit Agent: checks vendor legitimacy, duplicate history, math fraud, and price gouging
- Clarification Agent: decides whether a question is worth spending a turn on
- Coordinator Agent: selects which sub-agent acts next and merges outputs into the final action

Core capabilities expected from the agent system:

- structured information extraction
- entity normalization
- cross-document comparison
- anomaly and fraud detection
- selective clarification and question asking
- trajectory planning across multiple steps
- adaptation using prior grader feedback

## Tasks

The current environment already contains a natural difficulty ladder plus advanced variants:

1. Easy: Single Invoice Extraction
   Extract vendor, date, currency, total, and line items from one clean invoice.

2. Medium: Batch Invoice Cleaning
   Normalize multiple messy invoices with date-format chaos, vendor typos, mixed currencies, and arithmetic inconsistencies.

3. Hard: Invoice-PO Reconciliation
   Clean invoices and compare them against purchase orders to detect overcharges, extra items, and missing items.

4. Expert: Fraud Audit
   Use policy and reference data to classify phantom vendors, duplicate submissions, price gouging, and math fraud.

5. Adversarial: Trap-Resistant Extraction
   Ignore misleading subtotal, tax, adjustment, and FX lines while recovering the true invoice structure from OCR-corrupted text.

6. Negotiate: Clarification-then-Act
   Ask targeted questions before submitting the final extraction, with a reward bonus for solving accurately using few clarifications.

7. Supply Chain: Delivery Anomaly Detection
   Identify quantity shortfalls, price spikes, unauthorized substitutions, and phantom deliveries in logistics records that affect invoice trustworthiness.

These tasks support both the minimum competition requirement and a stronger story around agentic reasoning breadth.

## Reward Model and Evaluation Logic

The environment uses dense, trajectory-level reward rather than pure terminal success.

Key evaluation properties:

- scores are deterministic and normalized to `0.0–1.0`
- reward is provided every step
- partial credit is granted for partially correct structure and reasoning
- repeated failure is penalized through capped attempts and end-of-episode multipliers
- some tasks provide field-level or component-level breakdowns to support learning

Illustrative reward design:

- Extraction tasks score vendor, date, currency, total, and line-item accuracy separately
- Batch tasks average per-document correctness across the batch
- Reconciliation tasks combine extraction quality with discrepancy precision and recall
- Fraud tasks grade both verdict quality and correct fraud-type assignment
- Clarification tasks reward final correctness and add a bonus for solving with fewer questions
- Supply-chain tasks reward exact and partial anomaly matching using precision and recall

Why this is strong for judging:

- clear task objectives
- measurable, reproducible graders
- meaningful partial progress signals
- sensible episode boundaries
- realistic penalties for wasteful or unsafe behavior

## Long-Horizon and Multi-Agent Story

This environment is stronger than a one-shot extraction benchmark because good performance often requires a sequence of decisions:

- inspect noisy evidence
- choose whether to normalize or reconcile first
- decide whether ambiguity warrants clarification
- use feedback from the last failed attempt
- compare several external references before acting

That naturally creates a long-horizon control problem. In a multi-agent implementation, the coordinator can route subtasks to specialized workers and aggregate them into a single environment action. This gives the judges a concrete example of multi-agent interaction that is useful, realistic, and measurable.

## Post-Training and Self-Improvement Strategy

The self-improvement loop should operate at both the environment level and the agent level.

Environment-side self-improvement already fits well:

- dynamic difficulty adjusts generation parameters based on recent agent performance
- fresh synthetic scenarios prevent rote memorization
- richer failure cases can be added over time from observed model mistakes

Agent-side post-training strategy:

- collect trajectories with observation, action, reward, and grader feedback
- distill successful traces into task-specific demonstrations
- finetune or preference-optimize the coordinator on high-reward trajectories
- train specialist sub-agents on their own slices such as extraction, fraud, or reconciliation
- mine common failure modes and convert them into targeted curriculum tasks
- use reward breakdowns as auxiliary supervision for process improvement

An especially strong hackathon implementation would add a memory or policy-improvement module that:

- tracks recurrent failure categories per task
- updates prompting or tool routing after each episode batch
- increases confidence thresholds for high-risk actions
- learns when asking a clarification question is economically justified

## Why This Submission Should Score Well

Against the published judging criteria, this design is well positioned because it is:

- real-world: finance operations, procurement, and audit teams perform these workflows every day
- structured: tasks are concrete, typed, and easy to evaluate automatically
- agentic: success depends on planning, comparison, recovery from feedback, and selective questioning
- novel enough: it goes beyond invoice OCR into reconciliation, fraud, negotiation, and supply-chain reasoning
- extensible: more vendors, fraud patterns, policies, and document sources can be added without changing the API

## Short Submission Pitch

Autonomous Accounts Payable Control Tower is a real-world OpenEnv benchmark for training and evaluating AI agents on document-grounded finance operations. Agents must extract, normalize, reconcile, audit, question, and adapt across noisy multi-document workflows, receiving dense rewards and deterministic grading throughout. The environment supports single-agent and multi-agent systems, emphasizes long-horizon decision making, and includes a natural self-improvement loop through feedback-aware retries and adaptive difficulty.
