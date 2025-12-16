# Reconciliation Under Ambiguity

### Designing Defensible Financial Controls Without Shared Transaction IDs

---

## Overview

This project simulates **real-world payment, bank, and refund reconciliation** in fragmented financial environments where:

* Transaction IDs do **not** align across systems
* Narrations are manual, abbreviated, or rewritten
* Posting delays, fees, and batching distort expected values
* Refunds introduce elevated fraud and regulatory risk

Instead of forcing artificial precision, the reconciliation logic **explicitly models uncertainty**, encoding how experienced finance teams reason, review, and defend reconciliation decisions in practice.

This is not a matching exercise.
It is a **risk and controls case study**.

---

## Problem Statement

In many African and emerging-market financial systems:

* Payment platforms, banks, and refund systems operate independently
* No universal transaction identifier exists end-to-end
* Operations teams rely on narrations, timing, and contextual clues
* Traditional reconciliations assume clean joins and binary outcomes

Those assumptions fail in reality.

This project asks a different question:

> **How do competent finance teams reconcile under ambiguity — and how can that judgement be encoded into defensible, auditable logic?**

---

## Design Philosophy

### Reconciliation Logic — How Human Teams Think, Encoded in Code

This system does **not** aim for perfect matching.
It aims for **defensible confidence**.

---

## Payments ↔ Bank Reconciliation Logic

### Step 1: Stop Pretending IDs Exist

We never join on transaction IDs.

Each payment record is treated as:

> **“A claim that money should exist somewhere in the bank.”**

The task is to evaluate evidence, not enforce identity.

---

### Step 2: Narration Normalization (Critical Control Step)

Narrations are aggressively standardized because:

* Humans abbreviate
* Systems truncate
* Banks rewrite descriptions

Example:

```
“Cheque dep. for MBR-20056 / Jan fees”
→ “cheque deposit member 20056 january fees”
```

This transforms unstructured text into **machine-comparable evidence**.

---

### Step 3: Candidate Generation (TF-IDF)

Instead of brute-force matching:

* Only plausible bank rows are shortlisted
* Based on contextual similarity

This mirrors operational reality:

> “Scan the bank statement for entries that look related.”

This step is **scalable, auditable, and defensible**.

---

### Step 4: Multi-Signal Scoring (Core Logic)

Each candidate match is evaluated across independent signals:

| Signal               | Why It Exists in Real Life              |
| -------------------- | --------------------------------------- |
| Narration similarity | Primary identifier when IDs don’t exist |
| Amount proximity     | Fees, rounding, partial settlements     |
| Date proximity       | Posting delays and batching             |
| Customer inference   | Humans embed identifiers in text        |

No signal is trusted in isolation.

---

## Confidence Scoring (Designed for Audit, Not Accuracy Theatre)

### Confidence Formula

```
Confidence =
0.4 × Narration similarity
0.3 × Amount proximity
0.2 × Date proximity
0.1 × Customer inference
```

### Rationale

* Narration carries semantic truth
* Amounts drift due to fees and netting
* Dates drift due to operational delays
* Customer IDs help, but are unreliable

Weights are:

* Policy-tunable
* Auditor-reviewable
* Risk-committee defensible

---

## Classification — Turning Scores Into Decisions

Binary outcomes are avoided.

| Score Band | Status             | Business Meaning         |
| ---------- | ------------------ | ------------------------ |
| ≥ 85       | Auto-Reconciled    | Safe to close            |
| 70–84      | Review Recommended | Human judgement required |
| 50–69      | Weak Match         | Elevated risk            |
| < 50       | Unmatched          | Potential leakage        |

This creates **operational queues**, not noise.

---

## Audit-Friendly Outputs

### `reconciliation_results_detailed.csv`

Row-level transparency:

* Why a match was made
* Confidence score
* Signal contributions

No black boxes. Full traceability.

---

### `reconciliation_summary.csv`

Executive-level oversight:

| Metric                   | Insight              |
| ------------------------ | -------------------- |
| Auto-reconciliation rate | Control maturity     |
| Review rate              | Operational workload |
| Weak matches             | Risk exposure        |
| Average confidence       | Data quality health  |

---

## Refund Reconciliation (Higher Risk, Higher Standard)

### Refund Logic — What Changes

Refund reconciliation reverses the question:

> **“Which original payment does this refund legitimately belong to?”**

This demands stricter controls.

---

### Signals Used (With Regulatory Intent)

| Signal                  | Why Regulators Care   |
| ----------------------- | --------------------- |
| Narration similarity    | Evidence of linkage   |
| Refund ratio            | Over-refund detection |
| Timing (days to refund) | SLA and fraud signal  |
| Customer inference      | Ownership validation  |

---

## Refund Confidence Scoring (Stricter by Design)

```
Confidence =
0.45 × Narration similarity
0.25 × Amount logic
0.20 × Timing logic
0.10 × Customer inference
```

This explicitly:

* Prioritizes semantic linkage
* Penalizes excessive or delayed refunds
* Avoids single-factor dependence

This is **policy encoded in code**.

---

## Regulatory Classification

| Status           | Meaning                          |
| ---------------- | -------------------------------- |
| Valid Refund     | Acceptable, well-linked          |
| Review Required  | Human judgement needed           |
| High Risk Refund | Escalation candidate             |
| Unlinked Refund  | Control failure / potential loss |

---

## Explainability (Regulator-First Design)

Each refund includes:

```
explainability_note
```

Examples:

* “Refund exceeds original amount”
* “Unusually late refund”
* “Weak narration similarity”

Scores are not interpreted — **reasons are stated**.

---

## Refund Outputs

* `refund_reconciliation_detailed.csv` — inspection-ready
* `refund_reconciliation_summary.csv` — executive oversight

Designed for:

* Internal audit
* Risk & compliance
* External regulators

---

## Why This Matters 

Most reconciliation systems optimize for **technical correctness**.
This project optimizes for **organizational trust**.

It demonstrates how to:

* Operate effectively without perfect data
* Encode human financial judgement into transparent logic
* Balance automation with accountable review
* Produce outputs that auditors can inspect, not interpret
* Translate data decisions into risk, cost, and governance language

In real financial environments:

* Precision is aspirational
* Defensibility is mandatory

This case study shows how data systems can function as **financial controls**, not just reporting tools — enabling leadership to scale operations **without scaling risk**.
This is a realistic prototype for small-to-medium businesses dealing with straightforward payment reconciliation. 
It handles the core problems well: inconsistent data formats, timing mismatches, and imperfect matches. 
However, enterprise systems handling high volumes would need additional features like real-time processing, multi-currency support, and deeper integration with accounting systems.

---

This architecture mirrors how reconciliation evidence flows in real financial institutions:
**operations generate signals → risk evaluates exposure → audit inspects controls**.

### Logical Flow

```
┌────────────────────┐
│  Payments System   │
│  (Claims of Value) │
└─────────┬──────────┘
          │
          │   Unstructured narrations
          │   Amounts, dates, channels
          ▼
┌────────────────────┐
│  Bank Statements   │
│  (Evidence of Cash)│
└─────────┬──────────┘
          │
          │   TF-IDF candidate generation
          │   Multi-signal scoring
          ▼
┌────────────────────────────────┐
│  Reconciliation Engine         │
│                                │
│ • Text normalization           │
│ • Fuzzy matching               │
│ • Confidence scoring           │
│ • Classification logic         │
└─────────┬──────────────────────┘
          │
          │   Confidence bands
          │   Explainability notes
          ▼
┌────────────────────┐
│  Operations Queue  │
│                    │
│ • Auto-close       │
│ • Review required  │
│ • Escalations      │
└─────────┬──────────┘
          │
          │   Aggregated risk signals
          ▼
┌────────────────────┐
│  Risk & Compliance │
│                    │
│ • Exposure metrics │
│ • Control health   │
│ • Policy tuning    │
└─────────┬──────────┘
          │
          │   Full traceability
          ▼
┌────────────────────┐
│  Audit & Assurance │
│                    │
│ • Row-level logic  │
│ • Evidence review  │
│ • Control testing  │
└────────────────────┘
```

### Key Design Principle

> **The same data serves three audiences without reinterpretation**:

* Operations get queues
* Risk gets exposure
* Audit gets evidence

No parallel systems. No narrative gaps.

---

## Control Framework Mapping (COSO)

This project is explicitly designed as a **control system**, not an analytics experiment.

Below is how the logic maps to recognized risk frameworks.

---

## COSO Internal Control Framework Mapping

### Control Environment

| COSO Principle             | How This Project Implements It                       |
| -------------------------- | ---------------------------------------------------- |
| Integrity & Ethical Values | Explicitly avoids forced matches and false certainty |
| Accountability             | Every reconciliation decision is traceable           |
| Risk Awareness             | Ambiguity is acknowledged, not hidden                |

> Reconciliation outcomes are **defended**, not assumed.

---

### Risk Assessment

| COSO Principle              | Implementation                                      |
| --------------------------- | --------------------------------------------------- |
| Identify Risks              | Missing settlements, over-refunds, unlinked refunds |
| Analyze Likelihood & Impact | Confidence scores quantify uncertainty              |
| Fraud Risk Assessment       | Refund timing, amount ratios, narration mismatch    |

Risk is **measured continuously**, not discovered post-fact.

---

### Control Activities

| Control                | Embedded Logic                           |
| ---------------------- | ---------------------------------------- |
| Preventive controls    | Auto-reconciliation thresholds           |
| Detective controls     | Weak-match and high-risk classifications |
| Segregation of duties  | Machine flags, humans decide             |
| Authorization controls | Review queues for non-auto matches       |

Controls are **policy-encoded**, not manual checklists.

---

### Information & Communication

| COSO Requirement     | Delivery                    |
| -------------------- | --------------------------- |
| Relevant information | Signal-level explainability |
| Timely communication | Operational queues          |
| Clear reporting      | Executive summaries         |

Outputs speak **business**, not model language.

---

### Monitoring Activities

| Monitoring Type           | Evidence Produced             |
| ------------------------- | ----------------------------- |
| Ongoing monitoring        | Confidence distributions      |
| Separate evaluations      | Audit-ready CSVs              |
| Deficiency identification | Unmatched & high-risk volumes |

Control effectiveness is **observable over time**.

---

