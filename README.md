# Ola Domain Support Agent (CrewAI)

**Track completed: Ola (Business Operations / Customer Support).**

A production-minded support agent for Ola that answers support-policy questions from a hand-written
knowledge base, looks up support-ticket status from a seeded dataset, remembers a conversation, is guarded
against misuse, has every draft reviewed by an independent Autogen team, and runs behind FastAPI.

**Everything runs under `MOCK_LLM=1` (the default) with zero API keys and zero network access.** Every
transcript in [`transcripts/`](transcripts/) was generated in that mode.

---

## Dataset design choices (Part 1, Task 1), for deterministic reproduction

| Choice | Value |
|---|---|
| Seed | `SEED = 42` (`random.Random(42)`; the first draw passed every check, no other seed was needed) |
| Records | `N = 48` (`OLA-T-0001` … `OLA-T-0048`) |
| Category weights | Billing 0.30, Technical Issue 0.22, Account Access 0.16, Product Defect 0.14, General Inquiry 0.18 |
| Status weights | Open 0.25, In Progress 0.25, Escalated 0.12, Resolved 0.23, Closed 0.15 |
| Priority weights (depend on category, matching `kb/01_ticket_priority.md`) | Billing P2 .55 / P3 .40 / P4 .05 · Technical Issue P1 .25 / P2 .30 / P3 .40 / P4 .05 · Account Access P2 .70 / P3 .30 · Product Defect P1 .15 / P2 .15 / P3 .70 · General Inquiry P3 .20 / P4 .80 |
| `resolution_time_hours` range | **0.5–72 h**, drawn inside each priority's SLA resolution window (P1 0.5–4, P2 2–24, P3 6–48, P4 12–72), rounded to 0.5 h |
| `days_since_created` | integer 0–30, uniform |
| `escalated` | always `True` when status is `Escalated`; otherwise `True` with probability 0.08 (escalated earlier, since resolved) |
| Amount range | `fare_amount_inr` **50–1500 INR** (fabricated trip fare) |

**Resolution-time reasoning (one sentence):** resolution times follow the SLA policy in the knowledge base,
so urgent P1 safety or trip-blocking issues close within 4 hours while low-priority P4 inquiries may take up
to 72 hours, which gives an overall realistic range of 0.5–72 hours.

**Result** (`python dataset.py`, see [`transcripts/task01_dataset.txt`](transcripts/task01_dataset.txt)):

| Category | Count | | Status | Count |
|---|---|---|---|---|
| Billing | 15 | | Open | 12 |
| Technical Issue | 7 | | In Progress | 9 |
| Account Access | 10 | | Escalated | 5 |
| Product Defect | 7 | | Resolved | 11 |
| General Inquiry | 9 | | Closed | 11 |

`escalated=True`: **9/48 = 18.8 %** (required band 10–30 %). No record was edited by hand; `validate()` in
`dataset.py` checks every threshold and a test checks determinism.

---

## Setup

Requires **Python 3.12** (CrewAI does not install on Python 3.14).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `protobuf==5.29.6` and `googleapis-common-protos==1.70.0` because
`autogen-core 0.7.5` requires protobuf 5.x while the newest `googleapis-common-protos` requires protobuf 6.

The embedding model `sentence-transformers/all-MiniLM-L6-v2` (free, local) is downloaded from Hugging Face on
the very first run only. After that, `config.enable_hf_offline()` sets `HF_HUB_OFFLINE=1`, so no run
touches the network.

### Zero-network configuration

`config.py` is imported before any library and sets:

| Variable | Why |
|---|---|
| `CREWAI_DISABLE_TELEMETRY=true` | **CrewAI telemetry off** (required by the brief) |
| `OTEL_SDK_DISABLED=true` | OpenTelemetry SDK off (CrewAI checks both, `crewai_core/telemetry.py`) |
| `CREWAI_TRACING_ENABLED=false` | CrewAI's separate trace collection, forced off |
| `ANONYMIZED_TELEMETRY=False` | ChromaDB telemetry off (also `Settings(anonymized_telemetry=False)`) |
| `LANGCHAIN_TRACING_V2=false`, `LANGSMITH_TRACING=false` | LangSmith tracing off |
| `HF_HUB_OFFLINE=1` | Embedding model is loaded from the local cache only |
| `MOCK_LLM=1` (default) | Deterministic mock LLMs, no API keys |

---

## Part 1: Dataset and RAG core

### Task 2: Knowledge base
Twelve original documents in [`kb/`](kb/), one per required topic, each 4–5 sentences
([`transcripts/task02_knowledge_base.txt`](transcripts/task02_knowledge_base.txt)):

| Required topic | Document |
|---|---|
| Ticket-priority classification rules | `01_ticket_priority.md` |
| SLA-by-severity policy | `02_sla_by_severity.md` |
| Escalation matrix | `03_escalation_matrix.md` |
| Refund/compensation policy | `04_refund_compensation.md` |
| Customer-communication-channel policy | `05_communication_channels.md` |
| Business-hours/holiday-support policy | `06_business_hours.md` |
| Repeat-complaint-handling policy | `07_repeat_complaints.md` |
| Service-credit policy | `08_service_credits.md` |
| Feedback-collection process | `09_feedback_collection.md` |
| VIP-customer handling policy | `10_vip_handling.md` |
| Outage-communication protocol | `11_outage_communication.md` |
| Data-retention policy for tickets | `12_data_retention.md` |

### Task 3: Two chunking strategies, two ChromaDB collections
| | Fixed-size with overlap | Sentence-based |
|---|---|---|
| Rule | 200 chars, 50-char overlap, window edges snapped to spaces | one sentence per chunk |
| Collection | `ola_kb_fixed` (48 chunks) | `ola_kb_sentence` (55 chunks) |

Both are embedded with `all-MiniLM-L6-v2` (normalised, title prepended to every chunk for both strategies)
and written with `collection.upsert()` into cosine-space collections, so similarity = 1 − distance. Both
return the SLA policy as top hit for the sample query
([`transcripts/task03_indexing.txt`](transcripts/task03_indexing.txt)).

### Task 4: Grounded generation and the calibrated "I don't know" threshold
Top-1 cosine similarity measured on **both** collections for 8 in-scope and 6 deliberately out-of-scope
queries ([`transcripts/task04_threshold_calibration.txt`](transcripts/task04_threshold_calibration.txt)):

| | fixed-size | sentence |
|---|---|---|
| In-scope top-1 range | 0.4321 – 0.7464 | 0.4861 – 0.7840 |
| Out-of-scope top-1 range | 0.0365 – 0.2863 | 0.0529 – 0.3072 |
| Gap | +0.146 | +0.179 |

Lowest in-scope score across both collections is **0.4321** ("Is support available on Diwali?"); highest
out-of-scope score is **0.3072** ("What is Ola's share price today?"). **Chosen threshold = midpoint
(0.3072 + 0.4321) / 2 ≈ 0.37.** A tutorial preset would have failed here: 0.5 refuses the legitimate Diwali
question on both collections, and 0.6 refuses 4 of 8 in-scope questions on the fixed-size collection.

Under `MOCK_LLM`, generation goes through `MockTextLLM`, which receives the real grounded prompt
(CONTEXT + QUESTION) and extracts the context sentences most similar to the question, so it can only output
retrieved text and cites the `[doc_id]` of each sentence. Five in-scope queries are answered and the
out-of-scope query ("Can you recommend a good movie to watch this weekend?", top-1 = 0.17) triggers the
fallback ([`transcripts/task04_grounded_generation.txt`](transcripts/task04_grounded_generation.txt)).

### Task 5: Chunking evaluation (document-level, deduplicated, k = 3)
Full per-query arithmetic: [`transcripts/task05_chunking_eval.txt`](transcripts/task05_chunking_eval.txt).

| Query | fixed P | fixed R | sentence P | sentence R |
|---|---|---|---|---|
| Q1 P1 response/resolution time | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 |
| Q2 Who approves refunds > INR 2,000 | 1/2 = 0.50 | 1/2 = 0.50 | 1/2 = 0.50 | 1/2 = 0.50 |
| Q3 Compensation for a late driver | 1/2 = 0.50 | 1/2 = 0.50 | 2/3 = 0.67 | 2/2 = 1.00 |
| Q4 Outage communication | 1/2 = 0.50 | 1/1 = 1.00 | 1/2 = 0.50 | 1/1 = 1.00 |
| Q5 Ticket retention and deletion | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 | 1/1 = 1.00 |
| **Mean** | **0.700** | **0.800** | **0.733** | **0.900** |

**Recommendation:** I recommend deploying the sentence-based collection (`ola_kb_sentence`): on the same 5
queries it scored mean precision 0.733 vs 0.700 and mean recall 0.900 vs 0.800 for fixed-size chunks, and
the gap comes from Q3, where only sentence chunks retrieved both the service-credit and the
refund/compensation policy (R = 2/2 vs 1/2). Sentence chunks also gave the wider in-scope vs out-of-scope
similarity gap during calibration (+0.179 vs +0.146), which makes the 0.37 "I don't know" threshold safer,
and because every chunk is one complete policy statement, grounded answers never start or stop mid-sentence.

`config.DEPLOYED_COLLECTION = "ola_kb_sentence"` is the fixed RAG input for Parts 2–4.

---

## Part 2: CrewAI orchestration, tools, memory and guardrails
*(in progress)*

## Part 3: Evaluation, observability and FastAPI
*(in progress)*

## Part 4: Resilience and governance
*(in progress)*

---

## PII scope
Phone numbers have a fixed format and are masked by the input guardrail (Task 10) before they reach the
model, the cache or the logs. Rider/driver names, pickup/drop addresses and payment details are free text
with no single fixed format, so masking them is out of scope under a keyless `MOCK_LLM`-only masker. Only
fabricated examples are used for all of them.

## Task index

| Task | Script | Transcript |
|---|---|---|
| 1 Dataset | `dataset.py`, `demos/task01_dataset.py` | `transcripts/task01_dataset.txt` |
| 2 Knowledge base | `demos/task02_knowledge_base.py` | `transcripts/task02_knowledge_base.txt` |
| 3 Chunking and indexing | `demos/task03_indexing.py` | `transcripts/task03_indexing.txt` |
| 4 Threshold calibration | `demos/task04_calibrate.py` | `transcripts/task04_threshold_calibration.txt` |
| 4 Grounded generation | `demos/task04_grounded_generation.py` | `transcripts/task04_grounded_generation.txt` |
| 5 Chunking evaluation | `demos/task05_chunking_eval.py` | `transcripts/task05_chunking_eval.txt` |

Tests: `python -m pytest`
