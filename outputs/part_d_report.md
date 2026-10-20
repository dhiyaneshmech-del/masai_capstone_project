# Part D — Policy RAG assistant and refund agent

Run `python policy_rag_refund_agent.py` after Part A. This workflow uses local TF-IDF retrieval and the offline `mock_llm.py`; it does not make network calls or require an API key.

## Chunking and retrieval

Chunk size: 40 words; overlap: 10 words. Total: **15** chunks (delivery_policy.txt: 3, escalation_policy.txt: 3, payment_policy.txt: 3, privacy_policy.txt: 2, refund_policy.txt: 4).
Top three sources for `refund policy late delivery`: refund_policy.txt, refund_policy.txt, refund_policy.txt.
Term frequency (TF) measures how often a term occurs in a chunk, while inverse document frequency (IDF) downweights terms common across chunks and emphasizes rarer terms. TF-IDF is lexical rather than semantic, so two equivalent phrasings with different words may have little retrieval similarity.

## Retrieval and grounded answer evaluation

| Chunking | Answerable docs in top 3 | Expected facts in answers | Unanswerable fallback answers |
|---|---:|---:|---:|
| 40 words, overlap 10 | 12/12 | 12/12 | 2/2 |
| 40 words, overlap 0 | 12/12 | 10/12 | 2/2 |

- With zero overlap, **What delivery time makes a completed order eligible for a full refund?** fails the expected-fact check because the relevant answer is split across a chunk boundary.
- With zero overlap, **How soon is money returned when a payment fails but is deducted?** fails the expected-fact check because the relevant answer is split across a chunk boundary.
For every question, `answer()` sends the required two-message prompt; the user content is exactly the retrieved `[source]` context block and `QUESTION:` section. Evaluation is case-sensitive for expected facts and uses a fallback-prefix check for unanswerable questions.

## Refund-agent scenarios

| Scenario | Status | Refund amount | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 (expected INR 128.25) | 5 | 969 |
| S2 | REFUND | INR 135.00 (expected INR 135.00) | 5 | 969 |
| S3 | ESCALATE | INR 1,999.00 (expected INR 1,999.00) | 5 | 956 |
| S4 | DENIED | INR 0.00 (expected INR 0.00) | 4 | 675 |
| S5 | DENIED | INR 0.00 (expected INR 0.00) | 3 | 404 |
| S6 | NOT_FOUND | INR 0.00 (expected INR 0.00) | 2 | 142 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 (expected INR 0.00) | 6 | 2871 |

S1 token-limit demo status: **FALLBACK_TOKEN_LIMIT** (421 tokens before stopping).
S1 zero-second timeout demo status: **FALLBACK_TIMEOUT** (0 model calls).

## Checkpoint and resume

Simulated crash after the second tool checkpoint; tools across crash and resume: **get_order → search_policy → calculate_refund → issue_refund**. The refund completed as REFUND for INR 128.25 without repeating a tool. Checkpoints are stored in `state/` after every tool observation.

## ReAct versus plan-and-execute

ReAct used 5 model calls and 969 tokens for S1, 3 calls and 404 tokens for S5, and 2 calls and 142 tokens for S6 because it chooses each tool after seeing the prior observation. A plan-and-execute design could request the step list once, then replan only after an error, reducing call count and repeated prompt tokens for straightforward refunds. But it must still validate each observation: a NOT_FOUND result or cancelled order must stop the plan before any refund action. I would choose ReAct here because refund eligibility and escalation depend on live tool results, and the additional calls are a reasonable cost for explicit observation-driven payment safeguards.

The seven standard scenario runs and the fallback/resume demonstrations are recorded in `outputs/agent_runs.jsonl`. Refund issuance and escalations append auditable entries to `outputs/refund_ledger.jsonl`.
The 14 answers from `data/raw/policy_eval.csv` are also recorded as `POLICY_Q01` through `POLICY_Q14` lines in `outputs/agent_runs.jsonl` and as a readable table in `outputs/policy_eval_answers.csv`.
