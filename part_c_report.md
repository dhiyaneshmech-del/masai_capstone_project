# Part C — Support ticket triage

Run `python ticket_triage.py` after Part B. Tickets are processed in file order. The supplied `mock_llm.py` is offline and all prompts use redacted text only.

## Input guards and redaction

- Empty/whitespace tickets skipped without a model call: **4**.
- Tickets above 400 estimated input tokens flagged without a model call: **4**.
- Email substitutions: **55**; phone substitutions: **91**; order-ID substitutions: **132**; total: **278**.
- Reapplying all three redaction patterns found **0** matches.

The zero-shot and few-shot system prompts both include `CLASSIFY`, the four allowed categories, and the required JSON-only schema. Only the redacted ticket text is sent as the user-message content.

## Retry behavior

Each run resets the mock immediately before processing. Retry logs are saved in `outputs/triage_retry_log.txt`; total retries across the four independent runs: **67** (zero-shot: 19, few-shot: 19, budget-demo: 10, final-few-shot: 19). There were no permanent model-call errors.

## Zero-shot and few-shot comparison

| Prompt | Classified | Accuracy | Correct / classified | Total tokens | Tokens per classified ticket | Status counts |
|---|---:|---:|---:|---:|---:|---|
| Zero-shot | 147 | 85.7% | 126 / 147 | 11,428 | 77.74 | {'classified_json': 135, 'skipped_empty': 4, 'flagged_long': 4, 'classified_keyword': 12, 'flagged_invalid_output': 9} |
| Few-shot | 147 | 100.0% | 147 / 147 | 20,632 | 140.35 | {'classified_json': 135, 'skipped_empty': 4, 'flagged_long': 4, 'classified_keyword': 12, 'flagged_invalid_output': 9} |

### Zero-shot confusion table

| Actual / Predicted | billing | delivery | technical | fraud |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

Few-shot used 9,204 additional tokens total (62.61 additional tokens per classified ticket) and improved accuracy from 85.7% to 100.0%. The extra prompt cost is worthwhile when avoiding a misrouted fraud or technical ticket has higher operational cost than the incremental token spend. When review capacity or token budgets are tight and lower accuracy is acceptable, the zero-shot prompt is cheaper; the measured value of few-shot depends on the cost of the errors it prevents.

## Token-budget demonstration

The zero-shot demonstration used a budget of **6,000 tokens** and marked **74** tickets `budget_exhausted` without calling the model.

## Final few-shot routing

The final few-shot run used a 100,000-token budget and wrote all **164** ticket rows to `outputs/triage_results.csv`. JSON/keyword classifications are rule-validated; empty, long, invalid-output and budget-exhausted tickets are routed to human review. Fraud routes to the security team; remaining valid classifications go to the Premium desk or support queue based on the Part B segment flag.

Final status counts: `{'classified_json': 135, 'skipped_empty': 4, 'flagged_long': 4, 'classified_keyword': 12, 'flagged_invalid_output': 9}`. Human-review routes: **17**; rule-validated classifications: **147**.
