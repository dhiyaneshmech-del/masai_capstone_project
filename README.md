# Overview

This capstone builds a reproducible Instamart customer-intelligence workflow across four parts: dirty-data auditing and SQLite storage (Part A), customer segmentation and lapse prediction (Part B), privacy-aware support-ticket triage (Part C), and policy retrieval with a tool-using refund agent (Part D). The supplied data are synthetic and fictional; all monetary amounts are INR. Start from the repository's `main` branch. Raw exports in `data/raw/` are the source of truth for the downstream outputs.

# Reproduce

Run the following commands from the repository root in order. The sequence starts with the isolated environment and ends with the Part D script.

1. Create and activate the environment:
   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
   ```
2. Install pinned dependencies (the pins are recorded from `pip freeze`):
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Regenerate the synthetic raw data and policy fixtures:
   ```powershell
   python generate_data.py
   ```
4. Audit and clean the raw tables, creating CSV and SQLite outputs:
   ```powershell
   python clean_data.py
   ```
5. Validate Part A:
   ```powershell
   python validate_data.py
   ```
   Successful validation prints exactly `ALL CHECKPOINTS PASSED`.
6. Run Part A SQL and pandas analyses:
   ```powershell
   python analyze_data.py
   ```
7. Run Part B:
   ```powershell
   python customer_segments_and_lapse.py
   ```
8. Run Part C:
   ```powershell
   python ticket_triage.py
   ```
9. Run Part D:
   ```powershell
   python policy_rag_refund_agent.py
   ```

# Part A — Data foundation

## Raw-data audit

The three business tables and support tickets were loaded from `data/raw/`; `tickets.csv` uses `keep_default_na=False` so empty ticket text remains an empty string. Raw missing-value counts below are pandas `isna()` counts; the separate non-numeric amount count also includes nonblank text such as `N/A` and `error`.

| Table | Shape | Column | Raw dtype | Missing count | Missing % |
|---|---:|---|---|---:|---:|
| orders | 13,084 × 9 | order_id | str | 0 | 0.00% |
| orders | 13,084 × 9 | customer_id | str | 0 | 0.00% |
| orders | 13,084 × 9 | product_id | str | 0 | 0.00% |
| orders | 13,084 × 9 | quantity | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_ts | str | 0 | 0.00% |
| orders | 13,084 × 9 | status | str | 0 | 0.00% |
| orders | 13,084 × 9 | discount_pct | int64 | 0 | 0.00% |
| orders | 13,084 × 9 | order_amount | str | 53 | 0.41% |
| orders | 13,084 × 9 | delivery_minutes | float64 | 2,541 | 19.42% |
| customers | 1,200 × 5 | customer_id | str | 0 | 0.00% |
| customers | 1,200 × 5 | city | str | 0 | 0.00% |
| customers | 1,200 × 5 | signup_date | str | 0 | 0.00% |
| customers | 1,200 × 5 | age | float64 | 61 | 5.08% |
| customers | 1,200 × 5 | gender | str | 30 | 2.50% |
| products | 40 × 4 | product_id | str | 0 | 0.00% |
| products | 40 × 4 | product_name | str | 0 | 0.00% |
| products | 40 × 4 | category | str | 0 | 0.00% |
| products | 40 × 4 | price_inr | int64 | 0 | 0.00% |
| tickets | 164 × 5 | ticket_id | str | 0 | 0.00% |
| tickets | 164 × 5 | customer_id | str | 0 | 0.00% |
| tickets | 164 × 5 | order_id | str | 0 | 0.00% |
| tickets | 164 × 5 | text | str | 0 | 0.00% |
| tickets | 164 × 5 | true_category | str | 0 | 0.00% |

| Additional raw audit measure | Result |
|---|---:|
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery minutes | 106 |
| Blank delivery minutes | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |

## Cleaning rules and outputs

- **Customers:** trim and title-case city values; 476 values changed and 24 raw spellings became six cities. Normalize gender spellings to `male`, `female`, or `unknown`; 950 values changed, yielding 610 female, 560 male, and 30 unknown. Convert six impossible ages to missing, then impute 67 values (61 original blanks plus six invalid) with each city's median using `groupby` and `transform`; zero ages remain missing and all are 18–60. Convert 1,200 signup dates to datetimes (CSV format `YYYY-MM-DD`).
- **Orders:** convert 13,084 timestamps to datetimes; remove 119 exact duplicate rows; keep the latest timestamp for each order, removing 656 older records. Convert amounts to numeric and impute 112 remaining unparsable amounts as `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. Set 105 negative delivery times to missing; impute 774 missing completed-order times from the completed median of each product category with `groupby` and `transform`. Leave all 1,199 cancelled delivery times missing.
- **Clean checkpoints:** 12,309 unique orders: 11,110 completed and 1,199 cancelled; no pending orders, no missing amounts, no missing completed delivery times. Completed revenue is INR 2,998,668.80. The SQLite database `data/instamart.db` contains `customers`, `products`, and `orders`; timestamps are text in `YYYY-MM-DD HH:MM:SS` format. Clean CSVs are in `data/clean/`.

## SQL results

All seven queries are in `sql/queries.sql` and were run with `pd.read_sql_query` against `data/instamart.db`.

| Query | Result |
|---|---|
| Q1 — WHERE and HAVING | 30 customers with completed orders over INR 500 and qualifying total over INR 8,000. |
| Q2 — three-table join | 30 city/category rows; revenue total INR 2,998,668.80. Largest cell: Bengaluru / Electronics, INR 233,909.45. |
| Q3 — scalar subquery | 11 products are priced above the catalogue average. |
| Q4 — LEFT JOIN | 25 customers have no completed order. |
| Q5 — two CTEs | 550 customers exceed average completed spend; average is INR 2,552.06. |
| Q6 — ranking windows | Top spenders: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Eight rows have `RANK()` different from `ROW_NUMBER()` because ties share rank but each row has a distinct row number. |
| Q7 — running total | 183 daily rows; last cumulative completed revenue is INR 2,998,668.80. |

## Pandas and exploratory results

The Q2 `pivot_table(..., margins=True)` grand total is INR 2,998,668.80, matching SQL.

| City | Completed amount sum (INR) | Mean (INR) | Count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

| Completed amount statistic | Value |
|---|---:|
| Mean | INR 269.91 |
| Median / Q2 | INR 135.00 |
| Skewness | 4.61 |
| Q1 | INR 64.80 |
| Q3 | INR 297.00 |
| Lower 1.5×IQR fence | INR -283.50 |
| Upper 1.5×IQR fence | INR 645.30 |
| Outliers below / above fences | 0 / 897 |

Amounts are strongly right-skewed, with a long upper tail; IQR outliers are not necessarily errors. The largest absolute correlation is order amount versus discount percentage at −0.095 (amount versus quantity is +0.087); a correlation of this size does not establish causation.

# Part B — Customer segments and lapse prediction

## RFM and clustering

RFM uses completed orders through the snapshot `2025-10-01 00:00:00`.

| RFM measure | Result |
|---|---:|
| Customers with completed orders | 1,175 |
| Recency median / minimum / maximum | 19 / 0 / 180 days |
| Frequency mean / maximum | 9.46 / 33 |
| Monetary mean / maximum | INR 2,552.06 / INR 14,766.55 |

The three RFM variables were standardized before K-means. K-means used `n_init=10, random_state=42` for each K.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 was chosen from {3, 4, 5}: it aligns with the elbow, has the best silhouette in that candidate set, and yields four practical campaign groups. Inertia at K=8 must be lower than at K=2 because more centroids can only reduce or preserve each point's distance to its nearest centroid. Inertia always favors additional clusters and has no complexity penalty, so that decrease alone does not identify a useful K.

| Segment ID | Business name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---:|---|---:|---:|---:|---:|:---:|---|
| 0 | Steady Customers | 436 | 28.86 | 3.77 | 1,057.62 | No | Personalized cross-sell to grow basket value. |
| 1 | Premium High-Value | 138 | 24.12 | 6.77 | 7,129.70 | Yes | Concierge support and early product access. |
| 2 | Frequent Loyalists | 419 | 8.77 | 19.40 | 3,410.02 | No | Reward repeat orders with a loyalty benefit. |
| 3 | At-Risk | 182 | 109.04 | 2.20 | 685.99 | No | Time-limited win-back offer and proactive follow-up. |

Cluster numbers are arbitrary labels, not an ordered rating; Premium is the segment with the highest mean monetary value. PCA on scaled RFM explains 0.634 and 0.218 of variance (total 0.852). Segment assignments are saved to `outputs/customer_segments.csv`.

## Lapse cohort, preprocessing, and model evaluation

Features use only orders before `2025-09-01`; completed orders from September are used only to define the lapse label. Segment labels were not used as model features.

| Cohort / split measure | Result |
|---|---:|
| Cohort customers | 1,156 |
| Lapsed | 443 (38.32%) |
| Train / test rows | 924 / 232 |
| Baseline accuracy (always not lapsed) | 61.6% |
| Baseline lapsed-class F1 | 0.000 |
| Training-fitted frequency mean | 8.115 |
| Full-cohort frequency mean | 8.029 |

The frequency means differ because one is estimated on training rows only, while the other uses the full cohort. All feature columns were checked for missing values. Encoders and scalers were fit only on training rows within the model pipelines; no encoder, scaler, or resampler saw test rows.

| Model | Grid and best parameters | Best CV F1 | Test confusion matrix (actual rows 0/1; predicted columns 0/1) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C ∈ {0.01, 0.1, 1, 10, 100}; best C=100 | 0.651 | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | 45 fits: depth ∈ {3,5,8}, leaf ∈ {5,10,20}; best depth=8, leaf=10 | 0.683 | [[114, 29], [31, 58]] | 0.667 | 0.652 | 0.659 |

Random Forest was selected by best cross-validated F1 and achieved test F1 0.659 (above 0.55). Missing a lapsing customer can forfeit future orders and an opportunity to intervene; wrongly targeting a loyal customer wastes campaign spend and may irritate them. The chosen operating threshold should reflect those business costs rather than assuming they are equal.

## Class balance, interpretation, and artifacts

Training-only oversampling retained every original training row and added lapsed examples with replacement, producing 570 non-lapsed and 570 lapsed rows. Evaluated on the unchanged test set:

| Logistic regression | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.644 | 0.652 | 0.648 |
| Balanced training data | 0.589 | 0.854 | 0.697 |

Recall moved most, increasing the number of lapsing customers flagged while precision fell; retention outreach should account for the resulting extra false positives.

| Top random-forest feature | Importance |
|---|---:|
| Frequency | 0.2808 |
| Monetary | 0.2696 |
| Recency | 0.1136 |
| Average delivery minutes | 0.0618 |
| Cancellation rate | 0.0526 |

| Logistic coefficient feature | Coefficient sign | Business reading |
|---|---|---|
| Frequency | Negative (-1.6977) | More completed orders are associated with lower lapse odds, holding other features constant. |
| Favourite category: Electronics | Negative (-1.1353) | This encoded category is associated with lower lapse odds than the dropped reference category. |
| Favourite category: Grocery | Negative (-0.4494) | This encoded category is associated with lower lapse odds than the dropped reference category. |

The selected pipeline is saved to `outputs/lapse_model.joblib`; reloaded predictions exactly matched the original test predictions. The top 10 Premium test-set customers are in `outputs/retention_list.csv`; 2 actually lapsed. The Premium flag is a post-hoc whole-window label including the September outcome period, so that retention list is illustrative and not a leakage-free evaluation.

# Part C — Support ticket triage

The triage script reads `tickets.csv` with empty strings preserved and processes tickets in file order. Four empty tickets were skipped and four long tickets (734 estimated tokens each) were flagged, without model calls.

| Redaction type | Substitutions |
|---|---:|
| Emails | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| Total | 278 |
| Matches remaining after redaction | 0 |

Both system prompts contain `CLASSIFY`, the four categories, and the required JSON-only schema; the zero-shot prompt has no `EXAMPLES:` section, while few-shot includes labelled examples including fraud. Each user message contains exactly the redacted ticket text. The zero-shot, few-shot, and final few-shot runs each recovered all 19 transient errors with one retry per failed ticket; the budget-limited run recovered 10 before exhausting its budget. Retry waits were in the required 0.045–0.0575 second range.

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens / classified ticket | Status counts |
|---|---:|---:|---:|---:|---:|---|
| Zero-shot | 147 | 126 | 85.7% | 11,428 | 77.74 | 135 JSON, 12 keyword, 9 invalid, 4 empty, 4 long |
| Few-shot | 147 | 147 | 100.0% | 20,632 | 140.35 | 135 JSON, 12 keyword, 9 invalid, 4 empty, 4 long |

| Zero-shot confusion matrix | Predicted billing | Predicted delivery | Predicted technical | Predicted fraud |
|---|---:|---:|---:|---:|
| Actual billing | 38 | 0 | 0 | 0 |
| Actual delivery | 0 | 36 | 0 | 0 |
| Actual technical | 10 | 0 | 29 | 0 |
| Actual fraud | 11 | 0 | 0 | 23 |

Few-shot used 9,204 additional tokens total (62.61 more per classified ticket) and improved accuracy by 14.3 percentage points; paying this cost is justified when preventing misrouted fraud or technical tickets is valuable. Zero-shot is cheaper where token budgets or latency matter more and the measured error rate is acceptable.

The zero-shot budget demo used 6,000 tokens and stopped 74 tickets as `budget_exhausted`. The final few-shot run used a 100,000-token budget and wrote 164 rows to `outputs/triage_results.csv`: 147 rule-validated classifications and 17 human-review routes (4 empty, 4 long, 9 invalid output). The routing sends fraud to the security team; otherwise Premium customers go to the premium desk, and other valid tickets go to the support queue. Budget demo and retry logs are `outputs/triage_budget_demo.csv` and `outputs/triage_retry_log.txt`.

# Part D — Policy RAG assistant and refund agent

This part uses local TF-IDF retrieval over the five policy documents and `mock_llm.py`; it makes no network calls and requires no API key. The documents produced 15 chunks at 40 words with 10-word overlap: refund 4, delivery 3, escalation 3, payment 3, privacy 2. The top three chunks for `refund policy late delivery` all came from `refund_policy.txt`. TF measures a term's frequency in a chunk; IDF downweights terms common across chunks. TF-IDF is lexical, so phrasings with the same meaning but different words may not match well.

| Chunking | Expected document among top 3 | Expected facts in answers | Unanswerable fallback |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, no overlap | 12/12 | 10/12 | 2/2 |

Without overlap, the full-refund delivery-time question and the payment-failure return-time question failed because their answers crossed chunk boundaries. For all 14 questions, `outputs/agent_runs.jsonl` contains `POLICY_Q01`–`POLICY_Q14` records with answer and source information; 12 were answered and the two unsupported questions received the required fallback.

| Policy question ID | Question | Status |
|---|---|---|
| POLICY_Q01 | What delivery time makes a completed order eligible for a full refund? | ANSWERED |
| POLICY_Q02 | Who must approve a refund above INR 1000? | ANSWERED |
| POLICY_Q03 | How many days do I have to raise a refund request? | ANSWERED |
| POLICY_Q04 | What is the delivery fee on small orders? | ANSWERED |
| POLICY_Q05 | On orders of what value is delivery free? | ANSWERED |
| POLICY_Q06 | Why does the electronics delivery need a one time password? | ANSWERED |
| POLICY_Q07 | Is cash on delivery available for a large order? | ANSWERED |
| POLICY_Q08 | How soon is money returned when a payment fails but is deducted? | ANSWERED |
| POLICY_Q09 | Can wallet balance be refunded as cash? | ANSWERED |
| POLICY_Q10 | When are delivery addresses deleted from active systems? | ANSWERED |
| POLICY_Q11 | How fast must a fraud ticket reach the security team? | ANSWERED |
| POLICY_Q12 | After how many steps must an automated assistant hand over to a human? | ANSWERED |
| POLICY_Q13 | What is the favourite colour of the company mascot? | FALLBACK |
| POLICY_Q14 | Who won the cricket world cup final? | FALLBACK |

| Scenario | Status | Refund amount | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 969 |
| S2 | REFUND | INR 135.00 | 5 | 969 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 956 |
| S4 | DENIED | INR 0.00 | 4 | 675 |
| S5 | DENIED | INR 0.00 | 3 | 404 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 142 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 2,871 |

S1 token-limit demo status: `FALLBACK_TOKEN_LIMIT` (421 tokens when the limit was exceeded). The zero-second timeout demo status: `FALLBACK_TIMEOUT` (0 model calls). The simulated crash/resume tool sequence was `get_order → search_policy → calculate_refund → issue_refund`, exactly once each; it returned REFUND for INR 128.25.

ReAct used 5 model calls and 969 tokens for S1, 3 calls and 404 tokens for S5, and 2 calls and 142 tokens for S6. A plan-and-execute approach could reduce calls and repeated prompt tokens for ordinary refunds by requesting all steps once and replanning only on error. It must still validate each observation: NOT_FOUND or a cancelled order should stop the plan before a refund is attempted. ReAct is preferable here because eligibility and escalation depend on live order and policy observations, making the extra calls worthwhile for payment safety.

Run results and tool observations are saved to `outputs/agent_runs.jsonl`; the 14 policy Q&As are also available in `outputs/policy_eval_answers.csv`. Issued refunds and escalations are recorded in `outputs/refund_ledger.jsonl`, and resumable tool checkpoints are in `state/`.

# Limitations and Next Steps

The data are deterministic and synthetic, so model metrics do not establish production performance. Validate feature behavior, operating thresholds, campaign costs, and the trade-off between missed lapses and unnecessary outreach on prospective real-world data. The Premium flag is derived from whole-window RFM and includes the September period used to define lapse, so the retention list is illustrative and not a leakage-free evaluation. TF-IDF can miss semantically similar queries with different vocabulary; evaluate retrieval on a broader set of real support questions before production use. Refund automation should retain human oversight for high-value refunds and audit its ledger and policy assumptions before any live deployment.
