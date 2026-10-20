# Part B — Customer segments and lapse prediction

Reproduce with `python customer_segments_and_lapse.py` after completing Part A. All monetary values are INR.

## RFM summary

- Customers with at least one completed order: **1,175**.
- Recency: median 19 days, minimum 0, maximum 180.
- Frequency: mean 9.46, maximum 33.
- Monetary: mean INR 2,552.06, maximum INR 14,766.55.

## K-means selection

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 was selected from 3–5: it aligns with the elbow and scores best by silhouette among these choices, while producing a manageable set of campaign audiences.
Inertia cannot increase as K grows because adding centroids can only leave points equally close or assign them to a closer centroid, so K=8 will have lower inertia than K=2. But inertia always rewards adding clusters and does not penalize model complexity, so its decrease alone cannot identify a useful or interpretable K.

## Segment profile

| Segment ID | Segment name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---:|---|---:|---:|---:|---:|:---:|---|
| 0 | Steady Customers | 436 | 28.86 | 3.77 | 1,057.62 | No | Use personalized cross-sell recommendations to grow basket value. |
| 1 | Premium High-Value | 138 | 24.12 | 6.77 | 7,129.70 | Yes | Offer concierge support and early access to relevant products. |
| 2 | Frequent Loyalists | 419 | 8.77 | 19.40 | 3,410.02 | No | Reward repeat orders with a loyalty benefit or subscription offer. |
| 3 | At-Risk | 182 | 109.04 | 2.20 | 685.99 | No | Launch a time-limited win-back offer and follow up before the next cycle. |

Cluster numbers are arbitrary labels, not an ordered customer rating. Premium is assigned to the segment with the highest mean monetary value.

PCA explained variance ratios on scaled RFM: **0.634**, **0.218** (total **0.852**).

## Lapse cohort and split

- Cohort customers: **1,156**; lapsed: **443 (38.32%)**.
- Training rows: **924**; test rows: **232**.
- Always predict not lapsed baseline accuracy: **0.616**.
- Confusion matrix (rows actual 0/1, columns predicted 0/1): `[[143, 0], [89, 0]]`
- Lapsed-class precision: **0.000**
- Lapsed-class recall: **0.000**
- Lapsed-class F1: **0.000**

The training-fitted frequency scaler mean is **8.115**, versus the full-cohort frequency mean of **8.029**; the training mean differs because it is calculated on the training subset rather than the full cohort.
A code assertion verifies that all feature columns are complete before splitting. All numeric scaling and categorical encoding are inside a pipeline fitted on training rows only; no encoder, scaler or resampler sees test rows.
No orders dated 2025-09-01 or later enter the lapse features, and the whole-window segment labels from Task 3 are not model features.

## Grid search and untouched-test evaluation

- Logistic regression grid: `C ∈ {0.01, 0.1, 1, 10, 100}`; best parameters **{'model__C': 100}**, best 5-fold CV F1 **0.651**.
- Random forest grid: `max_depth ∈ {3, 5, 8}`, `min_samples_leaf ∈ {5, 10, 20}` (45 fits); best parameters **{'model__max_depth': 8, 'model__min_samples_leaf': 10}**, best 5-fold CV F1 **0.683**.
- Selected model by best CV F1: **Random Forest**; test F1 **0.659**.

### Tuned logistic regression

- Confusion matrix (rows actual 0/1, columns predicted 0/1): `[[111, 32], [31, 58]]`
- Lapsed-class precision: **0.644**
- Lapsed-class recall: **0.652**
- Lapsed-class F1: **0.648**

### Tuned random forest

- Confusion matrix (rows actual 0/1, columns predicted 0/1): `[[114, 29], [31, 58]]`
- Lapsed-class precision: **0.667**
- Lapsed-class recall: **0.652**
- Lapsed-class F1: **0.659**

Random Forest was selected based on cross-validated F1, then evaluated once on the untouched test set. Missing a customer who is about to lapse can forfeit future orders and the chance to intervene early. Wrongly targeting a loyal customer wastes campaign spend and can make promotions feel unnecessary. Compare both costs when setting the outreach threshold; the default 0.5 cutoff reflects neither cost explicitly.

## Training-only class-balance experiment

- Balanced training rows: **1,140** (570 not lapsed; 570 lapsed). All original training rows were retained and only the required number of lapsed training examples were duplicated with replacement.
- Original logistic model: precision 0.644, recall 0.652, F1 0.648.
- Balanced logistic model: precision 0.589, recall 0.854, F1 0.697.
Recall moved most: oversampling makes the model more likely to flag lapsing customers, which can catch more customers for retention but may increase false-positive outreach; assess precision alongside it.

## Interpretation and saved artifacts

| Random forest feature | Importance |
|---|---:|
| `numeric__frequency` | 0.2808 |
| `numeric__monetary` | 0.2696 |
| `numeric__recency` | 0.1136 |
| `numeric__average_delivery_minutes` | 0.0618 |
| `numeric__cancellation_rate` | 0.0526 |

| Logistic regression feature | Coefficient | Sign | Business reading |
|---|---:|:---:|---|
| `numeric__frequency` | -1.6977 | negative | Higher values of this feature are associated with lower predicted lapse odds, holding other encoded features constant. |
| `categorical__favourite_category_Electronics` | -1.1353 | negative | Higher values of this feature are associated with lower predicted lapse odds, holding other encoded features constant. |
| `categorical__favourite_category_Grocery` | -0.4494 | negative | Higher values of this feature are associated with lower predicted lapse odds, holding other encoded features constant. |

Saved selected model at `outputs/lapse_model.joblib`; reload predictions exactly match original test predictions: **True**.
Saved the top 10 Premium test-set customers by predicted lapse probability to `outputs/retention_list.csv`; **2** actually lapsed.
The Premium flag is a post-hoc whole-window label that includes the September label period, so this retention list is illustrative and not a leakage-free evaluation.
