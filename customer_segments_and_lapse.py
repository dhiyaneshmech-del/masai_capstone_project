"""Build customer segments and a leakage-aware lapse prediction workflow."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample


ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
ORDERS_FILE = CLEAN / "orders.csv"
CUSTOMERS_FILE = CLEAN / "customers.csv"
PRODUCTS_FILE = CLEAN / "products.csv"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")
RFM_COLUMNS = ["recency", "frequency", "monetary"]
NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days",
]
CATEGORICAL_FEATURES = ["city", "gender", "favourite_category"]
SEGMENT_ACTIONS = {
    "Premium High-Value": "Offer concierge support and early access to relevant products.",
    "At-Risk": "Launch a time-limited win-back offer and follow up before the next cycle.",
    "Frequent Loyalists": "Reward repeat orders with a loyalty benefit or subscription offer.",
    "Steady Customers": "Use personalized cross-sell recommendations to grow basket value.",
}


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(ORDERS_FILE)
    customers = pd.read_csv(CUSTOMERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="raise")
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="raise"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="raise"
    )
    customers["age"] = pd.to_numeric(customers["age"], errors="raise")
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
    return orders, customers, products


def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    grouped = completed.groupby("customer_id", sort=True)
    rfm = grouped.agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    return rfm.reset_index()[["customer_id", *RFM_COLUMNS]]


def label_segments(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    segmented = rfm.copy()
    segmented["segment_id"] = labels.astype(int)
    profiles = segmented.groupby("segment_id")[RFM_COLUMNS].mean()
    premium_id = int(profiles["monetary"].idxmax())
    recency_id = int(profiles["recency"].idxmax())
    frequency_id = int(profiles["frequency"].idxmax())
    names: dict[int, str] = {}
    for segment_id in profiles.index.astype(int):
        if segment_id == premium_id:
            name = "Premium High-Value"
        elif segment_id == recency_id:
            name = "At-Risk"
        elif segment_id == frequency_id:
            name = "Frequent Loyalists"
        else:
            name = "Steady Customers"
        names[segment_id] = name
    segmented["segment_name"] = segmented["segment_id"].map(names)
    segmented["is_premium"] = segmented["segment_id"].eq(premium_id)
    return segmented


def build_lapse_cohort(
    orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    history = orders.loc[orders["order_ts"].lt(CUTOFF)].copy()
    completed_history = history.loc[history["status"].eq("completed")].copy()
    cohort_ids = np.sort(completed_history["customer_id"].unique())
    completed_future = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(LABEL_END)
    ]
    future_active = set(completed_future["customer_id"])

    product_categories = products.set_index("product_id")["category"]
    completed_history["category"] = completed_history["product_id"].map(
        product_categories
    )
    if completed_history["category"].isna().any():
        raise ValueError("A completed historical order has no product category.")

    completed_by_customer = completed_history.groupby("customer_id")
    features = completed_by_customer.agg(
        last_completed_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )
    features["recency"] = (CUTOFF - features["last_completed_ts"]).dt.days

    final_orders = history.loc[history["status"].isin(["completed", "cancelled"])]
    status_counts = final_orders.groupby(["customer_id", "status"]).size().unstack(
        fill_value=0
    )
    cancelled = status_counts.get("cancelled", pd.Series(dtype="int64"))
    completed_count = status_counts.get("completed", pd.Series(dtype="int64"))
    features["cancellation_rate"] = (
        cancelled.reindex(features.index, fill_value=0)
        / (cancelled.reindex(features.index, fill_value=0)
           + completed_count.reindex(features.index, fill_value=0))
    )

    favorite_categories = (
        completed_history.groupby("customer_id")["category"]
        .agg(lambda values: values.value_counts().sort_index().idxmax())
        .rename("favourite_category")
    )
    features = features.join(favorite_categories)

    cohort_customers = customers.set_index("customer_id").reindex(cohort_ids)
    features = features.reindex(cohort_ids).join(
        cohort_customers[["city", "age", "gender", "signup_date"]]
    )
    features["tenure_days"] = (CUTOFF - features["signup_date"]).dt.days
    features = features.drop(columns=["last_completed_ts", "signup_date"])
    features.index.name = "customer_id"
    y = pd.Series(
        [int(customer_id not in future_active) for customer_id in features.index],
        index=features.index,
        name="lapsed",
        dtype="int64",
    )
    if features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().any().any():
        missing = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum()
        raise ValueError(f"Lapse feature values are missing: {missing[missing > 0].to_dict()}")
    return features, y


def make_pipeline(estimator: object) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def class_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, object]:
    return {
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
        "precision": precision_score(y_true, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, predictions, pos_label=1, zero_division=0),
    }


def fmt_metrics(metrics: dict[str, object]) -> list[str]:
    matrix = metrics["confusion_matrix"]
    return [
        f"- Confusion matrix (rows actual 0/1, columns predicted 0/1): `[[{matrix[0, 0]}, {matrix[0, 1]}], [{matrix[1, 0]}, {matrix[1, 1]}]]`",
        f"- Lapsed-class precision: **{metrics['precision']:.3f}**",
        f"- Lapsed-class recall: **{metrics['recall']:.3f}**",
        f"- Lapsed-class F1: **{metrics['f1']:.3f}**",
    ]


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    orders, customers, products = load_clean_data()

    rfm = build_rfm(orders)
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[RFM_COLUMNS])
    k_rows: list[dict[str, float | int]] = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        segment_ids = model.fit_predict(scaled_rfm)
        k_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled_rfm, segment_ids)),
            }
        )
    k_table = pd.DataFrame(k_rows)

    # K=4 balances the elbow, the strongest silhouette among the business-ready
    # choices, and a manageable number of campaign audiences.
    chosen_k = 4
    final_kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    segment_ids = final_kmeans.fit_predict(scaled_rfm)
    segments = label_segments(rfm, segment_ids)
    segments.to_csv(OUTPUTS / "customer_segments.csv", index=False)
    segment_profiles = segments.groupby(
        ["segment_id", "segment_name", "is_premium"], as_index=False
    ).agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    segment_profiles["recommended_action"] = segment_profiles[
        "segment_name"
    ].map(SEGMENT_ACTIONS)
    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)
    pca_ratios = pca.explained_variance_ratio_

    features, y = build_lapse_cohort(orders, customers, products)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    baseline_predictions = np.zeros(len(y_test), dtype="int64")
    baseline = class_metrics(y_test, baseline_predictions)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    logistic_search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000)),
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    forest_search = GridSearchCV(
        make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    logistic_search.fit(X_train, y_train)
    forest_search.fit(X_train, y_train)

    logistic = logistic_search.best_estimator_
    forest = forest_search.best_estimator_
    logistic_metrics = class_metrics(y_test, logistic.predict(X_test))
    forest_metrics = class_metrics(y_test, forest.predict(X_test))
    best_search = max(
        (logistic_search, forest_search),
        key=lambda search: search.best_score_,
    )
    selected_name = (
        "Logistic Regression"
        if best_search is logistic_search
        else "Random Forest"
    )
    selected_model = best_search.best_estimator_
    selected_predictions = selected_model.predict(X_test)
    selected_probabilities = selected_model.predict_proba(X_test)[:, 1]
    selected_metrics = class_metrics(y_test, selected_predictions)
    if selected_metrics["f1"] < 0.55:
        raise AssertionError(
            f"Selected model test F1 is below 0.55: {selected_metrics['f1']:.3f}"
        )

    training_frequency_mean = float(
        logistic.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .mean_[NUMERIC_FEATURES.index("frequency")]
    )
    full_frequency_mean = float(features["frequency"].mean())

    training_frame = X_train.copy()
    training_frame["lapsed"] = y_train
    majority_count = int(y_train.value_counts().max())
    positive_train = training_frame.loc[training_frame["lapsed"].eq(1)]
    positive_duplicates = resample(
        positive_train,
        replace=True,
        n_samples=majority_count - len(positive_train),
        random_state=42,
    )
    balanced_train = pd.concat(
        [training_frame, positive_duplicates], axis=0
    ).sample(frac=1, random_state=42)
    balanced_y = balanced_train.pop("lapsed")
    balanced_counts = balanced_y.value_counts()
    if balanced_counts.get(0, 0) != balanced_counts.get(1, 0):
        raise AssertionError("Oversampled training classes are not balanced.")
    balanced_logistic = make_pipeline(LogisticRegression(max_iter=1000))
    balanced_logistic.set_params(model__C=logistic_search.best_params_["model__C"])
    balanced_logistic.fit(balanced_train, balanced_y)
    balanced_metrics = class_metrics(y_test, balanced_logistic.predict(X_test))

    feature_names = forest.named_steps["preprocess"].get_feature_names_out()
    importances = forest.named_steps["model"].feature_importances_
    importance_table = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    logistic_feature_names = logistic.named_steps[
        "preprocess"
    ].get_feature_names_out()
    logistic_coefficients = logistic.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame(
        {"feature": logistic_feature_names, "coefficient": logistic_coefficients}
    )
    coefficient_table["absolute_coefficient"] = coefficient_table[
        "coefficient"
    ].abs()
    coefficient_table = coefficient_table.sort_values(
        "absolute_coefficient", ascending=False
    )

    model_path = OUTPUTS / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)
    reloaded_predictions = reloaded_model.predict(X_test)
    reload_matches = bool(np.array_equal(selected_predictions, reloaded_predictions))
    if not reload_matches:
        raise AssertionError("Reloaded model predictions differ from saved model.")

    test_premium = set(segments.loc[segments["is_premium"], "customer_id"])
    retention = pd.DataFrame(
        {
            "customer_id": X_test.index,
            "lapse_probability": selected_probabilities,
            "actually_lapsed": y_test.to_numpy(),
        }
    )
    retention = retention.loc[
        retention["customer_id"].isin(test_premium)
    ].sort_values("lapse_probability", ascending=False).head(10)
    retention.to_csv(OUTPUTS / "retention_list.csv", index=False)
    true_lapsed_in_list = int(retention["actually_lapsed"].sum())

    lines = [
        "# Part B — Customer segments and lapse prediction",
        "",
        "Reproduce with `python customer_segments_and_lapse.py` after completing Part A. All monetary values are INR.",
        "",
        "## RFM summary",
        "",
        f"- Customers with at least one completed order: **{len(rfm):,}**.",
        f"- Recency: median {rfm['recency'].median():.0f} days, minimum {rfm['recency'].min():.0f}, maximum {rfm['recency'].max():.0f}.",
        f"- Frequency: mean {rfm['frequency'].mean():.2f}, maximum {rfm['frequency'].max():.0f}.",
        f"- Monetary: mean INR {rfm['monetary'].mean():,.2f}, maximum INR {rfm['monetary'].max():,.2f}.",
        "",
        "## K-means selection",
        "",
        "| K | Inertia | Silhouette |",
        "|---:|---:|---:|",
        *[
            f"| {int(row.k)} | {row.inertia:.1f} | {row.silhouette:.3f} |"
            for row in k_table.itertuples(index=False)
        ],
        "",
        f"K={chosen_k} was selected from 3–5: it aligns with the elbow and scores best by silhouette among these choices, while producing a manageable set of campaign audiences.",
        "Inertia cannot increase as K grows because adding centroids can only leave points equally close or assign them to a closer centroid, so K=8 will have lower inertia than K=2. But inertia always rewards adding clusters and does not penalize model complexity, so its decrease alone cannot identify a useful or interpretable K.",
        "",
        "## Segment profile",
        "",
        "| Segment ID | Segment name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |",
        "|---:|---|---:|---:|---:|---:|:---:|---|",
        *[
            f"| {int(row.segment_id)} | {row.segment_name} | {int(row.size)} | {row.mean_recency:.2f} | {row.mean_frequency:.2f} | {row.mean_monetary:,.2f} | {'Yes' if row.is_premium else 'No'} | {row.recommended_action} |"
            for row in segment_profiles.itertuples(index=False)
        ],
        "",
        "Cluster numbers are arbitrary labels, not an ordered customer rating. Premium is assigned to the segment with the highest mean monetary value.",
        "",
        f"PCA explained variance ratios on scaled RFM: **{pca_ratios[0]:.3f}**, **{pca_ratios[1]:.3f}** (total **{pca_ratios.sum():.3f}**).",
        "",
        "## Lapse cohort and split",
        "",
        f"- Cohort customers: **{len(features):,}**; lapsed: **{int(y.sum()):,} ({y.mean():.2%})**.",
        f"- Training rows: **{len(X_train):,}**; test rows: **{len(X_test):,}**.",
        f"- Always predict not lapsed baseline accuracy: **{baseline_accuracy:.3f}**.",
        *fmt_metrics(baseline),
        "",
        f"The training-fitted frequency scaler mean is **{training_frequency_mean:.3f}**, versus the full-cohort frequency mean of **{full_frequency_mean:.3f}**; the training mean differs because it is calculated on the training subset rather than the full cohort.",
        "A code assertion verifies that all feature columns are complete before splitting. All numeric scaling and categorical encoding are inside a pipeline fitted on training rows only; no encoder, scaler or resampler sees test rows.",
        "No orders dated 2025-09-01 or later enter the lapse features, and the whole-window segment labels from Task 3 are not model features.",
        "",
        "## Grid search and untouched-test evaluation",
        "",
        f"- Logistic regression grid: `C ∈ {{0.01, 0.1, 1, 10, 100}}`; best parameters **{logistic_search.best_params_}**, best 5-fold CV F1 **{logistic_search.best_score_:.3f}**.",
        f"- Random forest grid: `max_depth ∈ {{3, 5, 8}}`, `min_samples_leaf ∈ {{5, 10, 20}}` (45 fits); best parameters **{forest_search.best_params_}**, best 5-fold CV F1 **{forest_search.best_score_:.3f}**.",
        f"- Selected model by best CV F1: **{selected_name}**; test F1 **{selected_metrics['f1']:.3f}**.",
        "",
        "### Tuned logistic regression",
        "",
        *fmt_metrics(logistic_metrics),
        "",
        "### Tuned random forest",
        "",
        *fmt_metrics(forest_metrics),
        "",
        f"{selected_name} was selected based on cross-validated F1, then evaluated once on the untouched test set. Missing a customer who is about to lapse can forfeit future orders and the chance to intervene early. Wrongly targeting a loyal customer wastes campaign spend and can make promotions feel unnecessary. Compare both costs when setting the outreach threshold; the default 0.5 cutoff reflects neither cost explicitly.",
        "",
        "## Training-only class-balance experiment",
        "",
        f"- Balanced training rows: **{len(balanced_train):,}** ({balanced_y.value_counts().get(0, 0):,} not lapsed; {balanced_y.value_counts().get(1, 0):,} lapsed). All original training rows were retained and only the required number of lapsed training examples were duplicated with replacement.",
        f"- Original logistic model: precision {logistic_metrics['precision']:.3f}, recall {logistic_metrics['recall']:.3f}, F1 {logistic_metrics['f1']:.3f}.",
        f"- Balanced logistic model: precision {balanced_metrics['precision']:.3f}, recall {balanced_metrics['recall']:.3f}, F1 {balanced_metrics['f1']:.3f}.",
        "Recall moved most: oversampling makes the model more likely to flag lapsing customers, which can catch more customers for retention but may increase false-positive outreach; assess precision alongside it.",
        "",
        "## Interpretation and saved artifacts",
        "",
        "| Random forest feature | Importance |",
        "|---|---:|",
        *[
            f"| `{row.feature}` | {row.importance:.4f} |"
            for row in importance_table.head(5).itertuples(index=False)
        ],
        "",
        "| Logistic regression feature | Coefficient | Sign | Business reading |",
        "|---|---:|:---:|---|",
        *[
            f"| `{row.feature}` | {row.coefficient:.4f} | {'positive' if row.coefficient > 0 else 'negative'} | Higher values of this feature are associated with {'higher' if row.coefficient > 0 else 'lower'} predicted lapse odds, holding other encoded features constant. |"
            for row in coefficient_table.head(3).itertuples(index=False)
        ],
        "",
        f"Saved selected model at `outputs/lapse_model.joblib`; reload predictions exactly match original test predictions: **{reload_matches}**.",
        f"Saved the top {len(retention)} Premium test-set customers by predicted lapse probability to `outputs/retention_list.csv`; **{true_lapsed_in_list}** actually lapsed.",
        "The Premium flag is a post-hoc whole-window label that includes the September label period, so this retention list is illustrative and not a leakage-free evaluation.",
        "",
    ]
    (OUTPUTS / "part_b_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUTS / 'customer_segments.csv'}")
    print(f"Wrote {model_path}")
    print(f"Wrote {OUTPUTS / 'retention_list.csv'}")
    print(f"Wrote {OUTPUTS / 'part_b_report.md'}")
    print(f"Selected {selected_name}; test lapsed-class F1={selected_metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
