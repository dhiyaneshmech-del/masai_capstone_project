"""Run the seven SQL analyses and pandas/statistical cross-checks."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "data" / "instamart.db"
QUERIES_FILE = ROOT / "sql" / "queries.sql"
REPORT = ROOT / "reports" / "part_a_results.md"


def load_queries() -> dict[str, str]:
    text = QUERIES_FILE.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^-- (Q[1-7])\b.*$", text))
    queries: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        statement = text[match.end() : end].strip()
        if statement.endswith(";"):
            statement = statement[:-1]
        queries[match.group(1)] = statement
    if set(queries) != {f"Q{i}" for i in range(1, 8)}:
        raise ValueError("sql/queries.sql must contain exactly Q1 through Q7.")
    return queries


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for row in frame.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if pd.isna(value):
                formatted.append("")
            elif isinstance(value, float):
                formatted.append(f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    with sqlite3.connect(DATABASE) as connection:
        queries = load_queries()
        results = {
            name: pd.read_sql_query(statement, connection)
            for name, statement in queries.items()
        }
        orders = pd.read_sql_query("SELECT * FROM orders", connection)
        customers = pd.read_sql_query("SELECT * FROM customers", connection)
        products = pd.read_sql_query("SELECT * FROM products", connection)

    completed = orders.loc[orders["status"].eq("completed")].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])
    pivot = pd.pivot_table(
        completed.merge(customers[["customer_id", "city"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id"),
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
    )
    city_summary = completed.merge(
        customers[["customer_id", "city"]], on="customer_id"
    ).groupby("city").agg(
        completed_revenue=("order_amount", "sum"),
        mean_order_amount=("order_amount", "mean"),
        completed_order_count=("order_amount", "count"),
    )
    descriptive = completed["order_amount"].describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    q1, q3 = descriptive["25%"], descriptive["75%"]
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    below_fence = int(completed["order_amount"].lt(lower_fence).sum())
    above_fence = int(completed["order_amount"].gt(upper_fence).sum())
    correlations = completed[
        ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    ].corr()
    pair_values = [
        (abs(float(correlations.loc[left, right])), left, right, float(correlations.loc[left, right]))
        for i, left in enumerate(correlations.columns)
        for right in correlations.columns[i + 1 :]
    ]
    _, corr_left, corr_right, corr_value = max(pair_values)

    sql_lines: list[str] = []
    for name, result in results.items():
        sql_lines.extend(
            [
                f"### {name}",
                "",
                f"- Rows returned: {len(result):,}",
                "",
                markdown_table(result),
                "",
            ]
        )
    stats = [
        ("Mean", float(completed["order_amount"].mean())),
        ("Median", float(completed["order_amount"].median())),
        ("Skewness", float(completed["order_amount"].skew())),
        ("Q1", float(q1)),
        ("Q2 (median)", float(descriptive["50%"])),
        ("Q3", float(q3)),
        ("Lower 1.5×IQR fence", float(lower_fence)),
        ("Upper 1.5×IQR fence", float(upper_fence)),
        ("Outliers below lower fence", below_fence),
        ("Outliers above upper fence", above_fence),
    ]
    stats_markdown = "\n".join(
        f"| {label} | {value:,.2f} |" for label, value in stats
    )

    report = [
        "# Part A — SQL and pandas results",
        "",
        "SQL results below were run from `sql/queries.sql` with `pd.read_sql_query` against `data/instamart.db`.",
        "",
        *sql_lines,
        "## Pandas cross-check",
        "",
        f"- Q2 pivot grand total: **INR {pivot.loc['All', 'All']:,.2f}**.",
        f"- Completed SQL Q2 revenue total: **INR {results['Q2']['revenue_inr'].sum():,.2f}**.",
        "",
        "### Completed order amount by city",
        "",
        "| City | Sum (INR) | Mean (INR) | Count |",
        "|---|---:|---:|---:|",
        *[
            f"| {city} | {row['completed_revenue']:,.2f} | {row['mean_order_amount']:,.2f} | {int(row['completed_order_count']):,} |"
            for city, row in city_summary.sort_index().iterrows()
        ],
        "",
        "### Q2 pandas pivot table (INR)",
        "",
        markdown_table(pivot.reset_index()),
        "",
        "### Completed order amount statistics",
        "",
        "| Statistic | Result |",
        "|---|---:|",
        stats_markdown,
        "",
        "### Correlation matrix",
        "",
        markdown_table(correlations.reset_index()),
        "",
        f"The largest absolute correlation is `{corr_left}` with `{corr_right}` ({corr_value:.3f}); a correlation of this size does not establish causation.",
        "",
        "### Exploratory summary",
        "",
        "The data describe Instamart customers, products and orders across six Indian cities, with order timestamps, statuses, discounts, quantities, amounts and delivery times. Completed order amounts are strongly right-skewed: the mean is INR 269.91, compared with a median of INR 135.00, and skewness is about 4.61. The middle half spans INR 64.80 to INR 297.00. The upper 1.5×IQR fence is INR 645.30; 897 amounts lie above it and none lie below the lower fence. These high amounts are outliers by the IQR rule, not necessarily errors. Amount has the strongest absolute pairwise correlation with discount percentage (about −0.095), slightly larger than its correlation with quantity (about +0.087); a correlation of this size does not establish causation.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote analysis report to {REPORT}")
    for name, result in results.items():
        print(f"{name}: {len(result):,} rows")
    print(
        f"Q2 completed revenue: INR {results['Q2']['revenue_inr'].sum():,.2f}"
    )
    print(
        f"Q7 final cumulative revenue: INR {results['Q7']['cumulative_revenue'].iloc[-1]:,.2f}"
    )
    print(
        f"Correlation max: {corr_left} / {corr_right} = {corr_value:.3f}; "
        f"amount outliers: {below_fence} below, {above_fence} above"
    )


if __name__ == "__main__":
    main()
