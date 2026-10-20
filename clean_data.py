"""Audit and clean the supplied Instamart exports into CSV and SQLite tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DATABASE = ROOT / "data" / "instamart.db"
REPORTS = ROOT / "reports"


def read_raw_tables() -> dict[str, pd.DataFrame]:
    """Load the raw tables, preserving empty ticket text as an empty string."""
    return {
        "orders": pd.read_csv(RAW / "orders.csv"),
        "customers": pd.read_csv(RAW / "customers.csv"),
        "products": pd.read_csv(RAW / "products.csv"),
        "tickets": pd.read_csv(RAW / "tickets.csv", keep_default_na=False),
    }


def audit_tables(tables: dict[str, pd.DataFrame]) -> str:
    lines = ["## Raw table audit", ""]
    for name, frame in tables.items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- Shape: {frame.shape[0]} rows × {frame.shape[1]} columns",
                "- Data types:",
                "",
                "| Column | Data type | Missing count | Missing percent |",
                "|---|---|---:|---:|",
            ]
        )
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            percent = missing / len(frame) * 100 if len(frame) else 0.0
            lines.append(
                f"| `{column}` | `{frame[column].dtype}` | {missing} | {percent:.2f}% |"
            )
        lines.append("")

    orders = tables["orders"]
    customers = tables["customers"]
    order_amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(customers["age"], errors="coerce")
    blank_delivery = (
        orders["delivery_minutes"].isna()
        | orders["delivery_minutes"].astype(str).str.strip().eq("")
    ).sum()
    blank_age = (
        customers["age"].isna()
        | customers["age"].astype(str).str.strip().eq("")
    ).sum()
    blank_gender = (
        customers["gender"].isna()
        | customers["gender"].astype(str).str.strip().eq("")
    ).sum()
    gender_nonblank = customers.loc[
        customers["gender"].notna()
        & customers["gender"].astype(str).str.strip().ne(""),
        "gender",
    ]
    audit = [
        ("Raw orders rows", len(orders)),
        ("Exact duplicate order rows", int(orders.duplicated().sum())),
        ("Rows with pending status", int(orders["status"].eq("pending").sum())),
        ("Distinct order_id values", int(orders["order_id"].nunique())),
        ("Non-numeric order_amount entries", int(order_amount.isna().sum())),
        ("Negative delivery_minutes", int(delivery.lt(0).sum())),
        ("Blank delivery_minutes", int(blank_delivery)),
        ("Blank customer ages", int(blank_age)),
        ("Ages outside 10–100", int((age.notna() & ~age.between(10, 100)).sum())),
        ("Distinct raw city spellings", int(customers["city"].nunique())),
        ("Distinct non-blank raw gender spellings", int(gender_nonblank.nunique())),
        ("Blank genders", int(blank_gender)),
    ]
    lines.extend(
        [
            "### Required raw-data figures",
            "",
            "| Measure | Result |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {label} | {value:,} |" for label, value in audit)
    return "\n".join(lines)


def clean_customers(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    customers = raw.copy()
    raw_city = customers["city"].astype(str)
    customers["city"] = raw_city.str.strip().str.title()
    city_changes = int(raw_city.ne(customers["city"]).sum())

    raw_gender = customers["gender"].astype(str)
    normalized_gender = raw_gender.str.strip().str.lower()
    gender_map = {
        "male": "male",
        "m": "male",
        "female": "female",
        "f": "female",
    }
    customers["gender"] = normalized_gender.map(gender_map).fillna("unknown")
    gender_changes = int(raw_gender.ne(customers["gender"]).sum())

    age = pd.to_numeric(customers["age"], errors="coerce")
    invalid_age = age.notna() & ~age.between(10, 100)
    age = age.mask(invalid_age)
    age_missing_before_fill = int(age.isna().sum())
    city_median = age.groupby(customers["city"]).transform("median")
    customers["age"] = age.fillna(city_median)
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="raise"
    )
    changes = {
        "city values standardized": city_changes,
        "gender values standardized": gender_changes,
        "invalid ages converted to missing": int(invalid_age.sum()),
        "ages imputed (original blanks plus invalid ages)": age_missing_before_fill,
        "signup dates converted": len(customers),
    }
    return customers, changes


def clean_orders(
    raw: pd.DataFrame, products: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    orders = raw.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    exact_duplicates = int(orders.duplicated().sum())
    orders = orders.drop_duplicates()
    after_exact_dedup = len(orders)
    orders = orders.sort_values("order_ts").drop_duplicates(
        subset="order_id", keep="last"
    )
    duplicate_order_records = after_exact_dedup - len(orders)

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise")
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="raise"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    price_by_product = products.set_index("product_id")["price_inr"]
    prices = orders["product_id"].map(price_by_product)
    missing_amount = orders["order_amount"].isna()
    orders.loc[missing_amount, "order_amount"] = (
        prices[missing_amount]
        * orders.loc[missing_amount, "quantity"]
        * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)

    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    negative_delivery = orders["delivery_minutes"].lt(0)
    orders.loc[negative_delivery, "delivery_minutes"] = pd.NA
    categories = orders["product_id"].map(
        products.set_index("product_id")["category"]
    )
    completed = orders["status"].eq("completed")
    completed_missing_before_fill = int(
        (completed & orders["delivery_minutes"].isna()).sum()
    )
    orders.loc[completed, "delivery_minutes"] = orders.loc[
        completed, "delivery_minutes"
    ].fillna(
        orders.loc[completed]
        .groupby(categories.loc[completed])["delivery_minutes"]
        .transform("median")
    )
    orders["delivery_minutes"] = orders["delivery_minutes"].round().astype("Int64")
    orders["order_amount"] = orders["order_amount"].round(2)
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    changes = {
        "order timestamps converted": len(raw),
        "exact duplicate rows removed": exact_duplicates,
        "older records removed after latest-per-order selection": duplicate_order_records,
        "unparseable amounts imputed from product price": int(missing_amount.sum()),
        "negative delivery values set missing": int(negative_delivery.sum()),
        "completed delivery values imputed from product median": completed_missing_before_fill,
        "cancelled delivery values left missing": int(
            (orders["status"].eq("cancelled") & orders["delivery_minutes"].isna()).sum()
        ),
    }
    if categories.isna().any():
        raise ValueError("An order product_id has no matching product category.")
    return orders.reset_index(drop=True), changes


def save_tables(
    tables: dict[str, pd.DataFrame],
) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        output = frame.copy()
        if name == "customers":
            output["signup_date"] = output["signup_date"].dt.strftime("%Y-%m-%d")
        output.to_csv(CLEAN / f"{name}.csv", index=False)

    with sqlite3.connect(DATABASE) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, if_exists="replace", index=False)


def main() -> None:
    raw_tables = read_raw_tables()
    audit = audit_tables(raw_tables)
    products = raw_tables["products"].copy()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
    customers, customer_changes = clean_customers(raw_tables["customers"])
    orders, order_changes = clean_orders(raw_tables["orders"], products)
    save_tables(
        {
            "customers": customers,
            "products": products,
            "orders": orders,
        }
    )

    changes = ["## Cleaning change counts", ""]
    changes.extend(
        f"- Customers — {label}: **{count:,}**."
        for label, count in customer_changes.items()
    )
    changes.extend(
        f"- Orders — {label}: **{count:,}**."
        for label, count in order_changes.items()
    )
    (REPORTS / "raw_audit.md").write_text(
        audit + "\n\n" + "\n".join(changes) + "\n", encoding="utf-8"
    )
    print(audit)
    print("\n".join(changes))
    print(f"\nWrote clean CSVs to {CLEAN} and SQLite database to {DATABASE}.")


if __name__ == "__main__":
    main()
