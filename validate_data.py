"""Assert the Part A customer and order cleaning checkpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "data" / "instamart.db"
CLEAN = ROOT / "data" / "clean"


def main() -> None:
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    products = pd.read_csv(CLEAN / "products.csv")

    assert customers["city"].value_counts().to_dict() == {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    assert customers["gender"].value_counts().to_dict() == {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }
    assert customers["age"].notna().all()
    assert customers["age"].between(18, 60).all()
    assert pd.to_datetime(customers["signup_date"], errors="raise").notna().all()
    assert len(orders) == 12_309
    assert orders["order_id"].is_unique
    assert orders["status"].value_counts().to_dict() == {
        "completed": 11_110,
        "cancelled": 1_199,
    }
    assert not orders["status"].eq("pending").any()
    assert orders["order_amount"].notna().all()
    assert orders.loc[orders["status"].eq("completed"), "delivery_minutes"].notna().all()
    assert orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum() == 1_199
    completed_revenue = orders.loc[
        orders["status"].eq("completed"), "order_amount"
    ].sum()
    assert round(completed_revenue, 2) == 2_998_668.80
    assert len(products) == 40

    with sqlite3.connect(DATABASE) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"customers", "products", "orders"}.issubset(tables)
        stored_timestamp = connection.execute(
            "SELECT order_ts FROM orders LIMIT 1"
        ).fetchone()[0]
        assert len(stored_timestamp) == 19
        assert stored_timestamp[10] == " "

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
