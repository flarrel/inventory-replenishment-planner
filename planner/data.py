"""Load transaction data, extract weekly sales, and generate an EDA report."""

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

PRODUCT = re.compile(r"[0-9]{5}[A-Z]{0,2}")
DEFAULT_WORK = Path(__file__).resolve().parents[2] / "work"


def transaction(row):
    required = ("InvoiceNo", "StockCode", "Quantity", "InvoiceDate")
    if any(not (row.get(key) or "").strip() for key in required):
        return "missing", None
    invoice, sku = (row[key].strip().upper() for key in required[:2])
    try:
        quantity = int(row["Quantity"])
        timestamp = row["InvoiceDate"].strip()
        try:
            day = datetime.fromisoformat(timestamp).date()
        except ValueError:
            day = datetime.strptime(timestamp, "%m/%d/%Y %H:%M").date()
    except ValueError:
        return "malformed", None
    if invoice.startswith("C") or quantity < 0:
        return "cancellation_or_return", None
    if quantity == 0 or not PRODUCT.fullmatch(sku):
        return "non_merchandise_or_zero", None
    return "accepted", (sku, day.isoformat(), quantity)


def ingest(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        pending = Path(temp) / "sales.sqlite"
        counts = Counter()
        source_units = 0
        with closing(sqlite3.connect(pending)) as db, source.open(
            encoding="utf-8-sig", newline=""
        ) as stream, db:
            db.execute(
                """CREATE TABLE transactions (
                    sku TEXT NOT NULL, day TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK (quantity > 0)
                )"""
            )
            reader = csv.DictReader(stream)
            if not {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate"}.issubset(
                reader.fieldnames or ()
            ):
                raise ValueError("CSV is missing required transaction columns")

            def accepted_rows():
                nonlocal source_units
                for row in reader:
                    category, result = transaction(row)
                    counts[category] += 1
                    if result is not None:
                        source_units += result[2]
                        yield result

            db.executemany("INSERT INTO transactions VALUES (?, ?, ?)", accepted_rows())
            db.execute("CREATE INDEX transactions_day_sku ON transactions(day, sku)")
            rows, units = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(quantity), 0) FROM transactions"
            ).fetchone()
            if rows == 0 or (rows, units) != (counts["accepted"], source_units):
                raise ValueError("Accepted source rows or units failed reconciliation")
            summary = {
                "source_rows": sum(counts.values()),
                "row_counts": dict(sorted(counts.items())),
                "accepted_source_units": source_units,
                "sqlite_rows": rows,
                "sqlite_units": units,
            }
            db.execute("CREATE TABLE ingestion (summary TEXT NOT NULL)")
            db.execute("INSERT INTO ingestion VALUES (?)", (json.dumps(summary),))
        pending.replace(destination)
    return summary


def week_start(day):
    return day - timedelta(days=day.weekday())


def analysis_window(db):
    first, last = db.execute("SELECT MIN(day), MAX(day) FROM transactions").fetchone()
    if first is None:
        raise ValueError("No accepted transactions")
    return (
        week_start(date.fromisoformat(first)) + timedelta(weeks=1),
        week_start(date.fromisoformat(last)),
    )


TOP_SKUS = """
    SELECT sku FROM transactions
    WHERE day >= :start AND day < :stop
    GROUP BY sku
    ORDER BY COUNT(DISTINCT day) DESC, SUM(quantity) DESC, sku
    LIMIT :count
"""

WEEKLY_SALES = """
    WITH RECURSIVE calendar(week) AS (
        SELECT :start WHERE :start < :stop
        UNION ALL
        SELECT date(week, '+7 days') FROM calendar WHERE date(week, '+7 days') < :stop
    ),
    selected(sku) AS (SELECT value FROM json_each(:skus)),
    totals AS (
        SELECT sku,
               date(day, '-' || ((CAST(strftime('%w', day) AS INTEGER) + 6) % 7) || ' days')
                   AS week,
               SUM(quantity) AS units
        FROM transactions JOIN selected USING (sku)
        WHERE day >= :start AND day < :stop
        GROUP BY sku, week
    )
    SELECT calendar.week, selected.sku, COALESCE(totals.units, 0)
    FROM calendar CROSS JOIN selected
    LEFT JOIN totals ON totals.week = calendar.week AND totals.sku = selected.sku
    ORDER BY calendar.week, selected.sku
"""


def selected_skus(db, start, stop, count):
    if count < 1 or start >= stop:
        raise ValueError("Selection requires a positive SKU count and a valid date window")
    parameters = {"start": str(start), "stop": str(stop), "count": count}
    return [row[0] for row in db.execute(TOP_SKUS, parameters)]


def weekly_sales(db, start, stop, skus):
    return db.execute(
        WEEKLY_SALES, {"start": str(start), "stop": str(stop), "skus": json.dumps(skus)}
    ).fetchall()


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def explore(work):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with closing(sqlite3.connect(work / "sales.sqlite")) as db:
        start, stop = analysis_window(db)
        weeks = (stop - start).days // 7
        if weeks < 20:
            raise ValueError("EDA requires at least 20 complete weeks")
        train_stop = start + timedelta(weeks=weeks - 16)
        skus = selected_skus(db, start, train_stop, 24)
        rows = weekly_sales(db, start, stop, skus)
        ingestion = json.loads(db.execute("SELECT summary FROM ingestion").fetchone()[0])
        totals_by_sku = dict(
            db.execute(
                "SELECT sku, SUM(quantity) FROM transactions "
                "WHERE day >= ? AND day < ? GROUP BY sku",
                (str(start), str(stop)),
            )
        )
    extract_totals = Counter()
    totals_by_week = Counter()
    for week, sku, units in rows:
        extract_totals[sku] += units
        totals_by_week[week] += units
    if any(extract_totals[sku] != totals_by_sku.get(sku, 0) for sku in skus):
        raise ValueError("Weekly extract failed source reconciliation")

    values = np.array([row[2] for row in rows])
    positive = values[values > 0]
    per_week = np.array(list(totals_by_week.values()))
    summary = {
        "ingestion": ingestion,
        "first_week": str(start),
        "end_exclusive": str(stop),
        "sku_selection_end_exclusive": str(train_stop),
        "sku_count": len(skus),
        "sku_weeks": len(rows),
        "reconciled_weekly_units": int(values.sum()),
        "zero_sku_week_share": float(np.mean(values == 0)),
        "median_sku_week_units": float(np.median(values)),
        "p90_sku_week_units": float(np.percentile(values, 90)),
        "median_nonzero_sku_week_units": float(np.median(positive)),
        "first_eight_week_median_units": float(np.median(per_week[:8])),
        "last_eight_week_median_units": float(np.median(per_week[-8:])),
        "largest_sku_week_units": int(values.max()),
        "p90_without_largest_sku_week": float(np.percentile(np.sort(values)[:-1], 90)),
    }
    save_json(work / "eda.json", summary)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
    axes[0].plot(list(totals_by_week), per_week, color="#146861")
    axes[0].set(title="Observed sales by week", ylabel="Units across selected SKUs")
    axes[0].set_xticks(range(0, weeks, 8), list(totals_by_week)[::8])
    axes[1].hist(positive, bins=25, color="#146861", log=True)
    axes[1].set(title="Nonzero SKU-week sales", xlabel="Units", ylabel="SKU-weeks (log scale)")
    fig.savefig(work / "eda.png", dpi=135)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ingest", "eda"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.stage == "ingest":
        source = args.source or args.work / "downloads" / "online-retail.csv"
        report = ingest(source, args.work / "sales.sqlite")
    else:
        report = explore(args.work)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
