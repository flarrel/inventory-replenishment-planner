import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from planner.data import ingest, selected_skus, weekly_sales


class TransactionsTest(unittest.TestCase):
    def test_ingestion_and_half_open_weekly_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sales.csv"
            source.write_text(
                "InvoiceNo,StockCode,Quantity,InvoiceDate\n"
                "1,12345,2,2011-01-03 08:00:00\n"
                "2,12345,3,2011-01-09 10:00:00\n"
                "3,12345,8,2011-01-17 08:00:00\n"
                "C4,12345,-2,2011-01-10 10:00:00\n"
                "5,POST,1,2011-01-10 10:00:00\n"
                "6,12346,,2011-01-13 10:00:00\n"
                "7,12346,1,invalid\n",
                encoding="utf-8",
            )
            db_path = Path(directory) / "sales.sqlite"
            result = ingest(source, db_path)
            self.assertEqual(result["source_rows"], 7)
            self.assertEqual(result["sqlite_units"], 13)
            self.assertEqual(
                result["row_counts"],
                {"accepted": 3, "cancellation_or_return": 1, "non_merchandise_or_zero": 1,
                 "missing": 1, "malformed": 1},
            )
            with closing(sqlite3.connect(db_path)) as db:
                selected = selected_skus(db, "2011-01-03", "2011-01-17", 1)
                self.assertEqual(selected, ["12345"])
                self.assertEqual(
                    weekly_sales(db, "2011-01-03", "2011-01-17", selected),
                    [("2011-01-03", "12345", 5), ("2011-01-10", "12345", 0)],
                )
            source.write_text("wrong,columns\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ingest(source, db_path)
            with closing(sqlite3.connect(db_path)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0], 3)
