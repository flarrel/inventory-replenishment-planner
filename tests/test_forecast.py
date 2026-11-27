import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from planner.forecast import fit, load_panel, predict, score, supervised


class ForecastTest(unittest.TestCase):
    def test_panel_records_the_sku_identity_of_each_column(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sales.sqlite"
            start = date(2011, 1, 3)
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("CREATE TABLE transactions (sku TEXT, day TEXT, quantity INT)")
                db.executemany(
                    "INSERT INTO transactions VALUES (?, ?, ?)",
                    [(sku, str(start + timedelta(weeks=i)), units)
                     for i in range(30) for sku, units in (("99999", 30), ("12345", 10))],
                )
            _, skus, sales = load_panel(database)
            self.assertEqual(skus, ["12345", "99999"])
            np.testing.assert_array_equal(sales[0], [10, 30])

    def test_target_and_future_do_not_affect_features_for_target_week(self):
        weeks = [date(2026, 1, 5) + timedelta(weeks=i) for i in range(14)]
        sales = np.arange(28).reshape(14, 2)
        before = supervised(weeks, sales)
        sales[10:] = 10000
        after = supervised(weeks, sales)
        np.testing.assert_array_equal(before[:3], after[:3])
        self.assertFalse(np.array_equal(before[3], after[3]))

    def test_prediction_shape_and_error_direction(self):
        weeks = [date(2026, 1, 5) + timedelta(weeks=i) for i in range(13)]
        sales = np.arange(52).reshape(13, 4)
        x = supervised(weeks, sales)
        model = fit(x, sales[8:])
        predictions = predict(model, x)
        self.assertEqual(predictions["gradient_boosting"].shape, (5, 4))
        np.testing.assert_array_equal(predictions["last_week"], sales[7:-1])
        self.assertEqual(score(np.asarray([2, 2]), np.asarray([0, 2]))["mean_bias_units"], 1)
        self.assertIsNone(score(np.asarray([1]), np.asarray([0]))["wape"])
