import unittest
from datetime import date, timedelta

import numpy as np

from planner.forecast import fit, predict, score, supervised


class ForecastTest(unittest.TestCase):
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
