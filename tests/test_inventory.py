import unittest
from itertools import product

import numpy as np

from planner.inventory import Economics, optimize, simulate


class InventoryTest(unittest.TestCase):
    def test_budgeted_plan_matches_exhaustive_optimum(self):
        forecast = np.array([3.6, 2.1])
        opening = np.array([1, 0])
        economics = [Economics(3, 7, 2), Economics(2, 9, 1)]

        def objective(orders):
            shortage = forecast - opening - orders
            return np.maximum(shortage, 0) @ [7, 9] + np.maximum(-shortage, 0) @ [2, 1]

        for budget in (0, 2, 5, 9, 20):
            candidates = (
                objective(np.array(orders))
                for orders in product(range(4), repeat=2)
                if 3 * orders[0] + 2 * orders[1] <= budget
            )
            result = optimize(forecast, opening, economics, budget)
            self.assertLessEqual(result @ [3, 2], budget)
            self.assertAlmostEqual(objective(result), min(candidates))

    def test_simulation_preserves_inventory_and_does_not_backorder(self):
        def schedule(predicted, opening, economics, budget):
            return np.array([2])

        actual = np.array([[4], [1], [2]])
        outcome = simulate(schedule, actual, actual, [Economics(5, 10, 1)], 10)
        weekly = outcome["by_week"]
        self.assertEqual([row["unmet_units"] for row in weekly], [2, 0, 0])
        self.assertEqual([row["ending_units"] for row in weekly], [0, 1, 1])
        self.assertEqual(outcome["fill_rate"], 5 / 7)
