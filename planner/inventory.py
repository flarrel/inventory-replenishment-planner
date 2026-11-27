"""Budget-constrained replenishment and a chronological inventory simulation."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from planner.data import DEFAULT_WORK, save_json

DEMO_BUDGET_CENTS = 2_800_000


@dataclass(frozen=True)
class Economics:
    unit_cost_cents: int
    shortage_penalty_cents: int
    holding_cost_cents: int


def demo_economics(count):
    return [
        Economics(
            unit_cost_cents=250 + 75 * (index % 8),
            shortage_penalty_cents=(250 + 75 * (index % 8)) * (2 + index % 3),
            holding_cost_cents=12 + 4 * (index % 5),
        )
        for index in range(count)
    ]


def optimize(forecast, on_hand, economics, budget_cents):
    costs = np.asarray([item.unit_cost_cents for item in economics])
    shortage = [item.shortage_penalty_cents for item in economics]
    holding = [item.holding_cost_cents for item in economics]
    deficit = np.asarray(forecast) - on_hand
    count = len(costs)
    identity = np.eye(count)

    # order + projected shortage - excess = forecast - stock before ordering
    balance = np.hstack((identity, identity, -identity))
    spend = np.concatenate((costs, np.zeros(2 * count)))
    upper_orders = np.minimum(np.ceil(np.maximum(deficit, 0)), budget_cents // costs)
    result = milp(
        c=np.concatenate((np.zeros(count), shortage, holding)),
        integrality=np.concatenate((np.ones(count), np.zeros(2 * count))),
        bounds=Bounds(np.zeros(3 * count), np.r_[upper_orders, np.full(2 * count, np.inf)]),
        constraints=[
            LinearConstraint(balance, deficit, deficit),
            LinearConstraint(spend, 0, budget_cents),
        ],
        options={"time_limit": 5.0, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"Could not solve replenishment plan: {result.message}")
    orders = np.rint(result.x[:count]).astype(int)
    if np.any(orders < 0) or np.any(orders > upper_orders) or costs @ orders > budget_cents:
        raise RuntimeError("Solver output failed feasibility checks")
    return orders


def greedy(forecast, on_hand, economics, budget_cents):
    deficits = np.ceil(np.maximum(np.asarray(forecast) - on_hand, 0)).astype(int)
    orders = np.zeros_like(deficits)
    for i in np.argsort(-deficits, kind="stable"):
        unit_cost = economics[i].unit_cost_cents
        orders[i] = min(deficits[i], budget_cents // unit_cost)
        budget_cents -= int(orders[i]) * unit_cost
    return orders


def simulate(policy, forecast, actual, economics, budget_cents):
    on_hand = np.zeros(len(economics), dtype=int)
    realized = []
    costs = np.array([item.unit_cost_cents for item in economics])
    penalties = np.array([item.shortage_penalty_cents for item in economics])
    holding = np.array([item.holding_cost_cents for item in economics])
    for predicted, demand in zip(forecast, actual, strict=True):
        orders = policy(predicted, on_hand, economics, budget_cents)
        available = on_hand + orders
        sold = np.minimum(available, demand)
        lost = demand - sold
        on_hand = available - sold
        realized.append(
            {
                "purchase_spend_cents": int(costs @ orders),
                "units_ordered": int(orders.sum()),
                "units_sold": int(sold.sum()),
                "unmet_units": int(lost.sum()),
                "ending_units": int(on_hand.sum()),
                "shortage_penalty_cents": int(penalties @ lost),
                "holding_cost_cents": int(holding @ on_hand),
            }
        )
    totals = {name: sum(row[name] for row in realized) for name in realized[0]}
    totals["inventory_unit_weeks"] = totals.pop("ending_units")
    units = int(actual.sum())
    return {
        "observed_sales_units": units,
        "fill_rate": totals["units_sold"] / units if units else None,
        "penalty_plus_holding_cents": totals["shortage_penalty_cents"] + totals["holding_cost_cents"],
        "totals": totals,
        "by_week": realized,
    }


def evaluate(work, budget_cents):
    with (work / "test-forecasts.json").open(encoding="utf-8") as stream:
        inputs = json.load(stream)
    forecast, actual = np.asarray(inputs["forecast"]), np.asarray(inputs["actual"])
    economics = demo_economics(actual.shape[1])
    summary = {
        "inventory_and_economics": "simulated",
        "forecast_method": inputs["method"],
        "budget_cents_per_week": budget_cents,
        "weeks": inputs["weeks"],
        "optimized": simulate(optimize, forecast, actual, economics, budget_cents),
        "greedy_largest_unit_deficit": simulate(greedy, forecast, actual, economics, budget_cents),
    }
    save_json(work / "policy-metrics.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--budget-cents", type=int, default=DEMO_BUDGET_CENTS)
    args = parser.parse_args()
    if args.budget_cents < 0:
        parser.error("--budget-cents must be nonnegative")
    print(json.dumps(evaluate(args.work, args.budget_cents), indent=2))


if __name__ == "__main__":
    main()
