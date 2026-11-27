"""Local forecasting and inventory planning HTTP service."""

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from planner.data import DEFAULT_WORK
from planner.forecast import next_forecast
from planner.inventory import DEMO_BUDGET_CENTS, Economics, demo_economics, optimize

ForecastMethod = Literal["last_week", "four_week_mean", "gradient_boosting"]


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(min_length=1, max_length=12)
    on_hand: int = Field(strict=True, ge=0, le=1_000_000)
    unit_cost_cents: int = Field(strict=True, gt=0, le=1_000_000)
    shortage_penalty_cents: int = Field(strict=True, gt=0, le=1_000_000)
    holding_cost_cents: int = Field(strict=True, gt=0, le=1_000_000)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_cents: int = Field(strict=True, ge=0, le=1_000_000_000)
    items: list[Item] = Field(min_length=1, max_length=24)
    forecast_method: ForecastMethod | None = None

    @model_validator(mode="after")
    def require_distinct_skus(self):
        if len({item.sku for item in self.items}) != len(self.items):
            raise ValueError("Item SKUs must be unique")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("PLANNER_WORK", DEFAULT_WORK)) / "forecast.joblib"
    with path.open("rb") as stream:
        app.state.artifact_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        # The artifact is generated locally by planner.forecast.
        artifact = joblib.load(stream)
    app.state.skus = artifact["skus"]
    app.state.sku_positions = {sku: index for index, sku in enumerate(artifact["skus"])}
    app.state.default_method = artifact["method"]
    app.state.week = artifact["next_week"]
    app.state.predictions = next_forecast(artifact)
    yield


app = FastAPI(title="Inventory Replenishment Planner", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "artifact_sha256": request.app.state.artifact_sha256}


@app.get("/forecast")
def forecast(request: Request):
    state = request.app.state
    return {
        "week": state.week,
        "default_method": state.default_method,
        "items": [
            {
                "sku": sku,
                "units": {method: round(float(units[i]), 2) for method, units in state.predictions.items()},
            }
            for i, sku in enumerate(state.skus)
        ],
    }


@app.get("/demo")
def demo(request: Request):
    skus = request.app.state.skus
    return {
        "scenario": "simulated starting stock and item economics",
        "request": PlanRequest(
            budget_cents=DEMO_BUDGET_CENTS,
            items=[
                Item(
                    sku=sku, on_hand=0, unit_cost_cents=economics.unit_cost_cents,
                    shortage_penalty_cents=economics.shortage_penalty_cents,
                    holding_cost_cents=economics.holding_cost_cents,
                )
                for sku, economics in zip(skus, demo_economics(len(skus)), strict=True)
            ],
        ),
    }


@app.post("/plan")
def plan(body: PlanRequest, request: Request):
    state = request.app.state
    unknown = [item.sku for item in body.items if item.sku not in state.sku_positions]
    if unknown:
        raise HTTPException(422, detail={"unknown_skus": unknown})
    method = body.forecast_method or state.default_method
    positions = [state.sku_positions[item.sku] for item in body.items]
    forecast_units = state.predictions[method][positions]
    on_hand = np.array([item.on_hand for item in body.items])
    economics = [
        Economics(item.unit_cost_cents, item.shortage_penalty_cents, item.holding_cost_cents)
        for item in body.items
    ]
    try:
        orders = optimize(forecast_units, on_hand, economics, body.budget_cents)
    except RuntimeError as error:
        raise HTTPException(503, detail="Could not compute a feasible order plan") from error
    line_spend = [int(order) * item.unit_cost_cents for item, order in zip(body.items, orders)]
    return {
        "week": state.week,
        "forecast_method": method,
        "budget_cents": body.budget_cents,
        "purchase_spend_cents": sum(line_spend),
        "unspent_budget_cents": body.budget_cents - sum(line_spend),
        "items": [
            {
                "sku": item.sku,
                "forecast_units": round(float(units), 2),
                "on_hand": item.on_hand,
                "order_units": int(order),
                "purchase_spend_cents": spend,
                "projected_unmet_units": round(max(0.0, float(units) - item.on_hand - int(order)), 2),
            }
            for item, units, order, spend in zip(body.items, forecast_units, orders, line_spend)
        ],
    }
