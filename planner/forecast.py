"""Train and compare one-week-ahead SKU sales forecasts."""

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from planner.data import DEFAULT_WORK, analysis_window, save_json, selected_skus, weekly_sales

HISTORY_WEEKS = 8
BASELINES = ("last_week", "four_week_mean")
METHODS = (*BASELINES, "gradient_boosting")


def features(history, target_week):
    history = np.asarray(history)
    if history.ndim != 2 or history.shape[0] < HISTORY_WEEKS:
        raise ValueError("Forecasting requires eight complete weeks of sales per SKU")
    angle = 2 * np.pi * (target_week.timetuple().tm_yday - 1) / 365.25
    size = history.shape[1]
    return np.column_stack(
        (
            history[-1],
            history[-2],
            history[-4],
            history[-8],
            history[-4:].mean(axis=0),
            history[-8:].mean(axis=0),
            (history[-8:] == 0).mean(axis=0),
            np.full(size, np.sin(angle)),
            np.full(size, np.cos(angle)),
        )
    )


def supervised(weeks, sales):
    return np.stack(
        [features(sales[:index], weeks[index]) for index in range(HISTORY_WEEKS, len(weeks))]
    )


def fit(x, actual):
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=15, min_samples_leaf=15,
        learning_rate=0.07, l2_regularization=1.0, early_stopping=False, random_state=42,
    )
    with threadpool_limits(limits=1):
        model.fit(x.reshape(-1, x.shape[-1]), actual.ravel())
    return model


def predict(model, x):
    with threadpool_limits(limits=1):
        learned = model.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1])
    return {
        "last_week": x[..., 0],
        "four_week_mean": x[..., 4],
        "gradient_boosting": np.maximum(learned, 0),
    }


def score(predicted, actual):
    errors = predicted - actual
    units = float(actual.sum())
    return {
        "sku_weeks": actual.size,
        "observed_units": int(units),
        "mae_units": float(np.abs(errors).mean()),
        "mean_bias_units": float(errors.mean()),
        "wape": float(np.abs(errors).sum() / units) if units else None,
    }


def evaluate(predictions, actual):
    return {method: score(predictions[method], actual) for method in METHODS}


def load_panel(database):
    with closing(sqlite3.connect(database)) as db:
        start, stop = analysis_window(db)
        week_count = (stop - start).days // 7
        if week_count < 28:
            raise ValueError("Training requires at least 28 complete weeks")
        train_end = start + timedelta(weeks=week_count - 16)
        # The SQL panel emits SKU columns in lexical order within each week.
        skus = sorted(selected_skus(db, start, train_end, 24))
        rows = weekly_sales(db, start, stop, skus)
    weeks = [start + timedelta(weeks=i) for i in range(week_count)]
    if not skus or len(rows) != week_count * len(skus):
        raise ValueError("Incomplete weekly panel")
    sales = np.array([row[2] for row in rows], dtype=float).reshape(week_count, len(skus))
    return weeks, skus, sales


def train(work):
    weeks, skus, sales = load_panel(work / "sales.sqlite")
    x = supervised(weeks, sales)
    actual = sales[HISTORY_WEEKS:]
    end_training = len(actual) - 16
    end_validation = len(actual) - 8
    model = fit(x[:end_training], actual[:end_training])
    validation = evaluate(
        predict(model, x[end_training:end_validation]), actual[end_training:end_validation]
    )
    chosen = min(METHODS, key=lambda method: validation[method]["mae_units"])

    model = fit(x[:end_validation], actual[:end_validation])
    test_forecasts = predict(model, x[end_validation:])
    test = evaluate(test_forecasts, actual[end_validation:])
    summary = {
        "chosen_method": chosen,
        "selection_metric": "lowest validation MAE",
        "sku_count": len(skus),
        "first_training_target": str(weeks[HISTORY_WEEKS]),
        "first_validation_target": str(weeks[-16]),
        "first_test_target": str(weeks[-8]),
        "last_test_target": str(weeks[-1]),
        "validation": validation,
        "test": test,
    }
    save_json(work / "forecast-metrics.json", summary)
    save_json(
        work / "test-forecasts.json",
        {
            "weeks": [str(week) for week in weeks[-8:]],
            "skus": skus,
            "method": chosen,
            "actual": actual[end_validation:].astype(int).tolist(),
            "forecast": test_forecasts[chosen].tolist(),
        },
    )
    artifact = {
        "model": model,
        "method": chosen,
        "skus": skus,
        "recent_sales": sales[-HISTORY_WEEKS:].astype(int),
        "next_week": weeks[-1] + timedelta(weeks=1),
    }
    with TemporaryDirectory(dir=work) as temp:
        prepared = Path(temp) / "forecast.joblib"
        joblib.dump(artifact, prepared, compress=3)
        prepared.replace(work / "forecast.joblib")
    return summary


def next_forecast(artifact, history=None, target_week: date | None = None):
    if history is None:
        history = artifact["recent_sales"]
    if target_week is None:
        target_week = artifact["next_week"]
    return predict(artifact["model"], features(history, target_week))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    print(json.dumps(train(args.work), indent=2))


if __name__ == "__main__":
    main()
