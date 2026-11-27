"""Exercise the actual HTTP service and verify artifact reload after restart."""

import json
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base, path, body=None):
    if body is None:
        query = Request(base + path)
    else:
        query = Request(
            base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
    try:
        with urlopen(query, timeout=8) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        with error:
            return error.code, json.load(error)


@contextmanager
def server():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "planner.api:app", "--host", "127.0.0.1",
         "--port", str(port), "--no-access-log"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = False
        for _ in range(120):
            if process.poll() is not None:
                break
            try:
                if request(base, "/health")[0] == 200:
                    ready = True
                    break
            except URLError:
                time.sleep(0.1)
        if not ready:
            raise RuntimeError("Local API did not become ready")
        yield base
    except Exception:
        process.terminate()
        try:
            logs, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            logs, _ = process.communicate()
        print(logs, file=sys.stderr)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


def main():
    with server() as base:
        health = request(base, "/health")
        forecast = request(base, "/forecast")
        status, demo = request(base, "/demo")
        assert status == 200 and forecast[0] == 200
        body = demo["request"]
        status, plan = request(base, "/plan", body)
        assert status == 200 and 0 < plan["purchase_spend_cents"] <= body["budget_cents"]
        assert plan["purchase_spend_cents"] == sum(row["purchase_spend_cents"] for row in plan["items"])
        assert len(plan["items"]) == len(body["items"])
        revised_budget = body["budget_cents"] // 2
        status, revised = request(base, "/plan", dict(body, budget_cents=revised_budget))
        assert status == 200 and 0 < revised["purchase_spend_cents"] <= revised_budget
        assert sum(row["order_units"] for row in revised["items"]) < sum(
            row["order_units"] for row in plan["items"]
        )
        status, zero = request(base, "/plan", dict(body, budget_cents=0))
        assert status == 200 and all(row["order_units"] == 0 for row in zero["items"])
        status, learned = request(base, "/plan", dict(body, forecast_method="gradient_boosting"))
        assert status == 200 and learned["forecast_method"] == "gradient_boosting"
        invalid = [dict(body["items"][0], sku="UNKNOWN")]
        assert request(base, "/plan", dict(body, items=invalid))[0] == 422
        assert request(base, "/plan", dict(body, budget_cents=-1))[0] == 422
        assert request(base, "/plan", dict(body, items=[body["items"][0]] * 2))[0] == 422
    with server() as base:
        assert request(base, "/health") == health
        assert request(base, "/forecast") == forecast
    print(
        f"HTTP verification passed for {len(plan['items'])} SKUs. "
        f"Default {plan['forecast_method']}; spent {plan['purchase_spend_cents']} cents. "
        "Validated zero budget, alternate ML forecast, bad inputs, and artifact reload."
    )


if __name__ == "__main__":
    main()
