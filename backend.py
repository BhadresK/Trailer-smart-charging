# backend.py
# -----------------------------
# Brain: read price data + cost calculation
# -----------------------------
from datetime import datetime, timedelta
import math
import pandas as pd
import requests
from io import BytesIO

# ---------- Reading price data ----------
def read_price_24_from_local(path: str):
    """
    Read 24 hourly prices from a local CSV or Excel file in your repo.
    Expected columns: hour, price (rows: hour = 0..23).
    """
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(".xlsx"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            raise ValueError("Unsupported file type. Use .csv or .xlsx")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return _normalize_price_df(df)


def read_price_24_from_github_raw(owner: str, repo: str, branch: str, path_in_repo: str):
    """
    Fetch file from GitHub RAW URL and read it (CSV or Excel).
    Example RAW URL format:
      https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path_in_repo}
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path_in_repo}"
    # Try CSV first
    try:
        if path_in_repo.lower().endswith(".csv"):
            df = pd.read_csv(raw_url)
            return _normalize_price_df(df)
        elif path_in_repo.lower().endswith(".xlsx"):
            resp = requests.get(raw_url, timeout=20)
            resp.raise_for_status()
            bio = BytesIO(resp.content)
            df = pd.read_excel(bio, engine="openpyxl")
            return _normalize_price_df(df)
        else:
            raise ValueError("Unsupported file type at URL. Use .csv or .xlsx")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch/parse GitHub raw file: {e}")


def _normalize_price_df(df: pd.DataFrame):
    """
    Validate and extract 24 prices.
    Must have 'price' column; 'hour' helps ensure correct order.
    """
    if "price" not in df.columns:
        # try lowercase fix or strip
        cols = {c.strip().lower(): c for c in df.columns}
        if "price" in cols:
            price_col = cols["price"]
        else:
            raise ValueError("The file must contain a 'price' column.")
    else:
        price_col = "price"

    # If 'hour' exists, use it to order and slice first 24 hours
    if "hour" in df.columns:
        df = df.copy()
        # Ensure numeric hour
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df = df.dropna(subset=["hour"])
        df = df.sort_values("hour")
        # Keep only 0..23
        df = df[df["hour"].between(0, 23)]
        # If fewer than 24 remain, error
        if len(df) < 24:
            raise ValueError("Need 24 rows with hours 0..23.")
        df = df.iloc[:24]
        prices = pd.to_numeric(df[price_col], errors="coerce").tolist()
    else:
        # No hour column: just take first 24 prices
        if len(df) < 24:
            raise ValueError("Need at least 24 rows in 'price' column.")
        prices = pd.to_numeric(df[price_col].iloc[:24], errors="coerce").tolist()

    # Validate numeric
    if any(pd.isna(prices)):
        raise ValueError("Found non-numeric values in 'price'. Please clean the file.")
    return prices  # list of 24 floats


# ---------- Calculation helpers ----------
def slice_price_window(price_24, arrival_time: str, departure_time: str):
    """
    Return the list of hourly prices covering the parked window
    starting at arrival hour and spanning whole hours until departure.
    Handles midnight wrap. Uses whole hours for simplicity (ELI10).
    """
    fmt = "%H:%M"
    t_arr = datetime.strptime(arrival_time, fmt)
    t_dep = datetime.strptime(departure_time, fmt)
    if t_dep <= t_arr:
        t_dep += timedelta(days=1)

    duration_hours = (t_dep - t_arr).total_seconds() / 3600.0
    # For a simple model, use ceiling to include a partial first/last hour
    hours = max(1, math.ceil(duration_hours))
    start_hour = t_arr.hour  # simple: start from arrival hour boundary
    window = [price_24[(start_hour + i) % 24] for i in range(hours)]
    return window, hours, start_hour


def compute_charging_cost(
    battery_kWh: float,
    soc_arrival_pc: float,
    soc_target_pc: float,
    arrival_time: str,
    departure_time: str,
    price_24,
    power_kW: float = 11.0,
):
    """
    Compute energy needed and cost for Dumb vs Smart charging
    across the parked window prices.
    Dumb: charge immediately each hour until target or time runs out.
    Smart: use the cheapest hours first within the window.
    """
    # Energy needed to reach target
    energy_needed = battery_kWh * (soc_target_pc - soc_arrival_pc) / 100.0
    energy_needed = max(0.0, energy_needed)

    # Get window prices
    window_prices, hours, start_hour = slice_price_window(price_24, arrival_time, departure_time)

    # Energy we can deliver within the window
    deliverable = hours * power_kW
    enough = deliverable >= energy_needed

    # ---- Dumb cost: sequential from arrival ----
    remaining = energy_needed
    dumb_cost = 0.0
    for p in window_prices:
        if remaining <= 0:
            break
        charge_this_hour = min(power_kW, remaining)
        dumb_cost += charge_this_hour * p
        remaining -= charge_this_hour

    # ---- Smart cost: sort window by cheapest hours ----
    remaining = energy_needed
    smart_cost = 0.0
    for p in sorted(window_prices):
        if remaining <= 0:
            break
        charge_this_hour = min(power_kW, remaining)
        smart_cost += charge_this_hour * p
        remaining -= charge_this_hour

    result = {
        "energy_needed_kWh": round(energy_needed, 2),
        "energy_deliverable_kWh": round(deliverable, 2),
        "enough_to_meet_target": bool(enough),
        "dumb_cost_EUR": round(dumb_cost, 2),
        "smart_cost_EUR": round(smart_cost, 2),
        "hours_in_window": int(hours),
        "arrival_start_hour": int(start_hour),
    }
    return result