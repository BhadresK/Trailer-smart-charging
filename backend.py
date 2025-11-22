# backend.py
from datetime import datetime, timedelta
import math
import pandas as pd
import requests
from io import BytesIO

def read_price_24_from_github_raw():
    """
    Auto-fetch hourly prices from GitHub RAW URL.
    Update this URL to match your repo and file path.
    """
    raw_url = "https://github.com/BhadresK/Trailer-smart-charging/blob/main/data/hourly_prices.csv?raw=true"
    try:
        df = pd.read_csv(raw_url)
        return _normalize_price_df(df)
    except:
        # Fallback to local file
        df = pd.read_csv("data/hourly_prices.csv")
        return _normalize_price_df(df)

def _normalize_price_df(df: pd.DataFrame):
    if "price" not in df.columns:
        raise ValueError("CSV must have a 'price' column.")
    if len(df) < 24:
        raise ValueError("CSV must have at least 24 rows.")
    prices = pd.to_numeric(df["price"].iloc[:24], errors="coerce").tolist()
    if any(pd.isna(prices)):
        raise ValueError("Non-numeric values found in 'price' column.")
    return prices

def slice_price_window(price_24, arrival_time, departure_time):
    fmt = "%H:%M"
    t_arr = datetime.strptime(arrival_time, fmt)
    t_dep = datetime.strptime(departure_time, fmt)
    if t_dep <= t_arr:
        t_dep += timedelta(days=1)
    duration_hours = (t_dep - t_arr).total_seconds() / 3600.0
    hours = max(1, math.ceil(duration_hours))
    start_hour = t_arr.hour
    window = [price_24[(start_hour + i) % 24] for i in range(hours)]
    return window, hours

def compute_charging_cost(battery_kWh, soc_arrival, soc_target, arrival_time, departure_time, price_24, power_kW=11):
    energy_needed = battery_kWh * (soc_target - soc_arrival) / 100
    energy_needed = max(0, energy_needed)
    window_prices, hours = slice_price_window(price_24, arrival_time, departure_time)
    deliverable = hours * power_kW
    enough = deliverable >= energy_needed

    # Dumb charging
    remaining = energy_needed
    dumb_cost = 0
    for p in window_prices:
        if remaining <= 0: break
        charge = min(power_kW, remaining)
        dumb_cost += charge * p
        remaining -= charge

    # Smart charging
    remaining = energy_needed
    smart_cost = 0
    for p in sorted(window_prices):
        if remaining <= 0: break
        charge = min(power_kW, remaining)
        smart_cost += charge * p
        remaining -= charge

    return {
        "energy_needed_kWh": round(energy_needed, 2),
        "energy_deliverable_kWh": round(deliverable, 2),
        "enough_to_meet_target": enough,
        "dumb_cost_EUR": round(dumb_cost, 2),
        "smart_cost_EUR": round(smart_cost, 2),
        "hours_in_window": hours
    }