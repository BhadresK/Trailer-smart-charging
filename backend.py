# backend.py
from datetime import datetime, timedelta
import math
import pandas as pd
import requests

# -------------------------------
# DATA: Read 24-hour price series
# -------------------------------
def read_price_24_from_github_raw(
    raw_url: str = "https://github.com/BhadresK/Trailer-smart-charging/blob/main/data/hourly_prices.csv?raw=true"
):
    """
    Auto-fetch CSV with columns: hour,price (hours 0..23).
    """
    df = pd.read_csv(raw_url)
    return _normalize_price_df(df)

def _normalize_price_df(df: pd.DataFrame):
    if "price" not in df.columns:
        raise ValueError("CSV must contain a 'price' column.")
    # Keep first 24 rows (hours 0..23) in order if 'hour' exists
    if "hour" in df.columns:
        df = df.copy()
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df = df.dropna(subset=["hour"]).sort_values("hour")
        df = df[df["hour"].between(0, 23)]
        if len(df) < 24:
            raise ValueError("Need 24 rows for hours 0..23.")
        prices = pd.to_numeric(df["price"].iloc[:24], errors="coerce").tolist()
    else:
        prices = pd.to_numeric(df["price"].iloc[:24], errors="coerce").tolist()
    if any(pd.isna(prices)):
        raise ValueError("Price column has non-numeric values.")
    return prices  # list of 24 floats (€/kWh)

# ------------------------------------
# TIME: Build parked window over hours
# ------------------------------------
def build_window_hours(arrival_time: str, departure_time: str):
    """
    Returns:
      window_hours_idx: list of hour indices within parked window (0..23, wraps midnight)
      hours_in_window: int (ceil of duration)
      arrival_hour: int (0..23)
    """
    fmt = "%H:%M"
    t_arr = datetime.strptime(arrival_time, fmt)
    t_dep = datetime.strptime(departure_time, fmt)
    if t_dep <= t_arr:
        t_dep += timedelta(days=1)
    duration_hours = (t_dep - t_arr).total_seconds() / 3600.0
    hours_in_window = min(24, max(1, math.ceil(duration_hours)))  # cap at 24 for chart
    arrival_hour = t_arr.hour
    window_hours_idx = [ (arrival_hour + i) % 24 for i in range(hours_in_window) ]
    return window_hours_idx, hours_in_window, arrival_hour

# ----------------------------------------------------
# CORE: Build full-day power & SoC traces (0..23 hours)
# ----------------------------------------------------
def compute_day_traces(ev: dict, price_24: list):
    """
    ev dict keys:
      battery_kWh, soc_arrival_pc, soc_target_pc, arrival_time, departure_time,
      max_charger_kW, batt_eff_frac, obc_eff_frac,
      grid_kVA, pf_site

    Returns dict with:
      dumb_power_grid_kW[24], smart_power_grid_kW[24],
      dumb_soc_pc[24], smart_soc_pc[24],
      dumb_cost_EUR, smart_cost_EUR,
      window_hours_idx, hourly_price_24
    """
    # Unpack
    battery_kWh      = float(ev["battery_kWh"])
    soc_arrival_pc   = float(ev["soc_arrival_pc"])
    soc_target_pc    = float(ev["soc_target_pc"])
    arrival_time     = str(ev["arrival_time"])
    departure_time   = str(ev["departure_time"])
    max_charger_kW   = float(ev["max_charger_kW"])
    batt_eff         = float(ev["batt_eff_frac"])   # 0..1
    obc_eff          = float(ev["obc_eff_frac"])    # 0..1
    grid_kVA         = float(ev["grid_kVA"])
    pf_site          = float(ev["pf_site"])         # 0..1

    # Energy to battery (kWh)
    energy_needed_batt_kWh = max(0.0, battery_kWh * (soc_target_pc - soc_arrival_pc) / 100.0)

    # Effective grid power cap (kW)
    effective_grid_kW = min(max_charger_kW, grid_kVA * pf_site)

    # Simple taper: halve max grid power once SoC >= 80%
    TAPER_SOC_PC = 80.0
    def taper_cap(current_soc_pc):
        return effective_grid_kW if current_soc_pc < TAPER_SOC_PC else 0.5 * effective_grid_kW

    # Window hours
    window_hours_idx, hours_in_window, arrival_hour = build_window_hours(arrival_time, departure_time)

    # Init full-day arrays
    dumb_power_grid_kW  = [0.0] * 24
    smart_power_grid_kW = [0.0] * 24
    dumb_soc_pc         = [None] * 24
    smart_soc_pc        = [None] * 24

    # --- Dumb charging: chronological from arrival ---
    remaining_batt_kWh = energy_needed_batt_kWh
    soc_pc = soc_arrival_pc
    for h in window_hours_idx:
        dumb_soc_pc[h] = soc_pc  # SOC at start of the hour
        if remaining_batt_kWh <= 1e-9:
            dumb_power_grid_kW[h] = 0.0
            continue
        cap_kW = taper_cap(soc_pc)
        # Grid energy this hour capped by what’s needed
        needed_grid_kWh = remaining_batt_kWh / max(1e-12, batt_eff * obc_eff)
        grid_energy_this_hour_kWh = min(cap_kW, needed_grid_kWh)
        dumb_power_grid_kW[h] = grid_energy_this_hour_kWh
        # Battery gain & SOC update
        batt_gain_kWh = grid_energy_this_hour_kWh * batt_eff * obc_eff
        remaining_batt_kWh -= batt_gain_kWh
        soc_pc = min(100.0, soc_pc + (batt_gain_kWh / battery_kWh) * 100.0)
    # Fill SOC where None with last known
    fill_soc_line(dumb_soc_pc, start_soc=soc_arrival_pc)

    # --- Smart charging: allocate to cheapest window hours ---
    remaining_batt_kWh = energy_needed_batt_kWh
    # Sort window hours by price (cheapest first)
    cheapest_order = sorted(window_hours_idx, key=lambda h: price_24[h])
    # First, decide allocations
    allocations = {h: 0.0 for h in window_hours_idx}
    temp_soc_pc = soc_arrival_pc
    for h in cheapest_order:
        if remaining_batt_kWh <= 1e-9:
            break
        cap_kW = taper_cap(temp_soc_pc)
        needed_grid_kWh = remaining_batt_kWh / max(1e-12, batt_eff * obc_eff)
        grid_energy_this_hour_kWh = min(cap_kW, needed_grid_kWh)
        allocations[h] = grid_energy_this_hour_kWh
        batt_gain_kWh = grid_energy_this_hour_kWh * batt_eff * obc_eff
        remaining_batt_kWh -= batt_gain_kWh
        temp_soc_pc = min(100.0, temp_soc_pc + (batt_gain_kWh / battery_kWh) * 100.0)
    # Then walk the day chronologically to build SOC line
    soc_pc = soc_arrival_pc
    for h in range(24):
        smart_soc_pc[h] = soc_pc
        grid_kWh = allocations.get(h, 0.0)
        smart_power_grid_kW[h] = grid_kWh
        batt_gain_kWh = grid_kWh * batt_eff * obc_eff
        soc_pc = min(100.0, soc_pc + (batt_gain_kWh / battery_kWh) * 100.0)

    # Costs
    dumb_cost_EUR  = sum(price_24[h] * dumb_power_grid_kW[h]  for h in range(24))
    smart_cost_EUR = sum(price_24[h] * smart_power_grid_kW[h] for h in range(24))

    return {
        "hourly_price_24": price_24,
        "window_hours_idx": window_hours_idx,
        "dumb_power_grid_kW": dumb_power_grid_kW,
        "smart_power_grid_kW": smart_power_grid_kW,
        "dumb_soc_pc": dumb_soc_pc,
        "smart_soc_pc": smart_soc_pc,
        "dumb_cost_EUR": round(dumb_cost_EUR, 2),
        "smart_cost_EUR": round(smart_cost_EUR, 2),
    }

# --------------------------
# UTIL: fill SOC line nicely
# --------------------------
def fill_soc_line(soc_line, start_soc):
    """
    Replace None values with previous known SOC (flat carry-forward).
    """
    last = start_soc
    for i in range(24):
        if soc_line[i] is None:
            soc_line[i] = last
        else:
            last = soc_line[i]