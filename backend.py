# backend.py
from datetime import datetime, timedelta

def compute_charging_cost(battery_kWh, soc_arrival, soc_target, arrival_time, departure_time, price_24):
    """
    Calculate energy needed and cost for dumb vs smart charging.
    """
    # 1. Energy needed
    energy_needed = battery_kWh * (soc_target - soc_arrival) / 100

    # 2. Time window
    fmt = "%H:%M"
    t_arr = datetime.strptime(arrival_time, fmt)
    t_dep = datetime.strptime(departure_time, fmt)
    if t_dep <= t_arr:
        t_dep += timedelta(days=1)
    parked_hours = int((t_dep - t_arr).total_seconds() / 3600)

    # 3. Dumb charging cost (average price)
    avg_price = sum(price_24[:parked_hours]) / parked_hours
    dumb_cost = energy_needed * avg_price

    # 4. Smart charging cost (cheapest hours first)
    sorted_prices = sorted(price_24[:parked_hours])
    smart_cost = 0
    remaining_energy = energy_needed
    power_kW = 11  # assume 11 kW charger
    for price in sorted_prices:
        if remaining_energy <= 0:
            break
        smart_cost += min(power_kW, remaining_energy) * price
        remaining_energy -= power_kW

    return {
        "energy_needed_kWh": round(energy_needed, 2),
        "dumb_cost_EUR": round(dumb_cost, 2),
        "smart_cost_EUR": round(smart_cost, 2)
    }