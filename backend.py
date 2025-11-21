# backend.py
from datetime import datetime, timedelta

def compute_charging_cost(battery_kWh, soc_arrival, soc_target, arrival_time, departure_time, price_24):
    # 1. Energy needed
    energy_needed = battery_kWh * (soc_target - soc_arrival) / 100
    
    # 2. Time window
    fmt = "%H:%M"
    t_arr = datetime.strptime(arrival_time, fmt)
    t_dep = datetime.strptime(departure_time, fmt)
    if t_dep <= t_arr:
        t_dep += timedelta(days=1)
    parked_hours = int((t_dep - t_arr).total_seconds() / 3600)
    
    # 3. Dumb charging cost (charge immediately)
    dumb_cost = energy_needed * sum(price_24[:parked_hours]) / parked_hours  # avg price
    
    # 4. Smart charging cost (sort by cheapest hours)
    sorted_prices = sorted(price_24[:parked_hours])
    smart_cost = 0
    remaining_energy = energy_needed
    power_kW = 11  # assume 11 kW charger
    for price in sorted_prices:
        if remaining_energy <= 0:
            break
        smart_cost += power_kW * price  # cost for 1 hour
        remaining_energy -= power_kW
    
    return {
        "energy_needed_kWh": round(energy_needed, 2),
        "dumb_cost_EUR": round(dumb_cost, 2),
        "smart_cost_EUR": round(smart_cost, 2)
    }

# Example usage
winter_prices = [0.35]*24  # dummy prices
result = compute_charging_cost(60, 40, 100, "16:30", "07:30", winter_prices)
print(result)