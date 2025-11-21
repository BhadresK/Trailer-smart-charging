# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backend import read_price_24_from_github_raw, compute_charging_cost

st.set_page_config(page_title="Trailer Smart Charging", page_icon="🚗", layout="centered")
st.title("🚗 Trailer Smart Charging Calculator")

# Auto-load prices from GitHub
st.subheader("Hourly Prices (Auto-loaded from GitHub)")
try:
    price_24 = read_price_24_from_github_raw()
    price_df = pd.DataFrame({"Hour": list(range(24)), "Price (€)": price_24})
    st.dataframe(price_df, use_container_width=True)
except Exception as e:
    st.error(f"Error loading prices: {e}")
    st.stop()

st.divider()
st.subheader("EV & Charging Window")
col1, col2 = st.columns(2)
with col1:
    battery_kWh = st.number_input("Battery Capacity (kWh)", 10, 200, 60)
    soc_arrival = st.slider("SoC at Arrival (%)", 0, 100, 40)
    soc_target = st.slider("Target SoC (%)", 0, 100, 100)
with col2:
    arrival_time = st.text_input("Arrival Time (HH:MM)", "16:30")
    departure_time = st.text_input("Departure Time (HH:MM)", "07:30")
    power_kW = st.number_input("Charger Power (kW)", 1.0, 50.0, 11.0)

if st.button("Compute Smart vs Dumb Cost"):
    result = compute_charging_cost(battery_kWh, soc_arrival, soc_target, arrival_time, departure_time, price_24, power_kW)
    st.subheader("Results")
    st.write(f"Energy Needed: {result['energy_needed_kWh']} kWh")
    st.write(f"Energy Deliverable: {result['energy_deliverable_kWh']} kWh")
    st.write(f"Enough to meet target? {'✅ Yes' if result['enough_to_meet_target'] else '❌ No'}")
    st.write(f"Dumb Charging Cost: €{result['dumb_cost_EUR']}")
    st.write(f"Smart Charging Cost: €{result['smart_cost_EUR']}")

    # Charts
    st.subheader("Cost Comparison")
    cost_df = pd.DataFrame({"Scenario": ["Dumb", "Smart"], "Cost (€)": [result["dumb_cost_EUR"], result["smart_cost_EUR"]]})
    st.bar_chart(cost_df.set_index("Scenario"))

    st.subheader("SoC Progression")
    fig, ax = plt.subplots()
    ax.plot(["Arrival", "Departure"], [soc_arrival, soc_target], marker="o")
    ax.set_ylim(0, 100)
    ax.set_ylabel("State of Charge (%)")
    ax.grid(True)
    st.pyplot(fig)