# app.py
import streamlit as st
from backend import compute_charging_cost

st.title("Trailer Smart Charging Calculator")

# Inputs
battery_kWh = st.number_input("Battery Capacity (kWh)", 40, 120, 60)
soc_arrival = st.slider("SoC at Arrival (%)", 0, 100, 40)
soc_target = st.slider("Target SoC (%)", 0, 100, 100)
arrival_time = st.text_input("Arrival Time (HH:MM)", "16:30")
departure_time = st.text_input("Departure Time (HH:MM)", "07:30")

# Dummy prices for now (24 hours)
prices = [0.35] * 24

if st.button("Compute Cost"):
    result = compute_charging_cost(battery_kWh, soc_arrival, soc_target, arrival_time, departure_time, prices)
    st.subheader("Results")
    st.write(f"Energy Needed: {result['energy_needed_kWh']} kWh")
    st.write(f"Dumb Charging Cost: €{result['dumb_cost_EUR']}")
    st.write(f"Smart Charging Cost: €{result['smart_cost_EUR']}")