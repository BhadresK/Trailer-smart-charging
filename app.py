# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend import (
    read_price_24_from_local,
    read_price_24_from_github_raw,
    compute_charging_cost,
)

st.set_page_config(page_title="Trailer Smart Charging", page_icon="🚗", layout="centered")
st.title("🚗 Trailer Smart Charging Calculator")

st.markdown("**Pick how to load your hourly prices:**")
source = st.radio("Data source", ["Local file in this repo", "GitHub RAW URL"], index=0)

# ---- Inputs for data source ----
price_24 = None
price_df_preview = None

if source == "Local file in this repo":
    st.markdown("Example: `data/hourly_prices.csv` or `data/hourly_prices.xlsx` (with columns: hour, price)")
    local_path = st.text_input("Local path (relative to repo root)", value="data/hourly_prices.csv")
    if st.button("Load prices from local file"):
        try:
            price_24 = read_price_24_from_local(local_path)
            # Preview
            price_df_preview = pd.DataFrame({"hour": list(range(24)), "price": price_24})
            st.success("Loaded 24 hourly prices from local file ✅")
            st.dataframe(price_df_preview, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.markdown("Fill your GitHub info (RAW file):")
    owner = st.text_input("GitHub username/org", value="")
    repo = st.text_input("Repository name", value="Trailer-smart-charging")
    branch = st.text_input("Branch (e.g., main)", value="main")
    path_in_repo = st.text_input("Path in repo (e.g., data/hourly_prices.csv)", value="data/hourly_prices.csv")
    if st.button("Fetch prices from GitHub RAW"):
        try:
            price_24 = read_price_24_from_github_raw(owner, repo, branch, path_in_repo)
            # Preview
            price_df_preview = pd.DataFrame({"hour": list(range(24)), "price": price_24})
            st.success("Fetched 24 hourly prices from GitHub RAW ✅")
            st.dataframe(price_df_preview, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# ---- EV & schedule inputs ----
st.subheader("EV & Charging Window")
col1, col2 = st.columns(2)
with col1:
    battery_kWh = st.number_input("Battery Capacity (kWh)", min_value=10, max_value=200, value=60)
    soc_arrival = st.slider("SoC at Arrival (%)", 0, 100, 40)
    soc_target = st.slider("Target SoC (%)", 0, 100, 100)
with col2:
    arrival_time = st.text_input("Arrival Time (HH:MM)", value="16:30")
    departure_time = st.text_input("Departure Time (HH:MM)", value="07:30")
    power_kW = st.number_input("Charger Power (kW)", min_value=1.0, max_value=50.0, value=11.0, step=0.5)

# ---- Compute button ----
if st.button("Compute Smart vs Dumb Cost"):
    if price_24 is None:
        st.warning("Load prices first (local file or GitHub RAW).")
    else:
        try:
            result = compute_charging_cost(
                battery_kWh=battery_kWh,
                soc_arrival_pc=soc_arrival,
                soc_target_pc=soc_target,
                arrival_time=arrival_time,
                departure_time=departure_time,
                price_24=price_24,
                power_kW=power_kW,
            )

            st.subheader("Results")
            st.write(f"**Energy Needed:** {result['energy_needed_kWh']} kWh")
            st.write(f"**Energy Deliverable in Window:** {result['energy_deliverable_kWh']} kWh")
            st.write(f"**Enough to meet target?** {'✅ Yes' if result['enough_to_meet_target'] else '❌ No'}")
            st.write(f"**Dumb Charging Cost:** €{result['dumb_cost_EUR']}")
            st.write(f"**Smart Charging Cost:** €{result['smart_cost_EUR']}")

            # ---- Bar chart: cost comparison ----
            st.subheader("Cost Comparison")
            cost_df = pd.DataFrame(
                {"Scenario": ["Dumb", "Smart"], "Cost (€)": [result["dumb_cost_EUR"], result["smart_cost_EUR"]]}
            )
            st.bar_chart(cost_df.set_index("Scenario"))

            # ---- Simple SoC progression chart (Arrival -> Departure) ----
            st.subheader("SoC Progression (simplified)")
            fig, ax = plt.subplots()
            ax.plot(["Arrival", "Departure"], [soc_arrival, soc_target], marker="o")
            ax.set_ylim(0, 100)
            ax.set_ylabel("State of Charge (%)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Computation error: {e}")