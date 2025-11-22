# app.py
import streamlit as st
import pandas as pd
from backend import read_price_24_from_github_raw, compute_day_traces
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Page config & custom styles
# ---------------------------
st.set_page_config(page_title="Trailer Smart Charging", layout="wide")
PRIMARY_BLUE = "#0f172a"   # navy blue left panel
ACCENT_BLUE  = "#1e3a8a"   # header blue
SMART_ORANGE = "#f59e0b"
DUMB_GRAY    = "#6b7280"
PRICE_TEAL   = "#14b8a6"
SOC_SMART    = "#22c55e"
SOC_DUMB     = "#3b82f6"

st.markdown(f"""
<style>
/* Left panel look */
.left-panel {{
    background: {PRIMARY_BLUE};
    padding: 16px 20px;
    border-radius: 10px;
    color: #ffffff;
}}
.left-panel h2 {{
    margin-top: 0;
    font-weight: 700;
}}

/* Compact label + input rows */
.input-row {{
    display: grid;
    grid-template-columns: 48% 48%;
    column-gap: 4%;
    align-items: center;
    margin: 6px 0;
}}
.input-row .label {{
    font-weight: 600;
    font-size: 0.95rem;
    opacity: 0.95;
}}
.input-row .field > div {{
    margin-top: -10px; /* tighten vertical space */
}}

/* Styled tables */
.table-card {{
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 12px;
}}
.table-card .title {{
    background: {ACCENT_BLUE};
    color: #ffffff;
    font-weight: 700;
    padding: 10px 14px;
    font-size: 1.05rem;
}}
.table-card table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}}
.table-card th, .table-card td {{
    padding: 10px 12px;
    border-bottom: 1px solid #e5e7eb;
}}
.table-card th {{
    background: #f8fafc;
    font-weight: 700;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Layout: 30% / 70%
# ---------------------------
left_col, right_col = st.columns([0.30, 0.70])

# ---------------------------
# LEFT: All inputs (compact)
# ---------------------------
with left_col:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Inputs")

    # Build compact rows: label + input side by side
    # Each control uses label_visibility='collapsed' to save space
    # and a separate text label to the left.
    def compact_number(label, key, minv, maxv, val, step=1.0, format_str=None):
        c1, c2 = st.columns([0.48, 0.48])
        with c1:
            st.markdown(f'<div class="label">{label}</div>', unsafe_allow_html=True)
        with c2:
            return st.number_input("", min_value=minv, max_value=maxv, value=val,
                                   step=step, key=key, format=format_str,
                                   label_visibility="collapsed")
    def compact_text(label, key, val):
        c1, c2 = st.columns([0.48, 0.48])
        with c1:
            st.markdown(f'<div class="label">{label}</div>', unsafe_allow_html=True)
        with c2:
            return st.text_input("", value=val, key=key, label_visibility="collapsed")

    battery_kWh    = compact_number("Battery Capacity (kWh)", "battery", 10.0, 200.0, 60.0, step=1.0)
    soc_arrival_pc = compact_number("SoC at Arrival (%)", "soc_in", 0.0, 100.0, 40.0, step=1.0)
    soc_target_pc  = compact_number("SoC Target (%)", "soc_tgt", 0.0, 100.0, 100.0, step=1.0)

    arrival_time   = compact_text("Arrival (HH:MM)", "arr", "16:30")
    departure_time = compact_text("Departure (HH:MM)", "dep", "07:30")

    max_charger_kW = compact_number("Max Charger (kW)", "maxkW", 1.0, 50.0, 11.0, step=0.5)
    batt_eff_frac  = compact_number("Battery Eff. (%)", "beff", 50.0, 100.0, 97.0, step=0.5) / 100.0
    obc_eff_frac   = compact_number("OBC Eff. (%)", "oeff", 50.0, 100.0, 96.0, step=0.5) / 100.0

    grid_kVA       = compact_number("Grid Limit (kVA)", "gkva", 5.0, 100.0, 22.0, step=0.5)
    pf_site        = compact_number("Power Factor", "pf", 0.50, 1.00, 0.98, step=0.01, format_str="%.2f")

    days_per_month = compact_number("Parked Days / Month", "dpm", 1.0, 31.0, 20.0, step=1.0)
    winter_months  = compact_number("Winter Months", "wm", 0.0, 12.0, 6.0, step=1.0)
    summer_months  = compact_number("Summer Months", "sm", 0.0, 12.0, 6.0, step=1.0)

    st.markdown("</div>", unsafe_allow_html=True)  # end left-panel

# ---------------------------
# RIGHT: Chart + Tables
# ---------------------------
with right_col:
    st.markdown("## 📊 Smart vs Dumb — Full 24 h")
    # 1) Load prices (auto)
    price_24 = read_price_24_from_github_raw()

    # 2) Compute traces
    ev = dict(
        battery_kWh=battery_kWh,
        soc_arrival_pc=soc_arrival_pc,
        soc_target_pc=soc_target_pc,
        arrival_time=arrival_time,
        departure_time=departure_time,
        max_charger_kW=max_charger_kW,
        batt_eff_frac=batt_eff_frac,
        obc_eff_frac=obc_eff_frac,
        grid_kVA=grid_kVA,
        pf_site=pf_site,
    )
    traces = compute_day_traces(ev, price_24)

    # 3) Build professional Plotly figure (two subplots)
    hours = list(range(24))
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=("Power & Price per Hour", "State of Charge (%) per Hour")
    )

    # Top subplot: side-by-side bars for dumb vs smart power, plus price line
    fig.add_bar(
        x=hours, y=traces["dumb_power_grid_kW"],
        name="Dumb Power (kW)", marker_color=DUMB_GRAY, opacity=0.55,
        hovertemplate="Hour %{x}<br>Dumb Power: %{y:.2f} kW<extra></extra>",
        row=1, col=1
    )
    fig.add_bar(
        x=hours, y=traces["smart_power_grid_kW"],
        name="Smart Power (kW)", marker_color=SMART_ORANGE, opacity=0.65,
        hovertemplate="Hour %{x}<br>Smart Power: %{y:.2f} kW<extra></extra>",
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=hours, y=traces["hourly_price_24"],
            name="Hourly Price (€/kWh)", mode="lines+markers",
            line=dict(color=PRICE_TEAL, width=3), marker=dict(size=6),
            hovertemplate="Hour %{x}<br>Price: €%{y:.3f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True
    )

    # Bottom subplot: SoC lines dumb vs smart
    fig.add_trace(
        go.Scatter(
            x=hours, y=traces["dumb_soc_pc"],
            name="Dumb SoC (%)", mode="lines+markers",
            line=dict(color=SOC_DUMB, width=3), marker=dict(size=5),
            hovertemplate="Hour %{x}<br>SoC Dumb: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=hours, y=traces["smart_soc_pc"],
            name="Smart SoC (%)", mode="lines+markers",
            line=dict(color=SOC_SMART, width=3), marker=dict(size=5),
            hovertemplate="Hour %{x}<br>SoC Smart: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1
    )

    # Styling axes, grid, ranges
    fig.update_xaxes(
        title_text="Hour of Day (0–23)", tickmode="linear", tick0=0, dtick=1,
        showgrid=True, gridcolor="#e5e7eb", row=2, col=1
    )
    fig.update_yaxes(
        title_text="Power (kW)", showgrid=True, gridcolor="#e5e7eb",
        rangemode="tozero", row=1, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Price (€/kWh)", showgrid=False,
        rangemode="tozero", row=1, col=1, secondary_y=True
    )
    fig.update_yaxes(
        title_text="SoC (%)", range=[0, 100], showgrid=True, gridcolor="#e5e7eb",
        row=2, col=1
    )

    # Layout aesthetics
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=20, r=20, t=60, b=20),
        height=640,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # Cost per day (2 rows)
    # ---------------------
    def render_table_html(title, df: pd.DataFrame):
        html = f"""
        <div class="table-card">
          <div class="title">{title}</div>
          <div style="padding: 10px 12px;">
            {df.to_html(index=False, justify='center')}
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    daily_df = pd.DataFrame({
        "Scenario": ["Dumb Charging", "Smart Charging"],
        "Cost per Day (€)": [f"{traces['dumb_cost_EUR']:.2f}", f"{traces['smart_cost_EUR']:.2f}"]
    })
    render_table_html("Cost Breakdown (per Day)", daily_df)

    # -----------------------------
    # Yearly cost & savings (table)
    # -----------------------------
    total_months = int(winter_months + summer_months)
    yearly_dumb  = float(traces["dumb_cost_EUR"])  * float(days_per_month) * total_months
    yearly_smart = float(traces["smart_cost_EUR"]) * float(days_per_month) * total_months
    yearly_save  = yearly_dumb - yearly_smart

    yearly_df = pd.DataFrame({
        "Metric": [
            "Yearly Dumb Cost (EUR)",
            "Yearly Smart Cost (EUR)",
            "Savings (Smart vs Dumb)",
        ],
        "Value": [
            f"€{yearly_dumb:.2f}",
            f"€{yearly_smart:.2f}",
            f"€{yearly_save:.2f}",
        ]
    })
    render_table_html("Yearly Cost & Savings", yearly_df)
