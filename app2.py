# app.py — Tkinter front-end for EV & Reefer smart charging
# Author: M365 Copilot (for Bhadreshvara, Kuldip)
# Date: 2025-11-22

import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import backend as be  # local module

APP_TITLE = 'EV Charging Cost — Interactive'
APP_SIZE = (1320, 860)
DAYS_PER_MONTH = 20

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        w, h = APP_SIZE
        self.geometry(f"{w}x{h}")
        self.configure(bg='white')
        self._setup_style()
        self._build_layout()
        self._load_data()
        self._wire_events()
        self._recompute()

    def _setup_style(self):
        style = ttk.Style(self)
        try: style.theme_use('clam')
        except Exception: pass
        style.configure('TFrame', background='white')
        style.configure('TLabel', background='white', font=('Segoe UI', 10))
        style.configure('Header.TLabel', background='white', font=('Segoe UI', 11, 'bold'))
        style.configure('Box.TLabelframe', background='white')
        style.configure('Box.TLabelframe.Label', background='white', font=('Segoe UI', 11, 'bold'))
        style.configure('Treeview', font=('Segoe UI', 10))
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # Left panel
        self.left = ttk.Frame(self); self.left.grid(row=0, column=0, sticky='nsew', padx=12, pady=12)
        self.left.grid_rowconfigure(2, weight=1); self.left.grid_columnconfigure(0, weight=1)

        # Middle panel
        self.mid = ttk.Frame(self); self.mid.grid(row=0, column=1, sticky='nsew', padx=8, pady=12)
        self.mid.grid_rowconfigure(0, weight=1); self.mid.grid_rowconfigure(1, weight=1); self.mid.grid_columnconfigure(0, weight=1)

        # Right panel
        self.right = ttk.Frame(self); self.right.grid(row=0, column=2, sticky='nsew', padx=12, pady=12)
        for r in range(5): self.right.grid_rowconfigure(r, weight=0)
        self.right.grid_columnconfigure(0, weight=1)

        # Inputs (left)
        frm_inputs = ttk.LabelFrame(self.left, text='Adjust inputs', style='Box.TLabelframe'); frm_inputs.grid(row=0, column=0, sticky='new')
        for r in range(7): frm_inputs.grid_rowconfigure(r, weight=0)
        frm_inputs.grid_columnconfigure(0, weight=0); frm_inputs.grid_columnconfigure(1, weight=1)

        self.var_arr = tk.StringVar(value='16:30')
        self.var_dep = tk.StringVar(value='07:30')
        self.var_soc_arr_w = tk.IntVar(value=80)
        self.var_soc_arr_s = tk.IntVar(value=40)
        self.var_soc_tgt = tk.IntVar(value=100)
        self.var_w_months = tk.IntVar(value=6)
        self.var_s_months = tk.IntVar(value=6)

        def add_row(r, label, widget):
            ttk.Label(frm_inputs, text=label, style='Header.TLabel').grid(row=r, column=0, sticky='e', padx=6, pady=4)
            widget.grid(row=r, column=1, sticky='ew', padx=6, pady=4)

        add_row(0, 'Arrival time:', ttk.Entry(frm_inputs, textvariable=self.var_arr, width=10))
        add_row(1, 'Departure time:', ttk.Entry(frm_inputs, textvariable=self.var_dep, width=10))
        add_row(2, 'Winter SoC at arrival (%):', ttk.Spinbox(frm_inputs, from_=0, to=100, textvariable=self.var_soc_arr_w, width=6))
        add_row(3, 'Summer SoC at arrival (%):', ttk.Spinbox(frm_inputs, from_=0, to=100, textvariable=self.var_soc_arr_s, width=6))
        add_row(4, 'Winter months:', ttk.Spinbox(frm_inputs, from_=0, to=12, textvariable=self.var_w_months, width=6))
        add_row(5, 'Summer months:', ttk.Spinbox(frm_inputs, from_=0, to=12, textvariable=self.var_s_months, width=6))
        add_row(6, 'SoC required at departure (%):', ttk.Spinbox(frm_inputs, from_=0, to=100, textvariable=self.var_soc_tgt, width=6))

        # Complementary months logic
        def on_w_months(*_): self.var_s_months.set(max(0, 12 - self.var_w_months.get())); self._recompute()
        def on_s_months(*_): self.var_w_months.set(max(0, 12 - self.var_s_months.get())); self._recompute()
        self.var_w_months.trace_add('write', on_w_months)
        self.var_s_months.trace_add('write', on_s_months)

        # Reefer selection
        frm_cycle = ttk.LabelFrame(self.left, text='Reefer cycle at stationary', style='Box.TLabelframe')
        frm_cycle.grid(row=1, column=0, sticky='new', pady=(8, 8))
        self.var_cycle = tk.StringVar(value='Continuous')
        ttk.Radiobutton(frm_cycle, text='Continuous', value='Continuous', variable=self.var_cycle, command=self._recompute).pack(anchor='w', padx=8, pady=4)
        ttk.Radiobutton(frm_cycle, text='Start-Stop', value='Start-Stop', variable=self.var_cycle, command=self._recompute).pack(anchor='w', padx=8, pady=4)
        ttk.Radiobutton(frm_cycle, text='Reefer OFF', value='NoReeferStationary', variable=self.var_cycle, command=self._recompute).pack(anchor='w', padx=8, pady=4)

        # Reefer preview
        self.frm_prev = ttk.LabelFrame(self.left, text='Reefer Cycle — Continuous', style='Box.TLabelframe')
        self.frm_prev.grid(row=2, column=0, sticky='nsew')
        self.fig_prev = Figure(figsize=(3.2, 2.2), dpi=100)
        self.ax_prev = self.fig_prev.add_subplot(111)
        self.ax_prev.set_xlabel('Time (HH:MM)'); self.ax_prev.set_ylabel('kW'); self.ax_prev.grid(True, linestyle=':', alpha=0.6)
        self.canvas_prev = FigureCanvasTkAgg(self.fig_prev, master=self.frm_prev)
        self.canvas_prev.get_tk_widget().pack(fill='both', expand=True)

        # Winter graph (middle)
        self.frm_w = ttk.LabelFrame(self.mid, text='Dumb vs Smart charging (Winter)', style='Box.TLabelframe'); self.frm_w.grid(row=0, column=0, sticky='nsew')
        self.fig_w = Figure(figsize=(6.0, 3.0), dpi=100); self.ax_w = self.fig_w.add_subplot(111)
        self.canvas_w = FigureCanvasTkAgg(self.fig_w, master=self.frm_w); self.canvas_w.get_tk_widget().pack(fill='both', expand=True)

        # Summer graph (middle)
        self.frm_s = ttk.LabelFrame(self.mid, text='Dumb vs Smart charging (Summer)', style='Box.TLabelframe'); self.frm_s.grid(row=1, column=0, sticky='nsew', pady=(8, 0))
        self.fig_s = Figure(figsize=(6.0, 3.0), dpi=100); self.ax_s = self.fig_s.add_subplot(111)
        self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=self.frm_s); self.canvas_s.get_tk_widget().pack(fill='both', expand=True)

        # Right: tables
        self.tbl_charge  = self._make_table(self.right, 'Battery Charging Cost Comparison (in €)', ['Metric', 'Winter', 'Summer'], row=0)
        self.tbl_trailer = self._make_table(self.right, 'Reefer Consumption Cost Comparison (in €)', ['Metric', 'Winter', 'Summer'], row=1)
        self.tbl_yearly  = self._make_table(self.right, 'Estimated Yearly Values (in € ; 20 Days/ Month)', ['Metric', 'Value'], row=2)

        # Info panel
        self.frm_info = ttk.LabelFrame(self.right, text='Understanding This Panel', style='Box.TLabelframe'); self.frm_info.grid(row=3, column=0, sticky='nsew', pady=(8,0))
        self.lbl_info = ttk.Label(self.frm_info, text='', justify='left', wraplength=320); self.lbl_info.pack(fill='both', expand=True, padx=8, pady=8)

        # Bottom buttons
        self.frm_btn = ttk.Frame(self.right); self.frm_btn.grid(row=4, column=0, sticky='ew', pady=(6,0))
        ttk.Button(self.frm_btn, text='Recompute', command=self._recompute).pack(side='left')

    def _make_table(self, parent, header, cols, row=0):
        frm = ttk.LabelFrame(parent, text=header, style='Box.TLabelframe'); frm.grid(row=row, column=0, sticky='nsew', pady=(0,8))
        tv = ttk.Treeview(frm, columns=cols, show='headings', height=6)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, anchor='center', width=120 if c != 'Metric' else 180)
        tv.pack(fill='both', expand=True)
        return tv

    def _load_data(self):
        try:
            winterWD, summerWD = be.read_price_excel('avg_price.xlsx')
            self.tariff = be.get_tariff_params()
            self.winterALL = be.compose_all_in_price(winterWD, self.tariff)
            self.summerALL = be.compose_all_in_price(summerWD, self.tariff)
        except Exception as e:
            messagebox.showerror('Missing price data', f"Cannot read avg_price.xlsx: {e}")
            self.destroy(); return
        try:
            self.socDF = be.read_taper_table('time_soc.xlsx')
        except Exception as e:
            messagebox.showerror('Missing taper data', f"Cannot read time_soc.xlsx: {e}")
            self.destroy(); return
        self.EV = be.EVParams(
            BatteryCapacity_kWh=70, UsableBatteryCap_kWh=60, BatteryChargingEffi_pc=97,
            OBC_Capacity_kW=22, OBC_UsableCapacity_kW=21.8, OBCEfficiency_pc=96,
            SOC_arrival_winter_pc=self.var_soc_arr_w.get(), SOC_arrival_summer_pc=self.var_soc_arr_s.get(),
            SOC_departure_target_pc=self.var_soc_tgt.get(), ArrivalTime_HHMM=self.var_arr.get(),
            DepartureTime_HHMM=self.var_dep.get(), MaxChargingPower_kW=22,
        )
        self.EV.finalize()

    def _wire_events(self):
        for v in [self.var_arr, self.var_dep, self.var_soc_arr_w, self.var_soc_arr_s, self.var_soc_tgt, self.var_cycle]:
            v.trace_add('write', lambda *_: self._recompute())

    def _recompute(self):
        try:
            self.EV.SOC_departure_target_pc = be.clamp(self.var_soc_tgt.get(), 0, 100)
            self.EV.ArrivalTime_HHMM = self.var_arr.get().strip()
            self.EV.DepartureTime_HHMM = self.var_dep.get().strip()
            self.EV.finalize()

            EVW = self._clone_ev(self.EV); EVS = self._clone_ev(self.EV)
            SOC_arrW_kWh = self.EV.UsableBatteryCap_kWh * (be.clamp(self.var_soc_arr_w.get(), 0, 100)/100.0)
            SOC_arrS_kWh = self.EV.UsableBatteryCap_kWh * (be.clamp(self.var_soc_arr_s.get(), 0, 100)/100.0)
            EVW.CurrentSOC_kWh = min(self.EV.UsableBatteryCap_kWh, max(0.0, SOC_arrW_kWh))
            EVS.CurrentSOC_kWh = min(self.EV.UsableBatteryCap_kWh, max(0.0, SOC_arrS_kWh))

            eff_frac2 = max(np.finfo(float).eps, (self.EV.BatteryChargingEffi_pc * self.EV.OBCEfficiency_pc) / 10000.0)
            soc_bp, Pcap_grid_bp_kW = be.build_taper_lookup(self.socDF, self.EV, eff_frac2)

            t_arr, t_dep, t, dt_hr = be.build_time_vector(self.EV.ArrivalTime_HHMM, self.EV.DepartureTime_HHMM)
            if len(t) == 0: return

            cycleUI = self.var_cycle.get()
            self.frm_prev.configure(text=f'Reefer Cycle — {"Reefer OFF" if cycleUI=="NoReeferStationary" else cycleUI}')
            t10_prev = [t_arr + timedelta(seconds=i*10) for i in range(int((70*60)/10))]
            if cycleUI == 'NoReeferStationary':
                P_reefer_1h_kW = np.zeros(len(t10_prev))
                self.ax_prev.clear(); self.ax_prev.plot(t10_prev, P_reefer_1h_kW, color=(0.35,0.35,0.35), lw=1.8)
                self.ax_prev.set_ylim(0, 1)
            else:
                P_reefer_1h_kW = be.get_reefer_cycle_trace(cycleUI, len(t10_prev), dt_sec=10)
                self.ax_prev.clear(); self.ax_prev.step(t10_prev, P_reefer_1h_kW, where='post', color=(0.00,0.45,0.10), lw=1.8)
                yMax = max(1.0, float(np.max(P_reefer_1h_kW))*1.2); self.ax_prev.set_ylim(0, yMax)
            self.ax_prev.set_xlabel('Time (HH:MM)'); self.ax_prev.set_ylabel('kW'); self.ax_prev.grid(True, linestyle=':', alpha=0.6)
            self.canvas_prev.draw()

            PF_reefer, kVA_refr_cap = 0.75, 19.765
            if cycleUI == 'NoReeferStationary':
                P_refr_min_kW = np.zeros(len(t)); kVA_refr_min = np.zeros(len(t))
            else:
                P_refr_min_kW, kVA_refr_min = be.build_reefer_stationary_minute_trace(cycleUI, t, t_arr, t_dep, 10, PF_reefer, kVA_refr_cap)

            target_kWh = self.EV.UsableBatteryCap_kWh * (self.var_soc_tgt.get()/100.0)
            needW_kWh = max(0.0, target_kWh - EVW.CurrentSOC_kWh)
            needS_kWh = max(0.0, target_kWh - EVS.CurrentSOC_kWh)

            Wsmart = be.plan_smart(t, dt_hr, self.winterALL, EVW, soc_bp, Pcap_grid_bp_kW, needW_kWh, eff_frac2, P_refr_min_kW, kVA_refr_min)
            Wbase  = be.plan_baseline(t, dt_hr, Wsmart['price_min'], EVW, soc_bp, Pcap_grid_bp_kW, eff_frac2, P_refr_min_kW, kVA_refr_min)
            Ssmart = be.plan_smart(t, dt_hr, self.summerALL, EVS, soc_bp, Pcap_grid_bp_kW, needS_kWh, eff_frac2, P_refr_min_kW, kVA_refr_min)
            Sbase  = be.plan_baseline(t, dt_hr, Ssmart['price_min'], EVS, soc_bp, Pcap_grid_bp_kW, eff_frac2, P_refr_min_kW, kVA_refr_min)

            Rparams = be.get_reefer_cost_params()
            RW = be.compute_reefer_cost_scenarios(P_refr_min_kW, dt_hr, Wsmart['price_min'], Rparams)
            RS = be.compute_reefer_cost_scenarios(P_refr_min_kW, dt_hr, Ssmart['price_min'], Rparams)

            rp = be.get_reefer_cost_params()
            cost_fixed_W_EV = Wbase['energy_kWh'] * rp.FixedPrice_EUR_per_kWh
            cost_fixed_S_EV = Sbase['energy_kWh'] * rp.FixedPrice_EUR_per_kWh
            self._fill_table(self.tbl_charge, [
                ('Energy needed (kWh)', f"{needW_kWh:.2f}", f"{needS_kWh:.2f}"),
                ('Fixed Price Charging', f"{cost_fixed_W_EV:.2f}", f"{cost_fixed_S_EV:.2f}"),
                ('Dumb Charging', f"{Wbase['cost_EUR']:.2f}", f"{Sbase['cost_EUR']:.2f}"),
                ('Smart Charging', f"{Wsmart['cost_EUR']:.2f}", f"{Ssmart['cost_EUR']:.2f}"),
            ])

            winterMonths = int(self.var_w_months.get()); summerMonths = int(self.var_s_months.get())
            yearly_fixed = cost_fixed_W_EV*DAYS_PER_MONTH*winterMonths + cost_fixed_S_EV*DAYS_PER_MONTH*summerMonths
            yearly_dumb  = Wbase['cost_EUR']*DAYS_PER_MONTH*winterMonths + Sbase['cost_EUR']*DAYS_PER_MONTH*summerMonths
            yearly_smart = Wsmart['cost_EUR']*DAYS_PER_MONTH*winterMonths + Ssmart['cost_EUR']*DAYS_PER_MONTH*summerMonths
            sav_smart_vs_fixed = yearly_fixed - yearly_smart
            sav_smart_vs_dumb  = yearly_dumb - yearly_smart
            self._fill_table(self.tbl_yearly, [
                ('Fixed Price Charging Cost', f"€{yearly_fixed:.2f}"),
                ('Dumb Charging Cost', f"€{yearly_dumb:.2f}"),
                ('Smart Charging Cost', f"€{yearly_smart:.2f}"),
                ('Savings (Smart vs Fixed Price Charging)', f"€{sav_smart_vs_fixed:.2f}"),
                ('Savings (Smart vs Dumb Charging)', f"€{sav_smart_vs_dumb:.2f}"),
            ])

            self._fill_table(self.tbl_trailer, [
                ('Energy used by trailer (kWh)', f"{RW['E_kWh']:.2f}", f"{RS['E_kWh']:.2f}"),
                ('Diesel powered', f"{RW['cost_diesel']:.2f}", f"{RS['cost_diesel']:.2f}"),
                ('Fixed electricity price', f"{RW['cost_fixed']:.2f}", f"{RS['cost_fixed']:.2f}"),
                ('Dumb Charging', f"{RW['cost_dynamic']:.2f}", f"{RS['cost_dynamic']:.2f}"),
                ('Smart Charging', f"{RW['cost_dynamic']:.2f}", f"{RS['cost_dynamic']:.2f}"),
            ])

            info = [
                "1. Arrival & Departure: Times define the parked window.",
                "2. SoC: Battery % at arrival and the target % at departure.",
                "3. Winter vs Summer: Separate scenarios with seasonal prices.",
                "4. Dumb Charging: Charges at max power without price optimization.",
                "5. Smart Charging: Shifts charging to cheaper hours; still meets target.",
                "6. Reefer Cycle: Trailer refrigeration load while parked (Continuous/Start-Stop/OFF).",
                "7. Trailer Energy: kWh consumed by the reefer during the parked window.",
                "8. Cost Comparison: Charging & reefer costs under fixed vs dynamic pricing.",
                "9. Yearly Savings: Assumes 20 parked days/month; multiplies seasonal costs.",
                "10. Graph Colors: Blue = Winter Smart Power, Orange = Summer Smart Power; Gray = Dumb.",
                f"11. Fixed electricity price: €{rp.FixedPrice_EUR_per_kWh:.2f} per kWh.",
                f"12. Diesel price: €{rp.DieselPrice_EUR_per_L:.2f}/L; DG efficiency: {int(100*rp.Genset_efficiency_frac)}%; Energy density: {rp.Diesel_kWh_per_L:.1f} kWh/L.",
            ]
            self.lbl_info.config(text='\n'.join(info))
            self._render_charts(t_arr, t, dt_hr, Wbase, Wsmart, Sbase, Ssmart, self.winterALL, self.summerALL, self.EV.UsableBatteryCap_kWh)

        except Exception:
            traceback.print_exc()
            messagebox.showerror('Error', 'An error occurred while recomputing. See console for details.')

    def _fill_table(self, tv: ttk.Treeview, rows):
        tv.delete(*tv.get_children())
        for r in rows: tv.insert('', 'end', values=r)

    def _clone_ev(self, EV: be.EVParams) -> be.EVParams:
        C = be.EVParams(**{k:v for k,v in EV.__dict__.items() if k in EV.__dataclass_fields__})
        C.finalize()
        return C

    def _render_charts(self, t_arr, t, dt_hr, Wbase, Wsmart, Sbase, Ssmart, winterALL, summerALL, EVcap_kWh):
        matplotlib.rcParams.update({'axes.facecolor':'#ffffff'})
        t0 = t_arr.replace(hour=0, minute=0, second=0, microsecond=0)
        t24 = [t0 + timedelta(minutes=i) for i in range(1440)]
        idx = [int(((ti - t0).total_seconds()/60.0) % 1440) for ti in t]
        eW24 = np.array(winterALL).reshape(24); eS24 = np.array(summerALL).reshape(24)
        eWst = np.concatenate([eW24, eW24[-1:]]); eSst = np.concatenate([eS24, eS24[-1:]])
        tp = [t0 + timedelta(hours=h) for h in range(25)]
        hMid = [t0 + timedelta(hours=h, minutes=30) for h in range(24)]

        P_norm_W = np.zeros(1440); P_smart_W = np.zeros(1440)
        P_norm_S = np.zeros(1440); P_smart_S = np.zeros(1440)
        SOC_norm_W = np.full(1440, np.nan); SOC_smart_W = np.full(1440, np.nan)
        SOC_norm_S = np.full(1440, np.nan); SOC_smart_S = np.full(1440, np.nan)

        P_norm_W[idx] = Wbase['P_trace']; P_smart_W[idx] = Wsmart['P_trace']
        P_norm_S[idx] = Sbase['P_trace']; P_smart_S[idx] = Ssmart['P_trace']
        SOC_norm_W[idx] = 100.0 * Wbase['SOC_trace']/EVcap_kWh
        SOC_smart_W[idx] = 100.0 * Wsmart['SOC_trace']/EVcap_kWh
        SOC_norm_S[idx] = 100.0 * Sbase['SOC_trace']/EVcap_kWh
        SOC_smart_S[idx] = 100.0 * Ssmart['SOC_trace']/EVcap_kWh

        colW_dumb = (0.70, 0.70, 0.70); colW_smart = (0.30, 0.75, 0.93)
        colS_dumb = (0.70, 0.70, 0.70); colS_smart = (1.00, 0.60, 0.20)
        colSoC_d = (0.50, 0.50, 0.50); colSoC_s = (0.00, 0.00, 0.00)
        colPrLn = (0.00, 0.60, 0.00); colPrTx = (0.00, 0.45, 0.15)

        # WINTER
        ax = self.ax_w; ax.clear(); ax.grid(True, linestyle=':', alpha=0.6)
        ax.fill_between(t24, P_norm_W, step='pre', color=colW_dumb, alpha=0.30, label='Dumb Power')
        ax.fill_between(t24, P_smart_W, step='pre', color=colW_smart, alpha=0.35, label='Winter Smart Power')
        ax.set_ylabel('Power (kW)')
        leftYL = ax.get_ylim(); 
        if leftYL[1]-leftYL[0] <= 0: ax.set_ylim(0, 1); leftYL = ax.get_ylim()
        padFrac = 0.04; bandFrac = 0.12
        rngW = leftYL[1]-leftYL[0]; yBase = leftYL[0]+padFrac*rngW; yTop = yBase+bandFrac*rngW
        eMinW = float(np.min(eWst)); eMaxW = float(np.max(eWst))
        yPriceW = (yBase + (eWst-eMinW)/(eMaxW-eMinW)*(yTop-yBase)) if eMaxW!=eMinW else np.full_like(eWst, yBase)
        ax.step(tp, yPriceW, where='post', color=colPrLn, lw=1.0, label='Hourly Electricity Price')
        ax2 = ax.twinx()
        ax2.plot(t24, SOC_norm_W, '-', color=colSoC_d, lw=1.1, label='SoC Dumb')
        ax2.plot(t24, SOC_smart_W, '-', color=colSoC_s, lw=1.1, label='SoC Smart')
        ax2.set_ylim(0, 100); ax2.set_ylabel('SoC (%)')
        ax.set_xlim(t0, t0+timedelta(hours=24))
        ax.set_xticks([t0+timedelta(hours=h) for h in range(0, 25, 2)])
        yTxtW = yBase + ((eW24 - eMinW)/(eMaxW-eMinW) if eMaxW!=eMinW else np.zeros(24))*(yTop-yBase)
        for h in range(24): ax.text(hMid[h], yTxtW[h], f"€{eW24[h]:.2f}", color=colPrTx, fontsize=9, fontweight='bold', ha='center', va='bottom')
        lines, labels = [], []
        for a in [ax, ax2]: L = a.get_legend_handles_labels(); lines += L[0]; labels += L[1]
        ax.legend(lines, labels, loc='upper center', ncol=3)
        self.canvas_w.draw()

        # SUMMER
        ax = self.ax_s; ax.clear(); ax.grid(True, linestyle=':', alpha=0.6)
        ax.fill_between(t24, P_norm_S, step='pre', color=colS_dumb, alpha=0.30)
        ax.fill_between(t24, P_smart_S, step='pre', color=colS_smart, alpha=0.30, label='Summer Smart Power')
        ax.set_ylabel('Power (kW)')
        leftYL = ax.get_ylim(); 
        if leftYL[1]-leftYL[0] <= 0: ax.set_ylim(0, 1); leftYL = ax.get_ylim()
        rngS = leftYL[1]-leftYL[0]; yBaseS = leftYL[0]+padFrac*rngS; yTopS = yBaseS+bandFrac*rngS
        eMinS = float(np.min(eSst)); eMaxS = float(np.max(eSst))
        yPriceS = (yBaseS + (eSst-eMinS)/(eMaxS-eMinS)*(yTopS-yBaseS)) if eMaxS!=eMinS else np.full_like(eSst, yBaseS)
        ax.step(tp, yPriceS, where='post', color=colPrLn, lw=1.0)
        ax2 = ax.twinx()
        ax2.plot(t24, SOC_norm_S, '-', color=colSoC_d, lw=1.1)
        ax2.plot(t24, SOC_smart_S, '-', color=colSoC_s, lw=1.1)
        ax2.set_ylim(0, 100); ax2.set_ylabel('SoC (%)')
        ax.set_xlim(t0, t0+timedelta(hours=24))
        ax.set_xticks([t0+timedelta(hours=h) for h in range(0, 25, 2)])
        yTxtS = yBaseS + ((eS24 - eMinS)/(eMaxS-eMinS) if eMaxS!=eMinS else np.zeros(24))*(yTopS-yBaseS)
        for h in range(24): ax.text(hMid[h], yTxtS[h], f"€{eS24[h]:.2f}", color=colPrTx, fontsize=9, fontweight='bold', ha='center', va='bottom')
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='upper center', ncol=2)
        self.canvas_s.draw()

if __name__ == '__main__':
    App().mainloop()
