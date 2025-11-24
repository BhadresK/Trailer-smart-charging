# backend.py — computational core for EV & Reefer smart charging
# Author: M365 Copilot (for Bhadreshvara, Kuldip)
# Date: 2025-11-22
# External data: avg_price.xlsx, time_soc.xlsx (same schema as MATLAB)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------- Utility structures -----------------------------
@dataclass
class EVParams:
    BatteryCapacity_kWh: float
    UsableBatteryCap_kWh: float
    BatteryChargingEffi_pc: float
    OBC_Capacity_kW: float
    OBC_UsableCapacity_kW: float
    OBCEfficiency_pc: float
    SOC_arrival_winter_pc: float
    SOC_arrival_summer_pc: float
    SOC_departure_target_pc: float
    ArrivalTime_HHMM: str
    DepartureTime_HHMM: str
    MaxChargingPower_kW: float

    # Fixed site defaults (mirroring MATLAB)
    PF_Reefer: float = 0.75
    GridPF_site: float = 0.98
    GridVoltage_V: float = 400
    GridCurrent_A: float = 32
    GridMax_kVA: float = None  # computed
    BattMaxCharge_kW: float = 14.148
    BattMaxDischarge_kW: float = 14.680
    EAxle_Efficiency_pc: float = 90
    EffectiveChargingPower_kW: float = None  # computed
    ReeferCycleInit: str = 'Continuous'  # 'Continuous'|'Start-Stop'|'NoReeferStationary'

    def finalize(self):
        self.GridMax_kVA = math.sqrt(3) * self.GridVoltage_V * self.GridCurrent_A / 1000.0
        self.EffectiveChargingPower_kW = min(self.OBC_UsableCapacity_kW, self.MaxChargingPower_kW)

# ----------------------------- Tariff components -----------------------------
def get_tariff_params() -> Dict[str, float]:
    # Same constants as in MATLAB EVSmartwithReefer10
    return {
        'ConcessionFee_ct': 1.992,
        'OffshoreGridLevy_ct': 0.816,
        'CHPLevy_ct': 0.277,
        'ElectricityTax_ct': 2.05,
        'NEV19Levy_ct': 1.558,
        'NetworkUsageFees_ct': 6.63,
        'SalesMargin_ct': 0.000,
        'VAT_pc': 19.0,
    }

def compose_all_in_price(spot_eur_24: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    fixed_ct = (
        params['ConcessionFee_ct'] + params['OffshoreGridLevy_ct'] +
        params['CHPLevy_ct'] + params['ElectricityTax_ct'] +
        params['NEV19Levy_ct'] + params['NetworkUsageFees_ct'] +
        params['SalesMargin_ct']
    )
    fixed_eur = fixed_ct / 100.0
    net = np.asarray(spot_eur_24, dtype=float) + fixed_eur
    all_in = net * (1.0 + params['VAT_pc']/100.0)
    return all_in

# ----------------------------- File I/O helpers ------------------------------
def read_price_excel(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read avg_price.xlsx (24 hourly values for WinterWD and SummerWD)."""
    df = pd.read_excel(path, engine='openpyxl')
    must = ['WinterWD', 'SummerWD']
    if not all(m in df.columns for m in must):
        raise ValueError(f"{path} must have columns: WinterWD, SummerWD (24 values).")
    winter = np.asarray(df['WinterWD']).reshape(-1)
    summer = np.asarray(df['SummerWD']).reshape(-1)
    if len(winter) != 24 or len(summer) != 24:
        raise ValueError("Price columns must each have 24 values.")
    if np.any(np.isnan(winter)) or np.any(np.isnan(summer)):
        raise ValueError("NaN in price columns.")
    return winter, summer

def read_taper_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine='openpyxl')
    if not {'Time', 'SoC'}.issubset(df.columns):
        raise ValueError("time_soc.xlsx must contain columns: Time (min), SoC (%).")
    df = df[['Time', 'SoC']].copy().dropna().sort_values('Time')
    minutes = df['Time'].to_numpy()
    if len(minutes) < 2 or np.any(np.diff(minutes) <= 0):
        raise ValueError('time_soc.xlsx "Time" must strictly increase.')
    return df

# ----------------------------- Core time helpers -----------------------------
def _parse_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.strip().split(':')
    return int(hh), int(mm)

def build_time_vector(arr_str: str, dep_str: str) -> Tuple[datetime, datetime, List[datetime], float]:
    base_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ah, am = _parse_hhmm(arr_str)
    dh, dm = _parse_hhmm(dep_str)
    t_arr = base_day + timedelta(hours=ah, minutes=am)
    t_dep = base_day + timedelta(hours=dh, minutes=dm)
    if t_dep <= t_arr:
        t_dep = t_dep + timedelta(days=1)
    dt_sec = 60  # 1-minute steps
    t = []
    cur = t_arr
    while cur <= t_dep - timedelta(seconds=dt_sec):
        t.append(cur)
        cur += timedelta(seconds=dt_sec)
    dt_hr = dt_sec / 3600.0
    return t_arr, t_dep, t, dt_hr

# ----------------------------- Charging taper -------------------------------
def build_taper_lookup(soc_df: pd.DataFrame, EV: EVParams, eff_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    minutes = soc_df['Time'].to_numpy()
    soc_pct = soc_df['SoC'].to_numpy()
    soc_frac = np.clip(soc_pct/100.0, 0.0, 1.0)
    SOC_kWh = soc_frac * EV.UsableBatteryCap_kWh
    dt_hr_vec = np.diff(minutes) / 60.0
    dSOC_kWh = np.diff(SOC_kWh)
    P_batt_kW = np.maximum(0.0, dSOC_kWh / dt_hr_vec)
    P_grid_kW = P_batt_kW / max(eff_frac, np.finfo(float).eps)
    if P_grid_kW.size >= 3:
        P_grid_kW = np.convolve(P_grid_kW, np.ones(3)/3.0, mode='same')
    soc_raw = soc_frac[:-1]
    P_raw = np.maximum(0.0, P_grid_kW)
    idx = np.argsort(soc_raw)
    soc_sorted = soc_raw[idx]
    P_sorted = P_raw[idx]
    uniq_soc, uniq_P, seen = [], [], set()
    for s, p in zip(soc_sorted, P_sorted):
        if s not in seen:
            seen.add(s)
            uniq_soc.append(s)
            uniq_P.append(p)
    return np.array(uniq_soc), np.array(uniq_P)

# ----------------------------- Reefer cycle ---------------------------------
REEFER_DEFAULTS = {
    'P_HIGH_C': 7.6,
    'P_LOW_C': 0.7,
    'P_HIGH_SS': 9.7,
    'P_MID_SS': 0.65,
    'P_LOW_SS': 0.0,
    'T_HIGH_C': 1717,
    'T_LOW_C': 292,
    'T_HIGH_SS': 975,
    'T_MID_SS': 295,
    'T_LOW_SS': 1207,
}

def get_reefer_cycle_trace(cycle_type: str, N: int, dt_sec: int = 60) -> np.ndarray:
    D = REEFER_DEFAULTS
    cycle_type_norm = cycle_type.replace(' ', '').replace('-', '')
    if cycle_type_norm.lower() in ['continuous']:
        pattern = np.array([[D['P_HIGH_C'], D['T_HIGH_C']],
                            [D['P_LOW_C'],  D['T_LOW_C']]])
    elif cycle_type_norm.lower() in ['startstop', 'startstop']:
        pattern = np.array([[D['P_HIGH_SS'], D['T_HIGH_SS']],
                            [D['P_MID_SS'],  D['T_MID_SS']],
                            [D['P_LOW_SS'],  D['T_LOW_SS']]])
    else:
        return np.zeros(N)
    steps = np.maximum(1, np.round(pattern[:,1]/dt_sec).astype(int))
    pw = np.repeat(pattern[:,0], steps)
    if pw.size == 0:
        return np.zeros(N)
    repeats = int(np.ceil(N/len(pw)))
    full = np.tile(pw, repeats)
    return full[:N]

def aggregate_to_minutes(t_hi: List[datetime], P_kW: np.ndarray) -> Tuple[List[datetime], np.ndarray]:
    assert len(t_hi) == len(P_kW)
    bins = [dt.replace(second=0, microsecond=0) for dt in t_hi]
    bucket_index, t_min = {}, []
    for i, b in enumerate(bins):
        if b not in bucket_index:
            bucket_index[b] = []
            t_min.append(b)
        bucket_index[b].append(i)
    Pmin = np.zeros(len(t_min))
    for j, b in enumerate(t_min):
        idxs = bucket_index[b]
        Pmin[j] = float(np.mean(P_kW[idxs])) if idxs else 0.0
    return t_min, Pmin

def build_reefer_stationary_minute_trace(cycle_type: str,
                                         t_minutes: List[datetime],
                                         t_arr: datetime,
                                         t_dep: datetime,
                                         dt_reefer_sec: int = 10,
                                         PF_reefer: float = 0.75,
                                         kVA_refr_cap: float = 19.765
                                         ) -> Tuple[np.ndarray, np.ndarray]:
    if cycle_type.strip().lower() == 'noreeferstationary':
        return np.zeros(len(t_minutes)), np.zeros(len(t_minutes))
    if t_dep <= t_arr:
        t_dep = t_dep + timedelta(days=1)
    N10 = int(max(0, (t_dep - t_arr).total_seconds() // dt_reefer_sec))
    if N10 <= 0:
        return np.zeros(len(t_minutes)), np.zeros(len(t_minutes))
    P10_kW = get_reefer_cycle_trace(cycle_type, N10, dt_reefer_sec)
    kVA10 = np.minimum(P10_kW / max(PF_reefer, np.finfo(float).eps), kVA_refr_cap)
    t10 = [t_arr + timedelta(seconds=i*dt_reefer_sec) for i in range(N10)]
    tMin, Pmin_kW = aggregate_to_minutes(t10, P10_kW)
    _, kVAmin = aggregate_to_minutes(t10, kVA10)
    P_refr_min_kW = np.zeros(len(t_minutes))
    kVA_refr_min = np.zeros(len(t_minutes))
    idx_lookup = {tm: j for j, tm in enumerate(tMin)}
    for i, tm in enumerate(t_minutes):
        j = idx_lookup.get(tm, None)
        if j is not None:
            P_refr_min_kW[i] = Pmin_kW[j]
            kVA_refr_min[i] = kVAmin[j]
    return P_refr_min_kW, kVA_refr_min

# ----------------------------- Simulation core -------------------------------
@dataclass
class SimResult:
    P_trace: np.ndarray
    SOC_trace: np.ndarray
    reachedTarget: bool
    delivered_kWh: float
    kVA_grid_total: np.ndarray

def simulate(t: List[datetime], dt_hr: float, price_min: np.ndarray, EV: EVParams,
             soc_bp: np.ndarray, Pcap_grid_bp_kW: np.ndarray, eff_frac: float, th: float,
             P_refr_min_kW: np.ndarray, kVA_refr_min: np.ndarray) -> SimResult:
    N = len(t)
    P_trace_gridAC_kW = np.zeros(N)
    SOC_trace = np.zeros(N)
    SOC = EV.UsableBatteryCap_kWh * (EV.SOC_arrival_winter_pc/100.0)
    if hasattr(EV, 'CurrentSOC_kWh'):
        SOC = getattr(EV, 'CurrentSOC_kWh')
    target = EV.UsableBatteryCap_kWh * (EV.SOC_departure_target_pc/100.0)
    reachedTarget = False
    GridMax_kVA = EV.GridMax_kVA
    PF_site = EV.GridPF_site
    eta_batt = max(np.finfo(float).eps, EV.BatteryChargingEffi_pc/100.0)
    eta_OBC = max(np.finfo(float).eps, EV.OBCEfficiency_pc/100.0)
    BattMaxChg = EV.BattMaxCharge_kW
    kVA_grid_total = np.zeros(N)

    for i in range(N):
        SOC_trace[i] = SOC
        if SOC >= target - 1e-9:
            reachedTarget = True
            SOC_trace[i:] = SOC
            break
        socFracNow = np.clip(SOC / EV.UsableBatteryCap_kWh, 0.0, 1.0)
        P_taper_grid_kW = float(np.interp(socFracNow, soc_bp, Pcap_grid_bp_kW,
                                          left=Pcap_grid_bp_kW[0], right=Pcap_grid_bp_kW[-1]))
        P_taper_battDC = P_taper_grid_kW * eta_batt * eta_OBC
        P_effective_battDC_cap = min(P_taper_battDC, EV.EffectiveChargingPower_kW*eta_batt*eta_OBC, BattMaxChg)
        kVA_refr = max(0.0, kVA_refr_min[i] if i < len(kVA_refr_min) else 0.0)
        rhs = (GridMax_kVA/eta_OBC) - kVA_refr
        P_battDC_cap_kVA = max(0.0, rhs / max(np.finfo(float).eps, PF_site*eta_batt))
        if price_min[i] <= th:
            P_needed_battDC = max(0.0, (target - SOC)/dt_hr)
            P_battDC = min(P_effective_battDC_cap, P_battDC_cap_kVA, P_needed_battDC)
        else:
            P_battDC = 0.0
        SOC = min(EV.UsableBatteryCap_kWh, SOC + P_battDC*dt_hr)
        P_gridAC_for_batt = P_battDC / (eta_batt*eta_OBC)
        P_trace_gridAC_kW[i] = P_gridAC_for_batt
        kVA_grid_total[i] = (kVA_refr + (P_battDC*eta_batt)*PF_site)*eta_OBC

    delivered_kWh = float(np.sum(P_trace_gridAC_kW*(eta_batt*eta_OBC))*dt_hr)
    return SimResult(P_trace=P_trace_gridAC_kW, SOC_trace=SOC_trace,
                     reachedTarget=reachedTarget, delivered_kWh=delivered_kWh,
                     kVA_grid_total=kVA_grid_total)

def empty_sim(t: List[datetime], EV: EVParams) -> SimResult:
    N = len(t)
    return SimResult(P_trace=np.zeros(N),
                     SOC_trace=np.full(N, EV.UsableBatteryCap_kWh*(EV.SOC_arrival_winter_pc/100.0)),
                     reachedTarget=True, delivered_kWh=0.0,
                     kVA_grid_total=np.zeros(N))

def finalize_res(sim: SimResult, price_min: np.ndarray, dt_hr: float, threshold_used: float) -> Dict[str, object]:
    res = {
        'P_trace': sim.P_trace,
        'SOC_trace': sim.SOC_trace,
        'reachedTarget': sim.reachedTarget,
        'delivered_kWh': sim.delivered_kWh,
        'price_min': price_min,
        'energy_kWh': float(np.sum(sim.P_trace) * dt_hr),
        'cost_EUR': float(np.sum(sim.P_trace * price_min) * dt_hr),
        'thresholdUsed': threshold_used,
        'kVA_grid_total': sim.kVA_grid_total,
    }
    return res

def plan_smart(t: List[datetime], dt_hr: float, price24: np.ndarray, EV: EVParams,
               soc_bp: np.ndarray, Pcap_grid_bp_kW: np.ndarray, need_kWh: float, eff_frac: float,
               P_refr_min_kW: np.ndarray, kVA_refr_min: np.ndarray) -> Dict[str, object]:
    price_min = np.asarray([price24[dt.hour] for dt in t], dtype=float)
    if need_kWh <= 1e-12:
        return finalize_res(empty_sim(t, EV), price_min, dt_hr, float('nan'))
    uniq = np.unique(price_min)
    best, usedTh, met = None, float('inf'), False
    for th in uniq:
        sim = simulate(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp_kW, eff_frac, th, P_refr_min_kW, kVA_refr_min)
        if sim.delivered_kWh >= need_kWh - 1e-9:
            best, usedTh, met = sim, float(th), True
            break
        else:
            best, usedTh = sim, float(th)
    if not met:
        best = simulate(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp_kW, eff_frac, float('inf'), P_refr_min_kW, kVA_refr_min)
        usedTh = float('inf')
    return finalize_res(best, price_min, dt_hr, usedTh)

def plan_baseline(t: List[datetime], dt_hr: float, price_min: np.ndarray, EV: EVParams,
                  soc_bp: np.ndarray, Pcap_grid_bp_kW: np.ndarray, eff_frac: float,
                  P_refr_min_kW: np.ndarray, kVA_refr_min: np.ndarray) -> Dict[str, object]:
    sim = simulate(t, dt_hr, price_min, EV, soc_bp, Pcap_grid_bp_kW, eff_frac, float('inf'), P_refr_min_kW, kVA_refr_min)
    return finalize_res(sim, price_min, dt_hr, float('inf'))

# ----------------------------- Reefer energy & cost ---------------------------
@dataclass
class ReeferCostParams:
    FixedPrice_EUR_per_kWh: float = 0.35
    DieselPrice_EUR_per_L: float = 1.80
    Diesel_kWh_per_L: float = 9.8
    Genset_efficiency_frac: float = 0.30
    DieselFixedCons_L_per_h: float = 2.5
    Method: str = 'energy'  # 'energy' or 'fixed'

def get_reefer_cost_params() -> ReeferCostParams:
    return ReeferCostParams()

def compute_reefer_grid_energy(cycle_type: str, t_arr: datetime, t_dep: datetime) -> float:
    if cycle_type.strip().lower() == 'noreeferstationary':
        return 0.0
    if t_dep <= t_arr:
        t_dep = t_dep + timedelta(days=1)
    dt_sec = 10
    N10 = int(max(0, (t_dep - t_arr).total_seconds() // dt_sec))
    if N10 <= 0:
        return 0.0
    P10_kW = get_reefer_cycle_trace(cycle_type, N10, dt_sec)
    return float(np.sum(P10_kW) * (dt_sec/3600.0))

def compute_reefer_cost_scenarios(P_refr_min_kW: np.ndarray, dt_hr: float,
                                  price_min: np.ndarray, params: ReeferCostParams) -> Dict[str, float]:
    E_kWh = float(np.sum(P_refr_min_kW) * dt_hr)
    cost_dynamic = float(np.sum(P_refr_min_kW * price_min) * dt_hr)
    cost_fixed = E_kWh * params.FixedPrice_EUR_per_kWh
    liters_energy = E_kWh / max(np.finfo(float).eps, params.Diesel_kWh_per_L * params.Genset_efficiency_frac)
    hours_total = len(P_refr_min_kW) * dt_hr
    liters_fixed = params.DieselFixedCons_L_per_h * hours_total
    liters_used = liters_fixed if params.Method.lower() == 'fixed' else liters_energy
    cost_diesel = liters_used * params.DieselPrice_EUR_per_L
    return {
        'E_kWh': E_kWh,
        'cost_dynamic': cost_dynamic,
        'cost_fixed': cost_fixed,
        'cost_diesel': cost_diesel,
        'diesel_liters_energy': liters_energy,
        'diesel_liters_fixed': liters_fixed,
        'diesel_liters_used': liters_used,
    }

# ----------------------------- Small helpers ---------------------------------
def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def fmt1(x: float) -> str:
    return f"{x:.1f}"

def fmt2(x: float) -> str:
    return f"{x:.2f}"
