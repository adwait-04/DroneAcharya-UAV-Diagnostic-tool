"""
Health Monitoring — Drone Condition & Maintenance Assessment
Inputs: flight log + .param file + questionnaire
Output: structured Drone Health Report
"""

import json
import sys
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import health_engine
from utils import data_processor, param_handler

st.set_page_config(page_title="Drone Health Monitoring", page_icon="🔬", layout="wide")
_css = _ROOT / "assests" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in {
    "parsed_signals": {}, "analysis_results": {}, "questionnaire": {},
    "param_summary": {}, "uploaded_file_name": None, "analysis_ran": False,
}.items():
    st.session_state.setdefault(_k, _v)

# ── Helpers ───────────────────────────────────────────────────────────────────
PLT = dict(template="plotly_white", plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
           font=dict(family="Inter, sans-serif", size=11, color="#334155"),
           margin=dict(l=10, r=10, t=40, b=10))

def _sf(v, default=float("nan")):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default

def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 55: return "D"
    if score >= 40: return "E"
    return "F"

def _risk(score: float) -> str:
    if score >= 75: return "Low"
    if score >= 50: return "Medium"
    return "High"

def _risk_color(risk: str) -> str:
    return {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}.get(risk, "#64748b")

def _status_color(status: str) -> str:
    return {"Good": "#16a34a", "Degraded": "#d97706", "Critical": "#dc2626",
            "Insufficient Data": "#94a3b8"}.get(status, "#64748b")

def _dominant_issue(alerts: list, metrics: dict, diag: dict) -> str:
    critical = [a["msg"] for a in alerts if a.get("level") == "critical"]
    if critical:
        return critical[0]
    warnings = [a["msg"] for a in alerts if a.get("level") == "warning"]
    if warnings:
        return warnings[0]
    rms = _sf(metrics.get("vibration_rms"))
    if not math.isnan(rms) and rms < 0.5:
        return "All monitored subsystems within normal operating parameters"
    return "Minor deviations detected — monitor on next flight"

def _subsystem_health(metrics: dict, diag: dict, secs: dict, alerts: list) -> dict:
    rms       = _sf(metrics.get("vibration_rms"),        float("nan"))
    shift     = _sf(metrics.get("resonance_shift_hz"),   float("nan"))
    imb       = _sf(metrics.get("motor_imbalance_pct"),  float("nan"))
    sp_e      = _sf(metrics.get("specific_energy_wh_per_km"), float("nan"))
    vib_sev   = diag.get("vibration_severity", "unknown")
    res_stat  = diag.get("resonance_status",   "N/A")
    mot_stat  = diag.get("motor_balance_status","N/A")
    en_stat   = diag.get("energy_status",      "N/A")
    n_crit    = sum(1 for a in alerts if a.get("level") == "critical")
    n_warn    = sum(1 for a in alerts if a.get("level") == "warning")

    def _vib_status():
        if math.isnan(rms): return "Insufficient Data"
        if rms < 0.5:  return "Good"
        if rms < 3.0:  return "Degraded"
        return "Critical"

    def _mot_status():
        if math.isnan(imb): return "Insufficient Data"
        if imb < 3.0:  return "Good"
        if imb < 8.0:  return "Degraded"
        return "Critical"

    def _pwr_status():
        if en_stat in ("NOMINAL", "NO_BASELINE"): return "Good"
        if en_stat == "SLIGHTLY_DEGRADED":        return "Degraded"
        if en_stat == "DEGRADED":                 return "Critical"
        return "Insufficient Data"

    vib_st = _vib_status()
    mot_st = _mot_status()
    pwr_st = _pwr_status()

    if vib_st == "Critical" or mot_st == "Critical":
        ctrl_st = "Critical"
        ctrl_issues = ["Control loop compensating for severe vibration or imbalance"]
    elif vib_st == "Degraded" or mot_st == "Degraded":
        ctrl_st = "Degraded"
        ctrl_issues = ["Elevated disturbances increasing corrective control effort"]
    else:
        ctrl_st = "Good"
        ctrl_issues = ["Control corrections within expected bounds"]

    has_gyro = any(len(secs.get("imu", {}).get(f"gyro_fft_f_{ax}", [])) > 0 for ax in ["x", "y", "z"])
    if not has_gyro and math.isnan(rms):
        sens_st = "Insufficient Data"
        sens_issues = ["No sensor data available in this log"]
    elif vib_st == "Critical":
        sens_st = "Critical"
        sens_issues = ["Critical vibration levels corrupting IMU readings"]
    elif vib_st == "Degraded":
        sens_st = "Degraded"
        sens_issues = ["Vibration coupling reducing IMU signal quality"]
    else:
        sens_st = "Good"
        sens_issues = ["IMU noise within acceptable limits"]

    if vib_st == "Critical" or (not math.isnan(shift) and abs(shift) > 3):
        ekf_st = "Critical"
        ekf_issues = ["State estimator under severe stress — position/attitude errors likely"]
    elif vib_st == "Degraded" or (not math.isnan(shift) and abs(shift) > 1):
        ekf_st = "Degraded"
        ekf_issues = ["Elevated vibration increasing estimator uncertainty"]
    else:
        ekf_st = "Good"
        ekf_issues = ["State estimation running within normal uncertainty bounds"]

    gspd = secs.get("imu", {}).get("ground_speed_ms", np.array([]))
    if len(gspd) > 10:
        nav_st = "Good"
        nav_issues = ["Navigation data present and consistent"]
    else:
        nav_st = "Insufficient Data"
        nav_issues = ["Limited GPS/navigation data in this log"]

    if n_crit > 0:
        sys_st = "Critical"
        sys_issues = [f"{n_crit} critical fault(s) detected — immediate inspection required"]
    elif n_warn > 0:
        sys_st = "Degraded"
        sys_issues = [f"{n_warn} warning(s) — schedule maintenance"]
    else:
        sys_st = "Good"
        sys_issues = ["No system faults detected"]

    total_events = n_crit + n_warn
    if n_crit > 0:
        rel_st = "Critical"
        rel_issues = ["Critical anomalous events in this flight"]
    elif total_events > 2:
        rel_st = "Degraded"
        rel_issues = [f"{total_events} anomalous events logged"]
    else:
        rel_st = "Good"
        rel_issues = ["No anomalous events"]

    return {
        "Propulsion":           {"status": mot_st, "issues": [a["msg"] for a in alerts if "motor" in a["msg"].lower() or "imbalance" in a["msg"].lower() or "esc" in a["msg"].lower()] or (["Motor output balanced"] if mot_st == "Good" else ["Motor imbalance detected"]), "meaning": "Rotor and motor thrust uniformity"},
        "Structure / Vibration":{"status": vib_st, "issues": [a["msg"] for a in alerts if "vibration" in a["msg"].lower() or "resonance" in a["msg"].lower() or "prop" in a["msg"].lower()] or (["Vibration within safe limits"] if vib_st == "Good" else ["Vibration outside normal range"]), "meaning": "Frame, arms and mount mechanical condition"},
        "Control":              {"status": ctrl_st,  "issues": ctrl_issues, "meaning": "Autopilot corrective effort and control authority"},
        "Power (Battery)":      {"status": pwr_st,   "issues": [a["msg"] for a in alerts if "efficiency" in a["msg"].lower() or "battery" in a["msg"].lower() or "current" in a["msg"].lower()] or (["Power draw nominal"] if pwr_st == "Good" else ["Power anomaly"]), "meaning": "Battery discharge health and energy efficiency"},
        "Sensors":              {"status": sens_st,  "issues": sens_issues, "meaning": "IMU signal quality and noise floor"},
        "Estimation (EKF)":     {"status": ekf_st,   "issues": ekf_issues, "meaning": "State estimator confidence and accuracy"},
        "Navigation":           {"status": nav_st,   "issues": nav_issues, "meaning": "Position and path tracking reliability"},
        "Autopilot / System":   {"status": sys_st,   "issues": sys_issues, "meaning": "Firmware-level fault and warning count"},
        "Reliability (Events)": {"status": rel_st,   "issues": rel_issues, "meaning": "Anomalous event frequency this flight"},
    }


def _degradation_indicators(metrics: dict, diag: dict, alerts: list) -> list:
    items = []
    rms = _sf(metrics.get("vibration_rms"))
    imb = _sf(metrics.get("motor_imbalance_pct"))
    shift = _sf(metrics.get("resonance_shift_hz"))
    en_stat = diag.get("energy_status", "N/A")

    if not math.isnan(rms) and rms >= 0.5:
        items.append(f"Elevated vibration ({rms:.2f} m/s²) — accelerating bearing and mount wear")
    if not math.isnan(imb) and imb >= 3.0:
        items.append(f"Motor output imbalance ({imb:.1f}%) — uneven thrust load on frame")
    if not math.isnan(shift) and abs(shift) >= 1.0:
        direction = "softening" if metrics.get("resonance_direction") == "downward" else "stiffening"
        items.append(f"Resonance shift ({shift:+.2f} Hz) — structural {direction} detected")
    if en_stat in ("DEGRADED", "SLIGHTLY_DEGRADED"):
        items.append("Energy efficiency below baseline — possible prop wear or increased drag")
    if not items:
        items.append("No active degradation indicators — system condition nominal")
    return items[:5]


def _root_cause_chain(metrics: dict, diag: dict) -> dict:
    rms   = _sf(metrics.get("vibration_rms"))
    imb   = _sf(metrics.get("motor_imbalance_pct"))
    shift = _sf(metrics.get("resonance_shift_hz"))
    en    = diag.get("energy_status", "N/A")

    vib_ded = min(40.0, rms * 13.33) if not math.isnan(rms) else 0
    res_ded = min(25.0, abs(shift) * 5.0) if not math.isnan(shift) else 0
    mot_ded = min(20.0, imb * 2.5) if not math.isnan(imb) else 0
    en_ded  = 15.0 if en == "DEGRADED" else 7.5 if en == "SLIGHTLY_DEGRADED" else 0

    dominant = max([(vib_ded, "vib"), (res_ded, "res"), (mot_ded, "mot"), (en_ded, "en")],
                   key=lambda x: x[0])

    if dominant[1] == "vib" and vib_ded > 0:
        return {"cause": "Propeller or motor imbalance generating excess vibration",
                "effect": "Vibration → IMU noise → EKF uncertainty → increased control corrections",
                "risk":   "Accelerated wear on motor bearings, frame joints, and electronic components"}
    if dominant[1] == "res" and res_ded > 0:
        direction = metrics.get("resonance_direction", "unknown")
        return {"cause": f"Structural resonance shift ({direction}) — loose mount or frame fatigue",
                "effect": "Resonance shift → altered vibration spectrum → filter mismatch → estimator error",
                "risk":   "Progressive structural weakening or mounting failure under load"}
    if dominant[1] == "mot" and mot_ded > 0:
        return {"cause": "Motor or ESC output imbalance — calibration drift or mechanical fault",
                "effect": "Uneven thrust → compensatory yaw → increased attitude error → higher power draw",
                "risk":   "Reduced control authority and uneven component aging"}
    if dominant[1] == "en" and en_ded > 0:
        return {"cause": "Energy efficiency below baseline — prop wear, drag increase, or battery aging",
                "effect": "Higher current draw → battery thermal stress → reduced flight time",
                "risk":   "Premature battery degradation and shortened mission range"}

    return {"cause": "No dominant fault detected in this flight",
            "effect": "All subsystems operating within expected parameters",
            "risk":   "Maintain regular inspection schedule"}


def _expected_vs_observed(metrics: dict, diag: dict) -> list:
    rows = []
    rms = _sf(metrics.get("vibration_rms"))
    if not math.isnan(rms):
        if rms >= 0.5:
            impact = "Structural stress and sensor degradation" if rms >= 1.5 else "Minor sensor noise increase"
            rows.append({"Expected": "Vibration RMS < 0.5 m/s² (nominal)",
                         "Observed":  f"Vibration RMS = {rms:.3f} m/s²",
                         "Impact":    impact})

    imb = _sf(metrics.get("motor_imbalance_pct"))
    if not math.isnan(imb) and imb >= 3.0:
        rows.append({"Expected": "Motor output spread < 3% (balanced)",
                     "Observed":  f"Motor spread = {imb:.1f}%",
                     "Impact":    "Uneven thrust loading and frame stress"})

    shift = _sf(metrics.get("resonance_shift_hz"))
    if not math.isnan(shift) and abs(shift) >= 1.0:
        direction = metrics.get("resonance_direction", "unknown")
        rows.append({"Expected": "Resonance frequency stable (< 1 Hz shift)",
                     "Observed":  f"Resonance {direction} by {abs(shift):.2f} Hz",
                     "Impact":    "Structural condition change — inspect mounts and arms"})

    en_stat = diag.get("energy_status", "N/A")
    if en_stat in ("DEGRADED", "SLIGHTLY_DEGRADED"):
        sp_e = _sf(metrics.get("specific_energy_wh_per_km"))
        rows.append({"Expected": "Specific energy near baseline Wh/km",
                     "Observed":  f"Specific energy = {sp_e:.2f} Wh/km (above baseline)",
                     "Impact":    "Reduced endurance per charge cycle"})

    if not rows:
        rows.append({"Expected": "All key metrics within normal operating ranges",
                     "Observed":  "All metrics nominal",
                     "Impact":    "No maintenance action required"})
    return rows[:3]


def _confidence(signals: dict, param_summary: dict) -> str:
    ts = signals.get("timestamp_s", [])
    has_imu  = len(signals.get("imu_accel_z", [])) > 50
    has_gps  = len(signals.get("ground_speed_ms", [])) > 10
    has_rcout = len(signals.get("rcout_1", [])) > 10
    has_param = bool(param_summary)
    score = sum([len(ts) > 500, has_imu, has_gps, has_rcout, has_param])
    if score >= 4: return "High"
    if score >= 2: return "Medium"
    return "Low"


def _build_health_json(score, grade, risk, dominant_issue, subsystems,
                       degradation, root_cause, evo, confidence,
                       questionnaire, param_summary, performance_model=None) -> str:
    params_clean = {}
    if param_summary:
        pids = param_summary.get("pid", {})
        for axis in ["roll", "pitch", "yaw"]:
            pg = pids.get(axis, {})
            for k in ["P", "I", "D"]:
                v = pg.get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    params_clean[f"ATC_RAT_{axis[:3].upper()}_{k}"] = v
        filt = param_summary.get("filters", {})
        for k, v in filt.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                params_clean[k] = v

    doc = {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_health": {
            "score": round(score, 1),
            "grade": grade,
            "risk_level": risk,
            "dominant_issue": dominant_issue,
        },
        "subsystem_health": {
            k: {"status": v["status"], "issues": v["issues"][:2], "meaning": v["meaning"]}
            for k, v in subsystems.items()
        },
        "degradation_indicators": degradation,
        "root_cause_chain": root_cause,
        "expected_vs_observed": evo,
        "confidence_level": confidence,
        "context": {k: v for k, v in questionnaire.items() if not callable(v)},
        "params": params_clean,
        "performance_model": performance_model or {
            "hover_power_w":     None,
            "cruise_power_w":    None,
            "mean_power_w":      None,
            "hover_throttle":    None,
            "voltage_sag_pct":   None,
            "energy_per_min_wh": None,
            "energy_per_km_wh":  None,
            "max_power_w":       None,
            "thrust_margin_pct": None,
            "efficiency_state":  None,
        },
    }
    return json.dumps(doc, indent=2)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigate")
    st.page_link("app.py",                          label="🏠  Home")
    st.page_link("pages/1_Health_Monitoring.py",    label="🔬  Health Monitoring")
    st.page_link("pages/2_Mission_Feasibility.py",  label="🗺️  Mission Feasibility")
    st.page_link("pages/3_AboutUs.py",              label="ℹ️  About Us")
    st.markdown("---")

    st.markdown("### Flight Log")
    log_file = st.file_uploader("Upload .bin or .csv", type=["bin","csv","log","txt"], key="hm_log")
    if log_file and log_file.name != st.session_state.get("uploaded_file_name"):
        with st.spinner("Parsing flight log…"):
            sigs, warns = data_processor.process_file(log_file, log_file.name)
        st.session_state["parsed_signals"]    = sigs
        st.session_state["uploaded_file_name"] = log_file.name
        st.session_state["analysis_ran"]      = False
        for w in warns: st.warning(w)
        dur = sigs.get("duration_s", 0.0)
        if dur > 0:
            st.success(f"{log_file.name}  ·  {dur:.1f} s")

    st.markdown("### Parameter File (.param)")
    param_file = st.file_uploader("Upload ArduPilot .param (required)", type=["param","txt"], key="hm_param")
    if param_file:
        raw_params, p_warns = param_handler.parse_param_file(param_file)
        summary = param_handler.summarize_params(raw_params)
        st.session_state["param_summary"] = summary
        prefill = summary["questionnaire_prefill"]
        q = st.session_state.setdefault("questionnaire", {})
        for k, v in prefill.items():
            q[k] = v
        for w in p_warns: st.warning(w)
        st.success(f"{len(raw_params)} parameters loaded.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero-badge'>Drone Condition & Maintenance Assessment · Physics-Driven · No AI</div>",
    unsafe_allow_html=True,
)
st.title("🔬 Drone Health Monitoring")
st.caption("Upload flight log + parameter file · Fill configuration · Run health analysis")
st.markdown("---")

# ── Questionnaire ─────────────────────────────────────────────────────────────
st.subheader("Drone Configuration")
st.info("All fields mandatory. Weights in **kg** · Frame & prop sizes in **inches**.")

q = st.session_state["questionnaire"]
frame_class = "Quadcopter"
CELL_OPTIONS  = ["", "2S", "3S", "4S", "5S", "6S", "8S", "10S", "12S"]
def _idx(opts, val): return opts.index(val) if val in opts else 0

c1, c2, c3 = st.columns(3)
with c1:
    frame_size_in = st.number_input("Frame Size (in) *",  0.0, 120.0, float(q.get("frame_size_in",0.0)), 0.5, format="%.1f",       key="q1_fs")
    dry_weight_kg = st.number_input("Dry Weight (kg) *",  0.0,  50.0, float(q.get("dry_weight_kg",0.0)), 0.05, format="%.3f",      key="q1_dw")
with c2:
    battery_cells = st.selectbox("Battery Cells (S) *",   CELL_OPTIONS,  index=_idx(CELL_OPTIONS,  q.get("battery_cells_str","")), key="q1_bc")
    battery_cap   = st.number_input("Battery (mAh) *",    0, 50000, int(q.get("battery_capacity_mah",0)), 100,                     key="q1_bap")
    motor_kv      = st.number_input("Motor KV *",         0, 10000, int(q.get("motor_kv",0)), 10,                                  key="q1_kv")
with c3:
    prop_diam     = st.number_input("Prop Diameter (in) *", 0.0, 50.0, float(q.get("prop_diameter_in",0.0)), 0.5, format="%.1f",   key="q1_pd")
    prop_pitch    = st.number_input("Prop Pitch (in) *",  0.0, 20.0, float(q.get("prop_pitch_in",0.0)), 0.5, format="%.1f",        key="q1_pp")
payload_kg = st.number_input("Current Payload (kg) *", 0.0, 20.0, float(q.get("payload_kg",0.0)), 0.05, format="%.3f", key="q1_pl")

cells_int = int(battery_cells.replace("S","")) if battery_cells else 0
st.session_state["questionnaire"].update({
    "frame_class": frame_class, "frame_size_in": frame_size_in, "dry_weight_kg": dry_weight_kg,
    "battery_cells_str": battery_cells, "battery_cells": cells_int, "battery_capacity_mah": battery_cap,
    "motor_kv": motor_kv, "prop_diameter_in": prop_diam, "prop_pitch_in": prop_pitch,
    "payload_kg": payload_kg,
    "num_motors": 4,
})

_MANDATORY = {"Frame Size": frame_size_in, "Dry Weight": dry_weight_kg,
              "Battery Cells": battery_cells, "Battery Cap": battery_cap, "Motor KV": motor_kv,
              "Prop Diameter": prop_diam, "Prop Pitch": prop_pitch}
_missing = [k for k, v in _MANDATORY.items() if not v or (isinstance(v, (int, float)) and v <= 0)]
if _missing: st.error(f"Missing: {', '.join(_missing)}")

st.markdown("---")

# ── Run ───────────────────────────────────────────────────────────────────────
can_run = not _missing and bool(st.session_state.get("parsed_signals"))
if not can_run:
    st.warning("Upload a flight log and complete all fields to run health analysis.")

if st.button("🔬  Analyse Drone Health", disabled=not can_run, type="primary", use_container_width=True):
    with st.spinner("Running health assessment pipeline…"):
        result = health_engine.analyze(
            signals=st.session_state["parsed_signals"],
            params=st.session_state["questionnaire"],
            context={},
        )
    st.session_state["analysis_results"] = result
    st.session_state["analysis_ran"]     = True
    st.success("Health assessment complete.")

# ── Guard ─────────────────────────────────────────────────────────────────────
if not (st.session_state.get("analysis_ran") and st.session_state.get("analysis_results")):
    st.stop()

res     = st.session_state["analysis_results"]
score   = res["score"]
metrics = res["metrics"]
secs    = res["sections"]
diag    = res["diagnostics"] if "diagnostics" in res else {}
alerts  = res["alerts"]
signals = st.session_state["parsed_signals"]
param_s = st.session_state.get("param_summary", {})
questionnaire = st.session_state["questionnaire"]

ts = np.array(signals.get("timestamp_s", []), dtype=float)

grade         = _grade(score)
risk          = _risk(score)
dom_issue     = _dominant_issue(alerts, metrics, diag)
subsystems    = _subsystem_health(metrics, diag, secs, alerts)
degradation   = _degradation_indicators(metrics, diag, alerts)
root_cause    = _root_cause_chain(metrics, diag)
evo           = _expected_vs_observed(metrics, diag)
confidence    = _confidence(signals, param_s)

# ══ PRESENTATION HELPERS ════════════════════════════════════════════════════

def _condition_label(score, alerts, subsystems):
    n_crit_alerts = sum(1 for a in alerts if a.get("level") == "critical")
    n_crit_subs   = sum(1 for s in subsystems.values() if s["status"] == "Critical")
    n_deg_subs    = sum(1 for s in subsystems.values() if s["status"] == "Degraded")
    if n_crit_alerts > 0 or n_crit_subs > 0 or score < 50:
        crit_names = [k for k, v in subsystems.items() if v["status"] == "Critical"]
        summary = (f"Critical failures in: {', '.join(crit_names[:2])}"
                   if crit_names else "Multiple critical alerts — immediate attention required")
        return {"label": "Critical", "summary": summary,
                "color": "#dc2626", "bg": "#fef2f2", "border": "#fca5a5"}
    if score < 75 or n_deg_subs > 0:
        deg_names = [k for k, v in subsystems.items() if v["status"] == "Degraded"]
        summary = (f"Degraded performance in: {', '.join(deg_names[:2])}"
                   if deg_names else "Performance below baseline in one or more subsystems")
        return {"label": "Partially Degraded", "summary": summary,
                "color": "#d97706", "bg": "#fffbeb", "border": "#fde68a"}
    return {"label": "Stable",
            "summary": "No significant degradation detected across all monitored subsystems",
            "color": "#16a34a", "bg": "#f0fdf4", "border": "#86efac"}


def _build_attention_items(metrics, diag, alerts, subsystems):
    items = []
    rms   = _sf(metrics.get("vibration_rms"))
    imb   = _sf(metrics.get("motor_imbalance_pct"))
    shift = _sf(metrics.get("resonance_shift_hz"))
    en    = diag.get("energy_status", "N/A")

    if not math.isnan(rms) and rms >= 1.5:
        items.append({
            "name": "Severe mechanical vibration",
            "meaning": "Vibration amplitude is outside the safe operating envelope for sensor-grade flight",
            "impact": "IMU readings are corrupted — state estimator and attitude control accuracy are reduced",
        })
    elif not math.isnan(rms) and rms >= 0.5:
        items.append({
            "name": "Elevated mechanical vibration",
            "meaning": "Vibration is above the acceptable threshold for clean inertial measurement",
            "impact": "IMU noise floor is raised; state estimation accuracy is reduced",
        })

    if not math.isnan(shift) and abs(shift) >= 1.0:
        direction = metrics.get("resonance_direction", "")
        meaning = ("Frame or mounting stiffness has decreased relative to the baseline flight"
                   if direction == "downward"
                   else "Mass distribution or mounting preload has changed relative to the baseline flight")
        items.append({
            "name": "Structural resonance shift",
            "meaning": meaning,
            "impact": "Vibration notch filter is mismatched — estimator performance degrades under load",
        })

    if not math.isnan(imb) and imb >= 8.0:
        items.append({
            "name": "Critical motor output imbalance",
            "meaning": "Significant asymmetry in motor thrust outputs across the frame",
            "impact": "Generates net yaw torque; control authority is reduced",
        })
    elif not math.isnan(imb) and imb >= 3.0:
        items.append({
            "name": "Motor output imbalance",
            "meaning": "Thrust is not symmetrically distributed across motors",
            "impact": "Increases corrective control demand and asymmetric frame loading",
        })

    if en in ("DEGRADED", "SLIGHTLY_DEGRADED"):
        items.append({
            "name": "Energy efficiency below baseline",
            "meaning": "Power consumption per flight distance exceeds the established baseline",
            "impact": "Mission range is reduced; battery undergoes increased thermal cycling",
        })

    for a in alerts:
        if len(items) >= 4:
            break
        if a.get("level") == "critical":
            msg = a["msg"]
            if not any(msg[:30].lower() in str(i).lower() for i in items):
                items.append({"name": "System fault", "meaning": msg, "impact": "Review full autopilot log"})

    return items[:4]


def _build_integrity_statements(subsystems):
    good = {k for k, v in subsystems.items() if v["status"] == "Good"}
    stmts = []
    nav_ekf  = [s for s in ["Navigation", "Estimation (EKF)"] if s in good]
    propuls  = [s for s in ["Propulsion", "Structure / Vibration"] if s in good]
    power    = [s for s in ["Power (Battery)"] if s in good]
    ctrl_sys = [s for s in ["Control", "Autopilot / System", "Reliability (Events)"] if s in good]
    sensors  = [s for s in ["Sensors"] if s in good]
    if nav_ekf:  stmts.append(f"{' and '.join(nav_ekf)} operating within normal bounds")
    if propuls:  stmts.append(f"{' and '.join(propuls)} within acceptable limits")
    if power:    stmts.append("Power rail stable — no efficiency anomaly detected")
    if ctrl_sys: stmts.append("Control response and autopilot fault count nominal")
    if sensors:  stmts.append("Sensor signal quality within acceptable range")
    return stmts[:3] if stmts else ["Core subsystems nominal"]


def _supporting_context_line(metrics, diag, alerts, subsystems):
    rms = _sf(metrics.get("vibration_rms"))
    imb = _sf(metrics.get("motor_imbalance_pct"))
    en  = diag.get("energy_status", "")
    if not math.isnan(rms) and rms >= 0.5 and not math.isnan(imb) and imb >= 3.0:
        return "Motor output asymmetry is a contributing factor to the elevated vibration signature."
    if not math.isnan(rms) and rms >= 1.5 and en in ("DEGRADED", "SLIGHTLY_DEGRADED"):
        return "Elevated vibration is increasing aerodynamic losses, contributing to higher energy consumption."
    return ""


# ── Compute display data ──────────────────────────────────────────────────────
cond       = _condition_label(score, alerts, subsystems)
attention  = _build_attention_items(metrics, diag, alerts, subsystems)
integrity  = _build_integrity_statements(subsystems)
ctx_line   = _supporting_context_line(metrics, diag, alerts, subsystems)

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYSTEM CONDITION BANNER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
fc_label = questionnaire.get("frame_class", "")
fs_label = questionnaire.get("frame_size_in", "")
dur_s    = metrics.get("duration_s", 0) or signals.get("duration_s", 0)

grade_color = {"A":"#16a34a","B":"#16a34a","C":"#d97706","D":"#d97706","E":"#dc2626","F":"#dc2626"}.get(grade,"#64748b")

banner_left, banner_right = st.columns([3, 1])
with banner_left:
    st.markdown(
        f"<div style='padding:1.4rem 1.6rem;background:{cond['bg']};"
        f"border:2px solid {cond['border']};border-radius:12px'>"
        f"<div style='font-size:0.7rem;font-weight:700;color:{cond['color']};"
        f"text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.3rem'>System Condition</div>"
        f"<div style='font-size:1.8rem;font-weight:900;color:{cond['color']};line-height:1.1'>{cond['label']}</div>"
        f"<div style='font-size:0.92rem;color:#475569;margin-top:0.5rem'>{cond['summary']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with banner_right:
    st.markdown(
        f"<div style='padding:1rem;border-radius:10px;background:#0f172a;"
        f"border:1px solid #1e293b;text-align:center;height:100%'>"
        f"<div style='font-size:2.6rem;font-weight:900;color:{grade_color};line-height:1'>{score:.0f}</div>"
        f"<div style='font-size:0.65rem;color:#64748b;margin-top:2px'>/ 100</div>"
        f"<div style='margin-top:0.5rem;padding:2px 10px;border-radius:4px;"
        f"background:{grade_color}22;color:{grade_color};font-weight:700;font-size:0.8rem;"
        f"display:inline-block'>Grade {grade} · {risk} Risk</div>"
        f"<div style='font-size:0.68rem;color:#475569;margin-top:0.4rem'>"
        f"{fc_label} · {dur_s:.0f} s · {datetime.now().strftime('%Y-%m-%d')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# 2. THINGS REQUIRING ATTENTION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")

if attention:
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;color:#dc2626;"
        "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.6rem'>Things Requiring Attention</div>",
        unsafe_allow_html=True,
    )
    for item in attention:
        st.markdown(
            f"<div style='background:#0f172a;border-left:3px solid #dc2626;"
            f"border-radius:0 8px 8px 0;padding:0.85rem 1.1rem;margin-bottom:0.5rem'>"
            f"<div style='font-weight:700;font-size:0.92rem;color:#f1f5f9'>{item['name']}</div>"
            f"<div style='color:#94a3b8;font-size:0.83rem;margin-top:0.25rem'>"
            f"→ {item['meaning']}</div>"
            f"<div style='color:#64748b;font-size:0.8rem;margin-top:0.2rem'>"
            f"→ {item['impact']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    if ctx_line:
        st.markdown(
            f"<div style='background:#1e293b;border-left:3px solid #d97706;"
            f"border-radius:0 6px 6px 0;padding:0.6rem 1rem;font-size:0.82rem;"
            f"color:#fbbf24;margin-top:0.25rem'>{ctx_line}</div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<div style='background:#0f172a;border-left:3px solid #16a34a;"
        "border-radius:0 8px 8px 0;padding:0.85rem 1.1rem'>"
        "<div style='font-weight:700;font-size:0.92rem;color:#86efac'>No issues requiring attention</div>"
        "<div style='color:#64748b;font-size:0.83rem;margin-top:0.2rem'>"
        "All monitored parameters are within normal operating bounds.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# 3. SYSTEM INTEGRITY
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#16a34a;"
    "text-transform:uppercase;letter-spacing:1.5px;margin:1.1rem 0 0.5rem 0'>System Integrity</div>",
    unsafe_allow_html=True,
)
integrity_html = "".join(
    f"<div style='color:#86efac;font-size:0.85rem;margin-bottom:0.2rem'>✓ {s}</div>"
    for s in integrity
)
st.markdown(
    f"<div style='background:#0f172a;border-left:3px solid #16a34a;"
    f"border-radius:0 8px 8px 0;padding:0.85rem 1.1rem'>{integrity_html}</div>",
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# 4. SUBSYSTEM HEALTH
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#94a3b8;"
    "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.7rem'>Subsystem Health</div>",
    unsafe_allow_html=True,
)

_STATUS_ICONS = {"Good": "✓", "Degraded": "!", "Critical": "✕", "Insufficient Data": "—"}
_STATUS_LABELS = {"Good": "Healthy", "Degraded": "Degraded", "Critical": "Critical", "Insufficient Data": "No Data"}

sub_names = list(subsystems.keys())
for row_start in range(0, len(sub_names), 3):
    row_names = sub_names[row_start:row_start + 3]
    cols = st.columns(len(row_names))
    for col, name in zip(cols, row_names):
        sub   = subsystems[name]
        st_val = sub["status"]
        sc    = _status_color(st_val)
        icon  = _STATUS_ICONS.get(st_val, "")
        lbl   = _STATUS_LABELS.get(st_val, st_val)
        insight = sub["issues"][0] if sub["issues"] else sub["meaning"]
        col.markdown(
            f"""<div style="border:1px solid {sc}44;border-top:3px solid {sc};
                border-radius:6px;padding:0.75rem 0.85rem;background:#0f172a;
                margin-bottom:0.5rem;min-height:110px">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:0.3rem">
                <div style="font-weight:700;font-size:0.82rem;color:#f1f5f9">{name}</div>
                <div style="padding:1px 7px;border-radius:3px;background:{sc}22;
                            color:{sc};font-size:0.72rem;font-weight:700">{icon} {lbl}</div>
              </div>
              <div style="font-size:0.78rem;color:#94a3b8;line-height:1.45">{insight}</div>
              <div style="font-size:0.68rem;color:#334155;margin-top:0.4rem;
                          border-top:1px solid #1e293b;padding-top:0.3rem">{sub['meaning']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# 5. ROOT CAUSE CHAIN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#94a3b8;"
    "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.7rem'>Root Cause Chain</div>",
    unsafe_allow_html=True,
)

rc1, rc_arrow1, rc2, rc_arrow2, rc3 = st.columns([5, 1, 5, 1, 5])
chain_base = "padding:0.9rem 1rem;border-radius:8px;background:#0f172a;min-height:90px"
with rc1:
    st.markdown(
        f"<div style='{chain_base};border:1px solid #334155'>"
        f"<div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:0.35rem'>Cause</div>"
        f"<div style='font-size:0.84rem;color:#f1f5f9;line-height:1.45'>{root_cause['cause']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with rc_arrow1:
    st.markdown("<div style='text-align:center;color:#475569;font-size:1.4rem;padding-top:1.5rem'>→</div>",
                unsafe_allow_html=True)
with rc2:
    st.markdown(
        f"<div style='{chain_base};border:1px solid #1d4ed8'>"
        f"<div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:0.35rem'>Effect</div>"
        f"<div style='font-size:0.84rem;color:#93c5fd;line-height:1.45'>{root_cause['effect']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with rc_arrow2:
    st.markdown("<div style='text-align:center;color:#475569;font-size:1.4rem;padding-top:1.5rem'>→</div>",
                unsafe_allow_html=True)
with rc3:
    st.markdown(
        f"<div style='{chain_base};border:1px solid #991b1b'>"
        f"<div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:0.35rem'>Risk</div>"
        f"<div style='font-size:0.84rem;color:#fca5a5;line-height:1.45'>{root_cause['risk']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# 6. SUPPORTING DATA
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#94a3b8;"
    "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.7rem'>Supporting Data</div>",
    unsafe_allow_html=True,
)

pwr   = secs.get("power", {})
vib   = secs.get("vibration", {})
imu_s = secs.get("imu", {})
mot   = secs.get("motors", {})

g1, g2 = st.columns(2)

with g1:
    volt_arr = pwr.get("voltage_v", np.array([]))
    curr_arr = pwr.get("current_a", np.array([]))
    if len(ts) and len(volt_arr) and len(curr_arr):
        n = min(len(ts), len(volt_arr), len(curr_arr), 4000)
        fig_bat = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Voltage (V)", "Current (A)"), vertical_spacing=0.1)
        fig_bat.add_trace(go.Scatter(x=ts[:n].tolist(), y=volt_arr[:n].tolist(),
                          mode="lines", line=dict(color="#2563eb", width=1.5), name="Voltage"),
                          row=1, col=1)
        fig_bat.add_trace(go.Scatter(x=ts[:n].tolist(), y=curr_arr[:n].tolist(),
                          mode="lines", fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                          line=dict(color="#dc2626", width=1.5), name="Current"),
                          row=2, col=1)
        fig_bat.update_layout(**PLT, title="Battery — Voltage & Current", height=300,
                              xaxis2_title="Time (s)")
        st.plotly_chart(fig_bat, use_container_width=True, key="hm_fig_bat")
    else:
        st.caption("No battery data in this log.")

with g2:
    rms_r   = vib.get("rolling_rms", np.array([]))
    rcout_1 = mot.get("rcout_1", np.array([]))
    if len(ts) and len(rms_r) and len(rcout_1):
        n = min(len(ts), len(rms_r), len(rcout_1), 4000)
        fig_vt = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Vibration RMS (m/s²)", "Motor 1 Output (µs)"),
                               vertical_spacing=0.1)
        fig_vt.add_trace(go.Scatter(x=ts[:n].tolist(), y=rms_r[:n].tolist(),
                         mode="lines", fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                         line=dict(color="#dc2626", width=1.3), name="Vib RMS"),
                         row=1, col=1)
        fig_vt.add_trace(go.Scatter(x=ts[:n].tolist(), y=rcout_1[:n].tolist(),
                         mode="lines", line=dict(color="#7c3aed", width=1.3), name="Motor 1"),
                         row=2, col=1)
        fig_vt.update_layout(**PLT, title="Vibration vs Throttle", height=300,
                             xaxis2_title="Time (s)")
        st.plotly_chart(fig_vt, use_container_width=True, key="hm_fig_vt")
    elif len(ts) and len(rms_r):
        n = min(len(ts), len(rms_r), 4000)
        fig_vt2 = go.Figure(go.Scatter(x=ts[:n].tolist(), y=rms_r[:n].tolist(),
                             mode="lines", fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                             line=dict(color="#dc2626", width=1.5)))
        for val, lbl, col in [(0.5,"Acceptable","#facc15"),(1.5,"Elevated","#f97316"),(3.0,"Critical","#dc2626")]:
            fig_vt2.add_hline(y=val, line_dash="dot", line_color=col, annotation_text=lbl)
        fig_vt2.update_layout(**PLT, title="Vibration RMS Timeline", height=300,
                              xaxis_title="Time (s)", yaxis_title="m/s²")
        st.plotly_chart(fig_vt2, use_container_width=True, key="hm_fig_vt2")
    else:
        st.caption("No vibration data in this log.")

g3, g4 = st.columns(2)

with g3:
    roll_arr  = imu_s.get("roll_deg",  np.array([]))
    pitch_arr = imu_s.get("pitch_deg", np.array([]))
    if len(ts) and len(roll_arr):
        n = min(len(ts), len(roll_arr), 4000)
        fig_att = go.Figure()
        fig_att.add_trace(go.Scatter(x=ts[:n].tolist(), y=np.abs(roll_arr[:n]).tolist(),
                          mode="lines", line=dict(color="#2563eb", width=1.2), name="|Roll|"))
        if len(pitch_arr):
            n2 = min(len(ts), len(pitch_arr), 4000)
            fig_att.add_trace(go.Scatter(x=ts[:n2].tolist(), y=np.abs(pitch_arr[:n2]).tolist(),
                              mode="lines", line=dict(color="#7c3aed", width=1.2), name="|Pitch|"))
        fig_att.add_hline(y=5, line_dash="dot", line_color="#d97706", annotation_text="5° threshold")
        fig_att.update_layout(**PLT, title="Attitude Error (|Roll|, |Pitch|)",
                              xaxis_title="Time (s)", yaxis_title="Degrees", height=280)
        st.plotly_chart(fig_att, use_container_width=True, key="hm_fig_att")
    else:
        st.caption("No attitude data in this log.")

with g4:
    sf     = vib.get("spec_f", np.array([]))
    st_arr = vib.get("spec_t", np.array([]))
    sdb    = vib.get("spec_db", np.array([[]]))
    if sf is not None and len(sf) and sdb.size > 1:
        fig_ekf = go.Figure(go.Heatmap(
            z=sdb.tolist(), x=st_arr.tolist(), y=sf.tolist(),
            colorscale="RdYlGn_r", colorbar=dict(title="dB"),
            zmin=float(np.percentile(sdb, 5)), zmax=float(np.percentile(sdb, 95)),
        ))
        fig_ekf.update_layout(**PLT, title="EKF Stress Proxy — Vibration Spectrogram",
                              xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                              yaxis=dict(range=[0, 200]), height=280)
        st.plotly_chart(fig_ekf, use_container_width=True, key="hm_fig_ekf")
    else:
        st.caption("Insufficient data for EKF stress chart.")

gspd = imu_s.get("ground_speed_ms", np.array([]))
if len(ts) and len(gspd) > 20:
    n = min(len(ts), len(gspd), 4000)
    gspd_arr = np.array(gspd[:n], dtype=float)
    window   = 50
    var_arr  = (np.array([np.std(gspd_arr[max(0,i-window):i+1]) for i in range(len(gspd_arr))])
                if len(gspd_arr) > window else gspd_arr)
    fig_path = go.Figure()
    fig_path.add_trace(go.Scatter(x=ts[:n].tolist(), y=gspd[:n].tolist(),
                       mode="lines", line=dict(color="#059669", width=1.2), name="Ground Speed",
                       opacity=0.5))
    fig_path.add_trace(go.Scatter(x=ts[:len(var_arr)].tolist(), y=var_arr.tolist(),
                       mode="lines", line=dict(color="#dc2626", width=1.5, dash="dot"),
                       name="Speed Variance", fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"))
    fig_path.update_layout(**PLT, title="Path Deviation — Ground Speed & Variance",
                           xaxis_title="Time (s)", yaxis_title="m/s", height=260)
    st.plotly_chart(fig_path, use_container_width=True, key="hm_fig_path")
else:
    st.caption("No GPS/speed data available for path deviation chart.")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FINAL HEALTH BRIEF
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#94a3b8;"
    "text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.8rem'>Final Health Brief</div>",
    unsafe_allow_html=True,
)

brief_col, conf_col = st.columns([3, 1])

with brief_col:
    condition_line = f"System condition: {cond['label']} — {cond['summary'].split('—')[-1].strip() if '—' in cond['summary'] else cond['summary']}"
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;"
        f"padding:1.2rem 1.4rem'>"
        f"<div style='font-weight:700;color:{cond['color']};font-size:0.95rem;"
        f"margin-bottom:0.8rem'>{condition_line}</div>",
        unsafe_allow_html=True,
    )
    if attention:
        issues_html = "".join(
            f"<div style='color:#94a3b8;font-size:0.84rem;margin-bottom:0.3rem'>"
            f"· {item['name']}</div>"
            for item in attention
        )
        st.markdown(
            f"<div style='color:#64748b;font-size:0.72rem;text-transform:uppercase;"
            f"letter-spacing:1px;margin-bottom:0.4rem'>Key Issues</div>"
            f"{issues_html}",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#64748b;font-size:0.84rem'>No key issues — all parameters nominal.</div>",
            unsafe_allow_html=True,
        )
    integrity_brief = integrity[0] if integrity else "Core subsystems nominal"
    st.markdown(
        f"<div style='margin-top:0.8rem;padding-top:0.7rem;border-top:1px solid #1e293b;"
        f"color:#86efac;font-size:0.84rem'>System integrity: {integrity_brief}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with conf_col:
    conf_colors = {"High": "#16a34a", "Medium": "#d97706", "Low": "#dc2626"}
    conf_c = conf_colors.get(confidence, "#64748b")
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid {conf_c}44;"
        f"border-radius:10px;padding:1.2rem;text-align:center;height:100%'>"
        f"<div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:0.5rem'>Assessment Confidence</div>"
        f"<div style='font-size:1.4rem;font-weight:900;color:{conf_c}'>{confidence}</div>"
        f"<div style='font-size:0.7rem;color:#475569;margin-top:0.4rem;line-height:1.4'>"
        f"Based on log completeness,<br>GPS coverage, and<br>parameter file presence</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# 8. DOWNLOAD + NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")

health_json = _build_health_json(
    score, grade, risk, dom_issue, subsystems,
    degradation, root_cause, evo, confidence,
    questionnaire, param_s,
    performance_model=res.get("performance_model"),
)

act1, act2 = st.columns(2)
with act1:
    st.download_button(
        "⬇  Download Drone Health File (.json)",
        data=health_json,
        file_name=f"drone_health_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )
    st.caption("Use this file in Mission Feasibility for health-aware planning.")
with act2:
    st.page_link("pages/2_Mission_Feasibility.py", label="🗺️  Go to Mission Feasibility", use_container_width=True)

st.markdown("---")
st.caption("DroneAcharya Health Monitoring · Condition assessment — no AI in the analysis layer.")
