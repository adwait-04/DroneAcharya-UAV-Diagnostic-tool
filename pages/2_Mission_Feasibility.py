"""
Mission Feasibility — Pre-flight Decision System
Constraint-based GO / RISK / NO-GO determination from physics first principles.
"""

import json
import math
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import mission_engine

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Mission Feasibility", page_icon="🗺️", layout="wide")

_css_path = _ROOT / "assests" / "style.css"
if _css_path.exists():
    with open(_css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "mf_health_json":    None,
    "mf_health_name":    None,
    "mf_results":        {},
    "mf_ran":            False,
    "mf_decision":       None,
    "mf_hard_fails":     [],
    "mf_risk_flags":     [],
    "mf_model_errors":   [],
    "mf_confirmed":      [],
    "mf_context_line":   "",
    "mf_signals":        {},
    "mf_drone_cfg_eff":  {},
}.items():
    st.session_state.setdefault(_k, _v)

# ── Environment / flight-type maps ────────────────────────────────────────────

_ENV_MAP = {
    "Calm (≤3 m/s wind, clear)":       {"wind_speed_ms": 2.0,  "ambient_temp_c": 20.0, "terrain_difficulty": 1.0},
    "Moderate (4–10 m/s, some gusts)": {"wind_speed_ms": 7.0,  "ambient_temp_c": 15.0, "terrain_difficulty": 1.2},
    "Harsh (>10 m/s, rain/fog/cold)":  {"wind_speed_ms": 14.0, "ambient_temp_c": 5.0,  "terrain_difficulty": 1.4},
}
_FT_MAP = {
    "Hover / Station-keeping": {"battery_margin_pct": 25.0},
    "Normal transit flight":   {"battery_margin_pct": 20.0},
    "Aggressive / High-speed": {"battery_margin_pct": 15.0},
}
_ENV_KEYS = list(_ENV_MAP.keys())
_FT_KEYS  = list(_FT_MAP.keys())

_DECISION_THEME = {
    "READY TO FLY":     {"color": "#16a34a", "bg": "#f0fdf4", "border": "#86efac", "icon": "✅"},
    "READY WITH RISK":  {"color": "#d97706", "bg": "#fffbeb", "border": "#fde68a", "icon": "⚠️"},
    "NOT READY TO FLY": {"color": "#dc2626", "bg": "#fef2f2", "border": "#fca5a5", "icon": "🚫"},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚁 DroneAcharya")
    st.markdown("---")
    st.markdown("### Navigate")
    st.page_link("app.py",                          label="🏠  Home")
    st.page_link("pages/1_Health_Monitoring.py",    label="🔬  Health Monitoring")
    st.page_link("pages/2_Mission_Feasibility.py",  label="🗺️  Mission Feasibility")
    st.page_link("pages/3_AboutUs.py",              label="ℹ️  About Us")
    st.markdown("---")

    st.markdown("### Drone Health File (.json)")
    st.caption("Export from Health Monitoring after running an analysis.")
    health_file = st.file_uploader(
        "Drone Health JSON", type=["json"], key="mf_health_upload",
    )
    if health_file is not None and health_file.name != st.session_state.get("mf_health_name"):
        try:
            raw = json.loads(health_file.read())
            st.session_state["mf_health_json"] = raw
            st.session_state["mf_health_name"] = health_file.name
            st.session_state["mf_ran"]         = False
            st.session_state["mf_results"]     = {}
            st.session_state["mf_decision"]    = None
            st.rerun()
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    if st.session_state.get("mf_health_name"):
        st.success(f"📋 {st.session_state['mf_health_name']}")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='font-size:2.2rem;font-weight:900;margin-bottom:0'>🗺️ Mission Feasibility</h1>",
    unsafe_allow_html=True,
)
st.caption("Performance-based pre-flight decision · measured power data · health-aware energy budget")
st.markdown("---")

# ── Require health JSON ───────────────────────────────────────────────────────

if st.session_state["mf_health_json"] is None:
    st.info(
        "Upload the **Drone Health JSON** from the sidebar to begin.\n\n"
        "Generate it from **Health Monitoring** after running an analysis."
    )
    st.page_link("pages/1_Health_Monitoring.py", label="🔬  Go to Health Monitoring")
    st.stop()

health_json = st.session_state["mf_health_json"]

# ── Extract drone config ──────────────────────────────────────────────────────

_ctx    = health_json.get("context", {})
_params = health_json.get("params",  {})

def _f(d, k, fallback=0.0):
    try:
        return float(d.get(k, fallback))
    except (TypeError, ValueError):
        return fallback

drone_config = {
    "dry_weight_kg":        _f(_ctx, "dry_weight_kg")        or _f(_params, "dry_weight_kg"),
    "battery_cells":        _f(_ctx, "battery_cells")         or _f(_params, "battery_cells",        4.0),
    "battery_capacity_mah": _f(_ctx, "battery_capacity_mah") or _f(_params, "battery_capacity_mah"),
    "motor_kv":             _f(_ctx, "motor_kv")              or _f(_params, "motor_kv"),
    "prop_diameter_in":     _f(_ctx, "prop_diameter_in")      or _f(_params, "prop_diameter_in",     10.0),
    "prop_pitch_in":        _f(_ctx, "prop_pitch_in")         or _f(_params, "prop_pitch_in",         4.5),
    "num_motors":           _f(_ctx, "num_motors")            or _f(_params, "num_motors",             4.0),
    "payload_kg":           _f(_ctx, "payload_kg")            or _f(_params, "payload_kg",             0.0),
    "frame_class":          _ctx.get("frame_class")           or _params.get("frame_class",             ""),
    "frame_size_in":        _f(_ctx, "frame_size_in")         or _f(_params, "frame_size_in",          18.0),
}

# ── Health data ───────────────────────────────────────────────────────────────

overall_health   = health_json.get("overall_health", {})
subsystem_health = health_json.get("subsystem_health", {})
health_score     = float(overall_health.get("score",   100.0))
health_grade     = overall_health.get("grade",         "?")
health_risk_lbl  = overall_health.get("risk_level",    overall_health.get("risk", "Unknown"))

def _sub_status(name):
    sub = subsystem_health.get(name, {})
    return sub.get("status", "Good") if isinstance(sub, dict) else "Good"

_critical_subs   = [
    n for n in subsystem_health
    if isinstance(subsystem_health[n], dict) and subsystem_health[n].get("status") == "Critical"
]
_batt_sub  = _sub_status("Power (Battery)")
_vib_sub   = _sub_status("Structure / Vibration")
_ekf_sub   = _sub_status("Estimation (EKF)")

_missing_config = [k for k, v in {
    "Dry Weight":       drone_config["dry_weight_kg"],
    "Battery Cells":    drone_config["battery_cells"],
    "Battery Capacity": drone_config["battery_capacity_mah"],
    "Prop Diameter":    drone_config["prop_diameter_in"],
}.items() if not v or v <= 0]

with st.expander("📋  Drone Configuration & Health Summary", expanded=False):
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric("Frame",       drone_config["frame_class"] or "—")
        st.metric("Dry Weight",  f"{drone_config['dry_weight_kg']:.3f} kg"  if drone_config["dry_weight_kg"]    else "—")
        st.metric("Motors",      str(int(drone_config["num_motors"]))        if drone_config["num_motors"]       else "—")
    with dc2:
        st.metric("Battery",
                  f"{int(drone_config['battery_cells'])}S / {int(drone_config['battery_capacity_mah'])} mAh"
                  if drone_config["battery_capacity_mah"] else "—")
        st.metric("Motor KV",    str(int(drone_config["motor_kv"]))          if drone_config["motor_kv"]         else "—")
    with dc3:
        st.metric("Prop",
                  f"{drone_config['prop_diameter_in']:.1f}×{drone_config['prop_pitch_in']:.1f} in"
                  if drone_config["prop_diameter_in"] else "—")
        st.metric("Payload (recorded)", f"{drone_config['payload_kg']:.3f} kg")
    st.markdown("---")
    hs1, hs2, hs3, hs4 = st.columns(4)
    hs1.metric("Health Score",     f"{health_score:.1f}/100")
    hs2.metric("Grade / Risk",     f"{health_grade} · {health_risk_lbl}")
    hs3.metric("Battery",          _batt_sub)
    hs4.metric("Vibration",        _vib_sub)
    if _critical_subs:
        st.error(f"Critical subsystems: **{', '.join(_critical_subs)}**")

if _missing_config:
    st.warning(f"Drone config incomplete — missing: {', '.join(_missing_config)}. Physics results may be inaccurate.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  Mission Questionnaire
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#94a3b8;text-transform:uppercase;"
    "letter-spacing:1.5px;margin-bottom:0.9rem'>Mission Parameters</div>",
    unsafe_allow_html=True,
)

mq1, mq2, mq3 = st.columns(3)

with mq1:
    mf_payload = st.number_input(
        "Mission Payload (kg)", 0.0, 50.0, 0.0, 0.05, format="%.3f", key="mf_payload",
        help="Additional payload for this mission. Zero is valid (drone only). Added to dry weight for TOW.",
    )
    mf_distance = st.number_input(
        "Mission Distance (km)", 0.0, 500.0, 0.0, 0.5, format="%.1f", key="mf_distance",
        help="One-way or total path distance. Drives transit energy budget.",
    )
    mf_duration = st.number_input(
        "Required Duration (min)", 0, 240, 0, 1, key="mf_duration",
        help="Minimum airborne time required. Drives hover energy budget. If both distance and duration are entered, the more demanding constraint applies.",
    )

with mq2:
    mf_env = st.selectbox(
        "Environment *", _ENV_KEYS, key="mf_env",
        help="Operating conditions. Maps to wind speed, temperature, and terrain factor.",
    )
    mf_flight_type = st.selectbox(
        "Flight Profile *", _FT_KEYS, key="mf_flight_type",
        help="Determines the battery safety reserve margin.",
    )
    mf_altitude = st.number_input(
        "Operating Altitude AMSL (m)", 0, 5000, 0, 10, key="mf_altitude",
        help="Altitude above mean sea level at the operating site. Used only for ISA air density correction — does not account for climb energy.",
    )

with mq3:
    mf_battery_charged = st.checkbox(
        "Battery is fully charged",
        value=True,
        key="mf_battery_charged",
        help="If unchecked, 80% state of charge is applied to the energy budget.",
    )

    dry_w       = drone_config["dry_weight_kg"] or 0.0
    tow_preview = dry_w + mf_payload
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;"
        f"padding:0.85rem 1rem;margin-top:0.5rem'>"
        f"<div style='font-size:0.65rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:0.4rem'>Total Takeoff Weight</div>"
        f"<div style='font-size:1.5rem;font-weight:900;color:#f1f5f9'>{tow_preview:.3f} kg</div>"
        f"<div style='font-size:0.72rem;color:#475569;margin-top:0.25rem'>"
        f"Dry {dry_w:.3f} + Payload {mf_payload:.3f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if not mf_battery_charged:
        st.markdown(
            "<div style='background:#1e293b;border-left:2px solid #d97706;padding:0.5rem 0.75rem;"
            "border-radius:0 6px 6px 0;font-size:0.78rem;color:#fbbf24;margin-top:0.4rem'>"
            "80% SOC applied to energy budget.</div>",
            unsafe_allow_html=True,
        )

_mf_missing = []
if mf_distance <= 0 and mf_duration <= 0:
    _mf_missing.append("Mission Distance or Duration (at least one required)")

if _mf_missing:
    st.warning(f"Required: {', '.join(_mf_missing)}")

st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────

run_btn = st.button(
    "⚡  Assess Mission Feasibility",
    disabled=bool(_mf_missing),
    type="primary",
    use_container_width=True,
)

if run_btn and not _mf_missing:
    env_p = _ENV_MAP[mf_env]
    ft_p  = _FT_MAP[mf_flight_type]

    cap_eff = drone_config["battery_capacity_mah"]
    if not mf_battery_charged:
        cap_eff *= 0.80
    if _batt_sub == "Critical":
        cap_eff *= 0.85
    elif _batt_sub == "Degraded":
        cap_eff *= 0.93

    drone_cfg_eff = {**drone_config, "battery_capacity_mah": cap_eff}

    signals = {
        "target_payload_kg":  mf_payload,
        "distance_km":        mf_distance if mf_distance > 0 else float("nan"),
        "duration_min":       float(mf_duration),
        "altitude_m":         mf_altitude,
        "wind_speed_ms":      env_p["wind_speed_ms"],
        "ambient_temp_c":     env_p["ambient_temp_c"],
        "terrain_difficulty": env_p["terrain_difficulty"],
        "battery_margin_pct": ft_p["battery_margin_pct"],
    }

    with st.spinner("Analysing mission feasibility…"):
        result = mission_engine.analyze(
            signals=signals,
            params=drone_cfg_eff,
            context={"performance_model": health_json.get("performance_model") or {}},
        )

        diag          = result.get("diagnostics", {})
        metrics       = result.get("metrics",     {})
        engine_alerts = result.get("alerts",      [])
        energy_feas   = diag.get("energy_feasibility", "N/A")
        payload_feas  = diag.get("payload_feasibility", "N/A")
        env_risk      = diag.get("environmental_risk",  "NOMINAL")
        engine_base   = diag.get("mission_decision",    "N/A")

        def _m(k, fb=float("nan")):
            v = metrics.get(k, fb)
            try:
                f = float(v)
                return float("nan") if math.isnan(f) else f
            except (TypeError, ValueError):
                return fb

        flight_h      = _m("flight_time_h")
        margin_pct    = _m("energy_margin_pct")
        req_wh        = _m("required_energy_wh")
        avail_wh      = _m("usable_energy_wh")
        max_pl_kg     = _m("max_payload_kg")
        hover_end_min = _m("hover_endurance_min")
        range_km      = _m("range_limited_km")
        physics_basis = metrics.get("physics_basis", "unknown")

        # ── Hard constraints → NOT READY TO FLY ──────────────────────────────
        hard_fails = []

        if energy_feas == "INFEASIBLE":
            hard_fails.append("Battery energy is insufficient for the stated mission requirements")

        if energy_feas == "BORDERLINE" and "Harsh" in mf_env:
            hard_fails.append("Energy reserve is too low to absorb harsh environment conditions")

        if payload_feas == "OVERLOADED":
            hard_fails.append("Mission payload exceeds the motor thrust capacity of this configuration")

        if _critical_subs:
            sub_str = ", ".join(_critical_subs)
            hard_fails.append(f"Drone health is Critical ({sub_str}) — airworthiness cannot be confirmed")

        if mf_duration > 0 and not math.isnan(flight_h) and physics_basis == "transit":
            avail_min = flight_h * 60.0
            if avail_min < mf_duration:
                hard_fails.append("Required flight duration exceeds available energy endurance")

        if mf_distance > 0 and not math.isnan(range_km) and mf_distance > range_km:
            hard_fails.append("Required mission distance exceeds maximum achievable range")

        # ── Risk flags → READY WITH RISK ─────────────────────────────────────
        risk_flags = []

        if energy_feas == "MARGINAL":
            risk_flags.append("Energy margin is below the 20% safe-operating threshold")

        if energy_feas == "BORDERLINE" and "Harsh" not in mf_env:
            risk_flags.append("Energy margin is critically low — no buffer for unplanned deviations")

        if "WIND" in env_risk:
            risk_flags.append("Wind speed exceeds 50% of cruise speed — actual consumption will exceed the modelled budget")

        if "DENSITY_ALTITUDE" in env_risk:
            risk_flags.append("Reduced air density at operating altitude increases hover power demand above the measured baseline")

        if payload_feas == "MARGINAL":
            risk_flags.append("Mission payload is close to the thrust limit — available margin is reduced")

        if health_score < 70 and not _critical_subs:
            risk_flags.append(f"Drone health is degraded (Grade {health_grade}) — system reliability is below nominal")

        if _vib_sub == "Degraded":
            risk_flags.append("Elevated vibration recorded in health report — IMU quality and control authority are reduced")

        if _ekf_sub == "Degraded":
            risk_flags.append("State estimator health is degraded — position and attitude accuracy are below nominal")

        if not mf_battery_charged:
            risk_flags.append("Battery is not at full charge — energy budget reflects an 80% state of charge")

        if _batt_sub in ("Degraded", "Critical") and not hard_fails:
            risk_flags.append(f"Battery health is {_batt_sub.lower()} — effective capacity is reduced")

        if mf_duration > 0 and not math.isnan(flight_h) and physics_basis == "transit":
            avail_min = flight_h * 60.0
            if mf_duration >= avail_min * 0.9 and not hard_fails:
                risk_flags.append("Required duration is within 10% of maximum transit endurance — margin is critically tight")

        if mf_distance > 0 and not math.isnan(range_km) and mf_distance >= range_km * 0.9 and not hard_fails:
            risk_flags.append("Required distance is within 10% of maximum achievable range — margin is critically tight")

        model_errors: list = []
        for _a in engine_alerts:
            _lvl = _a.get("level", "warning")
            _msg = _a.get("msg", "")
            if not _msg:
                continue
            if _lvl == "critical" and _msg not in hard_fails:
                hard_fails.append(_msg)
            elif _lvl == "warning" and _msg not in risk_flags:
                risk_flags.append(_msg)
            elif _lvl == "model_error" and _msg not in model_errors:
                model_errors.append(_msg)

        hard_fails   = hard_fails[:4]
        risk_flags   = risk_flags[:4]
        model_errors = model_errors[:4]

        # ── Decision ──────────────────────────────────────────────────────────
        if hard_fails:
            decision = "NOT READY TO FLY"
        elif (not math.isnan(margin_pct)
              and margin_pct >= 20.0
              and payload_feas != "OVERLOADED"):
            decision = "READY TO FLY"
        elif len(risk_flags) >= 2:
            decision = "READY WITH RISK"
        else:
            decision = "READY TO FLY"

        if engine_base == "NOT READY TO FLY" and decision != "NOT READY TO FLY":
            decision = "NOT READY TO FLY"
        elif engine_base == "READY WITH RISK" and decision == "READY TO FLY":
            decision = "READY WITH RISK"

        # ── Confirmed constraints ─────────────────────────────────────────────
        confirmed = []
        if energy_feas == "FEASIBLE" and not math.isnan(margin_pct):
            confirmed.append(f"Energy budget: {margin_pct:.0f}% margin above the safety reserve")
        if payload_feas == "SAFE" and not math.isnan(max_pl_kg):
            confirmed.append(f"Payload ({mf_payload:.3f} kg) is within motor thrust capacity ({max_pl_kg:.3f} kg max)")
        if health_score >= 75 and not _critical_subs:
            confirmed.append(f"Drone health is nominal (score {health_score:.0f}/100)")
        if env_risk == "NOMINAL":
            confirmed.append("Environmental conditions are within safe operating limits")

        # ── Context line ──────────────────────────────────────────────────────
        context_line = ""
        if _batt_sub in ("Degraded", "Critical") and energy_feas in ("MARGINAL", "BORDERLINE", "INFEASIBLE"):
            context_line = "Battery degradation recorded in the health report is constraining the effective energy budget."
        elif "WIND" in env_risk and energy_feas in ("MARGINAL", "BORDERLINE", "INFEASIBLE"):
            context_line = "Wind conditions are the primary driver of elevated energy demand for this mission."
        elif not mf_battery_charged and energy_feas != "FEASIBLE":
            context_line = "Non-full state of charge is the binding constraint on available energy."

    st.session_state["mf_results"]      = result
    st.session_state["mf_ran"]          = True
    st.session_state["mf_decision"]     = decision
    st.session_state["mf_hard_fails"]   = hard_fails
    st.session_state["mf_risk_flags"]   = risk_flags
    st.session_state["mf_model_errors"] = model_errors
    st.session_state["mf_confirmed"]    = confirmed
    st.session_state["mf_context_line"] = context_line
    st.session_state["mf_signals"]      = signals
    st.session_state["mf_drone_cfg_eff"]= drone_cfg_eff
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  Results
# ════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("mf_ran"):
    st.stop()

res          = st.session_state["mf_results"]
decision     = st.session_state["mf_decision"]
hard_fails   = st.session_state.get("mf_hard_fails",   [])
risk_flags   = st.session_state.get("mf_risk_flags",   [])
model_errors = st.session_state.get("mf_model_errors", [])
confirmed    = st.session_state.get("mf_confirmed",    [])
context_line = st.session_state.get("mf_context_line", "")
signals_used = st.session_state.get("mf_signals",      {})
dc_eff       = st.session_state.get("mf_drone_cfg_eff", drone_config)

metrics      = res.get("metrics",     {})
diag         = res.get("diagnostics", {})

def _safe(k, fb=float("nan")):
    v = metrics.get(k, fb)
    try:
        f = float(v)
        return float("nan") if math.isnan(f) else f
    except (TypeError, ValueError):
        return fb

margin_pct    = _safe("energy_margin_pct")
margin_wh     = _safe("energy_margin_wh")
range_km      = _safe("range_limited_km")
hover_end_min = _safe("hover_endurance_min")
req_wh        = _safe("required_energy_wh")
avail_wh      = _safe("usable_energy_wh")
total_kg      = _safe("total_mass_kg")
hover_w       = _safe("hover_power_w")
ff_w          = _safe("forward_flight_power_w")
cruise_ms     = _safe("cruise_speed_ms")
max_pl_kg     = _safe("max_payload_kg")
rho_val       = _safe("air_density_kgm3", 1.225)
flight_h      = _safe("flight_time_h")
physics_basis = metrics.get("physics_basis", "unknown")

payload_feas = diag.get("payload_feasibility", "N/A")
env_risk_str = diag.get("environmental_risk",  "NOMINAL")

theme = _DECISION_THEME[decision]

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# 1. DECISION BANNER
# ═════════════════════════════════════════════════════════════════════════════

_subtitles = {
    "READY TO FLY":     "All energy, payload, and health constraints are satisfied with adequate margin.",
    "READY WITH RISK":  "Mission is physically achievable but one or more operational margins are tight.",
    "NOT READY TO FLY": "One or more hard physical constraints are violated. The mission cannot proceed safely.",
}

banner_col, tow_col = st.columns([3, 1])

with banner_col:
    st.markdown(
        f"<div style='padding:1.6rem 1.8rem;background:{theme['bg']};"
        f"border:2px solid {theme['border']};border-left:5px solid {theme['color']};"
        f"border-radius:10px'>"
        f"<div style='font-size:0.68rem;font-weight:700;color:{theme['color']};"
        f"text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem'>"
        f"Pre-flight Decision</div>"
        f"<div style='font-size:2.4rem;font-weight:900;color:{theme['color']};line-height:1.1'>"
        f"{theme['icon']} {decision}</div>"
        f"<div style='font-size:0.9rem;color:#475569;margin-top:0.6rem'>"
        f"{_subtitles[decision]}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with tow_col:
    _pl_used  = signals_used.get("target_payload_kg", 0)
    _dry_used = dc_eff.get("dry_weight_kg", 0)
    tow_disp  = total_kg if not math.isnan(total_kg) else (_dry_used + _pl_used)
    st.markdown(
        f"<div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;"
        f"padding:1.2rem;text-align:center'>"
        f"<div style='font-size:0.63rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:0.3rem'>Takeoff Weight</div>"
        f"<div style='font-size:2rem;font-weight:900;color:#f1f5f9'>{tow_disp:.3f}</div>"
        f"<div style='font-size:0.72rem;color:#64748b'>kg</div>"
        f"<div style='font-size:0.68rem;color:#334155;margin-top:0.4rem'>"
        f"Dry {_dry_used:.3f} + Payload {_pl_used:.3f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# 2. CONSTRAINT REASONS
# ═════════════════════════════════════════════════════════════════════════════

if hard_fails:
    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;color:#dc2626;"
        "text-transform:uppercase;letter-spacing:1.5px;margin:1.2rem 0 0.5rem'>Constraints Not Met</div>",
        unsafe_allow_html=True,
    )
    for fail in hard_fails:
        st.markdown(
            f"<div style='background:#0f172a;border-left:3px solid #dc2626;"
            f"border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin-bottom:0.4rem;"
            f"color:#fca5a5;font-size:0.88rem;line-height:1.55'>{fail}</div>",
            unsafe_allow_html=True,
        )

if risk_flags:
    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;color:#d97706;"
        "text-transform:uppercase;letter-spacing:1.5px;margin:1rem 0 0.5rem'>Elevated Risk Factors</div>",
        unsafe_allow_html=True,
    )
    for flag in risk_flags:
        st.markdown(
            f"<div style='background:#0f172a;border-left:3px solid #d97706;"
            f"border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin-bottom:0.4rem;"
            f"color:#fbbf24;font-size:0.88rem;line-height:1.55'>{flag}</div>",
            unsafe_allow_html=True,
        )

if confirmed and decision != "NOT READY TO FLY":
    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;color:#16a34a;"
        "text-transform:uppercase;letter-spacing:1.5px;margin:1rem 0 0.5rem'>Confirmed</div>",
        unsafe_allow_html=True,
    )
    for item in confirmed[:3]:
        st.markdown(
            f"<div style='background:#0f172a;border-left:3px solid #16a34a;"
            f"border-radius:0 8px 8px 0;padding:0.65rem 1rem;margin-bottom:0.3rem;"
            f"color:#86efac;font-size:0.87rem'>✓ {item}</div>",
            unsafe_allow_html=True,
        )

if context_line:
    st.markdown(
        f"<div style='background:#1e293b;border-left:3px solid #64748b;"
        f"border-radius:0 6px 6px 0;padding:0.55rem 1rem;font-size:0.82rem;"
        f"color:#94a3b8;margin-top:0.5rem'>{context_line}</div>",
        unsafe_allow_html=True,
    )

if model_errors:
    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;color:#7c3aed;"
        "text-transform:uppercase;letter-spacing:1.5px;margin:1rem 0 0.5rem'>"
        "Physics Model Inconsistencies</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem'>"
        "These are input or model issues — they do not affect mission readiness.</div>",
        unsafe_allow_html=True,
    )
    for me in model_errors:
        st.markdown(
            f"<div style='background:#0f172a;border-left:3px solid #7c3aed;"
            f"border-radius:0 8px 8px 0;padding:0.65rem 1rem;margin-bottom:0.3rem;"
            f"color:#c4b5fd;font-size:0.87rem'>⚙ {me}</div>",
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# 3. KEY METRICS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")

st.markdown(
    "<style>"
    "[data-testid='stMetricValue']{font-size:1.05rem!important;white-space:nowrap;overflow:hidden;"
    "text-overflow:ellipsis}"
    "[data-testid='stMetricLabel']{font-size:0.72rem!important}"
    "[data-testid='stMetricDelta']{font-size:0.72rem!important}"
    "</style>",
    unsafe_allow_html=True,
)

km1, km2, km3, km4 = st.columns(4)

km1.metric(
    "Energy Margin",
    f"{margin_pct:.0f}%" if not math.isnan(margin_pct) else "N/A",
    delta=f"{margin_pct:+.0f}%" if not math.isnan(margin_pct) else None,
    delta_color="normal" if not math.isnan(margin_pct) and margin_pct >= 0 else "inverse",
)
km2.metric(
    "Req / Avail (Wh)",
    f"{req_wh:.0f} / {avail_wh:.0f}"
    if not (math.isnan(req_wh) or math.isnan(avail_wh)) else "N/A",
)

if physics_basis == "hover" or (math.isnan(range_km) and not math.isnan(hover_end_min)):
    km3.metric("Max Hover", f"{hover_end_min:.0f} min" if not math.isnan(hover_end_min) else "N/A")
else:
    km3.metric("Max Range", f"{range_km:.1f} km"       if not math.isnan(range_km)       else "N/A")

km4.metric("Hover Power", f"{hover_w:.0f} W" if not math.isnan(hover_w) else "N/A")

cap_orig     = drone_config["battery_capacity_mah"]
cap_eff_used = dc_eff.get("battery_capacity_mah", cap_orig)
if abs(cap_eff_used - cap_orig) > 1:
    reduction_pct = (1 - cap_eff_used / cap_orig) * 100
    soc_part    = "80% SOC" if not mf_battery_charged else ""
    health_part = "health degradation" if _batt_sub != "Good" else ""
    sep         = " + " if soc_part and health_part else ""
    st.caption(
        f"Battery: {cap_eff_used:.0f} mAh effective "
        f"(−{reduction_pct:.0f}% from {cap_orig:.0f} mAh nominal — {soc_part}{sep}{health_part})"
    )

# ═════════════════════════════════════════════════════════════════════════════
# 4. PHYSICS DETAIL (collapsed)
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")

with st.expander("📊  Physics Detail", expanded=False):
    import numpy as np
    _T = dict(template="plotly_white", plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff")

    try:
        import plotly.graph_objects as go

        ph1, ph2 = st.columns(2)
        env_p_used = _ENV_MAP[mf_env]

        with ph1:
            st.markdown("**Energy Budget**")
            if not (math.isnan(req_wh) or math.isnan(avail_wh)):
                cells_v       = dc_eff.get("battery_cells", 4.0)
                total_batt_wh = (cap_eff_used / 1000.0) * float(cells_v) * 3.8
                ft_margin_pct = _FT_MAP[mf_flight_type]["battery_margin_pct"]
                reserve_wh    = total_batt_wh * (ft_margin_pct / 100.0)
                remaining     = avail_wh - req_wh

                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Battery Total", f"Reserve ({ft_margin_pct:.0f}%)", "Mission Required", "Remaining"],
                    y=[total_batt_wh, -reserve_wh, -req_wh, 0],
                    text=[f"{total_batt_wh:.1f} Wh", f"−{reserve_wh:.1f} Wh",
                          f"−{req_wh:.1f} Wh", f"{remaining:.1f} Wh"],
                    textposition="outside",
                    connector={"line": {"color": "#94a3b8"}},
                    increasing={"marker": {"color": "#16a34a"}},
                    decreasing={"marker": {"color": "#dc2626"}},
                    totals={"marker": {"color": "#2563eb" if remaining >= 0 else "#dc2626"}},
                ))
                fig.update_layout(yaxis_title="Energy (Wh)", height=300,
                                  margin=dict(l=0, r=0, t=10, b=20), **_T)
                st.plotly_chart(fig, use_container_width=True, key="mf_fig_wf")

        with ph2:
            st.markdown("**Power Components**")
            if not (math.isnan(hover_w) or math.isnan(ff_w)):
                fig = go.Figure(go.Bar(
                    x=["Induced", "Profile Drag", "Parasitic"],
                    y=[hover_w * 0.85, hover_w * 0.15, max(0.0, ff_w - hover_w * 0.85)],
                    marker_color=["#2563eb", "#7c3aed", "#dc2626"],
                    text=[f"{hover_w * 0.85:.1f} W", f"{hover_w * 0.15:.1f} W",
                          f"{max(0.0, ff_w - hover_w * 0.85):.1f} W"],
                    textposition="outside",
                ))
                fig.update_layout(yaxis_title="Power (W)", height=300,
                                  margin=dict(l=0, r=0, t=10, b=20), **_T)
                st.plotly_chart(fig, use_container_width=True, key="mf_fig_pw")

        dry_kg  = dc_eff.get("dry_weight_kg", float("nan"))
        n_mot   = float(dc_eff.get("num_motors", 4.0))
        prop_in = dc_eff.get("prop_diameter_in", 10.0)
        pr_s    = (prop_in * 0.0254) / 2.0
        da_s    = math.pi * pr_s ** 2
        max_sw  = max(dry_kg * 0.8 if not math.isnan(dry_kg) else 0.5,
                      max_pl_kg * 1.2 if not math.isnan(max_pl_kg) else 0.5, 0.5)
        pl_sweep = np.linspace(0.0, max_sw, 60)

        if physics_basis == "hover" and not (math.isnan(dry_kg) or math.isnan(avail_wh)) and hover_w > 0:
            st.markdown("**Payload vs Hover Endurance**")
            end_arr = []
            for pl_s in pl_sweep:
                m_t = dry_kg + pl_s
                t_s = (m_t * 9.80665) / n_mot
                v_i = math.sqrt(t_s / (2.0 * rho_val * da_s + 1e-12))
                hp  = t_s * v_i * n_mot * 1.15
                end_arr.append((avail_wh / (hp + 1e-9)) * 60.0)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=pl_sweep.tolist(), y=end_arr, mode="lines",
                                       fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
                                       line=dict(color="#2563eb", width=2.5)))
            _mfp = signals_used.get("target_payload_kg", 0)
            if _mfp and _mfp > 0:
                fig_s.add_vline(x=float(_mfp), line_dash="dash", line_color="#dc2626",
                                annotation_text=f"Mission: {_mfp:.2f} kg", annotation_font_color="#dc2626")
            _dur = signals_used.get("duration_min", 0)
            if _dur and _dur > 0:
                fig_s.add_hline(y=float(_dur), line_dash="dot", line_color="#16a34a",
                                annotation_text=f"Required: {_dur:.0f} min")
            fig_s.update_layout(xaxis_title="Payload (kg)", yaxis_title="Max Hover Endurance (min)",
                                height=280, margin=dict(l=0, r=0, t=10, b=30), **_T)
            st.plotly_chart(fig_s, use_container_width=True, key="mf_fig_sweep")

        elif not (math.isnan(dry_kg) or math.isnan(avail_wh) or math.isnan(ff_w) or math.isnan(cruise_ms)) \
                and ff_w > 0 and cruise_ms > 0:
            st.markdown("**Payload vs Achievable Range**")
            rng_arr = []
            for pl_s in pl_sweep:
                m_t = dry_kg + pl_s
                w_s = m_t * 9.80665
                t_s = w_s / n_mot
                v_i = math.sqrt(t_s / (2.0 * rho_val * da_s + 1e-12))
                hp  = t_s * v_i * n_mot * 1.15
                vef = max(1.0, cruise_ms + env_p_used["wind_speed_ms"] * 0.5)
                par = 0.015 * 0.5 * rho_val * vef ** 2 * (w_s / (rho_val * da_s * n_mot + 1e-9))
                pw  = (hp * 0.85 + par) * env_p_used["terrain_difficulty"]
                rng_arr.append((avail_wh / (pw + 1e-9)) * cruise_ms * 3.6)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=pl_sweep.tolist(), y=rng_arr, mode="lines",
                                       fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
                                       line=dict(color="#2563eb", width=2.5)))
            _mfp = signals_used.get("target_payload_kg", 0)
            if _mfp and _mfp > 0:
                fig_s.add_vline(x=float(_mfp), line_dash="dash", line_color="#dc2626",
                                annotation_text=f"Mission: {_mfp:.2f} kg", annotation_font_color="#dc2626")
            _mfd = signals_used.get("distance_km", 0)
            if not math.isnan(_mfd) and _mfd > 0:
                fig_s.add_hline(y=float(_mfd), line_dash="dot", line_color="#16a34a",
                                annotation_text=f"Required: {_mfd:.1f} km")
            fig_s.update_layout(xaxis_title="Payload (kg)", yaxis_title="Maximum Range (km)",
                                height=280, margin=dict(l=0, r=0, t=10, b=30), **_T)
            st.plotly_chart(fig_s, use_container_width=True, key="mf_fig_sweep")

        _flight_label = "Hover Time" if physics_basis == "hover" else "Transit Time"
        _flight_val   = f"{flight_h * 60:.1f} min" if not math.isnan(flight_h) else "N/A"

        st.markdown("**Physics Metrics**")
        rows = [
            ("Energy Basis",          physics_basis.title()),
            ("Total Mass (TOW)",      f"{total_kg:.3f} kg"        if not math.isnan(total_kg)       else "N/A"),
            ("Hover Power",           f"{hover_w:.1f} W"          if not math.isnan(hover_w)         else "N/A"),
            ("Forward Flight Power",  f"{ff_w:.1f} W"             if not math.isnan(ff_w)            else "N/A"),
            ("Cruise Speed",          f"{cruise_ms:.2f} m/s"      if not math.isnan(cruise_ms)       else "N/A"),
            ("Max Hover Endurance",   f"{hover_end_min:.1f} min"  if not math.isnan(hover_end_min)   else "N/A"),
            ("Max Range",             f"{range_km:.2f} km"        if not math.isnan(range_km)        else "N/A"),
            ("Required Energy",       f"{req_wh:.2f} Wh"          if not math.isnan(req_wh)          else "N/A"),
            ("Available Energy",      f"{avail_wh:.2f} Wh"        if not math.isnan(avail_wh)        else "N/A"),
            ("Energy Margin",         f"{margin_wh:.2f} Wh / {margin_pct:.1f}%"
             if not (math.isnan(margin_wh) or math.isnan(margin_pct)) else "N/A"),
            ("Air Density",           f"{rho_val:.4f} kg/m³"),
            ("Effective Battery",     f"{cap_eff_used:.0f} mAh"),
            (_flight_label,           _flight_val),
        ]
        tc1, tc2 = st.columns(2)
        half = len(rows) // 2
        for col, chunk in [(tc1, rows[:half]), (tc2, rows[half:])]:
            with col:
                st.markdown(
                    "<div style='background:#f8fafc;border-radius:8px;padding:0.5rem 0.75rem;'>",
                    unsafe_allow_html=True,
                )
                for label, val in chunk:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:5px 0;border-bottom:1px solid #e2e8f0'>"
                        f"<span style='color:#374151;font-size:0.83rem'>{label}</span>"
                        f"<span style='font-weight:700;font-size:0.88rem;color:#111827'>{val}</span></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

    except ImportError:
        st.info("Install plotly for charts: `pip install plotly`")

st.markdown("---")
st.caption("DroneAcharya Mission Feasibility · Physics-driven pre-flight decision · Actuator disk model")

