"""
Mission Feasibility Engine — Performance-Based Energy & Constraint Evaluator

Pipeline (FIXED ORDER — NO EARLY RETURNS):
  1. Input Conditioning   — mission scenario + drone hardware config
  2. Performance Loading  — extract measured data from health JSON performance_model
  3. Energy Budget        — usable energy with voltage-sag + reserve derating
  4. Endurance & Range    — derived from measured power and observed GPS speed
  5. Constraint Evaluation — 5 hard constraints, margin-based risk classification
  6. Diagnostics          — structured issues, actions, risk level
  7. Scoring

Entrypoint: analyze(signals, params, context) → fixed result schema

context MUST contain {"performance_model": {...}} injected from the health JSON.
When performance_model is absent/null the engine flags a data-quality error and
degrades gracefully — it does NOT fall back to prop-geometry estimation.

Energy model:
  total_energy_wh  = battery_capacity_mah × cells × 3.8 / 1000
  usable_energy_wh = total_energy_wh × (1 − reserve_pct) × (1 − voltage_sag_pct)

Hard constraints (mission fails if ANY violated):
  C1  required_duration  > endurance_min   × 0.85
  C2  required_distance  > range_km        × 0.85
  C3  thrust_margin_pct  < 0.25
  C4  voltage_sag_pct    > 0.25
  C5  hover_throttle     > 0.70

Risk level:
  HIGH    — any hard constraint violated  OR  min margin < 15 %
  MEDIUM  — min margin in [15 %, 30 %)
  LOW     — all margins ≥ 30 %
"""

from typing import Any, Dict, List, Tuple

import numpy as np


# ── Module constants ──────────────────────────────────────────────────────────

_RHO_SL              = 1.225    # kg/m³ — sea-level air density
_T_REF_K             = 288.15   # K
_L_RATE              = 0.0065   # K/m
_FALLBACK_CRUISE_MS  = 5.0      # m/s — conservative GPS-free fallback
_BASE_RESERVE_PCT    = 20.0     # % — default battery reserve

# Hard-constraint thresholds (spec-mandated)
_C3_THRUST_MIN       = 0.25
_C4_SAG_MAX          = 0.25
_C5_THROTTLE_MAX     = 0.70
_CONSTRAINT_SAFETY   = 0.85     # 85 % rule for time/range feasibility

# Risk margin thresholds
_MARGIN_HIGH_RISK    = 15.0     # %
_MARGIN_MED_RISK     = 30.0     # %


# ── Public entrypoint ─────────────────────────────────────────────────────────


def analyze(
    signals: Dict[str, Any],
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Mission feasibility analysis.

    signals : mission scenario (payload, distance, duration, environment)
    params  : drone hardware config (from health JSON context + questionnaire)
    context : MUST include {"performance_model": {...}} from health JSON

    Returns:
        {"score": float, "metrics": {}, "sections": {}, "diagnostics": {}, "alerts": []}
    """
    result: Dict[str, Any] = {
        "score": 0.0, "metrics": {}, "sections": {}, "diagnostics": {}, "alerts": [],
    }

    mission, drone, env  = _condition_inputs(signals, params, context)
    perf                 = _load_perf_model(context)
    budget               = _compute_energy_budget(drone, perf, mission)
    er                   = _compute_endurance_range(budget, perf, mission)
    payload_info         = _assess_payload(perf, mission, drone)
    env_info             = _assess_environment(mission, er, env)
    constraints          = _evaluate_constraints(er, perf, payload_info)

    metrics              = _build_metrics(mission, drone, budget, er, perf,
                                          constraints, payload_info, env)
    result["metrics"]    = metrics

    diagnostics          = _build_diagnostics(
        constraints, er, perf, payload_info, env_info, mission, drone
    )
    result["diagnostics"] = diagnostics
    result["alerts"]      = diagnostics.pop("_alerts", [])
    result["score"]       = _compute_score(constraints, er, payload_info)
    result["sections"]    = _build_sections(metrics, diagnostics, mission, drone, env)

    return result


# ─── STAGE 1: Input Conditioning ─────────────────────────────────────────────


def _condition_inputs(
    signals: Dict[str, Any],
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[Dict, Dict, Dict]:
    def _f(d: Dict, k: str, fallback: float = float("nan")) -> float:
        v = d.get(k, fallback)
        try:
            return float(v)
        except (TypeError, ValueError):
            return fallback

    mission: Dict[str, Any] = {
        "target_payload_kg":  _f(signals, "target_payload_kg",  0.0),
        "distance_km":        _f(signals, "distance_km",        float("nan")),
        "duration_min":       _f(signals, "duration_min",       0.0),
        "battery_margin_pct": _f(signals, "battery_margin_pct", _BASE_RESERVE_PCT),
        "wind_speed_ms":      _f(signals, "wind_speed_ms",      0.0),
        "terrain_difficulty": _f(signals, "terrain_difficulty", 1.0),
        "altitude_m":         _f(signals, "altitude_m",         0.0),
        "ambient_temp_c":     _f(signals, "ambient_temp_c",     15.0),
    }

    drone: Dict[str, Any] = {
        "dry_weight_kg":        _f(params, "dry_weight_kg"),
        "battery_cells":        _f(params, "battery_cells",        4.0),
        "battery_capacity_mah": _f(params, "battery_capacity_mah"),
        "payload_kg":           _f(params, "payload_kg",           0.0),
        "frame_class":          params.get("frame_class", ""),
        "num_motors":           _f(params, "num_motors",            4.0),
    }
    cells = drone["battery_cells"]
    drone["nominal_voltage_v"] = cells * 3.8 if not np.isnan(cells) else float("nan")

    cap = drone["battery_capacity_mah"]
    v   = drone["nominal_voltage_v"]
    drone["total_energy_wh"] = (
        (cap / 1000.0) * v
        if not (np.isnan(cap) or np.isnan(v)) else float("nan")
    )

    # ISA air density correction
    tk  = mission["ambient_temp_c"] + 273.15
    alt = mission["altitude_m"]
    rho = _RHO_SL * (_T_REF_K / tk) * ((1.0 - _L_RATE * alt / _T_REF_K) ** 4.256)
    env: Dict[str, float] = {
        "air_density_kgm3": rho,
        "temperature_k":    tk,
        "altitude_m":       alt,
    }

    return mission, drone, env


# ─── STAGE 2: Performance Loading ────────────────────────────────────────────


def _load_perf_model(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract performance_model fields from context.
    Returns a normalised dict with float values (NaN when field is null/absent).
    The 'available' flag signals whether the health JSON contained a model.
    """
    raw = context.get("performance_model") or {}

    def _pf(k: str) -> float:
        v = raw.get(k)
        if v is None:
            return float("nan")
        try:
            f = float(v)
            return float("nan") if np.isnan(f) or np.isinf(f) else f
        except (TypeError, ValueError):
            return float("nan")

    return {
        "hover_power_w":     _pf("hover_power_w"),
        "cruise_power_w":    _pf("cruise_power_w"),
        "mean_power_w":      _pf("mean_power_w"),
        "hover_throttle":    _pf("hover_throttle"),
        "voltage_sag_pct":   _pf("voltage_sag_pct"),
        "energy_per_min_wh": _pf("energy_per_min_wh"),
        "energy_per_km_wh":  _pf("energy_per_km_wh"),
        "max_power_w":       _pf("max_power_w"),
        "thrust_margin_pct": _pf("thrust_margin_pct"),
        "efficiency_state":  raw.get("efficiency_state"),
        "available":         bool(raw),
    }


def _eff_power(perf: Dict, key_primary: str, key_fallback: str = "mean_power_w") -> float:
    """Return best available power value for a given use-case."""
    v = perf.get(key_primary, float("nan"))
    if not np.isnan(v):
        return v
    return perf.get(key_fallback, float("nan"))


# ─── STAGE 3: Energy Budget ───────────────────────────────────────────────────


def _compute_energy_budget(
    drone: Dict,
    perf: Dict,
    mission: Dict,
) -> Dict[str, float]:
    """
    usable_energy_wh = total_energy_wh × (1 − reserve_pct) × (1 − voltage_sag_pct)

    voltage_sag_pct is taken from the measured performance model.
    If unavailable, sag correction is skipped (conservative: assumes no sag).
    """
    nan            = float("nan")
    total_wh       = drone.get("total_energy_wh", nan)
    reserve_pct    = mission.get("battery_margin_pct", _BASE_RESERVE_PCT) / 100.0
    sag            = perf.get("voltage_sag_pct", nan)
    sag_factor     = (1.0 - sag) if not np.isnan(sag) else 1.0

    usable_wh = (
        total_wh * (1.0 - reserve_pct) * sag_factor
        if not np.isnan(total_wh) else nan
    )

    return {
        "total_energy_wh":  total_wh,
        "usable_energy_wh": usable_wh,
        "reserve_pct":      reserve_pct,
        "sag_factor":       sag_factor,
    }


# ─── STAGE 4: Endurance & Range ──────────────────────────────────────────────


def _compute_endurance_range(
    budget: Dict,
    perf: Dict,
    mission: Dict,
) -> Dict[str, Any]:
    """
    Derive endurance and range from measured power data.

    cruise_speed is derived from observed GPS metrics:
        speed_ms = (energy_per_min_wh / energy_per_km_wh) × 1000 / 60
    Falls back to _FALLBACK_CRUISE_MS when GPS data is absent.
    """
    nan      = float("nan")
    usable   = budget.get("usable_energy_wh", nan)
    dist_km  = mission.get("distance_km",  nan)
    dur_min  = mission.get("duration_min", 0.0)
    has_dist = not np.isnan(dist_km) and dist_km > 0
    has_dur  = dur_min > 0

    # Best available power values
    hover_p  = _eff_power(perf, "hover_power_w",  "mean_power_w")
    cruise_p = _eff_power(perf, "cruise_power_w", "mean_power_w")
    # If cruise still nan, fall back to hover
    if np.isnan(cruise_p):
        cruise_p = hover_p

    # Operative power for this mission type
    if has_dist and not has_dur:
        operative_p   = cruise_p
        physics_basis = "cruise"
    elif has_dur and not has_dist:
        operative_p   = hover_p
        physics_basis = "hover"
    elif has_dist and has_dur:
        # Most demanding constraint drives the budget
        operative_p   = (
            max(v for v in (hover_p, cruise_p) if not np.isnan(v))
            if not (np.isnan(hover_p) and np.isnan(cruise_p)) else nan
        )
        physics_basis = "mixed"
    else:
        operative_p   = hover_p
        physics_basis = "hover"

    # Hover endurance (reference — always computed)
    hover_endurance_min = nan
    if not (np.isnan(usable) or np.isnan(hover_p)) and hover_p > 0:
        hover_endurance_min = (usable / hover_p) * 60.0

    # Mission endurance from operative power
    endurance_min = nan
    if not (np.isnan(usable) or np.isnan(operative_p)) and operative_p > 0:
        endurance_min = (usable / operative_p) * 60.0

    # Observed cruise speed from GPS-derived energy ratios
    epk = perf.get("energy_per_km_wh", nan)
    epm = perf.get("energy_per_min_wh", nan)
    if not (np.isnan(epk) or np.isnan(epm)) and epk > 0:
        # speed = distance / time = (1 km) / (energy_per_km / energy_per_min minutes)
        cruise_speed_ms = (epm / epk) * 1000.0 / 60.0
        gps_speed_available = True
    else:
        cruise_speed_ms     = _FALLBACK_CRUISE_MS
        gps_speed_available = False

    # Clamp to physically realistic range for multirotors
    cruise_speed_ms = float(np.clip(cruise_speed_ms, 2.0, 25.0))

    # Range from endurance
    range_km = nan
    if not np.isnan(endurance_min) and cruise_speed_ms > 0:
        range_km = cruise_speed_ms * (endurance_min * 60.0) / 1000.0

    # Required energy (for margin calculation)
    required_energy_wh = nan
    if has_dist and not np.isnan(cruise_p) and cruise_speed_ms > 0:
        transit_h   = (dist_km * 1000.0) / (cruise_speed_ms * 3600.0)
        req_transit = cruise_p * transit_h
        if has_dur and not np.isnan(hover_p):
            req_hover          = hover_p * (dur_min / 60.0)
            required_energy_wh = max(req_transit, req_hover)
        else:
            required_energy_wh = req_transit
    elif has_dur and not np.isnan(hover_p):
        required_energy_wh = hover_p * (dur_min / 60.0)

    return {
        "endurance_min":        endurance_min,
        "hover_endurance_min":  hover_endurance_min,
        "range_km":             range_km,
        "cruise_speed_ms":      cruise_speed_ms,
        "operative_power_w":    operative_p,
        "physics_basis":        physics_basis,
        "required_energy_wh":   required_energy_wh,
        "dist_km":              dist_km,
        "dur_min":              dur_min,
        "gps_speed_available":  gps_speed_available,
    }


# ─── STAGE 4b: Payload Assessment ────────────────────────────────────────────


def _assess_payload(
    perf: Dict,
    mission: Dict,
    drone: Dict,
) -> Dict[str, Any]:
    """
    Evaluate payload feasibility using observed thrust_margin_pct.

    The health JSON records thrust_margin_pct AT the time of the health flight
    (which included drone["payload_kg"]).  For a new mission payload, we adjust
    for the weight delta without using propeller physics:

        delta_kg   = target_payload - recorded_payload
        weight_frac = delta_kg / (dry_weight + recorded_payload)   [if dry_weight known]
        adjusted_margin = recorded_margin - weight_frac

    This is a linear approximation valid for small deltas.
    """
    nan         = float("nan")
    rec_margin  = perf.get("thrust_margin_pct", nan)
    rec_payload = drone.get("payload_kg",        0.0)
    dry_kg      = drone.get("dry_weight_kg",     nan)
    target_kg   = mission.get("target_payload_kg", 0.0)

    # Adjust thrust margin for new payload
    adjusted_margin = rec_margin
    if not np.isnan(rec_margin) and not np.isnan(dry_kg) and dry_kg > 0:
        recorded_mass = dry_kg + rec_payload
        delta_kg      = target_kg - rec_payload
        if abs(delta_kg) > 0.001:
            weight_frac     = delta_kg / recorded_mass
            adjusted_margin = rec_margin - weight_frac

    # Feasibility classification
    if np.isnan(adjusted_margin):
        feasibility = "N/A"
    elif adjusted_margin < 0.0:
        feasibility = "OVERLOADED"
    elif adjusted_margin < _C3_THRUST_MIN:
        feasibility = "OVERLOADED"     # < 25 % is a hard constraint
    elif adjusted_margin < 0.35:
        feasibility = "MARGINAL"
    else:
        feasibility = "SAFE"

    # Estimated max additional payload (linear: margin zeroed out)
    max_payload_kg = nan
    if not np.isnan(rec_margin) and not np.isnan(dry_kg) and dry_kg > 0:
        recorded_mass   = dry_kg + rec_payload
        max_payload_kg  = rec_margin * recorded_mass  # kg that can be added before margin = 0

    return {
        "feasibility":       feasibility,
        "adjusted_margin":   adjusted_margin,
        "recorded_margin":   rec_margin,
        "target_payload_kg": target_kg,
        "max_payload_kg":    max_payload_kg,
    }


# ─── STAGE 4c: Environmental Assessment ──────────────────────────────────────


def _assess_environment(
    mission: Dict,
    er: Dict,
    env: Dict,
) -> Dict[str, Any]:
    risks = []
    wind  = mission.get("wind_speed_ms", 0.0)
    v_c   = er.get("cruise_speed_ms", _FALLBACK_CRUISE_MS)
    rho   = env.get("air_density_kgm3", _RHO_SL)

    if not np.isnan(wind) and not np.isnan(v_c) and wind > v_c * 0.5:
        risks.append("WIND")
    if rho < 1.1:
        risks.append("DENSITY_ALTITUDE")

    return {"risks": risks, "wind_ms": wind, "rho": rho}


# ─── STAGE 5: Constraint Evaluation ──────────────────────────────────────────


def _evaluate_constraints(
    er: Dict,
    perf: Dict,
    payload_info: Dict,
) -> Dict[str, Any]:
    """
    Evaluate the 5 hard constraints and compute risk level.

    Returns:
        hard_violations : list of {"code", "msg"} — mission CANNOT fly
        warnings        : list of {"code", "msg"} — proceed with caution
        risk_level      : "low" | "medium" | "high"
        margins_pct     : dict of named margins used for risk classification
    """
    nan         = float("nan")
    violations: List[Dict[str, str]] = []
    warnings:   List[Dict[str, str]] = []
    margins:    Dict[str, float]     = {}

    endurance   = er["endurance_min"]
    range_km    = er["range_km"]
    dur_min     = er["dur_min"]
    dist_km     = er["dist_km"]
    thrust_m    = perf.get("thrust_margin_pct", nan)
    sag         = perf.get("voltage_sag_pct",   nan)
    hover_thr   = perf.get("hover_throttle",    nan)

    # C1 — Time feasibility
    if not np.isnan(endurance) and dur_min > 0:
        time_margin_pct = (endurance - dur_min) / endurance * 100.0
        margins["time"] = time_margin_pct
        if dur_min > endurance * _CONSTRAINT_SAFETY:
            violations.append({
                "code": "C1_TIME",
                "msg": "Required flight duration exceeds the safe endurance limit — energy reserve will be depleted",
            })
        elif dur_min > endurance * 0.70:
            warnings.append({
                "code": "C1_TIME_WARN",
                "msg": "Required flight duration consumes most of the available endurance — energy reserve is tight",
            })

    # C2 — Range feasibility
    if not (np.isnan(range_km) or np.isnan(dist_km)) and dist_km > 0:
        range_margin_pct = (range_km - dist_km) / range_km * 100.0
        margins["range"] = range_margin_pct
        if dist_km > range_km * _CONSTRAINT_SAFETY:
            violations.append({
                "code": "C2_RANGE",
                "msg": "Required mission distance exceeds the safe range limit — mission cannot complete with reserve intact",
            })
        elif dist_km > range_km * 0.70:
            warnings.append({
                "code": "C2_RANGE_WARN",
                "msg": "Required mission distance consumes most of the available range — no buffer for course deviations",
            })

    # C3 — Thrust margin (from adjusted payload assessment)
    adj_m = payload_info.get("adjusted_margin", nan)
    if not np.isnan(adj_m):
        thrust_margin_pct = adj_m * 100.0
        margins["thrust"] = thrust_margin_pct
        if adj_m < _C3_THRUST_MIN:
            violations.append({
                "code": "C3_THRUST",
                "msg": "Thrust margin is below the safe minimum — insufficient headroom for gusts or control authority",
            })
        elif adj_m < 0.35:
            warnings.append({
                "code": "C3_THRUST_WARN",
                "msg": "Thrust margin is below 35% — limited reserve for payload variations or wind gusts",
            })

    # C4 — Voltage sag
    # margin = (1 − sag / C4_SAG_MAX) × 100  →  how far below the dangerous threshold
    if not np.isnan(sag):
        sag_headroom_pct = max(0.0, (1.0 - sag / _C4_SAG_MAX) * 100.0)
        margins["voltage_sag"] = sag_headroom_pct
        if sag > _C4_SAG_MAX:
            violations.append({
                "code": "C4_SAG",
                "msg": "Voltage sag exceeds the safe threshold — battery is degraded or undersized for this load",
            })
        elif sag > 0.15:
            warnings.append({
                "code": "C4_SAG_WARN",
                "msg": "Voltage sag is elevated — battery internal resistance may be increasing; inspect before flight",
            })

    # C5 — Hover throttle
    # margin = (1 − hover_thr) × 100  →  actual thrust reserve remaining
    if not np.isnan(hover_thr):
        margins["hover_throttle"] = max(0.0, (1.0 - hover_thr) * 100.0)
        if hover_thr > _C5_THROTTLE_MAX:
            violations.append({
                "code": "C5_THROTTLE",
                "msg": "Hover throttle exceeds the safe limit — aircraft is aerodynamically overloaded",
            })
        elif hover_thr > 0.55:
            warnings.append({
                "code": "C5_THROTTLE_WARN",
                "msg": "Hover throttle is above 55% — efficiency is reduced and thrust reserve is limited",
            })

    # Risk classification
    if violations:
        risk_level = "high"
    elif margins:
        min_m = min(margins.values())
        if min_m < _MARGIN_HIGH_RISK:
            risk_level = "high"
        elif min_m < _MARGIN_MED_RISK:
            risk_level = "medium"
        else:
            risk_level = "low"
    else:
        risk_level = "medium" if warnings else "low"

    return {
        "hard_violations": violations,
        "warnings":        warnings,
        "risk_level":      risk_level,
        "margins_pct":     margins,
    }


# ─── STAGE 6: Diagnostics ────────────────────────────────────────────────────


def _build_diagnostics(
    constraints: Dict,
    er: Dict,
    perf: Dict,
    payload_info: Dict,
    env_info: Dict,
    mission: Dict,
    drone: Dict,
) -> Dict[str, Any]:
    """
    Build the unified diagnostics block.
    Includes both the new strict output format (summary/issues/actions/tuning)
    and the UI-compatible legacy keys (energy_feasibility, mission_decision…).
    """
    nan         = float("nan")
    violations  = constraints["hard_violations"]
    warnings    = constraints["warnings"]
    risk_level  = constraints["risk_level"]

    issues:  List[str] = []
    actions: List[str] = []
    tuning:  List[str] = []
    alerts:  List[Dict[str, str]] = []

    # ── Issues from hard constraint violations ─────────────────────────────────
    for v in violations:
        issues.append(v["msg"])
        alerts.append({"level": "critical", "msg": v["msg"]})

    # ── Issues from warnings ───────────────────────────────────────────────────
    for w in warnings:
        issues.append(w["msg"])
        alerts.append({"level": "warning", "msg": w["msg"]})

    # ── Environmental issues ───────────────────────────────────────────────────
    env_risks = env_info.get("risks", [])
    wind_ms   = env_info.get("wind_ms", 0.0)
    for r in env_risks:
        if r == "WIND":
            msg = "Wind speed exceeds 50% of cruise speed — actual energy consumption will exceed the modelled budget"
            issues.append(msg)
            alerts.append({"level": "warning", "msg": msg})
        elif r == "DENSITY_ALTITUDE":
            msg = "Reduced air density at operating altitude — measured hover power underestimates actual demand"
            issues.append(msg)
            alerts.append({"level": "info", "msg": msg})

    # ── Performance model availability ────────────────────────────────────────
    if not perf.get("available"):
        msg = (
            "Performance model missing from health JSON — "
            "regenerate health JSON to enable physics-based constraint evaluation"
        )
        issues.append(msg)
        alerts.append({"level": "model_error", "msg": msg})
    elif not er.get("gps_speed_available"):
        alerts.append({
            "level": "model_error",
            "msg": "GPS speed data absent — range estimate uses a conservative cruise speed fallback",
        })

    # ── Actions — physical corrective actions ─────────────────────────────────
    codes = {v["code"] for v in violations} | {w["code"] for w in warnings}

    if "C1_TIME" in codes or "C1_TIME_WARN" in codes:
        actions.append("Reduce required duration or increase battery capacity")
    if "C2_RANGE" in codes or "C2_RANGE_WARN" in codes:
        actions.append("Reduce mission distance or increase battery capacity")
    if "C3_THRUST" in codes:
        actions.append("Reduce total takeoff weight or upgrade to higher-thrust motors/props")
    if "C4_SAG" in codes:
        actions.append("Replace battery — high internal resistance is reducing usable capacity")
    if "C5_THROTTLE" in codes:
        actions.append("Reduce payload or dry weight to bring hover throttle within safe limits")
    if "C3_THRUST_WARN" in codes or "C5_THROTTLE_WARN" in codes:
        actions.append("Verify takeoff weight is within the aircraft's design envelope")
    if "C4_SAG_WARN" in codes:
        actions.append("Test battery with a capacity checker — elevated sag indicates rising internal resistance")

    # ── Tuning recommendations (control/propulsion only when relevant) ─────────
    hover_thr = perf.get("hover_throttle", nan)
    if not np.isnan(hover_thr) and hover_thr > 0.55:
        tuning.append("Calibrate MOT_THST_HOVER to the measured hover throttle — prevents integrator windup in loiter")
    if perf.get("efficiency_state") == "low":
        tuning.append("High energy consumption detected — consider reducing D-gain to lower motor heating")

    # ── Summary ───────────────────────────────────────────────────────────────
    if violations:
        n       = len(violations)
        summary = (
            f"{n} hard constraint{'s' if n > 1 else ''} violated — mission cannot proceed safely"
        )
    elif warnings:
        summary = f"Mission is achievable but {len(warnings)} margin{'s are' if len(warnings) > 1 else ' is'} tight — review before committing to flight"
    else:
        endurance = er.get("endurance_min", nan)
        end_str   = f"{endurance:.1f} min endurance" if not np.isnan(endurance) else "—"
        summary   = f"All constraints satisfied ({end_str} available) — proceed as planned"

    # ── Mission decision (UI legacy key) ─────────────────────────────────────
    _risk_to_decision = {
        "high":   "NOT READY TO FLY",
        "medium": "READY WITH RISK",
        "low":    "READY TO FLY",
    }
    mission_decision = _risk_to_decision[risk_level]

    # ── Energy feasibility legacy key ─────────────────────────────────────────
    usable       = nan
    required_e   = er.get("required_energy_wh", nan)
    energy_feas  = "N/A"
    budget_usable = nan

    if not np.isnan(required_e):
        # Approximate from constraints
        if any(v["code"] == "C1_TIME" or v["code"] == "C2_RANGE" for v in violations):
            energy_feas = "INFEASIBLE"
        else:
            endurance = er.get("endurance_min", nan)
            dur_min   = er.get("dur_min", 0.0)
            range_km  = er.get("range_km", nan)
            dist_km   = er.get("dist_km", nan)

            time_margin  = (
                (endurance - dur_min) / endurance * 100.0
                if not np.isnan(endurance) and dur_min > 0 else nan
            )
            range_margin = (
                (range_km - dist_km) / range_km * 100.0
                if not (np.isnan(range_km) or np.isnan(dist_km)) and dist_km > 0 else nan
            )
            min_e_margin = min(
                [m for m in (time_margin, range_margin) if not np.isnan(m)],
                default=nan,
            )

            if np.isnan(min_e_margin):
                energy_feas = "N/A"
            elif min_e_margin >= 20.0:
                energy_feas = "FEASIBLE"
            elif min_e_margin >= 5.0:
                energy_feas = "MARGINAL"
            elif min_e_margin >= 0.0:
                energy_feas = "BORDERLINE"
            else:
                energy_feas = "INFEASIBLE"

    # ── Payload feasibility legacy key ────────────────────────────────────────
    payload_feas = payload_info.get("feasibility", "N/A")

    # ── Environmental risk legacy key ─────────────────────────────────────────
    env_risk_str = "|".join(env_risks) if env_risks else "NOMINAL"

    return {
        # New spec format
        "summary":     summary,
        "risk_level":  risk_level,
        "issues":      issues,
        "actions":     actions,
        "tuning":      tuning,
        # UI legacy keys (required by 3_Mission_Feasibility.py)
        "energy_feasibility":  energy_feas,
        "payload_feasibility": payload_feas,
        "environmental_risk":  env_risk_str,
        "mission_decision":    mission_decision,
        "_alerts":             alerts,
    }


# ─── STAGE 7: Scoring ────────────────────────────────────────────────────────


def _compute_score(
    constraints: Dict,
    er: Dict,
    payload_info: Dict,
) -> float:
    score = 100.0

    # Hard violations — heavy penalty per constraint breached
    score -= len(constraints["hard_violations"]) * 20.0

    # Warnings — moderate penalty
    score -= len(constraints["warnings"]) * 8.0

    # Margin bonus/penalty
    margins = constraints.get("margins_pct", {})
    if margins:
        min_m = min(margins.values())
        if min_m < 10:
            score -= 20.0
        elif min_m < 20:
            score -= 10.0
        elif min_m < 30:
            score -= 5.0

    # Payload overload
    if payload_info.get("feasibility") == "OVERLOADED":
        score -= 30.0
    elif payload_info.get("feasibility") == "MARGINAL":
        score -= 10.0

    return max(0.0, min(100.0, round(score, 2)))


# ─── Metrics builder ─────────────────────────────────────────────────────────


def _build_metrics(
    mission: Dict,
    drone: Dict,
    budget: Dict,
    er: Dict,
    perf: Dict,
    constraints: Dict,
    payload_info: Dict,
    env: Dict,
) -> Dict[str, Any]:
    nan = float("nan")

    def _pf(k: str) -> float:
        v = perf.get(k, nan)
        return nan if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    total_kg = nan
    dry_kg   = drone.get("dry_weight_kg", nan)
    if not np.isnan(dry_kg):
        total_kg = dry_kg + mission.get("target_payload_kg", 0.0)

    usable_wh = budget.get("usable_energy_wh", nan)
    total_wh  = budget.get("total_energy_wh",  nan)
    req_e     = er.get("required_energy_wh",   nan)

    energy_margin_wh  = nan
    energy_margin_pct = nan
    if not (np.isnan(req_e) or np.isnan(usable_wh)):
        energy_margin_wh  = round(usable_wh - req_e, 3)
        energy_margin_pct = round((usable_wh - req_e) / (usable_wh + 1e-9) * 100.0, 2)

    return {
        "physics_basis":               er.get("physics_basis", "unknown"),
        "total_mass_kg":               round(total_kg, 4) if not np.isnan(total_kg) else nan,
        "payload_fraction":            (
            round(mission.get("target_payload_kg", 0.0) / total_kg, 4)
            if not np.isnan(total_kg) and total_kg > 0 else nan
        ),
        "hover_power_w":               _pf("hover_power_w"),
        "cruise_power_w":              _pf("cruise_power_w"),
        "forward_flight_power_w":      _pf("cruise_power_w"),   # alias for UI compat
        "mean_power_w":                _pf("mean_power_w"),
        "cruise_speed_ms":             er.get("cruise_speed_ms", nan),
        "required_energy_wh":          req_e,
        "usable_energy_wh":            usable_wh,
        "total_energy_wh":             total_wh,
        "energy_margin_wh":            energy_margin_wh,
        "energy_margin_pct":           energy_margin_pct,
        "hover_endurance_min":         er.get("hover_endurance_min", nan),
        "endurance_min":               er.get("endurance_min",       nan),
        "range_limited_km":            er.get("range_km",            nan),
        "hover_throttle":              _pf("hover_throttle"),
        "thrust_margin_pct":           payload_info.get("adjusted_margin", nan),
        "voltage_sag_pct":             _pf("voltage_sag_pct"),
        "max_payload_kg":              payload_info.get("max_payload_kg", nan),
        "air_density_kgm3":            env.get("air_density_kgm3", nan),
        "risk_level":                  constraints["risk_level"],
        "efficiency_state":            perf.get("efficiency_state"),
        # Legacy alias
        "flight_time_h":               (
            er.get("endurance_min", nan) / 60.0
            if not np.isnan(er.get("endurance_min", nan)) else nan
        ),
    }


# ─── Section builder ─────────────────────────────────────────────────────────


def _build_sections(
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
    mission: Dict,
    drone: Dict,
    env: Dict,
) -> Dict[str, Any]:
    def fmt(v: Any, d: int = 2, unit: str = "") -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{round(float(v), d)}{' ' + unit if unit else ''}"

    nan = float("nan")

    return {
        "propulsion": {
            "title":                  "Power Budget",
            "physics_basis":          metrics.get("physics_basis", "unknown"),
            "hover_power_w":          fmt(metrics.get("hover_power_w"),         1, "W"),
            "cruise_power_w":         fmt(metrics.get("cruise_power_w"),        1, "W"),
            "forward_flight_power_w": fmt(metrics.get("cruise_power_w"),        1, "W"),
            "cruise_speed_ms":        fmt(metrics.get("cruise_speed_ms"),       2, "m/s"),
            "total_mass_kg":          fmt(metrics.get("total_mass_kg"),         3, "kg"),
            "air_density_kgm3":       fmt(metrics.get("air_density_kgm3"),      4, "kg/m³"),
            "hover_throttle":         fmt(metrics.get("hover_throttle"),        2),
            "thrust_margin_pct":      fmt(
                (metrics.get("thrust_margin_pct", nan) or nan) * 100.0
                if not np.isnan(metrics.get("thrust_margin_pct", nan) or nan) else nan,
                1, "%",
            ),
            "voltage_sag_pct": fmt(
                (metrics.get("voltage_sag_pct", nan) or nan) * 100.0
                if not np.isnan(metrics.get("voltage_sag_pct", nan) or nan) else nan,
                1, "%",
            ),
        },
        "energy_budget": {
            "title":                "Energy Budget",
            "physics_basis":        metrics.get("physics_basis", "unknown"),
            "required_wh":          fmt(metrics.get("required_energy_wh"),  2, "Wh"),
            "usable_wh":            fmt(metrics.get("usable_energy_wh"),    2, "Wh"),
            "total_wh":             fmt(metrics.get("total_energy_wh"),     2, "Wh"),
            "margin_wh":            fmt(metrics.get("energy_margin_wh"),    2, "Wh"),
            "margin_pct":           fmt(metrics.get("energy_margin_pct"),   1, "%"),
            "hover_endurance_min":  fmt(metrics.get("hover_endurance_min"), 1, "min"),
            "endurance_min":        fmt(metrics.get("endurance_min"),       1, "min"),
            "range_km":             fmt(metrics.get("range_limited_km"),    2, "km"),
            "feasibility":          diagnostics.get("energy_feasibility",  "N/A"),
            "mission_decision":     diagnostics.get("mission_decision",    "N/A"),
        },
        "payload_assessment": {
            "title":          "Payload Assessment",
            "target_kg":      fmt(mission.get("target_payload_kg", nan), 3, "kg"),
            "max_payload_kg": fmt(metrics.get("max_payload_kg",     nan), 3, "kg"),
            "status":         diagnostics.get("payload_feasibility", "N/A"),
        },
        "environment": {
            "title":            "Environmental Conditions",
            "wind_ms":          fmt(mission.get("wind_speed_ms",  0.0), 1, "m/s"),
            "ambient_temp_c":   fmt(mission.get("ambient_temp_c", 15.0), 1, "°C"),
            "altitude_m":       fmt(mission.get("altitude_m",     0.0),  0, "m AGL"),
            "air_density_kgm3": fmt(env.get("air_density_kgm3",  _RHO_SL), 4, "kg/m³"),
            "risk":             diagnostics.get("environmental_risk", "NOMINAL"),
        },
        "decision_report": {
            "title":      "Decision Report",
            "summary":    diagnostics.get("summary",    ""),
            "risk_level": diagnostics.get("risk_level", "high"),
            "issues":     diagnostics.get("issues",     []),
            "actions":    diagnostics.get("actions",    []),
            "tuning":     diagnostics.get("tuning",     []),
        },
    }
