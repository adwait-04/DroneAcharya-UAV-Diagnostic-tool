"""
ArduPilot Parameter File Handler (.param)
Parses key=value / key,value parameter files and maps them to
questionnaire fields, PID gains, filter config, and motor config.
"""

from typing import Any, Dict, List, Optional, Tuple
import math

# ── Frame class → motor count map ────────────────────────────────────────────
FRAME_CLASS_MAP: Dict[int, Dict[str, Any]] = {
    0:  {"name": "Undefined",      "motors": 4},
    1:  {"name": "Quadcopter",     "motors": 4},
    2:  {"name": "Hexacopter",     "motors": 6},
    3:  {"name": "Octocopter",     "motors": 8},
    4:  {"name": "OctoQuad",       "motors": 8},
    5:  {"name": "Y6",             "motors": 6},
    6:  {"name": "Hexacopter(+)",  "motors": 6},
    7:  {"name": "TriCopter",      "motors": 3},
    10: {"name": "Quadcopter",     "motors": 4},
    12: {"name": "DodecaHex",      "motors": 12},
    14: {"name": "Hexacopter",     "motors": 6},
}

# ── Params that map directly to questionnaire fields ─────────────────────────
_DIRECT_MAP: Dict[str, str] = {
    "BATT_CAPACITY": "battery_capacity_mah",
    "SCHED_LOOP_RATE": "loop_rate_hz",
}


def parse_param_file(file_obj: Any) -> Tuple[Dict[str, float], List[str]]:
    """
    Parse ArduPilot .param file (comma-separated or whitespace-separated).
    Returns (params_dict, warnings).
    """
    params: Dict[str, float] = {}
    warnings: List[str] = []

    try:
        raw = file_obj.read() if hasattr(file_obj, "read") else str(file_obj)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        warnings.append(f"Param file read error: {exc}")
        return params, warnings

    for line_no, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # Split on first comma, tab, or whitespace
        parts: Optional[List[str]] = None
        for sep in (",", "\t", None):
            split = line.split(sep, 1) if sep else line.split(None, 1)
            if len(split) == 2:
                parts = split
                break

        if not parts:
            continue

        name = parts[0].strip().upper()
        val_str = parts[1].strip()

        try:
            params[name] = float(val_str)
        except ValueError:
            warnings.append(f"L{line_no}: cannot parse {name}={val_str!r}")

    return params, warnings


def extract_questionnaire_fields(params: Dict[str, float]) -> Dict[str, Any]:
    """
    Map raw ArduPilot params → questionnaire field names.
    Only returns fields confidently derivable from params.
    """
    fields: Dict[str, Any] = {}

    # Direct mappings
    for param_key, q_key in _DIRECT_MAP.items():
        if param_key in params:
            fields[q_key] = int(params[param_key]) if q_key in ("battery_capacity_mah", "loop_rate_hz") else params[param_key]

    # Frame class → name + motor count
    if "FRAME_CLASS" in params:
        fc = int(params["FRAME_CLASS"])
        info = FRAME_CLASS_MAP.get(fc, FRAME_CLASS_MAP[1])
        fields["frame_class"] = info["name"]
        fields["num_motors"] = info["motors"]

    # ESC protocol from PWM range
    pwm_min = params.get("MOT_PWM_MIN", 1000)
    pwm_max = params.get("MOT_PWM_MAX", 2000)
    if pwm_max < 500:
        fields["esc_protocol"] = "DSHOT600"
    elif pwm_max < 900:
        fields["esc_protocol"] = "DSHOT300"
    elif 900 <= pwm_min <= 1100 and 1800 <= pwm_max <= 2200:
        fields["esc_protocol"] = "PWM"

    # Infer battery cells from low-voltage threshold (heuristic)
    low_v = params.get("BATT_LOW_VOLT", 0.0)
    if low_v > 3.0:
        cells_est = round(low_v / 3.4)
        if 2 <= cells_est <= 14:
            fields["battery_cells"] = cells_est

    return fields


def extract_pid_gains(params: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Return structured PID gains for Roll / Pitch / Yaw axes."""
    pids: Dict[str, Dict[str, float]] = {}

    axes = [
        ("roll",  "ATC_RAT_RLL"),
        ("pitch", "ATC_RAT_PIT"),
        ("yaw",   "ATC_RAT_YAW"),
    ]
    for axis, prefix in axes:
        pids[axis] = {
            "P":    params.get(f"{prefix}_P",    float("nan")),
            "I":    params.get(f"{prefix}_I",    float("nan")),
            "D":    params.get(f"{prefix}_D",    float("nan")),
            "FF":   params.get(f"{prefix}_FF",   float("nan")),
            "IMAX": params.get(f"{prefix}_IMAX", float("nan")),
            "FLTD": params.get(f"{prefix}_FLTD", float("nan")),
            "FLTT": params.get(f"{prefix}_FLTT", float("nan")),
        }

    pids["angle"] = {
        "roll_P":  params.get("ATC_ANG_RLL_P", float("nan")),
        "pitch_P": params.get("ATC_ANG_PIT_P", float("nan")),
        "yaw_P":   params.get("ATC_ANG_YAW_P", float("nan")),
    }

    return pids


def extract_filter_config(params: Dict[str, float]) -> Dict[str, Any]:
    """Return filter settings (gyro LP, notch, harmonic notch)."""
    return {
        "gyro_filter_hz":       params.get("INS_GYRO_FILTER",   float("nan")),
        "accel_filter_hz":      params.get("INS_ACCEL_FILTER",  float("nan")),
        # Static notch
        "notch_enabled":        bool(params.get("INS_NOTCH_ENABLE",  0)),
        "notch_freq_hz":        params.get("INS_NOTCH_FREQ",  float("nan")),
        "notch_bw_hz":          params.get("INS_NOTCH_BW",   float("nan")),
        "notch_atten_db":       params.get("INS_NOTCH_ATT",  float("nan")),
        # Harmonic notch
        "hnotch_enabled":       bool(params.get("INS_HNTCH_ENABLE", 0)),
        "hnotch_freq_hz":       params.get("INS_HNTCH_FREQ", float("nan")),
        "hnotch_bw_hz":         params.get("INS_HNTCH_BW",  float("nan")),
        "hnotch_atten_db":      params.get("INS_HNTCH_ATT", float("nan")),
        "hnotch_mode":          int(params.get("INS_HNTCH_MODE", 0)),
    }


def extract_motor_config(params: Dict[str, float]) -> Dict[str, Any]:
    """Return motor / ESC configuration."""
    return {
        "pwm_min":      params.get("MOT_PWM_MIN",    1000),
        "pwm_max":      params.get("MOT_PWM_MAX",    2000),
        "spin_min":     params.get("MOT_SPIN_MIN",   0.15),
        "spin_max":     params.get("MOT_SPIN_MAX",   0.95),
        "spin_arm":     params.get("MOT_SPIN_ARM",   0.10),
        "thrust_expo":  params.get("MOT_THST_EXPO",  0.65),
        "bat_volt_max": params.get("MOT_BAT_VOLT_MAX", 0.0),
        "bat_volt_min": params.get("MOT_BAT_VOLT_MIN", 0.0),
    }


def compute_theoretical_bandwidth(
    pid: Dict[str, float], filter_cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    Estimate closed-loop bandwidth and phase margin for a PD+I rate controller.

    Bandwidth: empirical rule bw_hz ≈ P × 100 (calibrated for ArduPilot rate loop
    with typical multirotor plant; P = 0.10–0.40 → BW = 10–40 Hz).

    Phase margin accounts for:
      - D-term lead (reduced by FLTD if configured)
      - Gyro LP filter lag
      - Estimated 10 ms total system delay (ESC + IMU + loop latency)

    Returns bandwidth_hz and phase_margin_deg (NaN if insufficient data).
    """
    P      = pid.get("P",    float("nan"))
    D      = pid.get("D",    float("nan"))
    fltd   = pid.get("FLTD", float("nan"))   # D-term filter Hz
    filt_d = filter_cfg.get("gyro_filter_hz", float("nan"))  # gyro LP Hz

    result = {"bandwidth_hz": float("nan"), "phase_margin_deg": float("nan")}

    if math.isnan(P) or P <= 0 or math.isnan(filt_d) or filt_d <= 0:
        return result

    if math.isnan(D):
        D = 0.0  # P-only controller (no D contribution)

    # Empirical bandwidth: P × 100 Hz (validated range 0.05–0.50 → 5–50 Hz)
    bw_hz   = P * 100.0
    omega_c = bw_hz * 2.0 * math.pi

    # D-term lead phase (attenuated by FLTD when configured)
    if not math.isnan(fltd) and fltd > 0:
        omega_d = fltd * 2.0 * math.pi
        # Effective D magnitude at omega_c through first-order FLTD filter
        d_lead = D * omega_c / math.sqrt(1.0 + (omega_c / omega_d) ** 2)
    else:
        d_lead = D * omega_c  # unfiltered D

    phi_d     = math.degrees(math.atan2(d_lead, P))
    phi_lp    = -math.degrees(math.atan(omega_c / (2.0 * math.pi * filt_d)))
    phi_delay = -math.degrees(omega_c * 0.010)  # 10 ms total system delay

    phase_margin = 90.0 + phi_d + phi_lp + phi_delay

    result["bandwidth_hz"]     = round(bw_hz, 3)
    result["phase_margin_deg"] = round(max(0.0, min(180.0, phase_margin)), 2)
    return result


def summarize_params(params: Dict[str, float]) -> Dict[str, Any]:
    """Return a consolidated summary dict for display and downstream use."""
    return {
        "raw":       params,
        "pid":       extract_pid_gains(params),
        "filters":   extract_filter_config(params),
        "motors":    extract_motor_config(params),
        "questionnaire_prefill": extract_questionnaire_fields(params),
    }
