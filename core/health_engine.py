"""
Health Engine — Physics-Based UAV Structural & Vibration Analysis

Pipeline (FIXED ORDER — NO EARLY RETURNS):
  1. Signal Conditioning
  2. Physics Reconstruction  (FFT · spectrogram · rolling RMS · motor analysis)
  3. Metric Computation
  4. Diagnostics
  5. Scoring
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid as _trapezoid


# ── Public entrypoint ─────────────────────────────────────────────────────────

def analyze(
    signals: Dict[str, Any],
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "score": 0.0, "metrics": {}, "sections": {}, "diagnostics": {},
        "alerts": [], "performance_model": {},
    }
    conditioned            = _condition_signals(signals)
    physics                = _reconstruct_physics(conditioned, params)
    metrics                = _compute_metrics(physics, params, context)
    result["metrics"]      = metrics
    diagnostics            = _run_diagnostics(metrics, params, context)
    result["diagnostics"]  = diagnostics
    result["alerts"]       = diagnostics.get("_alerts", [])
    result["score"]        = _compute_score(metrics, diagnostics)
    result["sections"]     = _build_sections(metrics, diagnostics, physics, conditioned)
    result["performance_model"] = _compute_performance_model(physics)
    return result


# ─── STAGE 1: Signal Conditioning ────────────────────────────────────────────

def _condition_signals(signals: Dict[str, Any]) -> Dict[str, np.ndarray]:
    ts_raw = np.array(signals.get("timestamp_s", []), dtype=np.float64)
    keys = [
        "imu_accel_x", "imu_accel_y", "imu_accel_z",
        "imu_gyro_x",  "imu_gyro_y",  "imu_gyro_z",
        "voltage_v", "current_a", "ground_speed_ms",
        "roll_deg", "pitch_deg", "yaw_deg",
        "rcout_1", "rcout_2", "rcout_3", "rcout_4",
    ]
    if len(ts_raw) < 4:
        return {"timestamp_s": ts_raw, **{k: np.array([]) for k in keys}}

    dt_arr = np.diff(ts_raw); dt_arr = dt_arr[dt_arr > 0]
    dt = float(np.median(dt_arr)) if len(dt_arr) else 0.01
    t_u = np.arange(ts_raw[0], ts_raw[-1], dt)
    out: Dict[str, np.ndarray] = {"timestamp_s": t_u, "_dt": np.array([dt])}

    for key in keys:
        raw = np.array(signals.get(key, []), dtype=np.float64)
        if len(raw) == 0:
            out[key] = np.zeros(len(t_u)); continue
        n = min(len(raw), len(ts_raw))
        raw, ts_src = raw[:n], ts_raw[:n]
        mask = np.isnan(raw)
        if mask.any() and (~mask).sum() >= 2:
            raw[mask] = np.interp(ts_src[mask], ts_src[~mask], raw[~mask])
        out[key] = np.interp(t_u, ts_src, raw)
    return out


# ─── STAGE 2: Physics Reconstruction ─────────────────────────────────────────

def _reconstruct_physics(
    cond: Dict[str, np.ndarray], params: Dict[str, Any]
) -> Dict[str, Any]:
    phy: Dict[str, Any] = {}
    ts = cond["timestamp_s"]
    if len(ts) < 4:
        return phy

    dt = float(cond["_dt"][0])
    fs = 1.0 / dt if dt > 0 else 1.0
    phy["sample_rate_hz"] = fs
    phy["duration_s"] = float(ts[-1] - ts[0])
    phy["timestamp_s"] = ts

    # ── Raw signals for time-series display ──────────────────────────────────
    for k in ["imu_accel_x", "imu_accel_y", "imu_accel_z",
              "imu_gyro_x",  "imu_gyro_y",  "imu_gyro_z",
              "voltage_v", "current_a",
              "roll_deg", "pitch_deg", "yaw_deg",
              "ground_speed_ms", "rcout_1", "rcout_2", "rcout_3", "rcout_4"]:
        phy[k] = cond.get(k, np.zeros(len(ts)))

    # ── Power & energy ───────────────────────────────────────────────────────
    volt = cond.get("voltage_v", np.zeros(len(ts)))
    curr = cond.get("current_a", np.zeros(len(ts)))
    phy["power_w"]       = volt * curr
    phy["energy_used_wh"] = float(_trapezoid(phy["power_w"], ts)) / 3600.0
    phy["mean_voltage_v"] = float(np.mean(volt[volt > 1])) if np.any(volt > 1) else float("nan")
    phy["mean_current_a"] = float(np.mean(curr[curr > 0])) if np.any(curr > 0) else float("nan")
    phy["peak_current_a"] = float(np.max(curr)) if len(curr) else float("nan")

    # ── Distance ─────────────────────────────────────────────────────────────
    spd = np.clip(cond.get("ground_speed_ms", np.zeros(len(ts))), 0, None)
    phy["distance_km"] = float(_trapezoid(spd, ts)) / 1000.0

    # ── Vibration isolation (high-pass > 5 Hz) ───────────────────────────────
    vib = {}
    for ax in ["x", "y", "z"]:
        raw = cond.get(f"imu_accel_{ax}", np.zeros(len(ts)))
        vib[ax] = _highpass(raw, 5.0, fs)
    vib_mag = np.sqrt(vib["x"]**2 + vib["y"]**2 + vib["z"]**2)
    phy["vibration_x"] = vib["x"]
    phy["vibration_y"] = vib["y"]
    phy["vibration_z"] = vib["z"]
    phy["vibration_magnitude"] = vib_mag

    # ── Rolling RMS (window = 1 s) ────────────────────────────────────────────
    win = max(1, int(fs * 1.0))
    rms_roll = np.array([
        float(np.sqrt(np.mean(vib_mag[max(0, i-win):i+1]**2)))
        for i in range(len(vib_mag))
    ])
    phy["vibration_rolling_rms"] = rms_roll

    # ── FFT of Z-axis vibration ───────────────────────────────────────────────
    n = len(vib["z"])
    if n >= 64:
        w = np.hanning(n)
        F = np.abs(fft(vib["z"] * w))
        freqs = fftfreq(n, d=dt)
        pos = freqs > 0
        fq, Fm = freqs[pos], F[pos]
        phy["fft_frequencies"] = fq
        phy["fft_magnitudes"]  = Fm

        valid = fq >= 1.0
        if valid.any():
            dom_idx = int(np.argmax(Fm[valid]))
            phy["dominant_frequency_hz"] = float(fq[valid][dom_idx])
            phy["dominant_magnitude"]    = float(Fm[valid][dom_idx])
        else:
            phy["dominant_frequency_hz"] = 0.0
            phy["dominant_magnitude"]    = 0.0

        # Top-10 peaks
        peak_idx, _ = sp_signal.find_peaks(
            Fm[valid], height=float(np.percentile(Fm[valid], 75)),
            distance=max(1, int(fs * 0.05)),
        )
        phy["spectral_peaks"] = sorted(
            [(float(fq[valid][i]), float(Fm[valid][i])) for i in peak_idx],
            key=lambda p: -p[1],
        )[:10]
    else:
        phy["fft_frequencies"] = np.array([])
        phy["fft_magnitudes"]  = np.array([])
        phy["dominant_frequency_hz"] = 0.0
        phy["dominant_magnitude"]    = 0.0
        phy["spectral_peaks"]        = []

    # ── Spectrogram (time-frequency) ──────────────────────────────────────────
    if n >= 256:
        nperseg = min(256, n // 8)
        f_s, t_s, Sxx = sp_signal.spectrogram(
            vib["z"], fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        )
        phy["spec_f"] = f_s
        phy["spec_t"] = t_s + float(ts[0])
        phy["spec_db"] = 10.0 * np.log10(Sxx + 1e-12)
    else:
        phy["spec_f"] = np.array([])
        phy["spec_t"] = np.array([])
        phy["spec_db"] = np.array([[]])

    # ── Per-axis FFT (gyro noise) ─────────────────────────────────────────────
    for ax in ["x", "y", "z"]:
        g = cond.get(f"imu_gyro_{ax}", np.zeros(len(ts)))
        if len(g) >= 64:
            Fg = np.abs(fft(g * np.hanning(len(g)))) / (len(g) / 2)
            fg = fftfreq(len(g), d=dt)
            pos = fg > 0
            phy[f"gyro_fft_f_{ax}"] = fg[pos]
            phy[f"gyro_fft_m_{ax}"] = Fg[pos]
        else:
            phy[f"gyro_fft_f_{ax}"] = np.array([])
            phy[f"gyro_fft_m_{ax}"] = np.array([])

    # ── Motor analysis ────────────────────────────────────────────────────────
    motor_means, motor_stds = [], []
    for ch in ["rcout_1", "rcout_2", "rcout_3", "rcout_4"]:
        arr = cond.get(ch, np.array([]))
        active = arr[arr > 900] if len(arr) else np.array([])
        motor_means.append(float(np.mean(active)) if len(active) else float("nan"))
        motor_stds.append(float(np.std(active)) if len(active) else float("nan"))

    phy["motor_mean_us"]  = motor_means
    phy["motor_std_us"]   = motor_stds
    valid_m = [m for m in motor_means if not np.isnan(m)]
    phy["motor_imbalance_pct"] = (
        float((max(valid_m) - min(valid_m)) / (np.mean(valid_m) + 1e-9) * 100.0)
        if len(valid_m) >= 2 else float("nan")
    )

    return phy


# ─── STAGE 3: Metric Computation ─────────────────────────────────────────────

def _compute_metrics(
    phy: Dict[str, Any],
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "vibration_rms":            float("nan"),
        "vibration_peak_to_peak":   float("nan"),
        "vibration_crest_factor":   float("nan"),
        "dominant_frequency_hz":    float("nan"),
        "resonance_shift_hz":       float("nan"),
        "resonance_direction":      "unknown",
        "motor_imbalance_pct":      float("nan"),
        "energy_used_wh":           float("nan"),
        "distance_km":              float("nan"),
        "specific_energy_wh_per_km": float("nan"),
        "motor_mean_us":            [float("nan")] * 4,
        "spectral_peaks":           [],
        "mean_voltage_v":           float("nan"),
        "mean_current_a":           float("nan"),
        "peak_current_a":           float("nan"),
        "sample_rate_hz":           float("nan"),
        "duration_s":               float("nan"),
    }

    vib = phy.get("vibration_magnitude", np.array([]))
    if len(vib):
        rms = float(np.sqrt(np.mean(vib**2)))
        m["vibration_rms"]          = rms
        m["vibration_peak_to_peak"] = float(np.ptp(vib))
        m["vibration_crest_factor"] = float(np.max(np.abs(vib)) / (rms + 1e-12))

    dom = phy.get("dominant_frequency_hz", 0.0)
    m["dominant_frequency_hz"] = dom

    baseline = context.get("resonance_baseline_hz")
    if baseline is not None and dom > 0:
        shift = dom - float(baseline)
        m["resonance_shift_hz"] = round(shift, 3)
        m["resonance_direction"] = "stable" if abs(shift) < 0.5 else ("downward" if shift < 0 else "upward")

    m["motor_imbalance_pct"] = phy.get("motor_imbalance_pct", float("nan"))
    m["motor_mean_us"]       = phy.get("motor_mean_us", [float("nan")] * 4)
    m["spectral_peaks"]      = phy.get("spectral_peaks", [])
    m["energy_used_wh"]      = phy.get("energy_used_wh", float("nan"))
    m["distance_km"]         = phy.get("distance_km", float("nan"))
    m["mean_voltage_v"]      = phy.get("mean_voltage_v", float("nan"))
    m["mean_current_a"]      = phy.get("mean_current_a", float("nan"))
    m["peak_current_a"]      = phy.get("peak_current_a", float("nan"))
    m["sample_rate_hz"]      = phy.get("sample_rate_hz", float("nan"))
    m["duration_s"]          = phy.get("duration_s", float("nan"))

    e, d = m["energy_used_wh"], m["distance_km"]
    if not np.isnan(e) and not np.isnan(d) and d > 0.01:
        m["specific_energy_wh_per_km"] = round(e / d, 4)

    return m


# ─── STAGE 4: Diagnostics ─────────────────────────────────────────────────────

def _run_diagnostics(
    m: Dict[str, Any],
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "vibration_status": "N/A", "vibration_severity": "unknown",
        "resonance_status": "N/A", "motor_balance_status": "N/A",
        "energy_status": "N/A", "_alerts": [],
    }
    alerts = []

    rms = m.get("vibration_rms", float("nan"))
    if not np.isnan(rms):
        if rms < 0.5:
            diag.update(vibration_status="OK",   vibration_severity="nominal")
        elif rms < 1.5:
            diag.update(vibration_status="WARN", vibration_severity="acceptable")
            alerts.append({"level": "warning", "msg": f"Elevated vibration RMS {rms:.3f} m/s² — inspect propellers and balancing."})
        elif rms < 3.0:
            diag.update(vibration_status="WARN", vibration_severity="elevated")
            alerts.append({"level": "warning", "msg": f"High vibration RMS {rms:.3f} m/s² — probable prop imbalance or loose motor mount."})
        else:
            diag.update(vibration_status="CRIT", vibration_severity="critical")
            alerts.append({"level": "critical", "msg": f"CRITICAL vibration {rms:.3f} m/s² — flight safety risk. Land immediately and inspect."})

    shift = m.get("resonance_shift_hz", float("nan"))
    direction = m.get("resonance_direction", "unknown")
    if not np.isnan(shift):
        if abs(shift) < 1.0:
            diag["resonance_status"] = "STABLE"
        elif direction == "downward":
            diag["resonance_status"] = "SOFTENING"
            alerts.append({"level": "warning", "msg": f"Resonance ↓{abs(shift):.2f} Hz — structural softening: check frame arms and motor mounts."})
        elif direction == "upward":
            diag["resonance_status"] = "STIFFENING"
            alerts.append({"level": "info", "msg": f"Resonance ↑{shift:.2f} Hz — stiffening or mass redistribution detected."})
        else:
            diag["resonance_status"] = "SHIFTED"

    imb = m.get("motor_imbalance_pct", float("nan"))
    if not np.isnan(imb):
        if imb < 3.0:
            diag["motor_balance_status"] = "BALANCED"
        elif imb < 8.0:
            diag["motor_balance_status"] = "MINOR_IMBALANCE"
            alerts.append({"level": "warning", "msg": f"Motor output spread {imb:.1f}% — verify ESC calibration and prop tracking."})
        else:
            diag["motor_balance_status"] = "IMBALANCED"
            alerts.append({"level": "critical", "msg": f"Motor imbalance {imb:.1f}% — mechanical or ESC fault. Check motor bearings."})

    sp_e = m.get("specific_energy_wh_per_km", float("nan"))
    baseline_sp_e = context.get("efficiency_baseline_wh_per_km")
    if not np.isnan(sp_e) and sp_e > 0:
        if baseline_sp_e is not None:
            deg = (sp_e - float(baseline_sp_e)) / (float(baseline_sp_e) + 1e-9) * 100.0
            diag["energy_status"] = "DEGRADED" if deg > 20 else "SLIGHTLY_DEGRADED" if deg > 10 else "NOMINAL"
            if deg > 20:
                alerts.append({"level": "warning", "msg": f"Efficiency degraded {deg:.1f}% vs baseline ({sp_e:.2f} vs {baseline_sp_e:.2f} Wh/km)."})
        else:
            diag["energy_status"] = "NO_BASELINE"

    diag["_alerts"] = alerts
    return diag


# ─── Performance model ───────────────────────────────────────────────────────

def _compute_performance_model(phy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive a serialisable performance model from reconstructed physics.
    All 10 fields are always present; value is None when data is unavailable.
    This block becomes the single source of truth for the mission engine.
    """
    def _f(v) -> Any:
        """Return None if NaN/missing, else rounded Python float."""
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else round(f, 6)
        except (TypeError, ValueError):
            return None

    pm: Dict[str, Any] = {
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
    }

    power_w    = phy.get("power_w",    np.array([]))
    ts         = phy.get("timestamp_s", np.array([]))
    spd        = phy.get("ground_speed_ms", np.array([]))
    volt       = phy.get("voltage_v",   np.array([]))
    curr       = phy.get("current_a",   np.array([]))
    motor_us   = phy.get("motor_mean_us", [])
    duration_s = phy.get("duration_s",  float("nan"))
    energy_wh  = phy.get("energy_used_wh", float("nan"))
    dist_km    = phy.get("distance_km", float("nan"))

    n = len(power_w)
    if n < 4:
        return pm

    # ── mean_power_w — from energy and duration ───────────────────────────────
    if not np.isnan(energy_wh) and not np.isnan(duration_s) and duration_s > 0:
        pm["mean_power_w"] = _f(energy_wh / (duration_s / 3600.0))

    # ── max_power_w — instantaneous peak ─────────────────────────────────────
    if len(volt) == n and len(curr) == n:
        pm["max_power_w"] = _f(np.max(volt * curr))
    elif n > 0:
        pm["max_power_w"] = _f(np.max(power_w))

    # ── velocity-segmented hover / cruise power ───────────────────────────────
    spd_n = n
    spd_arr = np.array(spd, dtype=float) if len(spd) == n else np.zeros(n)

    hover_mask  = spd_arr < 1.0   # < 1 m/s  → hovering
    cruise_mask = spd_arr > 2.0   # > 2 m/s  → forward flight

    if hover_mask.sum() >= 10:
        pm["hover_power_w"] = _f(np.mean(power_w[hover_mask]))
    else:
        # Fallback: lowest-20% velocity window
        p20 = float(np.percentile(spd_arr, 20))
        low_mask = spd_arr <= max(p20, 0.5)
        if low_mask.sum() >= 5:
            pm["hover_power_w"] = _f(np.mean(power_w[low_mask]))
        elif pm["mean_power_w"] is not None:
            pm["hover_power_w"] = pm["mean_power_w"]

    if cruise_mask.sum() >= 10:
        pm["cruise_power_w"] = _f(np.mean(power_w[cruise_mask]))
    elif pm["mean_power_w"] is not None:
        pm["cruise_power_w"] = pm["mean_power_w"]

    # ── hover_throttle — PWM motor outputs normalised to [0, 1] ──────────────
    valid_us = [m for m in motor_us if m is not None and not np.isnan(m) and m > 900]
    if valid_us:
        mean_us = float(np.mean(valid_us))
        # ArduPilot PWM: 1000 μs = 0% throttle, 2000 μs = 100%
        pm["hover_throttle"] = _f(np.clip((mean_us - 1000.0) / 1000.0, 0.0, 1.0))

    # ── voltage_sag_pct — (V_noload − V_loaded) / V_noload ───────────────────
    if len(volt) >= 10 and len(curr) >= 10:
        volt_arr = np.array(volt, dtype=float)
        curr_arr = np.array(curr, dtype=float)
        valid    = volt_arr > 1.0
        if valid.sum() >= 4:
            v_clean  = volt_arr[valid]
            c_clean  = curr_arr[valid]
            # No-load proxy: voltage when current is in lowest 10%
            c_thresh = float(np.percentile(c_clean, 10))
            v_noload = float(np.mean(v_clean[c_clean <= max(c_thresh, 0.1)]))
            # Loaded proxy: voltage when current is in highest 20%
            c_thresh2 = float(np.percentile(c_clean, 80))
            v_loaded  = float(np.mean(v_clean[c_clean >= c_thresh2]))
            if v_noload > 0 and v_noload > v_loaded:
                pm["voltage_sag_pct"] = _f((v_noload - v_loaded) / v_noload)

    # ── energy_per_min_wh ─────────────────────────────────────────────────────
    if not np.isnan(energy_wh) and not np.isnan(duration_s) and duration_s > 0:
        pm["energy_per_min_wh"] = _f(energy_wh / (duration_s / 60.0))

    # ── energy_per_km_wh — null when GPS data is missing or minimal ───────────
    if not np.isnan(energy_wh) and not np.isnan(dist_km) and dist_km > 0.05:
        pm["energy_per_km_wh"] = _f(energy_wh / dist_km)
    # else: stays None (flagged as degraded by mission engine)

    # ── thrust_margin_pct — derived from hover_throttle ───────────────────────
    if pm["hover_throttle"] is not None:
        pm["thrust_margin_pct"] = _f(1.0 - pm["hover_throttle"])

    # ── efficiency_state — based on energy_per_km if available, else throttle ─
    epk = pm["energy_per_km_wh"]
    ht  = pm["hover_throttle"]
    if epk is not None:
        pm["efficiency_state"] = "high" if epk < 25.0 else "moderate" if epk < 60.0 else "low"
    elif ht is not None:
        pm["efficiency_state"] = "high" if ht < 0.45 else "moderate" if ht < 0.65 else "low"

    return pm


# ─── STAGE 5: Scoring ─────────────────────────────────────────────────────────

def _compute_score(m: Dict[str, Any], diag: Dict[str, Any]) -> float:
    score = 100.0
    rms = m.get("vibration_rms", float("nan"))
    if not np.isnan(rms):       score -= min(40.0, rms * 13.33)
    shift = m.get("resonance_shift_hz", float("nan"))
    if not np.isnan(shift):     score -= min(25.0, abs(shift) * 5.0)
    imb = m.get("motor_imbalance_pct", float("nan"))
    if not np.isnan(imb):       score -= min(20.0, imb * 2.5)
    if diag.get("energy_status") == "DEGRADED":           score -= 15.0
    elif diag.get("energy_status") == "SLIGHTLY_DEGRADED": score -= 7.5
    return max(0.0, min(100.0, round(score, 2)))


# ─── Section Builder ──────────────────────────────────────────────────────────

def _build_sections(
    m: Dict[str, Any],
    diag: Dict[str, Any],
    phy: Dict[str, Any],
    cond: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    def fmt(v: Any, d: int = 3, u: str = "") -> str:
        return "N/A" if (isinstance(v, float) and np.isnan(v)) else f"{round(v, d)}{' ' + u if u else ''}"

    return {
        "vibration": {
            "title": "Vibration Analysis",
            "rms_ms2":          fmt(m.get("vibration_rms",          float("nan")), 3, "m/s²"),
            "peak_to_peak_ms2": fmt(m.get("vibration_peak_to_peak", float("nan")), 3, "m/s²"),
            "crest_factor":     fmt(m.get("vibration_crest_factor", float("nan")), 2),
            "severity":         diag.get("vibration_severity", "unknown"),
            "status":           diag.get("vibration_status",   "N/A"),
            "fft_frequencies":  phy.get("fft_frequencies", np.array([])),
            "fft_magnitudes":   phy.get("fft_magnitudes",  np.array([])),
            "spec_f":           phy.get("spec_f", np.array([])),
            "spec_t":           phy.get("spec_t", np.array([])),
            "spec_db":          phy.get("spec_db", np.array([[]])),
            "rolling_rms":      phy.get("vibration_rolling_rms", np.array([])),
        },
        "resonance": {
            "title": "Structural Resonance",
            "dominant_frequency_hz": fmt(m.get("dominant_frequency_hz", float("nan")), 2, "Hz"),
            "resonance_shift_hz":    fmt(m.get("resonance_shift_hz",    float("nan")), 3, "Hz"),
            "direction":             m.get("resonance_direction", "unknown"),
            "status":                diag.get("resonance_status", "N/A"),
            "spectral_peaks":        m.get("spectral_peaks", []),
        },
        "motors": {
            "title": "Motor Health",
            "mean_us":        m.get("motor_mean_us", [float("nan")] * 4),
            "std_us":         phy.get("motor_std_us", [float("nan")] * 4),
            "imbalance_pct":  fmt(m.get("motor_imbalance_pct", float("nan")), 1, "%"),
            "status":         diag.get("motor_balance_status", "N/A"),
            "rcout_1":        phy.get("rcout_1", np.array([])),
            "rcout_2":        phy.get("rcout_2", np.array([])),
            "rcout_3":        phy.get("rcout_3", np.array([])),
            "rcout_4":        phy.get("rcout_4", np.array([])),
        },
        "power": {
            "title": "Battery & Power",
            "energy_used_wh":            fmt(m.get("energy_used_wh",           float("nan")), 2, "Wh"),
            "distance_km":               fmt(m.get("distance_km",               float("nan")), 3, "km"),
            "specific_energy_wh_per_km": fmt(m.get("specific_energy_wh_per_km", float("nan")), 2, "Wh/km"),
            "mean_voltage_v":            fmt(m.get("mean_voltage_v",            float("nan")), 2, "V"),
            "mean_current_a":            fmt(m.get("mean_current_a",            float("nan")), 1, "A"),
            "peak_current_a":            fmt(m.get("peak_current_a",            float("nan")), 1, "A"),
            "status":                    diag.get("energy_status", "N/A"),
            "voltage_v":                 phy.get("voltage_v",  np.array([])),
            "current_a":                 phy.get("current_a",  np.array([])),
            "power_w":                   phy.get("power_w",    np.array([])),
        },
        "imu": {
            "imu_accel_x": phy.get("imu_accel_x", np.array([])),
            "imu_accel_y": phy.get("imu_accel_y", np.array([])),
            "imu_accel_z": phy.get("imu_accel_z", np.array([])),
            "roll_deg":    phy.get("roll_deg",     np.array([])),
            "pitch_deg":   phy.get("pitch_deg",    np.array([])),
            "yaw_deg":     phy.get("yaw_deg",      np.array([])),
            "ground_speed_ms": phy.get("ground_speed_ms", np.array([])),
            "gyro_fft_f_x":    phy.get("gyro_fft_f_x", np.array([])),
            "gyro_fft_m_x":    phy.get("gyro_fft_m_x", np.array([])),
            "gyro_fft_f_y":    phy.get("gyro_fft_f_y", np.array([])),
            "gyro_fft_m_y":    phy.get("gyro_fft_m_y", np.array([])),
            "gyro_fft_f_z":    phy.get("gyro_fft_f_z", np.array([])),
            "gyro_fft_m_z":    phy.get("gyro_fft_m_z", np.array([])),
        },
        "sample_rate_hz": m.get("sample_rate_hz", float("nan")),
        "duration_s":     m.get("duration_s",     float("nan")),
    }


# ─── DSP helpers ─────────────────────────────────────────────────────────────

def _highpass(data: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    if fs <= 0 or cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        return data.copy()
    b, a = sp_signal.butter(order, cutoff_hz / (fs / 2), btype="high")
    if len(data) < max(len(a), len(b)) * 3:
        return data.copy()
    return sp_signal.filtfilt(b, a, data)
