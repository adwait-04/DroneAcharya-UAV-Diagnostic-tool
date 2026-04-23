"""
Data Processor — Flight Log Ingestion & Signal Extraction
Supports: ArduPilot .bin (DataFlash), CSV telemetry, synthetic test signals.
All outputs follow a fixed signal schema regardless of source format.

Signal alignment: per-message TimeUS timestamps are extracted alongside values
so the tuning engine can interpolate each signal onto the IMU timeline correctly.
"""

import io
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Fixed output signal schema ────────────────────────────────────────────────
SIGNAL_SCHEMA: Dict[str, Any] = {
    # Master timeline (from IMU)
    "timestamp_s": [],
    # IMU — accelerometer (m/s²) and gyro (rad/s)
    "imu_accel_x": [], "imu_accel_y": [], "imu_accel_z": [],
    "imu_gyro_x":  [], "imu_gyro_y":  [], "imu_gyro_z":  [],
    # Battery & power
    "voltage_v": [], "current_a": [],
    # GPS / navigation
    "ground_speed_ms": [], "altitude_m": [],
    # Attitude (actual) — degrees
    "roll_deg": [], "pitch_deg": [], "yaw_deg": [],
    # Attitude (desired) — degrees  ← from ATT.DesRoll/DesPitch/DesYaw
    "desired_roll_deg": [], "desired_pitch_deg": [], "desired_yaw_deg": [],
    # Rate controller (inner loop) — deg/s  ← from RATE message
    "rate_roll_degs":          [],   # actual roll  rate (RATE.R)
    "rate_pitch_degs":         [],   # actual pitch rate (RATE.P)
    "rate_yaw_degs":           [],   # actual yaw   rate (RATE.Y)
    "desired_rate_roll_degs":  [],   # desired roll  rate (RATE.DR)
    "desired_rate_pitch_degs": [],   # desired pitch rate (RATE.DP)
    "desired_rate_yaw_degs":   [],   # desired yaw   rate (RATE.DY)
    # Motor outputs (µs PWM)
    "rcout_1": [], "rcout_2": [], "rcout_3": [], "rcout_4": [],
    # Per-message timestamps (seconds, normalised to IMU t0)
    # Used by the tuning engine for correct signal alignment.
    "att_timestamp_s":  [],   # ATT message times
    "rate_timestamp_s": [],   # RATE message times
    "rcou_timestamp_s": [],   # RCOU message times
    "bat_timestamp_s":  [],   # BAT/CURR message times
    "gps_timestamp_s":  [],   # GPS message times
    # Metadata
    "sample_rate_hz": 0.0,
    "duration_s":     0.0,
    "source_format":  "unknown",
    "parse_warnings": [],
}


def process_file(
    file_obj: Any, filename: str
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Main ingestion entrypoint.
    Returns (signals_dict, list_of_warnings).
    Never raises — errors become warnings, signals remain empty arrays.
    """
    signals  = _empty_signals()
    warnings: List[str] = []

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "bin":
        signals, warnings = _parse_bin(file_obj, warnings)
    elif ext in ("csv", "txt", "log"):
        signals, warnings = _parse_csv(file_obj, warnings)
    else:
        warnings.append(
            f"Unsupported file extension '{ext}'. "
            "Accepted: .bin (ArduPilot DataFlash), .csv, .txt, .log"
        )
        return signals, warnings

    signals = _finalize(signals)
    signals["parse_warnings"] = warnings
    return signals, warnings


# ─── BIN parser (ArduPilot DataFlash) ────────────────────────────────────────

_FMT_TYPE    = 128
_FMT_HEADER  = b'\xa3\x95'
_KNOWN_MESSAGES = {
    "IMU", "IMU2",
    "ATT",
    "RATE",
    "BARO", "BAT", "GPS",
    "RCOU", "CURR", "VIBE",
}


def _parse_bin(file_obj: Any, warnings: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    signals = _empty_signals()
    signals["source_format"] = "ardupilot_bin"

    try:
        raw = file_obj.read() if hasattr(file_obj, "read") else bytes(file_obj)
    except Exception as exc:
        warnings.append(f"BIN read error: {exc}")
        return signals, warnings

    formats: Dict[int, Dict] = {}
    records: Dict[str, List[Dict]] = {m: [] for m in _KNOWN_MESSAGES}

    i = 0
    while i < len(raw) - 3:
        if raw[i:i+2] != _FMT_HEADER:
            i += 1
            continue

        msg_type = raw[i+2]

        if msg_type == _FMT_TYPE:
            fmt = _parse_fmt_message(raw, i)
            if fmt:
                formats[fmt["type"]] = fmt
            i += 89  # FMT records are always 89 bytes (3-byte header + 86-byte body)
            continue

        if msg_type in formats:
            fmt = formats[msg_type]
            # fmt["length"] is total message length including the 3-byte header
            end = i + fmt["length"]
            if end > len(raw):
                break
            rec = _unpack_message(raw[i+3:end], fmt)
            name = fmt["name"]
            if name in records:
                records[name].append(rec)
            i = end
        else:
            i += 1

    signals = _map_bin_records_to_signals(records, signals, warnings)
    return signals, warnings


def _parse_fmt_message(raw: bytes, offset: int) -> Optional[Dict]:
    try:
        hdr_size      = 3
        fmt_body_size = 86  # FMT body: type(1)+length(1)+name(4)+format(16)+labels(64)
        if offset + hdr_size + fmt_body_size > len(raw):
            return None
        body      = raw[offset + hdr_size: offset + hdr_size + fmt_body_size]
        msg_type  = body[0]
        length    = body[1]
        name      = body[2:6].rstrip(b'\x00').decode("ascii", errors="ignore")
        fmt_str   = body[6:22].rstrip(b'\x00').decode("ascii", errors="ignore")
        labels_raw = body[22:].rstrip(b'\x00').decode("ascii", errors="ignore")
        labels    = [l.strip() for l in labels_raw.split(",") if l.strip()]
        return {"type": msg_type, "length": length, "name": name,
                "fmt": fmt_str, "labels": labels}
    except Exception:
        return None


def _unpack_message(data: bytes, fmt: Dict) -> Dict:
    rec: Dict[str, Any] = {}
    _TYPE_MAP = {
        'b': 'b', 'B': 'B', 'h': 'h', 'H': 'H',
        'i': 'i', 'I': 'I', 'f': 'f', 'd': 'd',
        'n': '4s', 'N': '16s', 'Z': '64s', 'M': 'B',
        'c': 'h', 'C': 'H', 'e': 'i', 'E': 'I',
        'L': 'i', 'q': 'q', 'Q': 'Q',
    }
    # Scaling for packed types ('c' = centidegrees → degrees, 'e' = centi → unit)
    _SCALE_MAP = {'c': 0.01, 'C': 0.01, 'e': 0.01, 'E': 0.01}

    fmt_str = "<"
    labels    = fmt.get("labels", [])
    fmt_chars = list(fmt.get("fmt", ""))
    scales    = []
    for ch in fmt_chars:
        mapped = _TYPE_MAP.get(ch)
        if mapped:
            fmt_str += mapped
            scales.append(_SCALE_MAP.get(ch, 1.0))
    try:
        values = struct.unpack_from(fmt_str, data)
        for label, val, scale in zip(labels, values, scales):
            if isinstance(val, bytes):
                val = val.rstrip(b'\x00').decode("ascii", errors="ignore")
            elif isinstance(val, (int, float)) and scale != 1.0:
                val = val * scale
            rec[label] = val
    except struct.error:
        pass
    return rec


def _map_bin_records_to_signals(
    records: Dict[str, List[Dict]], signals: Dict, warnings: List[str]
) -> Dict:
    def safe_list(recs: List[Dict], key: str, scale: float = 1.0) -> List[float]:
        out = []
        for r in recs:
            v = r.get(key)
            if v is not None:
                try:
                    out.append(float(v) * scale)
                except (TypeError, ValueError):
                    out.append(float("nan"))
        return out

    def safe_ts(recs: List[Dict], t0_us: float) -> List[float]:
        """Extract TimeUS from records, normalise to seconds relative to t0_us."""
        out = []
        for r in recs:
            t = r.get("TimeUS")
            if t is not None:
                try:
                    out.append((float(t) - t0_us) * 1e-6)
                except (TypeError, ValueError):
                    pass
        return out

    # ── IMU — establish global t0 ─────────────────────────────────────────────
    imu = records.get("IMU", []) or records.get("IMU2", [])
    t0_us = float("nan")
    if imu:
        ts_raw = [r.get("TimeUS") for r in imu]
        ts_raw = [float(t) for t in ts_raw if t is not None]
        if ts_raw:
            t0_us = ts_raw[0]
            signals["timestamp_s"] = [(t - t0_us) * 1e-6 for t in ts_raw]
        signals["imu_accel_x"] = safe_list(imu, "AccX")
        signals["imu_accel_y"] = safe_list(imu, "AccY")
        signals["imu_accel_z"] = safe_list(imu, "AccZ")
        signals["imu_gyro_x"]  = safe_list(imu, "GyrX")
        signals["imu_gyro_y"]  = safe_list(imu, "GyrY")
        signals["imu_gyro_z"]  = safe_list(imu, "GyrZ")

    # ── ATT — actual + desired attitude ──────────────────────────────────────
    att = records.get("ATT", [])
    if att:
        if not math.isnan(t0_us):
            signals["att_timestamp_s"] = safe_ts(att, t0_us)
        signals["roll_deg"]         = safe_list(att, "Roll")
        signals["pitch_deg"]        = safe_list(att, "Pitch")
        signals["yaw_deg"]          = safe_list(att, "Yaw")
        signals["desired_roll_deg"]  = safe_list(att, "DesRoll")
        signals["desired_pitch_deg"] = safe_list(att, "DesPitch")
        signals["desired_yaw_deg"]   = safe_list(att, "DesYaw")

    # ── RATE — rate controller (inner loop) inputs & outputs ─────────────────
    rate = records.get("RATE", [])
    if rate:
        if not math.isnan(t0_us):
            signals["rate_timestamp_s"] = safe_ts(rate, t0_us)
        signals["rate_roll_degs"]          = safe_list(rate, "R")
        signals["rate_pitch_degs"]         = safe_list(rate, "P")
        signals["rate_yaw_degs"]           = safe_list(rate, "Y")
        signals["desired_rate_roll_degs"]  = safe_list(rate, "DR")
        signals["desired_rate_pitch_degs"] = safe_list(rate, "DP")
        signals["desired_rate_yaw_degs"]   = safe_list(rate, "DY")

    # ── BAT / CURR ────────────────────────────────────────────────────────────
    bat = records.get("BAT", []) or records.get("CURR", [])
    if bat:
        if not math.isnan(t0_us):
            signals["bat_timestamp_s"] = safe_ts(bat, t0_us)
        signals["voltage_v"] = safe_list(bat, "Volt")
        signals["current_a"] = safe_list(bat, "Curr")

    # ── GPS ───────────────────────────────────────────────────────────────────
    gps = records.get("GPS", [])
    if gps:
        if not math.isnan(t0_us):
            signals["gps_timestamp_s"] = safe_ts(gps, t0_us)
        signals["ground_speed_ms"] = safe_list(gps, "Spd")
        signals["altitude_m"]      = safe_list(gps, "Alt")

    # ── RCOU — motor outputs ──────────────────────────────────────────────────
    rcou = records.get("RCOU", [])
    if rcou:
        if not math.isnan(t0_us):
            signals["rcou_timestamp_s"] = safe_ts(rcou, t0_us)
        signals["rcout_1"] = safe_list(rcou, "C1")
        signals["rcout_2"] = safe_list(rcou, "C2")
        signals["rcout_3"] = safe_list(rcou, "C3")
        signals["rcout_4"] = safe_list(rcou, "C4")

    if not signals["timestamp_s"] and not imu:
        warnings.append(
            "No IMU messages found in .bin file. "
            "Ensure the log contains IMU data (LOG_BITMASK must include IMU)."
        )

    return signals


# ─── CSV / TSV parser ─────────────────────────────────────────────────────────

_CSV_COLUMN_MAP = {
    # Timestamp
    "time_s": "timestamp_s", "time": "timestamp_s",
    "timestamp": "timestamp_s",
    "time_us": ("timestamp_s", 1e-6), "timeus": ("timestamp_s", 1e-6),
    "time_ms": ("timestamp_s", 1e-3),
    # IMU
    "accx": "imu_accel_x", "accy": "imu_accel_y", "accz": "imu_accel_z",
    "gyrx": "imu_gyro_x",  "gyry": "imu_gyro_y",  "gyrz": "imu_gyro_z",
    "ax": "imu_accel_x", "ay": "imu_accel_y", "az": "imu_accel_z",
    "gx": "imu_gyro_x",  "gy": "imu_gyro_y",  "gz": "imu_gyro_z",
    # Power
    "volt": "voltage_v", "voltage": "voltage_v", "vbat": "voltage_v",
    "curr": "current_a", "current": "current_a", "ibat": "current_a",
    # Navigation
    "spd": "ground_speed_ms", "speed": "ground_speed_ms",
    "groundspeed": "ground_speed_ms",
    "alt": "altitude_m", "altitude": "altitude_m",
    # Attitude actual
    "roll": "roll_deg", "pitch": "pitch_deg", "yaw": "yaw_deg",
    # Attitude desired
    "desroll": "desired_roll_deg", "despitch": "desired_pitch_deg",
    "desyaw": "desired_yaw_deg",
    # Rate controller
    "r": "rate_roll_degs",  "p": "rate_pitch_degs",  "y_r": "rate_yaw_degs",
    "dr": "desired_rate_roll_degs", "dp": "desired_rate_pitch_degs",
    "dy": "desired_rate_yaw_degs",
    # RCOUT
    "c1": "rcout_1", "c2": "rcout_2", "c3": "rcout_3", "c4": "rcout_4",
}


def _parse_csv(file_obj: Any, warnings: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    signals = _empty_signals()
    signals["source_format"] = "csv"

    try:
        if hasattr(file_obj, "read"):
            raw = file_obj.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
        else:
            raw = str(file_obj)

        df = pd.read_csv(
            io.StringIO(raw), sep=None, engine="python",
            comment="#", skip_blank_lines=True,
        )
    except Exception as exc:
        warnings.append(f"CSV parse error: {exc}")
        return signals, warnings

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    mapped: Dict[str, str] = {}
    for col in df.columns:
        entry = _CSV_COLUMN_MAP.get(col)
        if isinstance(entry, str):
            mapped[col] = entry
        elif isinstance(entry, tuple):
            target, scale = entry
            df[col] = pd.to_numeric(df[col], errors="coerce") * scale
            mapped[col] = target

    for src_col, tgt_key in mapped.items():
        arr = pd.to_numeric(df[src_col], errors="coerce").fillna(float("nan")).tolist()
        if tgt_key in signals:
            signals[tgt_key] = arr

    unmapped = [c for c in df.columns if c not in mapped]
    if unmapped:
        warnings.append(f"Unmapped CSV columns (ignored): {', '.join(unmapped[:10])}")

    if not signals["timestamp_s"] and "timestamp_s" not in mapped.values():
        n = len(next((v for v in signals.values() if isinstance(v, list) and v), []))
        if n:
            signals["timestamp_s"] = [float(i) * 0.01 for i in range(n)]
            warnings.append("No timestamp column found; assuming 100 Hz sample rate.")

    return signals, warnings


# ─── Finalization ─────────────────────────────────────────────────────────────

def _finalize(signals: Dict[str, Any]) -> Dict[str, Any]:
    ts = signals.get("timestamp_s", [])
    if len(ts) >= 2:
        dt_arr = np.diff(ts)
        dt_arr = dt_arr[dt_arr > 0]
        if len(dt_arr):
            median_dt = float(np.median(dt_arr))
            signals["sample_rate_hz"] = round(1.0 / median_dt, 2) if median_dt > 0 else 0.0
        signals["duration_s"] = float(ts[-1] - ts[0])
    return signals


def _empty_signals() -> Dict[str, Any]:
    s = {}
    for k, v in SIGNAL_SCHEMA.items():
        if isinstance(v, list):
            s[k] = []
        else:
            s[k] = v
    return s


# ─── Signal validation ────────────────────────────────────────────────────────

def validate_signals(signals: Dict[str, Any]) -> Tuple[bool, List[str]]:
    required = ["imu_accel_z", "timestamp_s"]
    missing  = [k for k in required if not signals.get(k)]
    return len(missing) == 0, missing


def signals_to_numpy(
    signals: Dict[str, Any], keys: List[str]
) -> Dict[str, np.ndarray]:
    arrays = {}
    lengths = []
    for k in keys:
        arr = np.array(signals.get(k, []), dtype=np.float64)
        arrays[k] = arr
        if len(arr):
            lengths.append(len(arr))

    if not lengths:
        return {k: np.array([], dtype=np.float64) for k in keys}

    min_len = min(lengths)
    result = {}
    for k in keys:
        arr = arrays[k]
        if len(arr) >= min_len:
            result[k] = arr[:min_len]
        else:
            padded = np.full(min_len, np.nan)
            padded[:len(arr)] = arr
            result[k] = padded
    return result
