"""
JSON Memory Handler — Flight Digital Twin Memory Layer
Stores and retrieves persistent flight metrics, baselines, and questionnaire state.
"""

import json
import io
from typing import Any, Dict, Optional


MEMORY_SCHEMA: Dict[str, Any] = {
    "version": "1.0",
    "resonance_baseline_hz": None,
    "efficiency_baseline_wh_per_km": None,
    "health_score_history": [],
    "specific_energy_history": [],
    "resonance_history": [],
    "questionnaire": {
        "frame_class": None,
        "frame_size_mm": None,
        "dry_weight_g": None,
        "battery_cells": None,
        "battery_capacity_mah": None,
        "motor_kv": None,
        "prop_diameter_in": None,
        "prop_pitch_in": None,
        "esc_protocol": None,
        "payload_g": None,
    },
    "last_flight": {
        "timestamp": None,
        "duration_s": None,
        "energy_used_wh": None,
        "distance_km": None,
        "max_vibration_rms": None,
    },
}


def load_memory(file_obj: Optional[Any]) -> Dict[str, Any]:
    """
    Load memory from uploaded JSON file object.
    Missing keys are filled from MEMORY_SCHEMA defaults (never fails).
    """
    memory = _deep_copy_schema(MEMORY_SCHEMA)
    if file_obj is None:
        return memory

    try:
        if hasattr(file_obj, "read"):
            raw = file_obj.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
        else:
            raw = str(file_obj)

        parsed = json.loads(raw)
        memory = _merge_with_schema(memory, parsed)
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        pass

    return memory


def save_memory(memory: Dict[str, Any]) -> bytes:
    """
    Serialize memory dict to JSON bytes for download.
    """
    return json.dumps(memory, indent=2, default=_json_serializer).encode("utf-8")


def update_memory_from_result(
    memory: Dict[str, Any],
    result: Dict[str, Any],
    questionnaire: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge latest analysis result into memory.
    Updates baselines and appends history entries.
    """
    metrics = result.get("metrics", {})

    # Ensure all schema keys exist (handles empty {} memory from no upload)
    for key, default in _deep_copy_schema(MEMORY_SCHEMA).items():
        if key not in memory:
            memory[key] = default

    # Update questionnaire snapshot
    if questionnaire:
        mem_q = memory.setdefault("questionnaire", {})
        for k, v in questionnaire.items():
            if v is not None:
                mem_q[k] = v

    # Update resonance baseline on first run or explicit reset
    dom_freq = metrics.get("dominant_frequency_hz")
    if dom_freq is not None and not (isinstance(dom_freq, float) and dom_freq != dom_freq):
        if memory.get("resonance_baseline_hz") is None:
            memory["resonance_baseline_hz"] = float(dom_freq)
        memory.setdefault("resonance_history", []).append(float(dom_freq))

    # Update specific energy baseline
    sp_energy = metrics.get("specific_energy_wh_per_km")
    _is_nan = lambda v: isinstance(v, float) and v != v
    if sp_energy is not None and not _is_nan(sp_energy):
        if memory.get("efficiency_baseline_wh_per_km") is None:
            memory["efficiency_baseline_wh_per_km"] = float(sp_energy)
        memory.setdefault("specific_energy_history", []).append(float(sp_energy))

    # Append health score
    score = result.get("score")
    if score is not None:
        memory.setdefault("health_score_history", []).append(float(score))

    # Update last flight snapshot
    lf = memory.setdefault("last_flight", {
        "timestamp": None, "duration_s": None,
        "energy_used_wh": None, "distance_km": None, "max_vibration_rms": None,
    })
    lf["energy_used_wh"] = metrics.get("energy_used_wh", lf.get("energy_used_wh"))
    lf["distance_km"] = metrics.get("distance_km", lf.get("distance_km"))
    lf["max_vibration_rms"] = metrics.get("vibration_rms", lf.get("max_vibration_rms"))

    return memory


def extract_questionnaire_prefill(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return questionnaire fields stored in memory for UI pre-population.
    Only returns non-None values.
    """
    raw = memory.get("questionnaire", {})
    return {k: v for k, v in raw.items() if v is not None}


# ─── Internal helpers ────────────────────────────────────────────────────────


def _deep_copy_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy MEMORY_SCHEMA so mutations don't bleed across sessions."""
    return json.loads(json.dumps(schema))


def _merge_with_schema(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge override into base without removing schema keys.
    """
    for k, v in override.items():
        if k in base:
            if isinstance(base[k], dict) and isinstance(v, dict):
                base[k] = _merge_with_schema(base[k], v)
            else:
                base[k] = v
        else:
            base[k] = v
    return base


def _json_serializer(obj: Any) -> Any:
    """Handle numpy types during serialization."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
