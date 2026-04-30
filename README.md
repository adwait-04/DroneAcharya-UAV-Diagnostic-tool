# DroneAcharya Flight Digital Twin

A physics-first UAV analytics platform for post-flight structural health assessment and pre-flight mission feasibility evaluation. Built on ArduPilot DataFlash logs, it extracts measured performance characteristics from real flight data and uses them — not theoretical models — to make mission GO/RISK/NO-GO decisions.

---

## Overview

DroneAcharya Flight Digital Twin is a two-engine analytics system designed for operators and engineers who need rigorous, data-driven insight into UAV health and mission readiness.

The **Health Engine** ingests raw ArduPilot binary logs, reconstructs physical signals from IMU, power, GPS, and motor outputs, and produces a structured health report including vibration diagnostics, structural resonance tracking, motor balance analysis, and an energy model grounded in observed flight data.

The **Mission Engine** consumes the `performance_model` produced by the Health Engine — not propeller geometry or theoretical curves — to evaluate whether a planned mission is feasible given the drone's actual measured power consumption, thrust margin, voltage behavior, and cruise speed. It enforces five hard constraints and classifies risk as LOW, MEDIUM, or HIGH.

The system is deployed as a multi-page Streamlit application with support for `.bin` log uploads, `.param` configuration files, and JSON memory persistence for baseline tracking across flights.

---

## Key Features

- Native ArduPilot DataFlash (`.bin`) parser with no dependency on MAVLink libraries; falls back to CSV/TXT telemetry
- FFT and STFT-based vibration analysis on IMU Z-axis with Hanning windowing, spectral peak detection, and rolling RMS
- Per-axis gyroscope FFT (roll, pitch, yaw) for independent structural resonance monitoring
- Resonance shift detection relative to a stored baseline (persisted in session memory)
- Motor output imbalance quantification from PWM microsecond spread across four channels
- Physics-integrated energy model: hover power, cruise power, voltage sag, thrust margin, and specific energy (Wh/km) derived from actual flight telemetry
- ISA-corrected air density computation for altitude and temperature-aware mission planning
- Five-constraint hard-failure system with margin-based risk classification
- ArduPilot `.param` file parser with PID gain extraction, filter configuration, ESC protocol inference, and cell count heuristics
- JSON memory layer for resonance baseline, efficiency baseline, health score history, and questionnaire state persistence across sessions
- Health score composed from vibration RMS, resonance shift, motor imbalance, and energy degradation penalties

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI Layer                    │
│   Health Monitoring  │  Mission Feasibility  │  About   │
└──────────┬──────────────────────┬────────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│  Health Engine   │   │        Mission Engine            │
│                  │   │                                  │
│ 1. Signal Cond.  │   │ 1. Input Conditioning            │
│ 2. Physics Recon.│──►│ 2. Performance Loading           │
│ 3. Metric Comp.  │   │ 3. Energy Budget                 │
│ 4. Diagnostics   │   │ 4. Endurance & Range             │
│ 5. Scoring       │   │ 5. Constraint Evaluation (×5)    │
└──────────────────┘   │ 6. Diagnostics                   │
         │             │ 7. Scoring                       │
         │             └──────────────────────────────────┘
         │
         ▼
┌──────────────────┐   ┌────────────────┐   ┌────────────────┐
│  data_processor  │   │  param_handler │   │  json_handler  │
│  (.bin / .csv)   │   │  (.param file) │   │  (memory JSON) │
└──────────────────┘   └────────────────┘   └────────────────┘
```

The Health Engine and Mission Engine share an identical three-argument interface: `analyze(signals, params, context)`. This allows both to be invoked uniformly from the UI while maintaining strict separation between post-flight analysis and pre-flight planning.

The `performance_model` output of the Health Engine is the sole bridge between the two engines. The Mission Engine does not fall back to propeller geometry or motor KV estimation when the performance model is absent — it flags a data-quality error and degrades gracefully, making the dependency on real flight data explicit and non-negotiable.

---

## Data Flow

```
Flight Log (.bin)
       │
       ▼
data_processor.process_file()
       │  Fixed signal schema:
       │  timestamp_s, imu_accel_{x,y,z}, imu_gyro_{x,y,z},
       │  voltage_v, current_a, ground_speed_ms,
       │  roll/pitch/yaw_deg, rcout_{1–4}, ...
       ▼
health_engine.analyze(signals, params, context)
       │
       ├── result["metrics"]          — computed physical quantities
       ├── result["diagnostics"]      — status codes and alert strings
       ├── result["score"]            — 0–100 composite health score
       ├── result["sections"]         — UI-ready structured output
       └── result["performance_model"]
                │
                ▼   (exported as JSON, re-ingested on Mission page)
       mission_engine.analyze(signals, params, context)
                │
                ├── result["metrics"]     — budget, endurance, range, constraints
                ├── result["diagnostics"] — constraint violations, risk level
                ├── result["score"]       — 0–100 mission feasibility score
                └── result["sections"]    — UI-ready structured output
```

---

## Module Breakdown

### `health_engine.py`

Implements the five-stage post-flight analysis pipeline in fixed execution order (no early returns).

**Stage 1 — Signal Conditioning:** Resamples all input signals onto a uniform time grid derived from the IMU timestamp median interval. NaN values are linearly interpolated from neighboring valid samples before resampling.

**Stage 2 — Physics Reconstruction:** Applies a 4th-order Butterworth high-pass filter (cutoff 5 Hz) to isolate vibration from the accelerometer DC component. Computes instantaneous power (`V × I`), energy via trapezoidal integration, and distance via speed integration. Runs full FFT with Hanning window on Z-axis vibration and per-axis gyroscope signals, generates a time-frequency spectrogram via STFT, detects spectral peaks above the 75th-percentile magnitude threshold, and computes 1-second rolling RMS on vibration magnitude.

**Stage 3 — Metric Computation:** Derives vibration RMS, peak-to-peak, crest factor, dominant frequency, resonance shift from stored baseline, motor imbalance from PWM spread, and energy efficiency metrics.

**Stage 4 — Diagnostics:** Applies threshold-based classification to produce status codes (`GOOD`, `WARNING`, `CRITICAL`) for vibration severity, resonance drift, motor balance, and energy degradation.

**Stage 5 — Scoring:** Composites a 0–100 score by penalizing vibration RMS (up to −40), resonance shift (up to −25), motor imbalance (up to −20), and energy degradation status (−7.5 or −15).

**Performance Model Output:** Extracts velocity-segmented hover and cruise power by classifying samples below 1 m/s (hover) and above 2 m/s (cruise), computes voltage sag from low-current vs. high-current voltage percentiles, normalizes PWM outputs to hover throttle fraction, and derives thrust margin from 1 − hover_throttle.

---

### `mission_engine.py`

Implements the pre-flight feasibility pipeline in fixed execution order.

**Performance Loading:** Reads `performance_model` from the injected health JSON context. All fields convert to float with NaN on null/absent values. The `available` flag gates downstream computation.

**Energy Budget:** `usable_energy_wh = total_energy_wh × (1 − reserve_pct) × (1 − voltage_sag_pct)`. Battery nominal voltage is computed as `cells × 3.8 V`.

**Endurance and Range:** Endurance is derived from `usable_energy_wh / hover_power_w`. Range is derived from `usable_energy_wh / energy_per_km_wh` when GPS data is available, otherwise falls back to a 5 m/s cruise speed assumption. Endurance values below 10 W or above 300 minutes are treated as sensor faults rather than valid readings.

**ISA Air Density Correction:** `ρ = ρ_SL × (T_ref / T_actual) × (1 − L × alt / T_ref)^4.256` — applied to adjust power estimates for altitude and temperature.

**Five Hard Constraints:**

| ID | Constraint | Threshold |
|----|-----------|-----------|
| C1 | Required duration exceeds 85% of endurance | Fail if true |
| C2 | Required distance exceeds 85% of range | Fail if true |
| C3 | Thrust margin below 25% | Fail if true |
| C4 | Voltage sag above 25% | Fail if true |
| C5 | Hover throttle above 70% | Fail if true |

**Risk Classification:** HIGH if any hard constraint is violated or minimum margin < 15%. MEDIUM if minimum margin in [15%, 30%). LOW if all margins ≥ 30%.

---

### `data_processor.py`

Parses ArduPilot DataFlash binary (`.bin`) format without external MAVLink libraries. Reads `FMT` descriptor records to build a dynamic message schema, then extracts the following message types: `IMU`/`IMU2`, `ATT`, `RATE`, `BARO`, `BAT`/`CURR`, `GPS`, `RCOU`, `VIBE`, `PARM`. Each message type's timestamps are preserved as separate arrays for correct signal alignment during interpolation. Falls back to structured CSV/TXT parsing for non-binary telemetry exports. All errors are non-fatal; they are collected as warnings and returned alongside the (potentially partial) signal dict.

---

### `param_handler.py`

Parses ArduPilot `.param` files (comma-separated or whitespace-separated key-value format). Provides four extraction functions:

- `extract_questionnaire_fields()` — maps `FRAME_CLASS`, `BATT_CAPACITY`, `BATT_LOW_VOLT`, and `MOT_PWM_MIN/MAX` to questionnaire field names including frame type, motor count, battery cell estimate (via 3.4 V/cell heuristic), and ESC protocol inference
- `extract_pid_gains()` — extracts P/I/D/FF/IMAX/FLTD/FLTT for Roll, Pitch, and Yaw rate controllers and angle P gains
- `extract_filter_config()` — extracts gyro/accel LP filter Hz, static notch (enable, frequency, bandwidth, attenuation), and harmonic notch configuration
- `extract_motor_config()` — extracts PWM range, spin thresholds, thrust expo, and battery voltage compensation limits
- `compute_theoretical_bandwidth()` — estimates closed-loop bandwidth (P × 100 Hz empirical rule) and phase margin accounting for D-term lead, FLTD attenuation, gyro LP filter lag, and 10 ms system delay

---

### `json_handler.py`

Manages a typed memory schema for cross-session persistence. Stores resonance and efficiency baselines (set on first analysis, used for drift detection in subsequent runs), health score history, specific energy history, resonance frequency history, questionnaire state, and last-flight metadata. `load_memory()` merges uploaded JSON against the schema defaults — missing keys are silently filled. `update_memory_from_result()` appends new measurements and sets baselines on first run. `save_memory()` serializes to JSON bytes for download, with a custom serializer handling NumPy scalar types.

---

### Streamlit UI (`1_Health_Monitoring.py`, `2_Mission_Feasibility.py`, `3_AboutUs.py`)

**Health Monitoring page:** Accepts `.bin` or `.csv` log files and an optional `.param` file. Renders vibration time series, FFT spectrum, STFT spectrogram, rolling RMS, gyroscope FFT per axis, motor PWM outputs, battery voltage/current/power traces, attitude signals, and health score. Exports the `performance_model` as a downloadable JSON file. Supports memory JSON upload to enable resonance shift tracking against historical baselines.

**Mission Feasibility page:** Accepts the health JSON from a prior analysis. Exposes mission parameters (payload, distance, duration, altitude, temperature, wind, terrain factor, battery reserve). Displays the five constraint evaluations with pass/fail/warn status, margin percentages, endurance and range estimates, energy budget breakdown, and the final GO/RISK/NO-GO verdict with risk level.

---

## Core Concepts

### Physics-Based Modeling vs. Estimation

The system avoids constructing power or thrust models from first principles (propeller disc theory, motor KV, propeller diameter). Instead, the health engine extracts observed quantities directly from flight telemetry. Hover power is the mean instantaneous power during periods where GPS speed < 1 m/s. Voltage sag is the delta between low-load and high-load voltage percentiles. Hover throttle is the mean normalized PWM output during those same low-speed periods. These values carry measurement uncertainty from the actual flight — but they are grounded in what the drone actually did, not what it should theoretically do.

### FFT and Vibration Analysis

The Z-axis accelerometer signal is high-pass filtered at 5 Hz (4th-order Butterworth, zero-phase via `filtfilt`) to remove gravitational bias and low-frequency attitude motion before spectral analysis. The FFT is windowed with a Hanning function to reduce spectral leakage. Spectral peaks are detected using `scipy.signal.find_peaks` with a height threshold at the 75th percentile of the positive-frequency magnitudes and a minimum separation of 5% of the sampling rate. Per-axis gyroscope FFTs are computed independently on roll, pitch, and yaw rate signals to isolate axis-specific structural modes.

### Energy-Based Mission Validation

Mission feasibility is not determined by comparing estimated range against required distance alone. It evaluates the usable energy envelope after accounting for voltage sag (internal resistance loss under load) and the user-specified reserve percentage. Endurance and range are then derived from this reduced budget. Thrust margin and hover throttle are checked independently because a drone with sufficient range may still be operating near its thrust ceiling, leaving no headroom for wind gusts, payload variation, or emergency maneuvers.

---

## Installation

```bash
pip install streamlit numpy scipy pandas
```

Run the application:

```bash
streamlit run app.py
```

Python 3.9 or later is recommended. No MAVLink or ArduPilot SDK dependencies are required.

---

## Usage

### Post-Flight Health Analysis

1. Navigate to **Health Monitoring** in the sidebar.
2. Upload an ArduPilot `.bin` DataFlash log (or `.csv` telemetry export).
3. Optionally upload a `.param` file to pre-populate the drone configuration fields.
4. Optionally upload a previously saved memory JSON to enable resonance baseline comparison.
5. Fill in any missing questionnaire fields (frame class, battery, motor configuration).
6. Run the analysis. Review vibration diagnostics, spectral plots, motor balance, power traces, and the composite health score.
7. Download the **Health JSON** output. This file contains the `performance_model` required by the Mission Feasibility engine.

### Pre-Flight Mission Feasibility

1. Navigate to **Mission Feasibility** in the sidebar.
2. Upload the Health JSON file exported from the previous step.
3. Enter mission parameters: target payload, required distance, required duration, operating altitude, ambient temperature, wind speed, terrain difficulty, and battery reserve percentage.
4. Run the analysis. Review the five constraint evaluations, endurance and range estimates, and the final risk verdict.

---

## Example Output

**Health Engine** produces:
- Vibration RMS, peak-to-peak, and crest factor in m/s²
- Dominant vibration frequency in Hz and shift from stored baseline
- Top spectral peaks (frequency, magnitude pairs)
- Per-axis gyroscope spectral plots
- Motor output mean and standard deviation per channel (µs), imbalance percentage
- Energy used (Wh), distance (km), specific energy (Wh/km), mean and peak current, mean voltage
- Composite health score (0–100) with per-category diagnostic status
- `performance_model` containing: `hover_power_w`, `cruise_power_w`, `mean_power_w`, `hover_throttle`, `voltage_sag_pct`, `energy_per_min_wh`, `energy_per_km_wh`, `thrust_margin_pct`, `max_power_w`, `efficiency_state`

**Mission Engine** produces:
- Total and usable energy budget (Wh) after sag and reserve derating
- Estimated endurance (minutes) and range (km)
- Pass/Fail/Warn status and margin percentage for each of the five hard constraints
- Payload and environmental impact assessment
- Risk level: LOW / MEDIUM / HIGH
- Mission feasibility score (0–100)

---

## Limitations

- **Log quality dependency:** The health engine requires IMU data at a sufficient sample rate (minimum 64 samples for FFT, 256 for spectrogram). Logs with sparse or missing IMU records will produce degraded or absent spectral output.
- **Current sensor calibration:** Hover power, energy metrics, and all mission feasibility calculations depend on an accurate current sensor reading. ArduPilot installations with uncalibrated `BATT_AMP_PERVLT` / `BATT_AMP_OFFSET` will produce near-zero current readings, which the mission engine detects (threshold: < 10 W mean power) and flags as a sensor fault rather than propagating a misleading endurance estimate.
- **GPS availability:** Specific energy (Wh/km), range estimation, and the hover/cruise power segmentation all require GPS ground speed data. When GPS records are absent or the logged distance is below 50 m, range-based constraint evaluation degrades to a conservative fallback (5 m/s cruise assumed) and is flagged as such.
- **Single-battery model:** The energy model assumes a single battery pack. Parallel or series-parallel configurations are not explicitly handled.
- **No aerodynamic fallback:** When the health JSON is absent or the `performance_model` is null, the mission engine does not substitute propeller-based thrust or power estimates. This is intentional — the system does not invent data.
- **Motor count:** Motor imbalance analysis currently processes four RCOUT channels (`rcout_1`–`rcout_4`). Hexacopter or octocopter configurations log additional outputs that are not included in the balance metric.
- **Static structural model:** The resonance baseline is a single scalar (dominant Z-axis vibration frequency). Mode coupling, axis-specific resonance patterns, and multi-modal drift are not tracked independently.

---

## Future Improvements

- Extend motor imbalance analysis to 6- and 8-motor configurations using RCOU channels 5–8
- Per-axis resonance baseline tracking (independent X/Y/Z dominant frequencies) to detect axis-specific prop or bearing faults
- Adaptive notch filter recommendation based on detected spectral peaks, formatted as ready-to-apply ArduPilot parameter changes
- Multi-flight trend analysis using the existing history arrays in the memory schema (health score trend, efficiency degradation curve)
- Current sensor auto-calibration advisory: detect near-zero current readings and prompt the user to verify `BATT_AMP_PERVLT` against the configured battery
- Support for hexacopter and octocopter RCOU channel mapping driven by the frame class extracted from the param file
- Wind-adjusted range estimation by incorporating headwind/crosswind components into the cruise power model

---

## About

Developed by [DroneAcharya Aerial Innovations](https://droneacharya.com), Pune, India.  
For inquiries: info@droneacharya.com
