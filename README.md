# UAV Flight Analytics Platform — Physics-Based Health Assessment and Mission Feasibility Evaluation

A dual-engine analytics system for ArduPilot-based multirotors. The Health Engine reconstructs physical signals from raw DataFlash logs and produces a measured performance model. The Mission Engine consumes that model — and only that model — to evaluate pre-flight mission feasibility against five hard constraints with margin-based risk classification.

---

## Overview

The platform operates in two sequential phases. Post-flight, the Health Engine processes IMU, power, GPS, and motor telemetry through a fixed five-stage pipeline — signal conditioning, physics reconstruction, metric computation, diagnostics, and scoring — producing a `performance_model` that captures the drone's observed energy and thrust behavior. Pre-flight, the Mission Engine loads that model and evaluates whether a planned mission falls within the drone's demonstrated operational envelope, without recourse to theoretical propulsion estimates.

The system is deployed as a multi-page Streamlit application. It accepts ArduPilot `.bin` DataFlash logs, `.param` configuration files, and a JSON memory file for cross-session baseline tracking.

---

## Why This System Is Different

Most UAV planning tools estimate performance from hardware specifications: propeller diameter, motor KV, and battery cell count feed into disc-theory models that produce theoretical hover power and range figures. This system does not do that.

Every performance figure used for mission planning — hover power, cruise power, voltage sag, hover throttle, thrust margin, specific energy — is extracted from actual flight telemetry. If the drone flew inefficiently due to imbalanced props, an aging battery, or high payload, that inefficiency is captured in the model and propagates into the mission evaluation.

The Mission Engine has no aerodynamic fallback. When a valid `performance_model` is absent, it flags a data-quality error and halts. This is an intentional design constraint: the system will not construct a mission decision from inputs it cannot verify. A GO verdict from this system means the drone has already demonstrated it can sustain the required power level — not that a model predicts it should.

---

## Key Features

### Signal Processing and Physics Reconstruction

- Native ArduPilot DataFlash (`.bin`) binary parser with no MAVLink dependency; falls back to CSV/TXT telemetry
- 4th-order Butterworth high-pass filter (5 Hz cutoff, zero-phase `filtfilt`) for vibration isolation from accelerometer DC
- Hanning-windowed FFT on Z-axis vibration with top-10 spectral peak detection (threshold: 75th-percentile magnitude, minimum separation: 5% of sample rate)
- Per-axis gyroscope FFT (roll, pitch, yaw) for independent structural resonance monitoring
- STFT spectrogram on Z-axis vibration (256-sample window, 50% overlap)
- 1-second rolling RMS on 3-axis vibration magnitude

### Energy and Performance Modeling

- Hover power from mean instantaneous power (`V x I`) at GPS speed < 1 m/s; cruise power from speed > 2 m/s segments
- Voltage sag as the normalized delta between low-current (10th percentile) and high-current (80th percentile) voltage windows
- Specific energy (Wh/km) from trapezoidal integration of power and GPS speed
- ISA air density correction: `rho = rho_SL x (T_ref / T) x (1 - L*alt / T_ref)^4.256`
- Composite health score (0-100): vibration RMS (up to -40), resonance shift (up to -25), motor imbalance (up to -20), energy degradation (-7.5 or -15)

### Mission Decision System

- Five hard constraints: duration feasibility (C1), range feasibility (C2), thrust margin >= 25% (C3), voltage sag <= 25% (C4), hover throttle <= 70% (C5)
- 85% safety factor applied to endurance and range before constraint evaluation
- Risk classification: HIGH (any constraint violated or minimum margin < 15%), MEDIUM (15-30%), LOW (>= 30%)
- Endurance sanity bounds: readings below 10 W mean power flagged as sensor fault; cap at 300 minutes

### System Integration

- `.param` file parser: PID gains (P/I/D/FF per axis), filter configuration (LP, static notch, harmonic notch), ESC protocol inference from PWM range, battery cell count heuristic from low-voltage threshold
- JSON memory layer: resonance baseline, efficiency baseline, health score history, specific energy history, questionnaire state — merged non-destructively on load
- Uniform `analyze(signals, params, context)` interface across both engines

---

## System Architecture

Both engines share an identical three-argument interface. The `performance_model` produced by the Health Engine is the sole data contract between them — no parameters flow back, and no shared mutable state exists between sessions.

```
+----------------------------------------------------------+
|                    Streamlit UI Layer                    |
|   Health Monitoring  |  Mission Feasibility  |  About   |
+------------+--------------------+------------------------+
             |                    |
             v                    v
+--------------------+  +---------------------------------+
|   Health Engine    |  |        Mission Engine           |
|                    |  |                                 |
| 1. Signal Cond.    |  | 1. Input Conditioning           |
| 2. Physics Recon.  +->| 2. Performance Loading          |
| 3. Metric Comp.    |  | 3. Energy Budget                |
| 4. Diagnostics     |  | 4. Endurance & Range            |
| 5. Scoring         |  | 5. Constraint Evaluation (x5)   |
+--------------------+  | 6. Diagnostics                  |
         |              | 7. Scoring                      |
         v              +---------------------------------+
+----------------+  +----------------+  +----------------+
| data_processor |  | param_handler  |  |  json_handler  |
| (.bin / .csv)  |  | (.param file)  |  | (memory JSON)  |
+----------------+  +----------------+  +----------------+
```

---

## Data Flow

```
Flight Log (.bin / .csv)
        |
        v
data_processor.process_file()
        |  --> timestamp_s, imu_accel_{x,y,z}, imu_gyro_{x,y,z}
        |      voltage_v, current_a, ground_speed_ms
        |      roll/pitch/yaw_deg, rcout_{1-4}, per-message timestamps
        v
health_engine.analyze(signals, params, context)
        |
        +-- result["metrics"]            computed physical quantities
        +-- result["diagnostics"]        status codes and alert strings
        +-- result["score"]              0-100 composite health score
        +-- result["sections"]           UI-ready structured output
        +-- result["performance_model"]  <-- exported as Health JSON
                |
                v   (re-ingested on Mission Feasibility page)
mission_engine.analyze(signals, params, context)
                |
                +-- result["metrics"]      budget, endurance, range, constraints
                +-- result["diagnostics"]  constraint violations, risk level
                +-- result["score"]        0-100 feasibility score
                +-- result["sections"]     UI-ready structured output
```

---

## Module Breakdown

### `health_engine.py`

Five-stage pipeline executed in fixed order with no early returns.

**Stage 1 — Signal Conditioning:** Resamples all signals onto a uniform time grid at the IMU median sample interval. NaN gaps are linearly interpolated before resampling.

**Stage 2 — Physics Reconstruction:** Isolates vibration via 4th-order high-pass filter (5 Hz). Computes instantaneous power (`V x I`), energy via trapezoidal integration, distance via speed integration. Runs Hanning-windowed FFT and STFT on Z-axis vibration; runs per-axis FFT on gyroscope signals; computes 1-second rolling RMS on vibration magnitude.

**Stage 3 — Metric Computation:** Derives vibration RMS, peak-to-peak, crest factor, dominant frequency, resonance shift from stored baseline, motor imbalance from PWM spread across channels, and energy efficiency metrics.

**Stage 4 — Diagnostics:** Threshold classification producing `GOOD` / `WARNING` / `CRITICAL` status for vibration severity, resonance drift, motor balance, and energy degradation.

**Stage 5 — Scoring:** 0-100 composite score with weighted penalties: vibration RMS (up to -40), resonance shift (up to -25), motor imbalance (up to -20), energy degradation (-7.5 or -15).

**Performance Model:** Velocity-segmented power extraction (hover: speed < 1 m/s; cruise: speed > 2 m/s). Voltage sag from percentile-based low/high current windows. Thrust margin from `1 - hover_throttle`. Falls back to mean-power percentile if insufficient GPS data.

---

### `mission_engine.py`

Seven-stage pipeline executed in fixed order.

**Performance Loading:** Extracts all `performance_model` fields from the injected health JSON context. Returns NaN for null or absent values. The `available` flag gates all downstream constraint evaluation.

**Energy Budget:** `usable_energy_wh = total_energy_wh x (1 - reserve_pct) x (1 - voltage_sag_pct)`. Nominal voltage computed as `cells x 3.8 V`.

**Endurance and Range:** Endurance from `usable_energy_wh / hover_power_w`. Range from `usable_energy_wh / energy_per_km_wh` when GPS data is present; conservative 5 m/s fallback otherwise. Power readings below 10 W are flagged as sensor faults.

**Constraint Evaluation:**

| ID | Description | Threshold |
|----|-------------|-----------|
| C1 | Required duration vs. estimated endurance | Fail if > 85% of endurance |
| C2 | Required distance vs. estimated range | Fail if > 85% of range |
| C3 | Thrust margin | Fail if < 25% |
| C4 | Voltage sag | Fail if > 25% |
| C5 | Hover throttle | Fail if > 70% |

---

### `data_processor.py`

Parses ArduPilot DataFlash binary format without external MAVLink libraries. Reads `FMT` descriptor records to dynamically build message schemas, then extracts `IMU`/`IMU2`, `ATT`, `RATE`, `BARO`, `BAT`/`CURR`, `GPS`, `RCOU`, `VIBE`, and `PARM` messages. Per-message timestamps are preserved as separate arrays for accurate signal alignment during interpolation. Falls back to structured CSV/TXT parsing. All errors are non-fatal — collected as warnings and returned alongside partial signal output.

---

### `param_handler.py`

Parses ArduPilot `.param` files (comma- or whitespace-delimited). Provides:

- `extract_questionnaire_fields()` — maps `FRAME_CLASS`, `BATT_CAPACITY`, `BATT_LOW_VOLT`, `MOT_PWM_MIN/MAX` to frame type, motor count, cell count estimate (3.4 V/cell heuristic), and ESC protocol
- `extract_pid_gains()` — P/I/D/FF/IMAX/FLTD/FLTT for Roll, Pitch, Yaw rate controllers and angle P gains
- `extract_filter_config()` — gyro/accel LP filter Hz, static notch, harmonic notch parameters
- `compute_theoretical_bandwidth()` — closed-loop bandwidth (`P x 100 Hz`) and phase margin accounting for D-term lead, FLTD attenuation, gyro LP lag, and 10 ms system delay

---

### `json_handler.py`

Typed memory schema for cross-session persistence. Resonance and efficiency baselines are set on the first analysis run and used for drift detection thereafter. `load_memory()` merges uploaded JSON against schema defaults non-destructively. `update_memory_from_result()` appends new measurements and conditionally updates baselines. `save_memory()` serializes to JSON bytes with a NumPy scalar serializer.

---

### Streamlit UI

**`1_Health_Monitoring.py`** — Accepts `.bin` or `.csv` log, optional `.param` file, and optional memory JSON. Renders vibration time series, FFT spectrum, STFT spectrogram, rolling RMS, per-axis gyroscope FFT, motor PWM outputs, battery power traces, and attitude signals. Exports `performance_model` as a downloadable Health JSON.

**`2_Mission_Feasibility.py`** — Accepts the Health JSON. Exposes mission parameters (payload, distance, duration, altitude, temperature, wind, terrain factor, reserve). Displays constraint evaluations with pass/fail/warn status, margin percentages, energy budget breakdown, and the final risk verdict.

---

## Core Concepts

### FFT and Vibration Analysis

Z-axis accelerometer signal is high-pass filtered at 5 Hz (4th-order Butterworth, zero-phase `filtfilt`) before spectral analysis. The FFT uses a Hanning window to reduce spectral leakage. Peaks are detected with `scipy.signal.find_peaks` at the 75th-percentile magnitude threshold with minimum frequency separation of 5% of the sample rate. Per-axis gyroscope FFTs run independently to isolate axis-specific structural modes.

### Energy-Based Mission Validation

Mission feasibility is evaluated against the usable energy envelope after voltage sag and reserve deductions — not against nameplate battery capacity. Thrust margin and hover throttle are constrained independently because a drone within its energy budget may still be operating near its thrust ceiling, with no headroom for wind, payload variation, or emergency response.

---

## Installation

```bash
pip install streamlit numpy scipy pandas
```

Python 3.9 or later. No MAVLink or ArduPilot SDK dependencies required.

---

## Quick Start

```bash
streamlit run app.py
```

1. Open **Health Monitoring** — upload an ArduPilot `.bin` log (optionally a `.param` file and prior memory JSON)
2. Run the health analysis and download the **Health JSON** output
3. Open **Mission Feasibility** — upload the Health JSON
4. Enter mission parameters and run the feasibility evaluation

---

## Example Output

**Health Engine** produces:
- Vibration RMS, peak-to-peak, crest factor (m/s²); dominant frequency and shift from baseline (Hz)
- Top spectral peaks (frequency/magnitude pairs); per-axis gyroscope spectra
- Motor PWM mean and standard deviation per channel (µs); imbalance percentage
- Energy used (Wh), distance (km), specific energy (Wh/km), mean and peak current, mean voltage
- `performance_model` fields: `hover_power_w`, `cruise_power_w`, `mean_power_w`, `hover_throttle`, `voltage_sag_pct`, `energy_per_min_wh`, `energy_per_km_wh`, `thrust_margin_pct`, `max_power_w`, `efficiency_state`

**Mission Engine** produces:
- Usable energy budget (Wh) after sag and reserve derating
- Estimated endurance (minutes) and range (km)
- Pass/Fail/Warn status and margin percentage for each of the five constraints
- Risk level: LOW / MEDIUM / HIGH; mission feasibility score (0-100)

---

## Limitations

- **Log sample rate:** Minimum 64 IMU samples required for FFT; 256 for spectrogram. Sparse logs produce degraded or absent spectral output.
- **Current sensor calibration:** Uncalibrated `BATT_AMP_PERVLT` / `BATT_AMP_OFFSET` produces near-zero current readings. The mission engine detects this (< 10 W threshold) and flags it as a sensor fault rather than propagating a misleading endurance figure.
- **GPS dependency:** Specific energy (Wh/km), range estimation, and hover/cruise power segmentation require GPS ground speed. Absent GPS or distances below 50 m degrade range evaluation to a 5 m/s fallback, flagged in diagnostics.
- **Single-battery model:** Parallel or series-parallel pack configurations are not modeled.
- **Motor count:** Imbalance analysis covers four RCOUT channels. Additional channels on hexacopter and octocopter configurations are not included in the balance metric.
- **Resonance model:** Baseline is a single scalar (dominant Z-axis frequency). Multi-modal drift and axis-specific resonance patterns are not tracked independently.
- **No aerodynamic fallback:** Absent `performance_model` halts mission evaluation. This is by design.

---

## Future Improvements

- Motor imbalance extended to 6- and 8-channel configurations, driven by frame class extracted from the param file
- Per-axis (X/Y/Z) resonance baseline tracking for axis-specific prop and bearing fault detection
- Adaptive notch filter recommendations derived from detected spectral peaks, output as ready-to-apply ArduPilot parameter deltas
- Multi-flight trend analysis over the existing health score and efficiency history arrays in the memory schema
- Current sensor advisory: auto-detect near-zero readings and surface `BATT_AMP_PERVLT` calibration guidance
- Wind-adjusted range estimation incorporating headwind component into cruise power scaling

---

## About

Developed by [DroneAcharya Aerial Innovations](https://droneacharya.com), Pune, India.  
Contact: info@droneacharya.com
