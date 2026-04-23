"""
DroneAcharya UAV Engineering & Diagnostic Suite
Landing page — logo/brand in hero, shared upload section, product photo, nav cards.
"""

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data_processor import process_file
from utils.param_handler import parse_param_file


def _load_css() -> None:
    css_path = _ROOT / "assests" / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _init_session_state() -> None:
    defaults = {
        "memory": {},
        "uploaded_file_name": None,
        "parsed_signals": {},
        "analysis_results": {},
        "mission_results": {},
        "questionnaire": {},
        "mission_params": {},
        "param_summary": {},
        "analysis_ran": False,
        "mission_ran": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _handle_log_upload(file_obj) -> None:
    if file_obj and file_obj.name != st.session_state.get("uploaded_file_name"):
        sigs, warns = process_file(file_obj, file_obj.name)
        st.session_state["parsed_signals"] = sigs
        st.session_state["uploaded_file_name"] = file_obj.name
        if warns:
            for w in warns:
                st.warning(w)
        st.rerun()


def _handle_param_upload(file_obj) -> None:
    if file_obj:
        summary, _ = parse_param_file(file_obj)
        st.session_state["param_summary"] = summary


def main() -> None:
    st.set_page_config(
        page_title="DroneAcharya — UAV Diagnostic Suite",
        page_icon="🚁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()
    _load_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Navigate")
        st.page_link("app.py",                          label="🏠  Home")
        st.page_link("pages/1_Health_Monitoring.py",    label="🔬  Health Monitoring")
        st.page_link("pages/2_Mission_Feasibility.py",  label="🗺️  Mission Feasibility")
        st.page_link("pages/3_AboutUs.py",              label="ℹ️  About Us")
        st.markdown("---")

        st.markdown("**Upload Files**")
        sb_log = st.file_uploader(
            "Flight log (.bin / .csv)",
            type=["bin", "csv", "log", "txt"],
            key="sb_log",
        )
        _handle_log_upload(sb_log)

        sb_param = st.file_uploader(
            "Parameter file (.param) — required",
            type=["param", "txt"],
            key="sb_param",
        )
        _handle_param_upload(sb_param)

        st.markdown("---")
        if st.session_state.get("uploaded_file_name"):
            st.success(f"📁 {st.session_state['uploaded_file_name']}")
        if st.session_state.get("memory"):
            st.info("🧠 Memory: active")

    # ── Hero: logo + brand ────────────────────────────────────────────────────
    logo_path = _ROOT / "assests" / "logo.webp"
    hero_left, hero_right = st.columns([1, 3], vertical_alignment="center")
    with hero_left:
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
    with hero_right:
        st.markdown(
            """
            <div style="padding:0.5rem 0">
              <div style="font-size:2.6rem;font-weight:900;line-height:1.1;color:#0f172a">
                DroneAcharya
              </div>
              <div style="font-size:1.15rem;font-weight:600;color:#e8534a;margin-top:0.3rem">
                UAV Diagnostic Suite
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Nav cards ─────────────────────────────────────────────────────────────
    st.markdown("### Select a Module")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="nav-card nc-health">
              <span class="nav-card-icon">🔬</span>
              <div class="nav-card-title">Health Monitoring</div>
              <div class="nav-card-desc">
                Vibration analysis · Motor balance · Energy efficiency ·
                Flight-to-flight trend tracking.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_Health_Monitoring.py", label="🔬  Open Health Monitoring", use_container_width=True)

    with c2:
        st.markdown(
            """
            <div class="nav-card nc-mission">
              <span class="nav-card-icon">🗺️</span>
              <div class="nav-card-title">Mission Feasibility</div>
              <div class="nav-card-desc">
                Hover power · Energy budget · Payload vs range ·
                Environmental risk assessment.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/2_Mission_Feasibility.py", label="🗺️  Open Mission Feasibility", use_container_width=True)

    with c3:
        st.markdown(
            """
            <div class="nav-card nc-about">
              <span class="nav-card-icon">ℹ️</span>
              <div class="nav-card-title">About Us</div>
              <div class="nav-card-desc">
                Who we are · What we do · Get in touch.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/3_AboutUs.py", label="ℹ️  Open About Us", use_container_width=True)

    # ── Product photos ────────────────────────────────────────────────────────
    st.markdown("---")

    p1_path = _ROOT / "assests" / "product_1.png"
    p2_path = _ROOT / "assests" / "product_2.png"
    pc1, pc2 = st.columns(2)
    with pc1:
        if p1_path.exists() and p1_path.stat().st_size > 100:
            st.image(str(p1_path), use_container_width=True)
    with pc2:
        if p2_path.exists() and p2_path.stat().st_size > 100:
            st.image(str(p2_path), use_container_width=True)

    # ── Quick-start steps ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Quick Start")

    s1, s2, s3, s4 = st.columns(4)
    for col, num, icon, title, desc in [
        (s1, "Step 1", "📤", "Upload Log",         "Use the sidebar to upload your .bin or .csv flight log"),
        (s2, "Step 2", "📋", "Fill Questionnaire",  "Frame class, battery, prop, motor KV — all mandatory"),
        (s3, "Step 3", "⚙️", "Run Analysis",        "Physics pipeline: condition → reconstruct → diagnose → score"),
        (s4, "Step 4", "💾", "Save Memory",         "Download .json memory to enable baseline comparison next flight"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                  <div class="step-num">{num}</div>
                  <div class="step-icon">{icon}</div>
                  <div class="step-title">{title}</div>
                  <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("DroneAcharya UAV Diagnostic Suite v1.0")


if __name__ == "__main__":
    main()
