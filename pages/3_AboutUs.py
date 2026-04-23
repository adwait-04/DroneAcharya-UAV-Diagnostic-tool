"""
About Us Page
"""

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="About Us", page_icon="ℹ️", layout="wide")

_css_path = _ROOT / "assests" / "style.css"
if _css_path.exists():
    with open(_css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚁 DroneAcharya")
    st.markdown("---")
    st.markdown("### Navigate")
    st.page_link("app.py",                          label="🏠  Home")
    st.page_link("pages/1_Health_Monitoring.py",    label="🔬  Health Monitoring")
    st.page_link("pages/2_Mission_Feasibility.py",  label="🗺️  Mission Feasibility")
    st.page_link("pages/3_AboutUs.py",              label="ℹ️  About Us")

# ── Content ───────────────────────────────────────────────────────────────────

st.title("About Us")
st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### DroneAcharya Aerial Innovations")
    st.markdown(
        """
DroneAcharya is a Pune-based drone tech leader and the first drone startup to go public in India.
They specialize in high-end UAV solutions, ranging from combat-ready defense systems to enterprise-grade data analytics.

**Training** — A top DGCA-certified pilot training organization in India.

**Defense** — Known for FPV "Kamikaze" drones and tactical labs for the Indian Army.

**Services** — Heavy focus on GIS, 3D mapping, and industrial drone services.

**Recent News** — Expanding globally (DroneEntry) and recently secured a $3.5M+ order book for 2026.
        """
    )

    st.markdown("---")
    st.markdown("### Contact Information")

    st.markdown(
        """
**Headquarters**
1st & 2nd Floor, Galore Tech IT Park, LMD Square, Bavdhan, Pune, Maharashtra 411021

**Phone**
+91 98900 03590 &nbsp;/&nbsp; +91 89564 44677
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    contact_rows = [
        ("General Inquiry", "info@droneacharya.com"),
        ("Training / Courses", "training@droneacharya.com"),
        ("Business / Sales", "sales@droneacharya.com"),
        ("Careers", "talent@droneacharya.com"),
    ]
    rows_html = "".join(
        f"<tr>"
        f"<td style='padding:6px 16px 6px 0;color:#94a3b8;font-size:0.9rem;white-space:nowrap'>{label}</td>"
        f"<td style='padding:6px 0;font-size:0.9rem'>"
        f"<a href='mailto:{email}' style='color:#e94560;text-decoration:none'>{email}</a>"
        f"</td></tr>"
        for label, email in contact_rows
    )
    st.markdown(
        f"<table style='border-collapse:collapse'>{rows_html}</table>",
        unsafe_allow_html=True,
    )

with col_right:
    product_path = _ROOT / "assests" / "product_1.png"
    if product_path.exists() and product_path.stat().st_size > 100:
        st.image(str(product_path), use_container_width=True)
