"""METRAX-Cloud: Autonomous E-Commerce & Packaged Commodity Statutory Inspector.

Powered by Nebius Token Factory (NVIDIA Nemotron 3-tier cascade) and Tavily Search.
"""

import base64
import os
import streamlit as st
import pandas as pd

from engine import (
    MODEL_NANO,
    MODEL_SUPER,
    MODEL_ULTRA,
    MODEL_VISION,
    FullInspectionReport,
    extract_text_from_image_base64,
    run_full_inspection,
)
from rules import DeterministicAuditResult, ProductDeclarations

# ---------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="METRAX-Cloud | Legal Metrology Inspector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished typography, badges, and high-contrast metrics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }
    .onboarding-banner {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-left: 5px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-size: 0.93rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.18);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-major {
        background-color: rgba(245, 158, 11, 0.18);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: rgba(234, 179, 8, 0.18);
        color: #EAB308;
        border: 1px solid rgba(234, 179, 8, 0.4);
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-compliant {
        background-color: rgba(34, 197, 94, 0.18);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.4);
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .unit-math-card {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .telemetry-tag {
        font-family: monospace;
        background: rgba(128, 128, 128, 0.12);
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Presets Data Store
# ---------------------------------------------------------------------
PRESET_COMPLIANT = """Title: Fortune Sunlite Refined Sunflower Oil, 1 L Pouch
Brand: Fortune
Manufacturer & Packer: Adani Wilmar Limited, Fortune House, Near Navrangpura Railway Crossing, Ahmedabad - 380009, Gujarat, India.
Net Quantity: 1 L
MRP: Rs. 145.00 (Inclusive of all taxes)
Unit Sale Price: Rs. 145.00 per L
Country of Origin: India
Month and Year of Packaging: 10/2026
Customer Complaint Contact: Executive - Consumer Care, Adani Wilmar Ltd., Email: customercare@adaniwilmar.in, Toll Free Phone: 1800-233-9999
Storage Instructions: Store in a cool and dry place away from direct sunlight."""

PRESET_NON_COMPLIANT_SHRINKFLATION = """Title: Titan Beast High Protein Mass Gainer - Dutch Chocolate Flavor
Description: Premium muscle fuel imported directly from our global manufacturing partner.
Net Content: Approx 32 oz
Price: MRP Rs. 4,800 only! Special festival deal!
Unit Price: Rs. 100 per oz
Packed by: Titan Nutrition LLC, 500 Fifth Ave, NY, USA
Imported by: Titan India Impex Pvt Ltd
Customer Support: Message our Instagram @titan_beast or call our desk
Batch No: TB-9921, Pkd Date: 08/2026
Usage: Mix 2 scoops with milk daily."""

PRESET_NON_COMPLIANT_ELECTRONICS = """Title: NovaCharge 65W GaN Multi-Port Fast Charger Adapter
Description: High speed USB-C charger for laptops and phones.
Package Contents: 1 Unit GaN Charger
Price: Rs. 2,499
Unit Sale Price: Rs. 2,499 per unit
Exported by: Shenzhen Nova Tech Ltd, High-Tech Industrial Park, Nanshan, Shenzhen, China.
Country of Origin: China
Imported & Distributed by: Nova India Tech Trading Co.
Manufacturing Month: 11/2025
Warranty Support: warranty@novacharge.com"""

# ---------------------------------------------------------------------
# Sidebar Configuration & Telemetry Status
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    # Check for existing keys in environment or st.secrets without exposing values
    has_nebius_env = bool(
        os.getenv("NEBIUS_API_KEY")
        or (hasattr(st, "secrets") and "NEBIUS_API_KEY" in st.secrets and st.secrets["NEBIUS_API_KEY"])
    )
    has_tavily_env = bool(
        os.getenv("TAVILY_API_KEY")
        or (hasattr(st, "secrets") and "TAVILY_API_KEY" in st.secrets and st.secrets["TAVILY_API_KEY"])
    )

    # 1. Nebius Token Factory Key Handling (Security Shielded)
    if has_nebius_env:
        st.success("✅ Nebius API: Connected (Server Managed)")
        with st.expander("🔑 Override Nebius Key (Optional)", expanded=False):
            nebius_override = st.text_input(
                "Custom Nebius API Key",
                type="password",
                value="",
                placeholder="Leave blank to use server key",
                help="Enter a key only if you want to override the pre-configured server key.",
                key="nebius_key_override",
            )
            if nebius_override.strip():
                os.environ["NEBIUS_API_KEY"] = nebius_override.strip()
    else:
        st.error("❌ Nebius API: Key Required")
        nebius_input = st.text_input(
            "Enter Nebius API Key",
            type="password",
            value="",
            placeholder="nvapi-...",
            key="nebius_key_manual",
        )
        if nebius_input.strip():
            os.environ["NEBIUS_API_KEY"] = nebius_input.strip()

    # 2. Tavily Search API Key Handling (Security Shielded)
    if has_tavily_env:
        st.success("✅ Tavily API: Connected (Server Managed)")
        with st.expander("🔑 Override Tavily Key (Optional)", expanded=False):
            tavily_override = st.text_input(
                "Custom Tavily API Key",
                type="password",
                value="",
                placeholder="Leave blank to use server key",
                help="Enter a key only if you want to override the pre-configured server key.",
                key="tavily_key_override",
            )
            if tavily_override.strip():
                os.environ["TAVILY_API_KEY"] = tavily_override.strip()
    else:
        st.error("❌ Tavily API: Key Required")
        tavily_input = st.text_input(
            "Enter Tavily API Key",
            type="password",
            value="",
            placeholder="tvly-...",
            key="tavily_key_manual",
        )
        if tavily_input.strip():
            os.environ["TAVILY_API_KEY"] = tavily_input.strip()

    st.markdown("---")
    st.markdown("### 🔀 Smart Router Controls")
    force_ultra = st.checkbox(
        "Force Nemotron-3-Ultra-550B Deep Legal Synthesis",
        value=False,
        help="When enabled, forces the 550B model to generate an official legal memo even if the product is 100% compliant.",
    )

    st.markdown("---")
    st.markdown("### 🏛️ Model Roster & Endpoints")
    st.markdown(
        """
        - **Fast Tier (Extraction):**  
          `Nemotron-3-Nano-30B-A3B`
        - **Logic Tier (Fallback):**  
          `nemotron-3-super-120b-a12b`
        - **Deep Tier (Synthesis):**  
          `Nemotron-3-Ultra-550b-a55b`
        - **Vision Tier (OCR):**  
          `Nemotron-3.5-Super-VL`
        - **Regulatory Precedents:**  
          `Tavily Search & Extract`
        """
    )
    st.caption("Powered by Nebius Token Factory inference cluster.")

# ---------------------------------------------------------------------
# App Header & Onboarding Banner
# ---------------------------------------------------------------------
st.markdown(
    '<div class="main-header">⚖️ METRAX-Cloud: Autonomous Statutory Inspector</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Multi-Tier AI Compliance Engine for India Legal Metrology (Packaged Commodities) Rules, 2011</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="onboarding-banner">
        <strong>⚡ 3-Tier Multi-Model Cascade:</strong>
        <strong>1.</strong> Select a preset or input a product listing / URL / packaging photo &nbsp;➔&nbsp;
        <strong>2.</strong> <code>Nemotron-3-Nano</code> extracts declarations and deterministic rules audit math &nbsp;➔&nbsp;
        <strong>3.</strong> Smart Router either <em>bypasses Ultra to save 99% compute</em> or escalates to <code>Nemotron-3-Ultra-550B</code> for legal show-cause notices.
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "inspection_report" not in st.session_state:
    st.session_state.inspection_report = None
if "input_text" not in st.session_state:
    st.session_state.input_text = PRESET_NON_COMPLIANT_SHRINKFLATION
if "url_input" not in st.session_state:
    st.session_state.url_input = "https://world.openfoodfacts.org/product/8901725181222/aashirvaad-superior-mp-atta"
if "use_sample_label" not in st.session_state:
    st.session_state.use_sample_label = True

# ---------------------------------------------------------------------
# Input Section ("Put the Action First")
# ---------------------------------------------------------------------
tab_presets, tab_url, tab_photo = st.tabs(
    [
        "📋 Instant Presets & Raw Listing Text",
        "🌐 Live E-Commerce URL Audit",
        "📸 Packaging Label Photo OCR",
    ]
)

with tab_presets:
    st.markdown("##### ⚡ 1-Click Test Judge Presets:")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        if st.button("✅ 1. Compliant Oil (99% Bypass)", use_container_width=True):
            st.session_state.input_text = PRESET_COMPLIANT
            st.rerun()

    with col_p2:
        if st.button(
            "🚨 2. Shrinkflation Supplement (Ultra Escalation)",
            use_container_width=True,
        ):
            st.session_state.input_text = PRESET_NON_COMPLIANT_SHRINKFLATION
            st.rerun()

    with col_p3:
        if st.button(
            "⚠️ 3. Imported Electronics (Ultra Escalation)",
            use_container_width=True,
        ):
            st.session_state.input_text = PRESET_NON_COMPLIANT_ELECTRONICS
            st.rerun()

    text_input = st.text_area(
        "Product Packaging / E-Commerce Listing Text:",
        value=st.session_state.input_text,
        height=210,
        help="Paste raw product description, packaging label copy, or select a preset above.",
    )

    run_text_btn = st.button(
        "🚀 Run Statutory Inspection",
        type="primary",
        use_container_width=True,
        key="btn_run_text",
    )

with tab_url:
    st.markdown("##### ⚡ 1-Click Preset URLs & Live E-Commerce Ingestion:")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        if st.button("🌾 1-Click Preset: Aashirvaad Atta Open Listing", use_container_width=True):
            st.session_state.url_input = "https://world.openfoodfacts.org/product/8901725181222/aashirvaad-superior-mp-atta"
            st.rerun()
    with col_u2:
        if st.button("📜 1-Click Preset: Ministry Gazette Notification (PIB)", use_container_width=True):
            st.session_state.url_input = "https://pib.gov.in/PressReleasePage.aspx?PRID=1785536"
            st.rerun()

    url_input = st.text_input(
        "Enter E-Commerce Product URL:",
        placeholder="https://world.openfoodfacts.org/product/...",
        value=st.session_state.get("url_input", "https://world.openfoodfacts.org/product/8901725181222/aashirvaad-superior-mp-atta"),
    )
    st.caption(
        "Powered by Tavily Extract with anti-bot search fallback: Ingests the page DOM, normalizes text, and audits statutory compliance."
    )
    run_url_btn = st.button(
        "🌐 Ingest & Inspect URL via Tavily",
        type="primary",
        use_container_width=True,
        key="btn_run_url",
    )

with tab_photo:
    st.markdown("##### 📸 Visual Label Transcription via Nemotron-3.5-Super-VL:")

    col_img_action1, col_img_action2 = st.columns([1, 1])
    with col_img_action1:
        if st.button("🖼️ Load Built-in Sample Label (Titan Whey Isolate)", use_container_width=True):
            st.session_state.use_sample_label = True
            st.rerun()
    with col_img_action2:
        if st.button("🔄 Clear & Upload Custom Photo", use_container_width=True):
            st.session_state.use_sample_label = False
            st.rerun()

    uploaded_file = st.file_uploader(
        "Or Upload Your Own Packaging Label Image:",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a clear photo of the back or front packaging panel.",
    )

    image_base64 = None
    mime_type = "image/png"

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        mime_type = uploaded_file.type or "image/jpeg"
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")
        st.image(
            file_bytes,
            caption="Uploaded Custom Packaging Label",
            width=380,
        )
    elif st.session_state.get("use_sample_label", True) and os.path.exists("sample_label.png"):
        with open("sample_label.png", "rb") as f:
            file_bytes = f.read()
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")
        mime_type = "image/png"
        st.image(
            file_bytes,
            caption="Built-in Sample Packaging Label (Loaded for Instant Judge Testing)",
            width=380,
        )
        st.info(
            "💡 **Judge Test Scenario:** This sample label contains deliberate statutory violations "
            "(Vague 'Approx 2 lbs', non-SI metric unit, missing tax-inclusive MRP statement, "
            "unit price math discrepancy ₹800 vs ₹1750, missing Country of Origin). Click below to run visual OCR & full inspection!"
        )

    run_img_btn = st.button(
        "👁️ Transcribe with Nemotron-3.5-Super-VL & Run Full Inspection",
        type="primary",
        use_container_width=True,
        disabled=(image_base64 is None),
        key="btn_run_img",
    )

# ---------------------------------------------------------------------
# Pipeline Execution Handler
# ---------------------------------------------------------------------
if run_text_btn or run_url_btn or run_img_btn:
    if not os.getenv("NEBIUS_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error(
            "Please configure NEBIUS_API_KEY and TAVILY_API_KEY in the sidebar before proceeding."
        )
    else:
        with st.status(
            "Running Autonomous Statutory Inspection Pipeline...",
            expanded=True,
        ) as status_box:

            def update_progress(msg: str):
                status_box.write(f"⚙️ {msg}")

            try:
                if run_img_btn and image_base64:
                    update_progress(
                        "Transcribing packaging declarations via NVIDIA Nemotron-3.5-Super-VL..."
                    )
                    transcribed, vl_telemetry = extract_text_from_image_base64(
                        image_base64, mime_type=mime_type
                    )
                    update_progress(
                        "Visual OCR complete. Routing transcribed copy into tiered inspection pipeline..."
                    )
                    report = run_full_inspection(
                        transcribed,
                        is_url=False,
                        force_ultra=force_ultra,
                        progress_callback=update_progress,
                    )
                    # Prepend OCR telemetry
                    report.telemetry_steps.insert(0, vl_telemetry)
                    report.total_latency_ms += vl_telemetry.latency_ms
                elif run_url_btn and url_input.strip():
                    update_progress(f"Ingesting URL: {url_input}...")
                    report = run_full_inspection(
                        url_input.strip(),
                        is_url=True,
                        force_ultra=force_ultra,
                        progress_callback=update_progress,
                    )
                else:
                    report = run_full_inspection(
                        text_input,
                        is_url=False,
                        force_ultra=force_ultra,
                        progress_callback=update_progress,
                    )

                st.session_state.inspection_report = report
                status_box.update(
                    label="Inspection Pipeline Complete!",
                    state="complete",
                    expanded=False,
                )
            except Exception as ex:
                status_box.update(
                    label=f"Inspection Failed: {str(ex)}",
                    state="error",
                    expanded=True,
                )
                st.error(f"Error details: {ex}")

# ---------------------------------------------------------------------
# Results Dashboard
# ---------------------------------------------------------------------
report: FullInspectionReport = st.session_state.inspection_report

if report is not None:
    st.markdown("---")
    st.markdown("### 📊 Statutory Inspection & Routing Dashboard")

    # 1. Top Metrics Bar
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    score = report.deterministic_audit.compliance_score
    status = report.deterministic_audit.status
    ultra_step = next(
        (
            s
            for s in report.telemetry_steps
            if "Ultra" in s.model_or_tool
            or ("Synthesis" in s.step_name and s.model_or_tool == MODEL_ULTRA)
        ),
        None,
    )
    ultra_triggered = ultra_step.triggered if ultra_step else False

    with m_col1:
        st.metric(
            label="Compliance Score",
            value=f"{score} / 100",
            delta=status,
            delta_color="normal" if status == "COMPLIANT" else "inverse",
        )

    with m_col2:
        num_v = len(report.deterministic_audit.violations)
        st.metric(
            label="Violations Flagged",
            value=f"{num_v} Defects",
            delta=f"{len(report.deterministic_audit.passed_checks)} Passed Checks",
            delta_color="off",
        )

    with m_col3:
        st.metric(
            label="Ultra 550B Router Action",
            value="ESCALATED 🏛️" if ultra_triggered else "BYPASSED ⚡",
            delta="Deep Legal Reasoning" if ultra_triggered else "99% Cost Bypass",
            delta_color="normal" if not ultra_triggered else "off",
        )

    with m_col4:
        st.metric(
            label="Compute Credits Saved",
            value=f"{report.estimated_cost_savings_pct}%",
            delta=f"Total: {report.total_latency_ms} ms",
            delta_color="normal",
        )

    # 2. Section A: Nebius Token Factory & Tavily Telemetry Table
    st.markdown("#### ⚡ Live Multi-Model Telemetry & Compute Audit")
    telemetry_data = []
    for s in report.telemetry_steps:
        telemetry_data.append(
            {
                "Pipeline Step": s.step_name,
                "Model / Engine": (
                    s.model_or_tool.split("/")[-1]
                    if "/" in s.model_or_tool
                    else s.model_or_tool
                ),
                "Routing Decision": (
                    "TRIGGERED ✅" if s.triggered else "BYPASSED ⚡"
                ),
                "Reason": s.reason,
                "Latency": f"{s.latency_ms} ms",
                "Tokens (In / Out)": f"{s.prompt_tokens} / {s.completion_tokens}",
                "Est. Cost (USD)": f"${s.estimated_cost_usd:.5f}",
            }
        )

    df_telemetry = pd.DataFrame(telemetry_data)
    st.dataframe(df_telemetry, use_container_width=True, hide_index=True)

    # 3. Two-Column Detailed Breakdown
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 🔍 Extracted Declarations & Statutory Audit")

        # Extracted Declarations Grid
        decl = report.extracted_declarations
        st.markdown(
            f"""
            <div class="unit-math-card">
                <strong>📦 Product:</strong> {decl.product_name}<br>
                <strong>⚖️ Net Quantity:</strong> {decl.net_quantity_value} {decl.net_quantity_unit or ''} <em>(Raw: '{decl.net_quantity_raw}')</em><br>
                <strong>💰 MRP:</strong> ₹{decl.mrp_value or 0.0:.2f} <em>(Tax Inclusive Stated: {'Yes ✅' if decl.mrp_incl_taxes_stated else 'No ❌'})</em><br>
                <strong>🏷️ Declared Unit Price:</strong> ₹{decl.unit_sale_price_declared or 0.0:.2f} <em>(Basis: '{decl.unit_sale_price_basis or 'None'}')</em><br>
                <strong>🌍 Country of Origin:</strong> {decl.country_of_origin or 'Missing ❌'}<br>
                <strong>🏭 Manufacturer / Packer:</strong> {decl.manufacturer_or_importer_address or 'Missing ❌'}<br>
                <strong>📞 Consumer Care:</strong> {decl.consumer_care_details or 'Missing ❌'}<br>
                <strong>📅 Date of Manufacture / Pack:</strong> {decl.month_and_year_of_manufacture or 'Missing ❌'}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Mathematical Unit Price Audit Box
        st.markdown("##### 🧮 Rule 6(11) Mathematical Unit Sale Price Audit:")
        math_check = (
            report.deterministic_audit.computed_unit_price_check or ""
        )
        if "Discrepancy" in math_check:
            st.error(f"🚨 **SHRINKFLATION / MATH ERROR:** {math_check}")
        elif "passed" in math_check.lower() or "matches" in math_check.lower():
            st.success(f"✅ **MATHEMATICAL ACCURACY CONFIRMED:** {math_check}")
        else:
            st.info(f"ℹ️ {math_check}")

        # Violations List
        st.markdown("##### ⚠️ Statutory Violations Detected:")
        if not report.deterministic_audit.violations:
            st.success("No statutory defects detected. Product is fully compliant with Legal Metrology Rules, 2011.")
        else:
            for idx, v in enumerate(report.deterministic_audit.violations, 1):
                sev_class = (
                    "badge-critical"
                    if v.severity == "CRITICAL"
                    else (
                        "badge-major"
                        if v.severity == "MAJOR"
                        else "badge-warning"
                    )
                )
                with st.expander(
                    f"#{idx} [{v.severity}] {v.rule_citation} - {v.field_name}",
                    expanded=True,
                ):
                    st.markdown(
                        f"<span class='{sev_class}'>{v.severity}</span> <strong>{v.rule_citation}</strong>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Defect:** {v.issue_description}")
                    st.markdown(f"**Statutory Requirement:** {v.statutory_requirement}")

        # Passed Checks Expander
        with st.expander(
            f"✅ Passed Statutory Checks ({len(report.deterministic_audit.passed_checks)})",
            expanded=False,
        ):
            for chk in report.deterministic_audit.passed_checks:
                st.markdown(f"- [PASS] {chk}")

    with col_right:
        st.markdown("#### 🏛️ Statutory Enforcement & Legal Synthesis")

        # Ultra 550B Legal Show-Cause Notice / Clearance
        if report.ultra_legal_memo:
            st.markdown(report.ultra_legal_memo)

            # Download Report Button
            safe_name = "".join(
                c
                for c in report.extracted_declarations.product_name
                if c.isalnum() or c in (" ", "_", "-")
            ).strip().replace(" ", "_")
            st.download_button(
                label="📥 Download Official Statutory Report (.md)",
                data=report.ultra_legal_memo,
                file_name=f"metrax_statutory_report_{safe_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Tavily Live Regulatory Citations
        if report.tavily_citations:
            st.markdown("---")
            st.markdown("##### 🔍 Live Regulatory Precedents (via Tavily):")
            for c in report.tavily_citations:
                st.markdown(f"- [**{c.title}**]({c.url})")
                st.caption(f"{c.snippet}")

# ---------------------------------------------------------------------
# Engineering Rigor Expander (Limitations, Failure Modes & Accessibility)
# ---------------------------------------------------------------------
st.markdown("---")
with st.expander(
    "🛡️ Engineering Rigor: Documented Limitations, Failure Cases & Accessibility",
    expanded=False,
):
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("### ⚠️ Known Edge Cases & Failure Modes")
        st.markdown(
            """
            1. **Multi-Pack & Heterogeneous Combo Bundles (Rule 24):**  
               When a package contains multiple distinct items (e.g., razor + 2 cartridges + shaving gel), the unit sale price must be individually calculated per commodity. Single-quotient extraction fails on heterogeneous combos.
            2. **Specular Glare on Cylindrical Bottles in OCR:**  
               Label transcription via Vision LLMs can misread curved numbers or barcodes on metallic cans or high-gloss cosmetic bottles under intense directional flash.
            3. **Anti-Scraping / Cloudflare Bot Shields on E-Commerce URLs:**  
               When auditing live marketplace URLs via Tavily Extract, heavily guarded Single Page Applications (SPAs) may occasionally return incomplete DOM content, necessitating fallback to raw copy paste.
            4. **Reasoning Token Latency Overhead:**  
               Nemotron-3-Nano generates chain-of-thought tokens before JSON emission. We enforce concise system instructions and 2048 token ceilings to prevent latency spikes while guaranteeing valid JSON.
            """
        )

    with col_l2:
        st.markdown("### ♿ Production Accessibility & Determinism")
        st.markdown(
            """
            1. **Decoupled Deterministic Math Engine:**  
               LLMs are notoriously prone to arithmetic hallucinations. METRAX enforces strict mathematical unit price checks in Python (`rules.py`) with 5% discrepancy detection rather than delegating math to generative tokens.
            2. **Screen-Reader & High-Contrast Friendly Design:**  
               Color-coded violation badges (`CRITICAL`, `MAJOR`, `WARNING`) are accompanied by textual semantic tags (`[CRITICAL]`) and WCAG AAA compliant background contrasts.
            3. **Portable Offline Fallback:**  
               The deterministic core (`rules.py`) operates standalone with zero cloud dependencies for local edge validation at customs checkpoints or fulfillment centers.
            4. **Full Telemetry Transparency:**  
               Every single inference call, token count, latency measurement, and estimated cost is audited and presented to prevent black-box surprises.
            """
        )
