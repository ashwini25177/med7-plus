"""
Med7-Plus Streamlit Interactive Web Application.
Provides a modern clinical dashboard for visualizing named entities,
inspecting linked medication events, checking negation/certainty, and exporting to FHIR.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path so med7_plus is always importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import json
import html
from med7_plus.extractor import Med7ClinicalExtractor
from med7_plus.config import ENTITY_COLORS, ALL_ENTITIES

# Page configuration
st.set_page_config(
    page_title="Med7-Plus: Clinical Medication Information Extraction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom medical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3799;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .entity-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.15rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .med-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .status-negated {
        color: #eb4d4b;
        font-weight: bold;
    }
    .status-active {
        color: #20bf6b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize extractor (cached)
@st.cache_resource
def get_extractor():
    return Med7ClinicalExtractor()

extractor = get_extractor()

# Sample clinical texts for instant exploration
SAMPLE_TEXTS = {
    "Cardiology Discharge": (
        "Patient admitted with acute decompensated heart failure. Discharge medications: "
        "Furosemide 40 mg tablet PO once daily for 14 days. Metoprolol succinate 25 mg PO daily. "
        "Discontinued Lisinopril due to persistent cough. Patient instructed to take Aspirin 81 mg oral daily."
    ),
    "Endocrinology Note": (
        "Assessment: Type 2 Diabetes with elevated HbA1c (8.8%). "
        "Initiate Metformin 500 mg PO BID with meals. Continue Lantus 20 units subcutaneous at bedtime. "
        "Patient denies taking any sulfonylureas. Avoid SGLT2 inhibitors at present."
    ),
    "Infectious Disease & IV Antibiotics": (
        "Patient completed 7 days of IV Vancomycin 1 g IV q12h for MRSA bacteremia. "
        "Step-down regimen: Amoxicillin 500 mg capsule PO TID for 10 days to treat residual infection."
    ),
    "ICU Critical Care & Pain": (
        "Started on Heparin continuous IV infusion at 18 units/kg/hr. Patient denies taking Warfarin at home. "
        "Morphine 2 mg IV q4h PRN for severe chest pain."
    )
}

# Sidebar
st.sidebar.title("💊 Med7-Plus Settings")
st.sidebar.markdown("**Clinical Information Extraction System**")

selected_sample = st.sidebar.selectbox("Select a Sample Note:", list(SAMPLE_TEXTS.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Supported Entities")
for ent, color in ENTITY_COLORS.items():
    st.sidebar.markdown(
        f"<span style='background-color:{color}; color:white; padding:3px 8px; border-radius:4px; font-weight:bold;'>{ent}</span>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **About Med7-Plus:**\n"
    "An enhanced implementation of Med7 (Kormilitzin et al., 2021) "
    "incorporating context extraction (NegEx/n2c2), RxNorm code normalization, "
    "and FHIR R4 interoperability."
)

# Header
st.markdown("<div class='main-header'>🏥 Med7-Plus Clinical Information Extractor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Extract medication concepts, link dosage attributes, "
    "detect discontinued status, and generate HL7 FHIR statements from free-text EHR narratives.</div>",
    unsafe_allow_html=True
)

# Main input text area
text_input = st.text_area(
    "Clinical Note / Electronic Health Record Narrative:",
    value=SAMPLE_TEXTS[selected_sample],
    height=140
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_button = st.button("🚀 Extract Information", type="primary")

if run_button or text_input:
    result = extractor.extract(text_input)
    entities = result["raw_entities"]
    prescriptions = result["prescriptions"]
    fhir_bundle = result["fhir_bundle"]

    tab_visual, tab_table, tab_fhir, tab_raw = st.tabs([
        "🎨 Highlighted Entities",
        "📊 Structured Medication Table",
        "🔥 HL7 FHIR Bundle",
        "💻 JSON Response"
    ])

    # Tab 1: Highlighted Entities Visualizer
    with tab_visual:
        st.subheader("Named Entity Recognition (NER)")
        
        # Render highlighted HTML
        highlighted_html = ""
        last_idx = 0
        sorted_entities = sorted(entities, key=lambda x: x["start"])

        for ent in sorted_entities:
            start, end, label, text = ent["start"], ent["end"], ent["label"], ent["text"]
            highlighted_html += html.escape(text_input[last_idx:start])
            color = ENTITY_COLORS.get(label, "#333333")
            highlighted_html += (
                f"<mark style='background-color:{color}; color:white; padding:0.2em 0.4em; "
                f"margin:0 0.1em; border-radius:0.35em; font-weight:bold;'>"
                f"{html.escape(text)} "
                f"<span style='font-size:0.7em; text-transform:uppercase; opacity:0.85; margin-left:3px;'>{label}</span>"
                f"</mark>"
            )
            last_idx = end
        highlighted_html += html.escape(text_input[last_idx:])

        st.markdown(
            f"<div style='background-color:#ffffff; border:1px solid #dcdde1; border-radius:8px; padding:1.2rem; font-size:1.1rem; line-height:2.0;'>{highlighted_html}</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"**Total Extracted Entities:** {len(entities)}")

    # Tab 2: Structured Medication Table
    with tab_table:
        st.subheader("Resolved Medication Events (Relation Linked)")
        if not prescriptions:
            st.warning("No explicit DRUG entities detected in narrative.")
        else:
            table_data = []
            for p in prescriptions:
                drug = p.get("drug", {})
                norm = drug.get("normalization", {})
                ctx = drug.get("context", {})

                table_data.append({
                    "Drug": drug.get("text", "N/A"),
                    "Generic (RxNorm)": f"{norm.get('generic_name', 'N/A')} (CUI: {norm.get('rxcui', 'N/A')})",
                    "Strength": p["strength"]["text"] if p.get("strength") else "—",
                    "Route": p["route"]["text"] if p.get("route") else "—",
                    "Frequency": p["frequency"]["text"] if p.get("frequency") else "—",
                    "Duration": p["duration"]["text"] if p.get("duration") else "—",
                    "Form": p["form"]["text"] if p.get("form") else "—",
                    "Status": "❌ DISCONTINUED / NEGATED" if ctx.get("is_negated") else "✅ ACTIVE",
                    "Certainty": ctx.get("certainty", "CONFIRMED")
                })

            try:
                st.dataframe(table_data, width="stretch")
            except TypeError:
                st.dataframe(table_data, use_container_width=True)

    # Tab 3: FHIR Bundle
    with tab_fhir:
        st.subheader("HL7 FHIR R4 MedicationStatement Collection")
        st.caption("Standardized Health IT interoperability payload compliant with FHIR R4 specifications.")
        st.json(fhir_bundle)

    # Tab 4: Raw JSON Response
    with tab_raw:
        st.subheader("Raw Pipeline Result")
        st.json(result)
