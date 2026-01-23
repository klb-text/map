import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from pathlib import Path

# ===================== Page Config =====================
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ===================== Files =====================
BASE_DIR = Path(__file__).parent
CADS_FILE = BASE_DIR / "CADS.csv"
VEHICLE_REF_FILE = BASE_DIR / "vehicle_example.txt"

# ===================== Loaders =====================
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_vehicle_ref(path):
    return pd.read_csv(path, sep="\t")

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_vehicle_ref(VEHICLE_REF_FILE)

# Normalize CADS columns
cads_df.columns = [c.strip() for c in cads_df.columns]

# ===================== Vehicle Input =====================
st.subheader("Enter Vehicle (freeform)")
vehicle_input = st.text_input("Vehicle")

search_clicked = st.button("Search")

matches_df = pd.DataFrame()

# ===================== Smart Match =====================
if search_clicked and vehicle_input.strip():
    vehicle_choices = vehicle_ref_df["Vehicle"].dropna().tolist()

    best = process.extractOne(
        vehicle_input,
        vehicle_choices,
        scorer=fuzz.token_sort_ratio
    )

    if best and best[1] >= 85:
        matched_vehicle = best[0]
        score = best[1]

        st.success(f"Smart match found: {matched_vehicle} (score {score})")

        ref_row = vehicle_ref_df[vehicle_ref_df["Vehicle"] == matched_vehicle].iloc[0]

        year = ref_row.get("Year")
        make = ref_row.get("Make")
        model = ref_row.get("Model")
        trim = ref_row.get("Trim")

        matches_df = cads_df.copy()

        if pd.notna(year):
            matches_df = matches_df[matches_df["MODEL_YEAR"] == int(year)]
        if pd.notna(make):
            matches_df = matches_df[matches_df["DIVISION_NAME"].str.contains(str(make), case=False, na=False)]
        if pd.notna(model):
            matches_df = matches_df[matches_df["MODEL_NAME"].str.contains(str(model), case=False, na=False)]
        if pd.notna(trim) and str(trim).strip():
            matches_df = matches_df[
                matches_df["STYLE_NAME"].str.contains(str(trim), case=False, na=False)
            ]

    else:
        st.warning("No strong vehicle match found. Use YMMT filters below.")

# ===================== YMMT Fallback =====================
st.subheader("YMMT Filter (optional)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    year_filter = st.selectbox("Year", [""] + sorted(cads_df["MODEL_YEAR"].dropna().unique().astype(int).tolist()))
with col2:
    make_filter = st.selectbox("Make", [""] + sorted(cads_df["DIVISION_NAME"].dropna().unique().tolist()))
with col3:
    model_filter = st.selectbox("Model", [""] + sorted(cads_df["MODEL_NAME"].dropna().unique().tolist()))
with col4:
    trim_filter = st.text_input("Trim")

if year_filter or make_filter or model_filter or trim_filter:
    filtered = cads_df.copy()

    if year_filter:
        filtered = filtered[filtered["MODEL_YEAR"] == int(year_filter)]
    if make_filter:
        filtered = filtered[filtered["DIVISION_NAME"] == make_filter]
    if model_filter:
        filtered = filtered[filtered["MODEL_NAME"] == model_filter]
    if trim_filter:
        filtered = filtered[filtered["STYLE_NAME"].str.contains(trim_filter, case=False, na=False)]

    matches_df = filtered

# ===================== Results Table =====================
st.subheader("Applicable Vehicle Lines")

if not matches_df.empty:
    display_df = matches_df[
        [
            "MODEL_YEAR",
            "DIVISION_NAME",
            "MODEL_NAME",
            "STYLE_NAME",
            "AD_MFGCODE",
            "STYLE_ID",
        ]
    ].copy()

    display_df.rename(
        columns={
            "MODEL_YEAR": "Year",
            "DIVISION_NAME": "Make",
            "MODEL_NAME": "Model",
            "STYLE_NAME": "Trim",
            "AD_MFGCODE": "Model Code",
            "STYLE_ID": "Style ID",
        },
        inplace=True,
    )

    display_df.insert(0, "Select", False)

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(required=True)
        }
    )

    selected = edited_df[edited_df["Select"]]

    st.write(f"Selected rows: {len(selected)}")

    if not selected.empty:
        st.dataframe(
            selected.drop(columns=["Select"]),
            use_container_width=True
        )

        # ===================== Submit Mapping =====================
        st.markdown("---")
        st.subheader("Submit Mapping")

        if st.button("Submit Mapping"):
            mapping_payload = {
                "vehicle_input": vehicle_input,
                "mapped_rows": selected.drop(columns=["Select"]).to_dict(orient="records")
            }

            st.success("Mapping submitted successfully.")
            st.json(mapping_payload)

else:
    st.info("No vehicle lines to display.")
