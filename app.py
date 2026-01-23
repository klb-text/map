import streamlit as st
import pandas as pd

st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

st.title("AFF Vehicle Mapping")

# -----------------------------
# Config
# -----------------------------
CADS_FILE = "CADS.csv"

@st.cache_data
def load_cads(path):
    return pd.read_csv(path)

cads_df = load_cads(CADS_FILE)

# -----------------------------
# Helpers
# -----------------------------
def filter_by_vehicle_text(df, text):
    """
    Token-based, regex-safe vehicle matching
    """
    tokens = text.lower().split()

    search_blob = (
        df["MODEL_YEAR"].astype(str) + " " +
        df["DIVISION_NAME"].fillna("") + " " +
        df["MODEL_NAME"].fillna("") + " " +
        df["STYLE_NAME"].fillna("")
    ).str.lower()

    mask = search_blob.apply(lambda x: all(tok in x for tok in tokens))
    return df[mask]

# -----------------------------
# Inputs
# -----------------------------
vehicle_input = st.text_input(
    "Enter Vehicle (freeform)",
    placeholder="e.g. 2025 TLX TECH 10 Speed Automatic"
)

st.subheader("YMMT Filter (optional)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    year = st.selectbox(
        "Year",
        options=[""] + sorted(cads_df["MODEL_YEAR"].dropna().unique().tolist())
    )

with col2:
    make = st.selectbox(
        "Make",
        options=[""] + sorted(cads_df["DIVISION_NAME"].dropna().unique().tolist())
    )

with col3:
    model = st.selectbox(
        "Model",
        options=[""] + sorted(cads_df["MODEL_NAME"].dropna().unique().tolist())
    )

with col4:
    trim = st.selectbox(
        "Trim",
        options=[""] + sorted(cads_df["TRIM"].dropna().unique().tolist())
    )

search_clicked = st.button("🔍 Search Vehicles")

# -----------------------------
# Search Logic (ONLY runs on click)
# -----------------------------
if search_clicked:
    filtered_df = cads_df.copy()

    # 1️⃣ Vehicle text filter FIRST
    if vehicle_input.strip():
        filtered_df = filter_by_vehicle_text(filtered_df, vehicle_input)

    # 2️⃣ YMMT refinement
    if year:
        filtered_df = filtered_df[filtered_df["MODEL_YEAR"] == year]
    if make:
        filtered_df = filtered_df[filtered_df["DIVISION_NAME"] == make]
    if model:
        filtered_df = filtered_df[filtered_df["MODEL_NAME"] == model]
    if trim:
        filtered_df = filtered_df[filtered_df["TRIM"] == trim]

    if filtered_df.empty:
        st.warning("No matching vehicles found.")
        st.stop()

    st.subheader("Applicable Vehicle Lines")

    display_df = filtered_df[
        [
            "MODEL_YEAR",
            "DIVISION_NAME",
            "MODEL_NAME",
            "TRIM",
            "AD_MFGCODE",
            "STYLE_ID"
        ]
    ].copy()

    display_df.rename(
        columns={
            "DIVISION_NAME": "Make",
            "MODEL_NAME": "Model",
            "TRIM": "Trim",
            "AD_MFGCODE": "Model Code"
        },
        inplace=True
    )

    # Checkbox column
    display_df.insert(0, "Select", False)

    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Submit Mapping
    # -----------------------------
    submit_clicked = st.button("✅ Submit Mapping")

    if submit_clicked:
        selected = edited_df[edited_df["Select"] == True]

        if selected.empty:
            st.warning("Please select at least one vehicle to map.")
        else:
            st.success(f"{len(selected)} vehicle(s) mapped successfully.")
            st.dataframe(
                selected.drop(columns=["Select"]),
                use_container_width=True
            )
