# app.py
# AFF Vehicle Mapping – Search-gated + Manual Multi-Select + Submit Mapping

import pandas as pd
import streamlit as st
import difflib

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# -------------------------------------------------
# Load CADS
# -------------------------------------------------
@st.cache_data
def load_cads():
    return pd.read_csv("CADS.csv", dtype=str)

cads_df = load_cads()

for col in ["MODEL_YEAR", "DIVISION_NAME", "MODEL_NAME", "TRIM"]:
    if col in cads_df.columns:
        cads_df[col] = cads_df[col].fillna("").str.strip()

# -------------------------------------------------
# Vehicle input
# -------------------------------------------------
vehicle_input = st.text_input("Enter Vehicle (freeform)")

# -------------------------------------------------
# Optional YMMT filters
# -------------------------------------------------
st.subheader("YMMT Filter (optional)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    year = st.selectbox("Year", [""] + sorted(cads_df["MODEL_YEAR"].unique().tolist()))

with col2:
    make = st.selectbox("Make", [""] + sorted(cads_df["DIVISION_NAME"].unique().tolist()))

with col3:
    model = st.selectbox("Model", [""] + sorted(cads_df["MODEL_NAME"].unique().tolist()))

with col4:
    trim = st.selectbox("Trim", [""] + sorted(cads_df["TRIM"].unique().tolist()))

# -------------------------------------------------
# Search trigger
# -------------------------------------------------
search_clicked = st.button("Search Vehicles")

# -------------------------------------------------
# Smart match (hint only)
# -------------------------------------------------
def smart_match(input_text, df):
    choices = (
        df["MODEL_YEAR"] + " " +
        df["DIVISION_NAME"] + " " +
        df["MODEL_NAME"] + " " +
        df["TRIM"]
    ).str.strip()

    matches = difflib.get_close_matches(input_text, choices.tolist(), n=1, cutoff=0.6)

    if matches:
        best = matches[0]
        score = difflib.SequenceMatcher(None, input_text, best).ratio() * 100
        return best, round(score, 1)

    return None, None

# -------------------------------------------------
# Run search ONLY after button click
# -------------------------------------------------
if search_clicked:

    if vehicle_input:
        best_match, score = smart_match(vehicle_input, cads_df)
        if best_match:
            st.success(f"Smart match found: {best_match} (score {score})")

    filtered_df = cads_df.copy()

    if year:
        filtered_df = filtered_df[filtered_df["MODEL_YEAR"] == year]
    if make:
        filtered_df = filtered_df[filtered_df["DIVISION_NAME"] == make]
    if model:
        filtered_df = filtered_df[filtered_df["MODEL_NAME"] == model]
    if trim:
        filtered_df = filtered_df[filtered_df["TRIM"] == trim]

    st.subheader("Applicable Vehicle Lines")

    if filtered_df.empty:
        st.warning("No vehicles match the selected criteria.")
        st.stop()

    display_df = filtered_df[
        [
            "MODEL_YEAR",
            "DIVISION_NAME",
            "MODEL_NAME",
            "TRIM",
            "AD_MFGCODE",
            "STYLE_ID"
        ]
    ].rename(columns={
        "MODEL_YEAR": "Year",
        "DIVISION_NAME": "Make",
        "MODEL_NAME": "Model",
        "TRIM": "Trim",
        "AD_MFGCODE": "Model Code"
    }).reset_index(drop=True)

    selected_rows = []

    hdr = st.columns([0.5, 1, 1.5, 2, 2, 1.2])
    hdr[0].markdown("**Select**")
    hdr[1].markdown("**Year**")
    hdr[2].markdown("**Make**")
    hdr[3].markdown("**Model**")
    hdr[4].markdown("**Trim**")
    hdr[5].markdown("**Model Code**")

    st.divider()

    for idx, row in display_df.iterrows():
        cols = st.columns([0.5, 1, 1.5, 2, 2, 1.2])

        with cols[0]:
            checked = st.checkbox("", key=f"select_{idx}")

        cols[1].write(row["Year"])
        cols[2].write(row["Make"])
        cols[3].write(row["Model"])
        cols[4].write(row["Trim"])
        cols[5].write(row["Model Code"])

        if checked:
            selected_rows.append(row)

    # -------------------------------------------------
    # Submit mapping
    # -------------------------------------------------
    if selected_rows:
        selected_df = pd.DataFrame(selected_rows)

        st.subheader("Selected Vehicles for Mapping")
        st.dataframe(selected_df, use_container_width=True)

        if st.button("Submit Mapping"):
            mapping_payload = {
                "vehicle_input": vehicle_input,
                "mapped_vehicles": selected_df.to_dict(orient="records")
            }

            st.success("Mapping submitted successfully")
            st.json(mapping_payload)
