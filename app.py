import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Config
# --------------------------------------------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

BASE_DIR = Path(__file__).parent
CADS_FILE = BASE_DIR / "CADS.csv"

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_cads(path):
    return pd.read_csv(path, low_memory=False)

cads_df = load_cads(CADS_FILE)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def normalize_text(s):
    return str(s).lower().strip()

def smart_vehicle_match(df, user_text, min_ratio=0.5):
    tokens = [t for t in normalize_text(user_text).split() if len(t) > 1]
    if not tokens:
        return df.iloc[0:0]

    searchable = (
        df["MODEL_YEAR"].astype(str) + " " +
        df["DIVISION_NAME"].fillna("") + " " +
        df["MODEL_NAME"].fillna("") + " " +
        df["TRIM"].fillna("") + " " +
        df["STYLE_NAME"].fillna("")
    ).str.lower()

    def score(text):
        hits = sum(token in text for token in tokens)
        return hits / len(tokens)

    scores = searchable.apply(score)
    return df[scores >= min_ratio]

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

search_clicked = st.button("Search Vehicles")

# --------------------------------------------------
# Search Logic
# --------------------------------------------------
if search_clicked:

    if not vehicle_input.strip():
        st.warning("Please enter a vehicle description.")
        st.stop()

    matches_df = smart_vehicle_match(cads_df, vehicle_input)

    if matches_df.empty:
        st.error("No matching vehicles found.")
        st.stop()

    st.subheader("Applicable Vehicle Lines")

    display_df = matches_df[
        [
            "MODEL_YEAR",
            "DIVISION_NAME",
            "MODEL_NAME",
            "TRIM",
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
            "TRIM": "Trim",
            "STYLE_NAME": "Style",
            "AD_MFGCODE": "Model Code",
            "STYLE_ID": "Style ID",
        },
        inplace=True,
    )

    # Add checkbox column
    display_df.insert(0, "Select", False)

    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(required=False)
        },
        disabled=[
            "Year",
            "Make",
            "Model",
            "Trim",
            "Style",
            "Model Code",
            "Style ID",
        ],
    )

    selected_rows = edited_df[edited_df["Select"]]

    st.markdown("---")

    if st.button("Submit Mapping"):
        if selected_rows.empty:
            st.warning("Please select at least one vehicle line to map.")
        else:
            st.success(f"{len(selected_rows)} vehicle line(s) mapped successfully.")

            st.subheader("Mapped Vehicles")
            st.dataframe(
                selected_rows.drop(columns=["Select"]),
                use_container_width=True,
            )
