import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# --- File paths ---
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# --- Load CSV with safe handling ---
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path, dtype=str, on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
mappings_df = load_csv(MAPPINGS_FILE)

# --- Ensure columns exist ---
for col in ["AD_YEAR","AD_MAKE","AD_MODEL","AD_TRIM","STYLE_ID","AD_MFGCODE","MODEL_YEAR","MAKE","MODEL_NAME","TRIM"]:
    if col not in cads_df.columns:
        cads_df[col] = ""

# --- Build a combined search field for fuzzy matching ---
cads_df['vehicle_search'] = cads_df[['MODEL_YEAR','AD_MAKE','MODEL_NAME','AD_TRIM']].astype(str).agg(' '.join, axis=1)

# --- Fuzzy vehicle match ---
def smart_vehicle_match(df, input_vehicle, limit=20):
    choices = df['vehicle_search'].tolist()
    matches = process.extract(
        input_vehicle, choices,
        scorer=fuzz.token_sort_ratio,
        limit=limit
    )
    results = []
    for match_str, score, idx in matches:
        row = df.iloc[idx].copy()
        row['match_score'] = score
        results.append(row)
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("AFF Vehicle Mapping")
vehicle_input = st.text_input("Enter Vehicle (freeform)")

with st.expander("Optional YMMT Filters"):
    year_filter = st.text_input("Year")
    make_filter = st.text_input("Make")
    model_filter = st.text_input("Model")
    trim_filter = st.text_input("Trim")

search_button = st.button("Search")

selected_rows = []

if search_button and vehicle_input:
    matches_df = smart_vehicle_match(cads_df, vehicle_input)

    # --- Apply YMMT filters only if provided ---
    if year_filter:
        matches_df = matches_df[matches_df['MODEL_YEAR'] == year_filter]
    if make_filter:
        matches_df = matches_df[matches_df['AD_MAKE'].str.lower() == make_filter.lower()]
    if model_filter:
        matches_df = matches_df[matches_df['MODEL_NAME'].str.lower() == model_filter.lower()]
    if trim_filter:
        matches_df = matches_df[matches_df['AD_TRIM'].str.lower() == trim_filter.lower()]

    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.write("Select applicable vehicle(s) to map:")
        # --- Multi-select table ---
        selected_rows = []
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['MODEL_NAME']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']} | Score: {row['match_score']}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_rows.append(row)

# --- Submit Mapping ---
if selected_rows:
    if st.button("Submit Mapping"):
        selected_df = pd.DataFrame(selected_rows)
        # Append to Mappings.csv
        if not mappings_df.empty:
            mappings_df = pd.concat([mappings_df, selected_df], ignore_index=True)
        else:
            mappings_df = selected_df
        mappings_df.to_csv(MAPPINGS_FILE, index=False)
        st.success(f"Mapping submitted for {len(selected_rows)} vehicle(s).")
