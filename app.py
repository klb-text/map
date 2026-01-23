import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os
import csv

# --- File paths ---
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# --- Helper functions ---
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, sep=None, engine='python', encoding='utf-8', error_bad_lines=False)

@st.cache_data
def load_vehicle_ref(path):
    return pd.read_csv(path, sep=None, engine='python', encoding='utf-8', error_bad_lines=False)

def smart_vehicle_match(cads_df, vehicle_input, ref_make=None):
    df = cads_df.copy()

    # Pre-filter CADS using ref_make if available
    if ref_make:
        df = df[df['AD_MAKE'].str.lower() == ref_make.lower()]

    # Create combined vehicle string for fuzzy matching
    df['vehicle_search'] = df[['MODEL_YEAR','AD_MAKE','MODEL_NAME','TRIM']].astype(str).agg(' '.join, axis=1)
    choices = df['vehicle_search'].tolist()
    matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=20)
    
    results = []
    for match_str, score, idx in matches:
        row = df.iloc[idx].copy()
        row['match_score'] = score
        results.append(row)
    if results:
        return pd.DataFrame(results), matches
    else:
        return pd.DataFrame(), matches

def save_mappings(selected_rows):
    if not os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['VehicleInput','MODEL_YEAR','AD_MAKE','MODEL_NAME','TRIM','AD_MFGCODE'])
    with open(MAPPINGS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in selected_rows:
            writer.writerow([row['vehicle_input'], row['MODEL_YEAR'], row['AD_MAKE'], row['MODEL_NAME'], row['TRIM'], row['AD_MFGCODE']])

# --- Load data ---
try:
    cads_df = load_csv(CADS_FILE)
except Exception as e:
    st.error(f"Error loading CADS.csv: {e}")
    st.stop()

try:
    vehicle_ref_df = load_vehicle_ref(VEHICLE_REF_FILE)
except Exception as e:
    st.error(f"Error loading vehicle reference file: {e}")
    st.stop()

if not os.path.exists(MAPPINGS_FILE):
    pd.DataFrame(columns=['VehicleInput','MODEL_YEAR','AD_MAKE','MODEL_NAME','TRIM','AD_MFGCODE']).to_csv(MAPPINGS_FILE, index=False)

mappings_df = pd.read_csv(MAPPINGS_FILE)

# --- Streamlit App ---
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

search_clicked = st.button("Search")

# Pre-fill YMMT filters
col1, col2, col3, col4 = st.columns(4)
year_filter = col1.text_input("Year")
make_filter = col2.text_input("Make")
model_filter = col3.text_input("Model")
trim_filter = col4.text_input("Trim")

matched_df = pd.DataFrame()
raw_matches = []

if search_clicked and vehicle_input:
    # Check vehicle_example first
    ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    ref_make = None
    if not ref_row.empty:
        ref_make = ref_row.iloc[0]['Make']

    matched_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, ref_make=ref_make)

    # Apply YMMT filters if provided
    if not matched_df.empty:
        if year_filter:
            matched_df = matched_df[matched_df['MODEL_YEAR'].astype(str) == str(year_filter)]
        if make_filter:
            matched_df = matched_df[matched_df['AD_MAKE'].str.lower() == make_filter.lower()]
        if model_filter:
            matched_df = matched_df[matched_df['MODEL_NAME'].str.lower() == model_filter.lower()]
        if trim_filter:
            matched_df = matched_df[matched_df['TRIM'].str.lower() == trim_filter.lower()]

    if matched_df.empty:
        st.info("No matching vehicles found.")
    else:
        st.subheader("Matching Vehicles")
        selected_rows = []
        for idx, row in matched_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['MODEL_NAME']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']} | Score: {row['match_score']}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_rows.append(row)
        if st.button("Submit Mapping") and selected_rows:
            # Add input vehicle for reference
            for r in selected_rows:
                r['vehicle_input'] = vehicle_input
            save_mappings(selected_rows)
            st.success("Mapping saved!")
