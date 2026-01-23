import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------
# File paths
# ---------------------------
VEHICLE_REF_FILE = "vehicle_example.txt"
CADS_FILE = "CADS.csv"

# ---------------------------
# Load CSV safely
# ---------------------------
@st.cache_data
def load_csv(path, sep='\t'):
    try:
        return pd.read_csv(path, sep=sep, dtype=str)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
cads_df = load_csv(CADS_FILE)

# Ensure required CADS columns exist as strings
required_cads_cols = ['AD_VEH_ID','AD_YEAR','AD_MAKE','AD_MODEL','AD_MFGCODE',
                      'AD_SERIES','AD_TRIM','STYLE_ID','MODEL_YEAR','DIVISION_NAME',
                      'MODEL_NAME','STYLE_NAME','TRIM','STYLE_CODE']
for col in required_cads_cols:
    if col not in cads_df.columns:
        cads_df[col] = ""
    cads_df[col] = cads_df[col].astype(str).fillna('')

# ---------------------------
# Smart vehicle match function
# ---------------------------
def smart_vehicle_match(df, vehicle_input, example_make=None, example_model=None):
    df = df.copy()

    # Pre-filter by make/model if provided
    if example_make and 'AD_MAKE' in df.columns:
        df = df[df['AD_MAKE'].str.lower() == example_make.lower()]
    if example_model and 'AD_MODEL' in df.columns:
        df = df[df['AD_MODEL'].str.lower() == example_model.lower()]

    # Ensure all key columns exist
    key_cols = ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM']
    for col in key_cols:
        if col not in df.columns:
            df[col] = ''
        else:
            # Convert each column to string and replace NaN with empty string
            df[col] = df[col].astype(str).fillna('')

    # Safely create a combined search string
    df['vehicle_search'] = df[key_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Fuzzy matching
    from thefuzz import process, fuzz
    matches = process.extract(vehicle_input, df['vehicle_search'].tolist(), scorer=fuzz.token_sort_ratio, limit=50)

    # Select matches with score >= 60
    matched_indices = [idx for _, score, idx in matches if score >= 60]
    matched_df = df.iloc[matched_indices]

    return matched_df, matches


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# YMMT optional filters
st.subheader("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
ymmt_year = col1.text_input("Year")
ymmt_make = col2.text_input("Make")
ymmt_model = col3.text_input("Model")
ymmt_trim = col4.text_input("Trim")

# Search button
search_clicked = st.button("Search")

if search_clicked and vehicle_input:
    # Try to get make/model from reference
    example_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    if not example_row.empty:
        example_make = example_row.iloc[0].get('Make', None)
        example_model = example_row.iloc[0].get('Model', None)
    else:
        example_make = None
        example_model = None

    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)

    # Apply YMMT filters if provided
    if ymmt_year:
        matches_df = matches_df[matches_df['MODEL_YEAR'] == ymmt_year]
    if ymmt_make:
        matches_df = matches_df[matches_df['AD_MAKE'].str.lower() == ymmt_make.lower()]
    if ymmt_model:
        matches_df = matches_df[matches_df['AD_MODEL'].str.lower() == ymmt_model.lower()]
    if ymmt_trim:
        matches_df = matches_df[matches_df['TRIM'].str.lower() == ymmt_trim.lower()]

    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.subheader("Matching Vehicles")
        # Checkbox selection for multiple vehicles
        selected_indices = []
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_indices.append(idx)

        # Submit mapping button
        if st.button("Submit Mapping"):
            if not selected_indices:
                st.warning("Please select at least one vehicle to map.")
            else:
                selected_df = matches_df.loc[selected_indices, ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']]
                st.success(f"Mapping submitted for {len(selected_df)} vehicle(s).")
                st.dataframe(selected_df)
