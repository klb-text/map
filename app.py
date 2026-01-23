import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os
import csv

# ------------------------------
# File paths
# ------------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "mappings.csv"

# ------------------------------
# Load CSV safely
# ------------------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

# ------------------------------
# Load CADS and vehicle reference
# ------------------------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
if not vehicle_ref_df.empty:
    vehicle_ref_df.columns = [col.strip() for col in vehicle_ref_df.columns]

# ------------------------------
# Load existing mappings
# ------------------------------
if os.path.exists(MAPPINGS_FILE):
    mappings_df = load_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["Vehicle", "AD_VEH_ID", "AD_YEAR", "AD_MAKE", "AD_MODEL", "AD_TRIM", "AD_MFGCODE", "STYLE_ID"])

# ------------------------------
# Helper functions
# ------------------------------
def get_make_model_from_example(vehicle_name):
    if vehicle_ref_df.empty:
        return None, None
    ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_name.lower()]
    if not ref_row.empty:
        return ref_row.iloc[0]['Make'], ref_row.iloc[0]['Model']
    return None, None

def smart_vehicle_match(df, vehicle_name, limit=10):
    if df.empty:
        return pd.DataFrame(), []

    # Create a search column combining relevant fields
    df['vehicle_search'] = df[['AD_YEAR','AD_MAKE','AD_MODEL','AD_TRIM']].astype(str).agg(' '.join, axis=1)
    choices = df['vehicle_search'].tolist()
    results = process.extract(vehicle_name, choices, scorer=fuzz.token_sort_ratio, limit=limit)
    
    matched_indices = [r[2] for r in results]
    matched_df = df.iloc[matched_indices].copy()
    return matched_df, results

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# YMMT optional filters
with st.expander("YMMT Filter (optional)"):
    filter_year = st.text_input("Year")
    filter_make = st.text_input("Make")
    filter_model = st.text_input("Model")
    filter_trim = st.text_input("Trim")

if st.button("Search Vehicle"):
    if not vehicle_input:
        st.warning("Please enter a vehicle name.")
    else:
        # Try to get make/model from vehicle_example
        example_make, example_model = get_make_model_from_example(vehicle_input)

        matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input)

        # Apply example make/model filter if found
        if example_make:
            matches_df = matches_df[matches_df['AD_MAKE'].str.lower() == example_make.lower()]
        if example_model:
            matches_df = matches_df[matches_df['AD_MODEL'].str.lower() == example_model.lower()]

        # Apply YMMT filters if entered
        if filter_year:
            matches_df = matches_df[matches_df['AD_YEAR'].astype(str).str.contains(filter_year)]
        if filter_make:
            matches_df = matches_df[matches_df['AD_MAKE'].str.contains(filter_make, case=False, na=False)]
        if filter_model:
            matches_df = matches_df[matches_df['AD_MODEL'].str.contains(filter_model, case=False, na=False)]
        if filter_trim:
            matches_df = matches_df[matches_df['AD_TRIM'].str.contains(filter_trim, case=False, na=False)]

        if matches_df.empty:
            st.info("No matching vehicles found.")
        else:
            st.subheader("Matching Vehicles")
            # Display table with checkboxes for selection
            matches_df = matches_df.reset_index(drop=True)
            selected_rows = []
            for idx, row in matches_df.iterrows():
                if st.checkbox(f"{row['AD_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']}", key=f"chk_{idx}"):
                    selected_rows.append(row)

            # Submit Mapping
            if st.button("Submit Mapping"):
                if not selected_rows:
                    st.warning("Please select at least one vehicle to map.")
                else:
                    for row in selected_rows:
                        mappings_df = pd.concat([mappings_df, pd.DataFrame([{
                            "Vehicle": vehicle_input,
                            "AD_VEH_ID": row.get("AD_VEH_ID",""),
                            "AD_YEAR": row.get("AD_YEAR",""),
                            "AD_MAKE": row.get("AD_MAKE",""),
                            "AD_MODEL": row.get("AD_MODEL",""),
                            "AD_TRIM": row.get("AD_TRIM",""),
                            "AD_MFGCODE": row.get("AD_MFGCODE",""),
                            "STYLE_ID": row.get("STYLE_ID","")
                        }])], ignore_index=True)
                    # Save locally
                    mappings_df.to_csv(MAPPINGS_FILE, index=False)
                    st.success("Mapping saved!")
                    st.dataframe(mappings_df.tail(10))
