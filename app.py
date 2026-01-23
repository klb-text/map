import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os
import csv

# --- FILE PATHS ---
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "mappings.csv"

# --- LOAD CSV UTILITY ---
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# --- LOAD CADS & VEHICLE REFERENCE ---
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# --- LOAD OR CREATE MAPPINGS ---
if os.path.exists(MAPPINGS_FILE):
    mappings_df = pd.read_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["vehicle_input", "AD_VEH_ID", "AD_YEAR", "AD_MAKE", "AD_MODEL", "AD_TRIM", "Model Code"])

# --- SMART VEHICLE MATCH FUNCTION ---
def smart_vehicle_match(df, vehicle_input, year=None, make=None, model=None, trim=None):
    # Create search string for each CADS row
    df['vehicle_search'] = df[['AD_YEAR','AD_MAKE','AD_MODEL','AD_TRIM']].astype(str).agg(' '.join, axis=1)

    # Fuzzy match
    choices = df['vehicle_search'].tolist()
    results = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=50)
    
    # Get rows with score >=70
    matched_indices = [i for i, (match, score, idx) in enumerate(results) if score >= 70]
    if not matched_indices:
        return pd.DataFrame(columns=df.columns)
    
    matched_df = df.iloc[[idx for _, score, idx in results if score >= 70]].copy()
    
    # Apply optional YMMT filter
    if year:
        matched_df = matched_df[matched_df['AD_YEAR'].astype(str).str.contains(str(year))]
    if make:
        matched_df = matched_df[matched_df['AD_MAKE'].str.contains(make, case=False, na=False)]
    if model:
        matched_df = matched_df[matched_df['AD_MODEL'].str.contains(model, case=False, na=False)]
    if trim:
        matched_df = matched_df[matched_df['AD_TRIM'].str.contains(trim, case=False, na=False)]
    
    return matched_df

# --- STREAMLIT UI ---
st.title("AFF Vehicle Mapping")

# Freeform vehicle input
vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filter
st.subheader("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_input = col1.text_input("Year")
make_input = col2.text_input("Make")
model_input = col3.text_input("Model")
trim_input = col4.text_input("Trim")

# Search button
search_clicked = st.button("Search Vehicles")

if search_clicked and vehicle_input:
    matches_df = smart_vehicle_match(cads_df, vehicle_input, year_input, make_input, model_input, trim_input)
    
    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.success(f"Found {len(matches_df)} possible matches.")

        # Display table with checkboxes
        selected = []
        st.subheader("Applicable Vehicle Lines")
        for i, row in matches_df.iterrows():
            checkbox = st.checkbox(
                f"{row['AD_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']}",
                key=f"{row['AD_VEH_ID']}"
            )
            if checkbox:
                selected.append(i)

        # Submit Mapping button
        if st.button("Submit Mapping"):
            if not selected:
                st.warning("Select at least one vehicle line to map.")
            else:
                for idx in selected:
                    row = matches_df.loc[idx]
                    mappings_df = pd.concat([mappings_df, pd.DataFrame([{
                        "vehicle_input": vehicle_input,
                        "AD_VEH_ID": row['AD_VEH_ID'],
                        "AD_YEAR": row['AD_YEAR'],
                        "AD_MAKE": row['AD_MAKE'],
                        "AD_MODEL": row['AD_MODEL'],
                        "AD_TRIM": row['AD_TRIM'],
                        "Model Code": row['AD_MFGCODE']
                    }])], ignore_index=True)
                
                # Save mappings locally
                mappings_df.to_csv(MAPPINGS_FILE, index=False)
                st.success(f"Mapping saved for {len(selected)} vehicle(s).")

