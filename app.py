import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import os

# ----------------------
# File paths
# ----------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# ----------------------
# Load CSV safely
# ----------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

# ----------------------
# Smart Vehicle Match
# ----------------------
def smart_vehicle_match(cads_df, vehicle_input, vehicle_ref_df):
    # Attempt to detect Make/Model from reference
    ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    if not ref_row.empty:
        make = ref_row.iloc[0]['Make']
        model = ref_row.iloc[0]['Model']
        trim_input = ref_row.iloc[0]['VehicleAttributes']
        year_input = ref_row.iloc[0]['Year']
    else:
        make = None
        model = None
        trim_input = None
        year_input = None

    filtered = cads_df.copy()
    
    # Filter by Make/Model if found
    if make:
        filtered = filtered[filtered['AD_MAKE'].str.lower() == make.lower()]
    if model:
        filtered = filtered[filtered['AD_MODEL'].str.lower() == model.lower()]
    if year_input:
        filtered = filtered[filtered['AD_YEAR'] == int(year_input)]

    if filtered.empty:
        return pd.DataFrame(), []

    # Fuzzy matching only on trim (optional)
    if trim_input:
        filtered['TrimScore'] = filtered['AD_TRIM'].fillna('').apply(lambda x: fuzz.partial_ratio(str(x).lower(), trim_input.lower()))
        filtered = filtered.sort_values(by='TrimScore', ascending=False)

    return filtered, ref_row

# ----------------------
# Load Data
# ----------------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
mappings_df = load_csv(MAPPINGS_FILE)

# ----------------------
# Streamlit UI
# ----------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT Filters
st.markdown("### YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_filter = col1.text_input("Year")
make_filter = col2.text_input("Make")
model_filter = col3.text_input("Model")
trim_filter = col4.text_input("Trim")

search_btn = st.button("Search CADS")

if search_btn and vehicle_input:
    matches_df, ref_row = smart_vehicle_match(cads_df, vehicle_input, vehicle_ref_df)

    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        # Display table with checkboxes
        st.markdown("### Matching Vehicles")
        selected_rows = []
        for idx, row in matches_df.iterrows():
            label = f"{row['AD_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_rows.append(idx)

        if selected_rows:
            if st.button("Submit Mapping"):
                for idx in selected_rows:
                    selected_row = matches_df.loc[idx]
                    # Add mapping to Mappings.csv (persist)
                    mapping_entry = {
                        'VehicleInput': vehicle_input,
                        'AD_VEH_ID': selected_row['AD_VEH_ID'],
                        'Year': selected_row['AD_YEAR'],
                        'Make': selected_row['AD_MAKE'],
                        'Model': selected_row['AD_MODEL'],
                        'Trim': selected_row['AD_TRIM'],
                        'ModelCode': selected_row['AD_MFGCODE']
                    }
                    mappings_df = pd.concat([mappings_df, pd.DataFrame([mapping_entry])], ignore_index=True)
                # Save updated mappings
                mappings_df.to_csv(MAPPINGS_FILE, index=False)
                st.success(f"Mapping saved for {len(selected_rows)} vehicle(s).")
