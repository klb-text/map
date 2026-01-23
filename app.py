import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# ----------------------------
# File paths
# ----------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "mappings.csv"

# ----------------------------
# Utility functions
# ----------------------------
@st.cache_data
def load_csv(path, sep=","):
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip()  # remove accidental spaces
    return df

def smart_vehicle_match(df, vehicle_input, limit=10):
    """Return top matching rows from CADS based on vehicle_input"""
    # Combine columns to create search target
    df['vehicle_search'] = df[['MODEL_YEAR','MAKE','MODEL_NAME','STYLE_NAME']].astype(str).agg(' '.join, axis=1)
    choices = df['vehicle_search'].tolist()
    results = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=limit)
    
    matched_indices = [r[2] for r in results if r[1] > 60]  # keep score > 60
    return df.iloc[matched_indices].copy(), results

# ----------------------------
# Load data
# ----------------------------
try:
    cads_df = load_csv(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load CADS.csv: {e}")

try:
    vehicle_ref_df = load_csv(VEHICLE_REF_FILE, sep="\t")
except Exception as e:
    st.error(f"Failed to load vehicle_example.txt: {e}")

# Ensure mappings file exists
if not os.path.exists(MAPPINGS_FILE):
    pd.DataFrame(columns=["vehicle_input","MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"]).to_csv(MAPPINGS_FILE, index=False)

mappings_df = load_csv(MAPPINGS_FILE)

# ----------------------------
# App UI
# ----------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

search_button = st.button("Search Vehicle")

# Store selected rows in session state
if "selected_rows" not in st.session_state:
    st.session_state.selected_rows = []

if search_button and vehicle_input:
    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input)
    
    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.write(f"Top matches for: {vehicle_input}")
        display_df = matches_df[["MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"]].copy()
        display_df.rename(columns={"AD_MFGCODE":"Model Code"}, inplace=True)
        
        # Show checkboxes for selection
        for idx, row in display_df.iterrows():
            key = f"row_{idx}"
            if key not in st.session_state:
                st.session_state[key] = False
            st.session_state[key] = st.checkbox(f"{row['MODEL_YEAR']} {row['MAKE']} {row['MODEL_NAME']} {row['TRIM']} (Model Code: {row['Model Code']})", value=st.session_state[key])
        
        st.session_state.selected_rows = [idx for idx, row in display_df.iterrows() if st.session_state[f"row_{idx}"]]

# ----------------------------
# Commit mapping
# ----------------------------
if st.session_state.selected_rows:
    if st.button("Submit Mapping"):
        selected_df = matches_df.loc[st.session_state.selected_rows, ["MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"]].copy()
        selected_df.rename(columns={"AD_MFGCODE":"Model Code"}, inplace=True)
        # Add original input for mapping
        selected_df["vehicle_input"] = vehicle_input
        mappings_df = pd.concat([mappings_df, selected_df], ignore_index=True)
        mappings_df.to_csv(MAPPINGS_FILE, index=False)
        st.success(f"Mapping saved for {vehicle_input}")
        st.dataframe(selected_df)

# ----------------------------
# Optional: Show full mappings table
# ----------------------------
if st.checkbox("Show all mappings"):
    st.dataframe(mappings_df)
