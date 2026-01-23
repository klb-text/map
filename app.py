import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# -----------------------------
# File paths
# -----------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# -----------------------------
# Load CSVs
# -----------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, sep=None, engine='python')  # auto-detect delimiter

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# Ensure Mappings.csv exists
if not os.path.exists(MAPPINGS_FILE):
    pd.DataFrame(columns=["Vehicle_Input", "MODEL_YEAR", "MAKE", "MODEL_NAME", "TRIM", "AD_MFGCODE", "STYLE_ID"]).to_csv(MAPPINGS_FILE, index=False)
mappings_df = pd.read_csv(MAPPINGS_FILE)

# -----------------------------
# Smart vehicle matching
# -----------------------------
def smart_vehicle_match(df, vehicle_input, top_n=10):
    # Create a searchable vehicle string in CADS
    df['vehicle_search'] = df[['MODEL_YEAR','MAKE','MODEL_NAME','TRIM']].astype(str).agg(' '.join, axis=1)

    # Use RapidFuzz to find top matches
    choices = df['vehicle_search'].tolist()
    matches = process.extract(vehicle_input, choices, scorer=fuzz.WRatio, limit=top_n)
    
    matched_indices = [i for match, score, i in matches if score >= 60]
    matched_df = df.iloc[matched_indices].copy()
    return matched_df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

st.subheader("Optional YMMT Filter")
col1, col2, col3, col4 = st.columns(4)
filter_year = col1.text_input("Year")
filter_make = col2.text_input("Make")
filter_model = col3.text_input("Model")
filter_trim = col4.text_input("Trim")

# -----------------------------
# Search button
# -----------------------------
search_clicked = st.button("Search CADS")
matches_df = pd.DataFrame()
if search_clicked and vehicle_input:
    matches_df = smart_vehicle_match(cads_df, vehicle_input)
    
    # Apply YMMT filters if provided
    if filter_year:
        matches_df = matches_df[matches_df["MODEL_YEAR"].astype(str).str.contains(filter_year)]
    if filter_make:
        matches_df = matches_df[matches_df["MAKE"].str.contains(filter_make, case=False, na=False)]
    if filter_model:
        matches_df = matches_df[matches_df["MODEL_NAME"].str.contains(filter_model, case=False, na=False)]
    if filter_trim:
        matches_df = matches_df[matches_df["TRIM"].str.contains(filter_trim, case=False, na=False)]
    
    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.subheader("Matching Vehicles")
        # Display checkboxes
        selected_indices = []
        for idx, row in matches_df.iterrows():
            if st.checkbox(f"{row['MODEL_YEAR']} {row['MAKE']} {row['MODEL_NAME']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}", key=f"chk_{idx}"):
                selected_indices.append(idx)

        # Submit mapping button
        if st.button("Submit Mapping"):
            if selected_indices:
                to_add = matches_df.loc[selected_indices, ["MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"]].copy()
                to_add["Vehicle_Input"] = vehicle_input
                mappings_df = pd.concat([mappings_df, to_add], ignore_index=True)
                mappings_df.to_csv(MAPPINGS_FILE, index=False)
                st.success(f"Mapping submitted for {len(selected_indices)} vehicle(s).")
            else:
                st.warning("No vehicles selected to map.")

# -----------------------------
# Show existing mappings
# -----------------------------
if not mappings_df.empty:
    st.subheader("Existing Mappings")
    st.dataframe(mappings_df)
