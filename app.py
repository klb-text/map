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
# Load CSV safely
# -----------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

# -----------------------------
# Load CADS and Vehicle Reference
# -----------------------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
mappings_df = load_csv(MAPPINGS_FILE)

if mappings_df.empty:
    mappings_df = pd.DataFrame(columns=["Vehicle", "AD_VEH_ID", "AD_YEAR","AD_MAKE","AD_MODEL","AD_TRIM","AD_MFGCODE"])

# -----------------------------
# Helper: smart match
# -----------------------------
def smart_vehicle_match(df, vehicle_input, max_matches=10):
    # Create a search string for each CADS line
    df['vehicle_search'] = df[['AD_YEAR', 'AD_MAKE', 'AD_MODEL', 'AD_TRIM']].astype(str).agg(' '.join, axis=1)
    choices = df['vehicle_search'].tolist()
    matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=max_matches)
    
    matched_indices = [df.index[i] for i, _, _ in matches]
    matched_scores = [score for _, score, _ in matches]
    
    matched_df = df.loc[matched_indices].copy()
    matched_df['score'] = matched_scores
    return matched_df, matches

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
st.write("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_filter = col1.text_input("Year")
make_filter = col2.text_input("Make")
model_filter = col3.text_input("Model")
trim_filter = col4.text_input("Trim")

search_clicked = st.button("Search")

if search_clicked and vehicle_input:
    # Use vehicle_example to infer Make if possible
    example_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    example_make = None
    if not example_row.empty and 'Make' in example_row.columns:
        example_make = example_row.iloc[0]['Make']
    
    # Filter CADS based on optional Make
    filtered_cads = cads_df.copy()
    if example_make:
        filtered_cads = filtered_cads[filtered_cads['AD_MAKE'].str.lower() == example_make.lower()]
    
    # Apply optional YMMT filters
    if year_filter:
        filtered_cads = filtered_cads[filtered_cads['AD_YEAR'].astype(str) == year_filter]
    if make_filter:
        filtered_cads = filtered_cads[filtered_cads['AD_MAKE'].str.lower() == make_filter.lower()]
    if model_filter:
        filtered_cads = filtered_cads[filtered_cads['AD_MODEL'].str.lower() == model_filter.lower()]
    if trim_filter:
        filtered_cads = filtered_cads[filtered_cads['AD_TRIM'].str.lower() == trim_filter.lower()]
    
    # Fuzzy match against filtered CADS
    if filtered_cads.empty:
        st.warning("No matching vehicles found.")
    else:
        matches_df, raw_matches = smart_vehicle_match(filtered_cads, vehicle_input)
        
        if matches_df.empty:
            st.warning("No close matches found.")
        else:
            st.write(f"Smart matches for: {vehicle_input}")
            selected_vehicle_ids = []
            
            # Table with checkboxes
            for idx, row in matches_df.iterrows():
                label = f"{row['AD_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']} | Score: {row['score']}"
                if st.checkbox(label, key=f"chk_{idx}"):
                    selected_vehicle_ids.append(row['AD_VEH_ID'])
            
            # Commit button
            if st.button("Commit Mapping"):
                for veh_id in selected_vehicle_ids:
                    # Avoid duplicates
                    if not ((mappings_df['Vehicle'] == vehicle_input) & (mappings_df['AD_VEH_ID'] == veh_id)).any():
                        new_row = matches_df[matches_df['AD_VEH_ID'] == veh_id][
                            ['AD_VEH_ID','AD_YEAR','AD_MAKE','AD_MODEL','AD_TRIM','AD_MFGCODE']
                        ].copy()
                        new_row.insert(0, 'Vehicle', vehicle_input)
                        mappings_df = pd.concat([mappings_df, new_row], ignore_index=True)
                st.success(f"Mapping committed for {len(selected_vehicle_ids)} vehicle(s).")
                # Save mappings locally
                mappings_df.to_csv(MAPPINGS_FILE, index=False)

# Show current mappings
st.subheader("Current Mappings")
st.dataframe(mappings_df)
