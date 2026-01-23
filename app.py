import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import os

# ----------------------------
# Constants
# ----------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# ----------------------------
# Load CSVs
# ----------------------------
@st.cache_data
def load_csv(path, sep=","):
    try:
        df = pd.read_csv(path, sep=sep, engine='python', encoding='utf-8', on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

# Load CADS and Vehicle Example
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE, sep='\t')
mappings_df = load_csv(MAPPINGS_FILE)

# Initialize session state
if 'selected_matches' not in st.session_state:
    st.session_state.selected_matches = []

# ----------------------------
# Helper Functions
# ----------------------------
def smart_vehicle_match(df, vehicle_input, top_n=10):
    """Return fuzzy matches from CADS based on vehicle input string."""
    # Prepare a search column
    df['vehicle_search'] = df[['MODEL_YEAR','MAKE','MODEL_NAME','STYLE_NAME','TRIM']].astype(str).agg(' '.join, axis=1)
    choices = df['vehicle_search'].tolist()
    matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=top_n)
    
    matched_indices = [idx for (val, score, idx) in matches if score > 50]
    matched_df = df.iloc[matched_indices].copy()
    return matched_df, matches

def get_make_model_from_example(vehicle_input):
    """Check vehicle_example.txt for a reference row to infer Make/Model."""
    vehicle_lower = vehicle_input.lower()
    ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_lower]
    if not ref_row.empty:
        return ref_row.iloc[0]['Make'], ref_row.iloc[0]['Model']
    return None, None

def save_mapping(selected_rows):
    """Append selected rows to Mappings.csv and update session."""
    global mappings_df
    for idx, row in selected_rows.iterrows():
        new_mapping = {
            'VehicleInput': row['vehicle_input'],
            'MODEL_YEAR': row['MODEL_YEAR'],
            'MAKE': row['MAKE'],
            'MODEL_NAME': row['MODEL_NAME'],
            'TRIM': row['TRIM'],
            'AD_MFGCODE': row['AD_MFGCODE'],
            'STYLE_ID': row['STYLE_ID']
        }
        mappings_df = pd.concat([mappings_df, pd.DataFrame([new_mapping])], ignore_index=True)
    mappings_df.to_csv(MAPPINGS_FILE, index=False)
    st.success("Mappings saved successfully!")

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT Filters
with st.expander("YMMT Filter (optional)"):
    filter_year = st.text_input("Year")
    filter_make = st.text_input("Make")
    filter_model = st.text_input("Model")
    filter_trim = st.text_input("Trim")

# Search button
if st.button("Search CADS"):
    if not vehicle_input.strip():
        st.warning("Enter a vehicle first.")
    else:
        # Check vehicle_example.txt first
        example_make, example_model = get_make_model_from_example(vehicle_input)
        filtered_cads = cads_df.copy()
        
        if example_make:
            filtered_cads = filtered_cads[filtered_cads['MAKE'].str.lower() == example_make.lower()]
        if example_model:
            filtered_cads = filtered_cads[filtered_cads['MODEL_NAME'].str.lower() == example_model.lower()]

        # Apply additional YMMT filters
        if filter_year:
            filtered_cads = filtered_cads[filtered_cads['MODEL_YEAR'].astype(str).str.contains(filter_year)]
        if filter_make:
            filtered_cads = filtered_cads[filtered_cads['MAKE'].str.contains(filter_make, case=False, na=False)]
        if filter_model:
            filtered_cads = filtered_cads[filtered_cads['MODEL_NAME'].str.contains(filter_model, case=False, na=False)]
        if filter_trim:
            filtered_cads = filtered_cads[filtered_cads['TRIM'].str.contains(filter_trim, case=False, na=False)]

        # Smart match fuzzy search
        matches_df, raw_matches = smart_vehicle_match(filtered_cads, vehicle_input)
        matches_df['vehicle_input'] = vehicle_input

        if matches_df.empty:
            st.info("No matching vehicles found.")
        else:
            st.write("Matching Vehicles")
            selected_rows = []
            for idx, row in matches_df.iterrows():
                chk = st.checkbox(
                    f"{row['MODEL_YEAR']} {row['MAKE']} {row['MODEL_NAME']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}",
                    key=f"chk_{idx}"
                )
                if chk:
                    selected_rows.append(row)

            # Store selected in session state for submit
            st.session_state.selected_matches = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame()

# Submit Mapping button
if st.session_state.selected_matches is not None and not st.session_state.selected_matches.empty:
    if st.button("Submit Mapping"):
        save_mapping(st.session_state.selected_matches)
        st.session_state.selected_matches = pd.DataFrame()  # clear selection
