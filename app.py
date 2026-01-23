import streamlit as st
import pandas as pd
from thefuzz import fuzz, process

# --------------------------
# Load CSV helper
# --------------------------
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

# --------------------------
# Smart vehicle match
# --------------------------
def smart_vehicle_match(df, vehicle_input, example_make=None, example_model=None):
    df_filtered = df.copy()
    
    # Filter by make/model if we have reference
    if example_make and example_model:
        df_filtered = df_filtered[
            (df_filtered['AD_MAKE'].str.lower() == example_make.lower()) &
            (df_filtered['AD_MODEL'].str.lower() == example_model.lower())
        ]
    
    # Combine columns for fuzzy search
    def combine_row(row):
        trim_val = row['TRIM'] if pd.notna(row['TRIM']) else ''
        return f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {trim_val}"
    
    df_filtered['vehicle_search'] = df_filtered.apply(combine_row, axis=1)
    
    # Fuzzy match
    choices = df_filtered['vehicle_search'].tolist()
    raw_matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=50)
    match_values = [m[0] for m in raw_matches if m[1] >= 60]  # threshold 60
    
    matches_df = df_filtered[df_filtered['vehicle_search'].isin(match_values)].copy()
    
    return matches_df, raw_matches

# --------------------------
# Main app
# --------------------------
st.title("AFF Vehicle Mapping")

# Input vehicle
vehicle_input = st.text_input("Enter Vehicle (freeform)")

# YMMT filter (optional)
st.subheader("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_input = col1.text_input("Year")
make_input = col2.text_input("Make")
model_input = col3.text_input("Model")
trim_input = col4.text_input("Trim")

# Load CSVs
cads_df = load_csv("CADS.csv")
vehicle_ref_df = load_csv("vehicle_example.txt")

# Only search when button clicked
if st.button("Search"):
    example_make = example_model = None
    # Try to get make/model from vehicle_example.txt if available
    if not cads_df.empty and not vehicle_ref_df.empty:
        ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
        if not ref_row.empty:
            example_make = ref_row.iloc[0]['Make']
            example_model = ref_row.iloc[0]['Model']
    
    if cads_df.empty:
        st.warning("CADS.csv is empty or failed to load")
    else:
        matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)
        
        if matches_df.empty:
            st.info(f"No matching vehicles found for: {vehicle_input}")
        else:
            st.subheader("Matching Vehicles")
            selected_indices = []
            for idx, row in matches_df.iterrows():
                label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}"
                if st.checkbox(label, key=f"chk_{idx}"):
                    selected_indices.append(idx)
            
            # Submit mapping
            if st.button("Submit Mapping"):
                if not selected_indices:
                    st.warning("No vehicles selected for mapping")
                else:
                    mapped_df = matches_df.loc[selected_indices, ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']]
                    st.success("Mapping submitted successfully!")
                    st.dataframe(mapped_df)
