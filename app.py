import streamlit as st
import pandas as pd
from thefuzz import process, fuzz

# ------------------------------
# Constants
# ------------------------------
VEHICLE_REF_FILE = "vehicle_example.txt"
CADS_FILE = "CADS.csv"

# ------------------------------
# Load CSVs safely
# ------------------------------
@st.cache_data
def load_vehicle_ref(path):
    try:
        df = pd.read_csv(path, sep="\t", engine='python')
        # Ensure missing columns exist
        for col in ['VehicleAttributes', 'Trim']:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception as e:
        st.error(f"Error loading vehicle_example.txt: {e}")
        return pd.DataFrame()

@st.cache_data
def load_cads(path):
    try:
        df = pd.read_csv(path, dtype=str)
        # Ensure necessary columns exist
        for col in ['AD_YEAR','AD_MAKE','AD_MODEL','AD_MFGCODE','TRIM']:
            if col not in df.columns:
                df[col] = ''
        return df
    except Exception as e:
        st.error(f"Error loading CADS.csv: {e}")
        return pd.DataFrame()

vehicle_ref_df = load_vehicle_ref(VEHICLE_REF_FILE)
cads_df = load_cads(CADS_FILE)

# ------------------------------
# App Layout
# ------------------------------
st.title("AFF Vehicle Mapping")

# Vehicle input
vehicle_input = st.text_input("Enter Vehicle (freeform)")

# YMMT filters (left to right)
with st.form(key="ymmt_form"):
    cols = st.columns(4)
    year_input = cols[0].text_input("Year")
    make_input = cols[1].text_input("Make")
    model_input = cols[2].text_input("Model")
    trim_input = cols[3].text_input("Trim")
    search_button = st.form_submit_button("Search")

# ------------------------------
# Fuzzy Matching Function
# ------------------------------
def smart_vehicle_match(df, input_vehicle, example_make=None, example_model=None):
    # Combine relevant fields for fuzzy search
    def combine_row(row):
        return ' '.join([
            str(row.get('AD_YEAR','')),
            str(row.get('AD_MAKE','')),
            str(row.get('AD_MODEL','')),
            str(row.get('TRIM',''))
        ]).strip()
    
    df = df.copy()
    df['vehicle_search'] = df.apply(combine_row, axis=1)
    
    # If make/model known, filter first
    if example_make:
        df = df[df['AD_MAKE'].str.lower() == example_make.lower()]
    if example_model:
        df = df[df['AD_MODEL'].str.lower() == example_model.lower()]
    
    # Fuzzy match against combined vehicle string
    choices = df['vehicle_search'].tolist()
    matches = process.extract(input_vehicle, choices, scorer=fuzz.token_sort_ratio, limit=20)
    
    matched_texts = [match[0] for match in matches if match[1] > 50]  # Only reasonably close matches
    filtered_df = df[df['vehicle_search'].isin(matched_texts)]
    
    return filtered_df, matches

# ------------------------------
# Search Logic
# ------------------------------
matches_df = pd.DataFrame()
if search_button and vehicle_input.strip() != "":
    # Attempt to get make/model from vehicle_example.txt
    example_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    example_make = example_row['Make'].values[0] if not example_row.empty else None
    example_model = example_row['Model'].values[0] if not example_row.empty else None
    
    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)
    
    if matches_df.empty:
        st.warning(f"No matching vehicles found. {vehicle_input}")
    else:
        st.subheader("Matching Vehicles")
        selected_indices = []
        for idx, row in matches_df.iterrows():
            checkbox_label = f"{row['AD_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}"
            if st.checkbox(checkbox_label, key=f"chk_{idx}"):
                selected_indices.append(idx)
        
        if st.button("Submit Mapping") and selected_indices:
            mapped_df = matches_df.loc[selected_indices, ['AD_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']]
            st.success("Mapping submitted!")
            st.dataframe(mapped_df)

# ------------------------------
# Optional: YMMT Filtering Table Display
# ------------------------------
if not matches_df.empty:
    st.subheader("Filtered Table Preview")
    st.dataframe(matches_df[['AD_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']])
