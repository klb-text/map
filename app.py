import streamlit as st
import pandas as pd
from thefuzz import process, fuzz

# -----------------------------
# File paths
# -----------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"

# -----------------------------
# Load CSVs safely
# -----------------------------
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine='python')  # use python engine for flexible parsing
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# -----------------------------
# Helper function for fuzzy matching
# -----------------------------
def smart_vehicle_match(df, vehicle_input, example_make=None, example_model=None):
    # Filter by Make/Model if known
    filtered = df.copy()
    if example_make:
        filtered = filtered[filtered['AD_MAKE'].str.lower() == example_make.lower()]
    if example_model:
        filtered = filtered[filtered['AD_MODEL'].str.lower() == example_model.lower()]

    if filtered.empty:
        return pd.DataFrame(), []

    # Combine searchable string (only use TRIM and STYLE_NAME)
    def combine_row(row):
        parts = []
        for col in ['TRIM', 'STYLE_NAME']:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        return ' '.join(parts)

    filtered['vehicle_search'] = filtered.apply(combine_row, axis=1)

    # Use fuzzy matching on vehicle_input against vehicle_search
    choices = filtered['vehicle_search'].tolist()
    raw_matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=50)
    # Merge score back to filtered
    score_map = {val: score for val, score in raw_matches}
    filtered['score'] = filtered['vehicle_search'].map(score_map)
    filtered = filtered.dropna(subset=['score'])
    filtered = filtered.sort_values(by='score', ascending=False)

    return filtered, raw_matches

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

st.write("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    year_filter = st.text_input("Year")
with col2:
    make_filter = st.text_input("Make")
with col3:
    model_filter = st.text_input("Model")
with col4:
    trim_filter = st.text_input("Trim")

if st.button("Search"):
    # Determine Make/Model from reference file if available
    example_make = None
    example_model = None
    if not make_filter or not model_filter:
        ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
        if not ref_row.empty:
            example_make = ref_row.iloc[0]['Make'] if 'Make' in ref_row.columns else None
            example_model = ref_row.iloc[0]['Model'] if 'Model' in ref_row.columns else None
    else:
        example_make = make_filter
        example_model = model_filter

    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)

    if matches_df.empty:
        st.warning(f"No matching vehicles found for: {vehicle_input}")
    else:
        st.write("Select applicable vehicle(s) to map:")

        # Track selections
        selected_indices = []
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']} | Score: {row['score']:.1f}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_indices.append(idx)

        if st.button("Submit Mapping"):
            if selected_indices:
                selected_df = matches_df.loc[selected_indices, ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID','STYLE_NAME']]
                st.success("Mapping submitted for the following vehicles:")
                st.dataframe(selected_df)
            else:
                st.warning("No vehicles selected for mapping.")
