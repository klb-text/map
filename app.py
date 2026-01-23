import streamlit as st
import pandas as pd
from thefuzz import process, fuzz

# --------------------------------------------
# File paths
# --------------------------------------------
VEHICLE_REF_FILE = "vehicle_example.txt"
CADS_FILE = "CADS.csv"

# --------------------------------------------
# Helper functions
# --------------------------------------------
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine='python', dtype=str)
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
        df.fillna('', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

def smart_vehicle_match(df, vehicle_input, example_make=None, example_model=None):
    df = df.copy()

    # Ensure required columns exist
    for col in ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].astype(str).fillna('')

    # Pre-filter by make/model if available
    if example_make:
        df = df[df['AD_MAKE'].str.lower() == example_make.lower()]
    if example_model:
        df = df[df['AD_MODEL'].str.lower() == example_model.lower()]

    df = df.reset_index(drop=True)

    # Create combined search string
    df['vehicle_search'] = df.apply(
        lambda row: ' '.join(filter(None, [row['MODEL_YEAR'], row['AD_MAKE'], row['AD_MODEL'], row['TRIM']])),
        axis=1
    )

    # Fuzzy match vehicle_input
    choices = df['vehicle_search'].tolist()
    raw_matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=50)

    # Filter matches with reasonable threshold (>=60)
    matched_indices = [i for i, (val, score) in enumerate(raw_matches) if score >= 60]
    matches_df = df.iloc[matched_indices].copy()

    return matches_df, raw_matches

# --------------------------------------------
# Load data
# --------------------------------------------
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
cads_df = load_csv(CADS_FILE)

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("AFF Vehicle Mapping")

# Vehicle input
vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
with st.expander("YMMT Filter (optional)"):
    year_filter = st.text_input("Year")
    make_filter = st.text_input("Make")
    model_filter = st.text_input("Model")
    trim_filter = st.text_input("Trim")

# Search button
search_clicked = st.button("Search")

# Process search
matches_df = pd.DataFrame()
if search_clicked and vehicle_input.strip():
    # Try to get example make/model from vehicle_ref_df
    example_make = example_model = None
    if not vehicle_ref_df.empty and 'Vehicle' in vehicle_ref_df.columns:
        ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
        if not ref_row.empty:
            example_make = ref_row.iloc[0]['Make'] if 'Make' in ref_row.columns else None
            example_model = ref_row.iloc[0]['Model'] if 'Model' in ref_row.columns else None

    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)

    if matches_df.empty:
        st.info("No matching vehicles found.")
    else:
        st.subheader("Matching Vehicles")
        # Create selection checkboxes
        selected_indices = []
        for idx, row in matches_df.iterrows():
            checkbox_label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row.get('AD_MFGCODE','')}"
            if st.checkbox(checkbox_label, key=f"chk_{idx}"):
                selected_indices.append(idx)

        # Submit mapping button
        if st.button("Submit Mapping"):
            if selected_indices:
                submitted_df = matches_df.loc[selected_indices, ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']]
                st.success("Mapping submitted:")
                st.dataframe(submitted_df)
            else:
                st.warning("No vehicles selected for mapping.")

# --------------------------------------------
# End of app
# --------------------------------------------
