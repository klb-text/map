import streamlit as st
import pandas as pd
from thefuzz import process, fuzz

st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# -------------------
# Load CSV helper
# -------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

# -------------------
# Smart vehicle match
# -------------------
def smart_vehicle_match(df, vehicle_input, example_make=None, example_model=None, limit=25):
    if df.empty:
        return pd.DataFrame(), []

    # Filter by make/model if provided
    filtered = df.copy()
    if example_make:
        filtered = filtered[filtered['AD_MAKE'].str.lower() == example_make.lower()]
    if example_model:
        filtered = filtered[filtered['AD_MODEL'].str.lower() == example_model.lower()]

    # Combine row for fuzzy matching
    def combine_row(row):
        return ' '.join([str(row.get(col, '') or '') for col in ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM']])

    filtered['vehicle_search'] = filtered.apply(combine_row, axis=1)

    # Use fuzzy matching
    choices = filtered['vehicle_search'].tolist()
    if not choices:
        return pd.DataFrame(), []

    matches = process.extract(vehicle_input, choices, limit=limit, scorer=fuzz.token_sort_ratio)
    scores = [m[1] for m in matches]
    matched_texts = [m[0] for m in matches]

    # Get corresponding rows
    matched_rows = filtered[filtered['vehicle_search'].isin(matched_texts)].copy()
    matched_rows['score'] = scores[:len(matched_rows)]
    return matched_rows, matches

# -------------------
# Load data
# -------------------
VEHICLE_REF_FILE = "vehicle_example.txt"
CADS_FILE = "CADS.csv"

vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
cads_df = load_csv(CADS_FILE)

# -------------------
# Input fields
# -------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

with st.expander("YMMT Filter (optional)"):
    col1, col2, col3, col4 = st.columns(4)
    year_filter = col1.text_input("Year")
    make_filter = col2.text_input("Make")
    model_filter = col3.text_input("Model")
    trim_filter = col4.text_input("Trim")

# -------------------
# Search button
# -------------------
if st.button("Search Vehicles") and vehicle_input.strip() != "":
    # Find example row in vehicle_ref_df to get make/model
    example_make = None
    example_model = None
    if not vehicle_ref_df.empty and 'Vehicle' in vehicle_ref_df.columns:
        ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
        if not ref_row.empty:
            example_make = ref_row.iloc[0].get('Make')
            example_model = ref_row.iloc[0].get('Model')

    # Override with YMMT filters if filled
    if make_filter.strip():
        example_make = make_filter.strip()
    if model_filter.strip():
        example_model = model_filter.strip()

    # Smart match
    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)

    if matches_df.empty:
        st.warning(f"No matching vehicles found. {vehicle_input}")
    else:
        st.subheader("Matching Vehicles")

        # Initialize session state for selected vehicles
        if 'selected_vehicles' not in st.session_state:
            st.session_state.selected_vehicles = []

        # Display checkboxes for each match
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row.get('TRIM','')} | Model Code: {row['AD_MFGCODE']} | Score: {row['score']:.1f}"
            checked = idx in st.session_state.selected_vehicles
            new_val = st.checkbox(label, key=f"chk_{idx}", value=checked)
            if new_val and idx not in st.session_state.selected_vehicles:
                st.session_state.selected_vehicles.append(idx)
            elif not new_val and idx in st.session_state.selected_vehicles:
                st.session_state.selected_vehicles.remove(idx)

        # Submit Mapping button
        if st.button("Submit Mapping"):
            if st.session_state.selected_vehicles:
                selected_df = matches_df.loc[st.session_state.selected_vehicles, 
                                             ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID','STYLE_NAME']]
                st.success("Mapping submitted for the following vehicles:")
                st.dataframe(selected_df)
                st.session_state.selected_vehicles = []  # Clear after submit
            else:
                st.warning("No vehicles selected for mapping.")
