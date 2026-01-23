import streamlit as st
import pandas as pd
from thefuzz import fuzz

# --- File paths ---
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"

# --- Load CSV safely ---
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path, sep=None, engine='python')  # auto-detect separator
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# --- Input ---
st.title("AFF Vehicle Mapping")
vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    year_input = st.text_input("Year")
with col2:
    make_input = st.text_input("Make")
with col3:
    model_input = st.text_input("Model")
with col4:
    trim_input = st.text_input("Trim")

search_clicked = st.button("Search Vehicles")

# --- Smart vehicle matching ---
def smart_vehicle_match(df, vehicle_input, example_make=None, example_model=None):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Prepare search string for fuzzy matching
    def combine_row(row):
        parts = [
            str(row.get('MODEL_YEAR', '')),
            str(row.get('AD_MAKE', '')),
            str(row.get('AD_MODEL', '')),
            str(row.get('TRIM', ''))
        ]
        return ' '.join([p for p in parts if p])

    df['vehicle_search'] = df.apply(combine_row, axis=1)

    # Filter on Make if provided
    df_filtered = df.copy()
    if example_make:
        df_filtered = df_filtered[df_filtered['AD_MAKE'].str.lower() == example_make.lower()]

    # If Model is provided, do partial / fuzzy match
    if example_model:
        df_filtered['model_score'] = df_filtered['AD_MODEL'].apply(
            lambda x: fuzz.partial_ratio(str(x).lower(), example_model.lower())
        )
        df_filtered = df_filtered[df_filtered['model_score'] > 60]  # threshold
        df_filtered = df_filtered.sort_values(by='model_score', ascending=False)
        df_filtered.drop(columns=['model_score'], inplace=True)

    # Apply YMMT filters
    if year_input:
        df_filtered = df_filtered[df_filtered['MODEL_YEAR'].astype(str) == str(year_input)]
    if make_input:
        df_filtered = df_filtered[df_filtered['AD_MAKE'].str.lower() == make_input.lower()]
    if model_input:
        df_filtered = df_filtered[df_filtered['AD_MODEL'].str.lower().str.contains(model_input.lower())]
    if trim_input:
        df_filtered = df_filtered[df_filtered['TRIM'].str.lower().str.contains(trim_input.lower(), na=False)]

    return df_filtered, df_filtered.copy()

# --- Search logic ---
matches_df = pd.DataFrame()
if search_clicked and vehicle_input:
    # Try to get Make/Model from vehicle_example
    example_row = vehicle_ref_df[
        vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()
    ]
    example_make = example_row['Make'].values[0] if not example_row.empty else None
    example_model = example_row['Model'].values[0] if not example_row.empty else None

    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)

    if matches_df.empty:
        st.warning(f"No matching vehicles found for: {vehicle_input}")
    else:
        st.subheader("Matching Vehicles")
        selected_vehicles = []

        # Checkbox for each matching vehicle
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_vehicles.append(idx)

        if selected_vehicles:
            if st.button("Submit Mapping"):
                final_df = matches_df.loc[selected_vehicles, ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']]
                st.success("Mapping submitted!")
                st.dataframe(final_df)

