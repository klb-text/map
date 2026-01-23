import streamlit as st
import pandas as pd
from thefuzz import process, fuzz

# --- File paths ---
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"

# --- Load CSVs ---
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# --- YMMT Inputs ---
st.title("AFF Vehicle Mapping")
st.write("Enter Vehicle (freeform)")

vehicle_input = st.text_input("Vehicle Name")

st.write("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_input = col1.text_input("Year")
make_input = col2.text_input("Make")
model_input = col3.text_input("Model")
trim_input = col4.text_input("Trim")

# --- Helper: get example make/model from reference file ---
def get_example_make_model(vehicle_name):
    if 'Vehicle' not in vehicle_ref_df.columns:
        return None, None
    ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_name.lower()]
    if not ref_row.empty:
        make = ref_row['Make'].values[0] if 'Make' in ref_row.columns else None
        model = ref_row['Model'].values[0] if 'Model' in ref_row.columns else None
        return make, model
    return None, None

example_make, example_model = get_example_make_model(vehicle_input)

# --- Smart Match Function ---
def smart_vehicle_match(df, vehicle_input, example_make=None, example_model=None):
    if df.empty or not vehicle_input:
        return pd.DataFrame(), []

    # Ensure columns exist
    for col in ['AD_MAKE','AD_MODEL','TRIM','MODEL_YEAR','AD_MFGCODE','STYLE_ID']:
        if col not in df.columns:
            df[col] = ''

    # Filter by year if available
    if year_input:
        df_filtered = df[df['MODEL_YEAR'].astype(str) == str(year_input)]
    elif example_make and example_model:
        df_filtered = df[(df['AD_MAKE'].str.lower() == example_make.lower()) &
                         (df['AD_MODEL'].str.lower() == example_model.lower())]
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return pd.DataFrame(), []

    # Combine columns for fuzzy search
    def combine_row(row):
        return ' '.join([str(row.get('MODEL_YEAR','')),
                         str(row.get('AD_MAKE','')),
                         str(row.get('AD_MODEL','')),
                         str(row.get('TRIM',''))]).strip()

    df_filtered['vehicle_search'] = df_filtered.apply(combine_row, axis=1)

    # Fuzzy match
    choices = df_filtered['vehicle_search'].tolist()
    results = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=20)

    matched_indices = [df_filtered.index[i] for i, choice in enumerate(choices) if choice in [r[0] for r in results]]
    matched_df = df_filtered.loc[matched_indices].copy()

    return matched_df, results

# --- Search Button ---
if st.button("Search Vehicles") and vehicle_input:
    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)

    if matches_df.empty:
        st.warning(f"No matching vehicles found for: {vehicle_input}")
    else:
        st.subheader("Matching Vehicles")

        # --- Initialize session_state for selections ---
        if 'selected_vehicles' not in st.session_state:
            st.session_state['selected_vehicles'] = []

        # --- Render checkboxes ---
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}"
            checked = idx in st.session_state['selected_vehicles']
            if st.checkbox(label, key=f"chk_{idx}", value=checked):
                if idx not in st.session_state['selected_vehicles']:
                    st.session_state['selected_vehicles'].append(idx)
            else:
                if idx in st.session_state['selected_vehicles']:
                    st.session_state['selected_vehicles'].remove(idx)

        # --- Submit Mapping ---
        if st.session_state['selected_vehicles']:
            if st.button("Submit Mapping"):
                final_df = matches_df.loc[st.session_state['selected_vehicles'],
                                          ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']]
                st.success("Mapping submitted!")
                st.dataframe(final_df)
