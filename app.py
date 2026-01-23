import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

# -------------------------------
# File paths
# -------------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# -------------------------------
# Load CSV with error handling
# -------------------------------
@st.cache_data
def load_csv(path, sep=","):
    try:
        df = pd.read_csv(path, sep=sep)
        df.columns = df.columns.str.strip()  # Strip whitespace
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE, sep="\t")
mappings_df = load_csv(MAPPINGS_FILE)

# -------------------------------
# Fuzzy matching function
# -------------------------------
def smart_vehicle_match(df, query, top_n=20):
    if df.empty:
        return pd.DataFrame(), []

    # Create a combined searchable string
    for col in ['MODEL_YEAR','AD_MAKE','MODEL_NAME','AD_TRIM']:
        if col not in df.columns:
            st.warning(f"Column {col} not in CADS.csv")
            return pd.DataFrame(), []
    df['vehicle_search'] = df[['MODEL_YEAR','AD_MAKE','MODEL_NAME','AD_TRIM']].astype(str).agg(' '.join, axis=1)

    choices = df['vehicle_search'].tolist()
    matches = process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=top_n)
    matched_indices = [i[2] for i in matches if i[1] > 50]  # score threshold 50
    return df.iloc[matched_indices].copy(), matches

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
st.write("YMMT Filters (optional)")
col1, col2, col3, col4 = st.columns(4)
year_filter = col1.text_input("Year")
make_filter = col2.text_input("Make")
model_filter = col3.text_input("Model")
trim_filter = col4.text_input("Trim")

search_clicked = st.button("Search")

if search_clicked:
    # Apply YMMT filters if entered
    filtered_cads = cads_df.copy()
    if year_filter:
        filtered_cads = filtered_cads[filtered_cads['MODEL_YEAR'].astype(str).str.contains(year_filter)]
    if make_filter:
        filtered_cads = filtered_cads[filtered_cads['AD_MAKE'].astype(str).str.contains(make_filter, case=False)]
    if model_filter:
        filtered_cads = filtered_cads[filtered_cads['MODEL_NAME'].astype(str).str.contains(model_filter, case=False)]
    if trim_filter:
        filtered_cads = filtered_cads[filtered_cads['AD_TRIM'].astype(str).str.contains(trim_filter, case=False)]

    # Use vehicle_example.txt to help identify make if needed
    example_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    if not example_row.empty and 'Make' in example_row.columns:
        example_make = example_row.iloc[0]['Make']
        filtered_cads = filtered_cads[filtered_cads['AD_MAKE'].str.lower() == str(example_make).lower()]

    matches_df, raw_matches = smart_vehicle_match(filtered_cads, vehicle_input)

    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.subheader("Matching Vehicles")

        selected_indices = []
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['MODEL_NAME']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_indices.append(idx)

        if st.button("Submit Mapping"):
            if not selected_indices:
                st.warning("No vehicles selected.")
            else:
                selected_rows = matches_df.loc[selected_indices]
                st.success("Mapping submitted:")
                st.dataframe(selected_rows[['MODEL_YEAR','AD_MAKE','MODEL_NAME','AD_TRIM','AD_MFGCODE','STYLE_ID']])
                # Optionally append to Mappings.csv here
                # selected_rows.to_csv(MAPPINGS_FILE, mode='a', index=False, header=False)
