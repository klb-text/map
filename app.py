import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# -----------------------------
# Config / File Paths
# -----------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "mappings.csv"

# -----------------------------
# Load CSVs
# -----------------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_vehicle_example(path):
    try:
        # tab-delimited
        df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines='skip')
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_mappings(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=["VehicleInput","MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"])

# -----------------------------
# Load data
# -----------------------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_vehicle_example(VEHICLE_REF_FILE)
mappings_df = load_mappings(MAPPINGS_FILE)

# -----------------------------
# Helper Functions
# -----------------------------
def smart_vehicle_match(df, vehicle_input):
    if df.empty:
        return pd.DataFrame(), []
    # create combined search field
    df = df.copy()
    df['vehicle_search'] = df[['MODEL_YEAR','MAKE','MODEL_NAME','TRIM']].astype(str).agg(' '.join, axis=1)
    # fuzzy match
    matches = process.extract(vehicle_input, df['vehicle_search'], scorer=fuzz.token_sort_ratio, limit=50)
    if not matches:
        return pd.DataFrame(), []
    scores = [m[1] for m in matches]
    matched_indices = [m[2] for m in matches]
    matched_df = df.iloc[matched_indices].copy()
    matched_df['score'] = scores
    return matched_df.sort_values(by='score', ascending=False), matches

def infer_make(vehicle_input):
    """Try to find make from vehicle_example"""
    if vehicle_ref_df.empty:
        return ""
    row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    if not row.empty:
        return row.iloc[0]['Make']
    return ""

def save_mapping(selected_rows):
    global mappings_df
    new_rows = []
    for row in selected_rows:
        new_rows.append({
            "VehicleInput": vehicle_input,
            "MODEL_YEAR": row["MODEL_YEAR"],
            "MAKE": row["MAKE"],
            "MODEL_NAME": row["MODEL_NAME"],
            "TRIM": row["TRIM"],
            "AD_MFGCODE": row["AD_MFGCODE"],
            "STYLE_ID": row["STYLE_ID"]
        })
    mappings_df = pd.concat([mappings_df, pd.DataFrame(new_rows)], ignore_index=True)
    mappings_df.to_csv(MAPPINGS_FILE, index=False)
    st.success("Mapping saved!")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

search_button = st.button("Search")

if search_button and vehicle_input:
    example_make = infer_make(vehicle_input)
    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input)
    
    if example_make:
        matches_df = matches_df[matches_df['MAKE'].str.lower() == example_make.lower()]

    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.subheader("YMMT Filter (optional)")
        col1, col2, col3, col4 = st.columns(4)
        year_filter = col1.text_input("Year")
        make_filter = col2.text_input("Make")
        model_filter = col3.text_input("Model")
        trim_filter = col4.text_input("Trim")

        filtered_df = matches_df.copy()
        if year_filter:
            filtered_df = filtered_df[filtered_df['MODEL_YEAR'].astype(str).str.contains(year_filter)]
        if make_filter:
            filtered_df = filtered_df[filtered_df['MAKE'].str.contains(make_filter, case=False)]
        if model_filter:
            filtered_df = filtered_df[filtered_df['MODEL_NAME'].str.contains(model_filter, case=False)]
        if trim_filter:
            filtered_df = filtered_df[filtered_df['TRIM'].str.contains(trim_filter, case=False)]

        st.subheader("Matching Vehicles")
        selected_rows = []
        for idx, row in filtered_df.iterrows():
            checkbox = st.checkbox(f"{row['MODEL_YEAR']} {row['MAKE']} {row['MODEL_NAME']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}", key=f"chk_{idx}")
            if checkbox:
                selected_rows.append(row)

        if st.button("Submit Mapping"):
            if selected_rows:
                save_mapping(selected_rows)
            else:
                st.warning("Please select at least one vehicle to map.")

st.subheader("Current Mappings")
st.dataframe(mappings_df)
