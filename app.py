import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# ---------------------------
# File paths
# ---------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # remove spaces
    return df

@st.cache_data
def load_vehicle_ref(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    return df

def smart_vehicle_match(df, user_input):
    """
    Fuzzy match vehicle input against CADS dataframe
    """
    # Construct a search string for each row
    df['vehicle_search'] = df[['MODEL_YEAR','AD_MAKE','MODEL_NAME','STYLE_NAME']].astype(str).agg(' '.join, axis=1)
    
    # Get top matches using RapidFuzz
    choices = df['vehicle_search'].tolist()
    matches = process.extract(user_input, choices, scorer=fuzz.token_sort_ratio, limit=20)
    
    # Get DataFrame of matches
    matched_rows = []
    for match_str, score, idx in matches:
        if score >= 60:  # adjustable threshold
            matched_rows.append(df.iloc[idx])
    if matched_rows:
        return pd.DataFrame(matched_rows)
    else:
        return pd.DataFrame()

def save_mapping(selected_df, vehicle_input):
    """
    Append new mapping to Mappings.csv
    """
    if os.path.exists(MAPPINGS_FILE):
        mappings_df = pd.read_csv(MAPPINGS_FILE)
    else:
        mappings_df = pd.DataFrame()
    
    selected_df = selected_df.copy()
    selected_df["Vehicle_Input"] = vehicle_input
    mappings_df = pd.concat([mappings_df, selected_df], ignore_index=True)
    mappings_df.to_csv(MAPPINGS_FILE, index=False)
    return mappings_df

# ---------------------------
# Load data
# ---------------------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_vehicle_ref(VEHICLE_REF_FILE)
if os.path.exists(MAPPINGS_FILE):
    mappings_df = load_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame()

st.title("AFF Vehicle Mapping")

# ---------------------------
# Vehicle input
# ---------------------------
vehicle_input = st.text_input("Enter Vehicle (freeform)")
search_clicked = st.button("Search")

matches_df = pd.DataFrame()
if search_clicked and vehicle_input:
    matches_df = smart_vehicle_match(cads_df, vehicle_input)
    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        # Optional YMMT filters
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
            filtered_df = filtered_df[filtered_df['AD_MAKE'].str.contains(make_filter, case=False, na=False)]
        if model_filter:
            filtered_df = filtered_df[filtered_df['MODEL_NAME'].str.contains(model_filter, case=False, na=False)]
        if trim_filter:
            filtered_df = filtered_df[filtered_df['STYLE_NAME'].str.contains(trim_filter, case=False, na=False)]

        if not filtered_df.empty:
            st.subheader("Applicable Vehicle Lines")
            # Checkbox selection
            filtered_df["Select"] = False
            for i, row in filtered_df.iterrows():
                filtered_df.at[i, "Select"] = st.checkbox(
                    f"{row['MODEL_YEAR']} | {row['AD_MAKE']} | {row['MODEL_NAME']} | {row['STYLE_NAME']} | Model Code: {row['AD_MFGCODE']}",
                    value=False,
                    key=f"chk_{i}"
                )

            # Submit mapping button
            if st.button("Submit Mapping"):
                selected = filtered_df[filtered_df["Select"] == True]
                if not selected.empty:
                    mappings_df = save_mapping(selected[['MODEL_YEAR','AD_MAKE','MODEL_NAME','STYLE_NAME','AD_MFGCODE']], vehicle_input)
                    st.success(f"Mapping saved for {len(selected)} vehicle(s).")
                else:
                    st.warning("No vehicles selected to map.")

# Display existing mappings
if not mappings_df.empty:
    st.subheader("Existing Mappings")
    st.dataframe(mappings_df)

