import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os
from github import Github
from io import StringIO

# ------------------------------
# Configuration
# ------------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# GitHub secrets
GITHUB_TOKEN = "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"
GITHUB_OWNER = "klb-text"
GITHUB_REPO = "map"
GITHUB_BRANCH = "main"

# ------------------------------
# Helper Functions
# ------------------------------

@st.cache_data
def load_csv(path, delimiter=","):
    try:
        df = pd.read_csv(path, delimiter=delimiter)
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

def smart_vehicle_match(cads_df, vehicle_input, top_n=20):
    # Create a combined search field
    cads_df['vehicle_search'] = (
        cads_df['MODEL_YEAR'].astype(str) + " " +
        cads_df['AD_MAKE'].astype(str) + " " +
        cads_df['MODEL_NAME'].astype(str) + " " +
        cads_df['AD_TRIM'].astype(str)
    )
    choices = cads_df['vehicle_search'].tolist()
    results = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=top_n)
    matched_indices = [idx for _, _, idx in results if _ > 60]
    matched_df = cads_df.iloc[matched_indices]
    return matched_df, results

def save_mapping_to_github(selected_df):
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(f"{GITHUB_OWNER}/{GITHUB_REPO}")
        content_file = repo.get_contents(MAPPINGS_FILE, ref=GITHUB_BRANCH)
        old_content = content_file.decoded_content.decode()
        # Append new mappings
        csv_buffer = StringIO()
        selected_df.to_csv(csv_buffer, index=False, header=False)
        new_content = old_content + csv_buffer.getvalue()
        repo.update_file(MAPPINGS_FILE, "Add new mappings", new_content, content_file.sha, branch=GITHUB_BRANCH)
        st.success("Mappings saved to GitHub!")
    except Exception as e:
        st.error(f"Error saving to GitHub: {e}")

# ------------------------------
# Load Data
# ------------------------------
st.title("AFF Vehicle Mapping")

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE, delimiter="\t")
mappings_df = load_csv(MAPPINGS_FILE)

# ------------------------------
# Vehicle Input
# ------------------------------
vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
st.subheader("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_filter = col1.text_input("Year")
make_filter = col2.text_input("Make")
model_filter = col3.text_input("Model")
trim_filter = col4.text_input("Trim")

# Search button
search_clicked = st.button("Search Vehicles")

# ------------------------------
# Search Logic
# ------------------------------
if search_clicked and vehicle_input:
    # Try to identify make from vehicle_example
    ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    example_make = None
    if not ref_row.empty:
        example_make = ref_row.iloc[0]['Make']

    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input)

    # Apply optional filters
    if year_filter:
        matches_df = matches_df[matches_df['MODEL_YEAR'].astype(str) == year_filter]
    if make_filter:
        matches_df = matches_df[matches_df['AD_MAKE'].str.lower() == make_filter.lower()]
    elif example_make:
        matches_df = matches_df[matches_df['AD_MAKE'].str.lower() == example_make.lower()]
    if model_filter:
        matches_df = matches_df[matches_df['MODEL_NAME'].str.lower() == model_filter.lower()]
    if trim_filter:
        matches_df = matches_df[matches_df['AD_TRIM'].str.lower() == trim_filter.lower()]

    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.subheader("Matching Vehicles")

        # Initialize session state for checkboxes
        if "selected_indices" not in st.session_state:
            st.session_state.selected_indices = []

        # Display checkboxes for selection
        for idx, row in matches_df.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['MODEL_NAME']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']}"
            checked = st.checkbox(label, key=f"chk_{idx}", value=(idx in st.session_state.selected_indices))
            if checked:
                if idx not in st.session_state.selected_indices:
                    st.session_state.selected_indices.append(idx)
            else:
                if idx in st.session_state.selected_indices:
                    st.session_state.selected_indices.remove(idx)

        # Commit Mapping button
        if st.button("Submit Mapping"):
            if not st.session_state.selected_indices:
                st.warning("No vehicles selected.")
            else:
                selected_rows = matches_df.loc[st.session_state.selected_indices]
                st.success("Mapping submitted:")
                st.dataframe(selected_rows[['MODEL_YEAR','AD_MAKE','MODEL_NAME','AD_TRIM','AD_MFGCODE','STYLE_ID']])
                # Optionally save to GitHub
                save_mapping_to_github(selected_rows)
                # Clear selection
                st.session_state.selected_indices = []
