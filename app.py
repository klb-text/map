import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import os
import requests

# --- CONFIG ---
VEHICLE_REF_FILE = "vehicle_example.txt"
CADS_FILE = "CADS.csv"
MAPPINGS_FILE = "mappings.csv"

# GitHub settings
GITHUB_TOKEN = "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"
GITHUB_OWNER = "klb-text"
GITHUB_REPO = "map"
GITHUB_BRANCH = "main"

# --- UTILITY FUNCTIONS ---
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            on_bad_lines='skip'
        )
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()
    return df

def smart_vehicle_match(cads_df, vehicle_input, example_make="", example_model=""):
    """
    Filters CADS by make/model if available, then fuzzy matches vehicle_input.
    Returns sorted matches and raw scores.
    """
    df = cads_df.copy()
    # Filter by make/model if provided
    if example_make:
        df = df[df['AD_MAKE'].str.lower() == example_make.lower()]
    if example_model:
        df = df[df['AD_MODEL'].str.lower() == example_model.lower()]

    if df.empty:
        return pd.DataFrame(), []

    # Create searchable string
    df['vehicle_search'] = df[['AD_YEAR','AD_MAKE','AD_MODEL','AD_TRIM']].fillna("").astype(str).agg(' '.join, axis=1)

    choices = df['vehicle_search'].tolist()
    raw_matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=20)
    matched_indices = [i for i, _ in sorted(enumerate(raw_matches), key=lambda x: -x[1][1])]
    matches_df = df.iloc[[x[0] for x in raw_matches if x[1] >= 50]].copy()  # threshold 50
    return matches_df, raw_matches

def save_mapping(mapping_df):
    mapping_df.to_csv(MAPPINGS_FILE, index=False)
    # GitHub sync
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{MAPPINGS_FILE}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    # Get SHA
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha", None)
    import base64
    content = base64.b64encode(mapping_df.to_csv(index=False).encode()).decode()
    data = {"message":"update mappings","content":content,"branch":GITHUB_BRANCH}
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=headers, json=data)
    if r.status_code in [200,201]:
        st.success("Mappings synced to GitHub")
    else:
        st.warning(f"Failed to sync GitHub: {r.text}")

# --- LOAD FILES ---
# --- LOAD FILES ---
def load_vehicle_ref(path):
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines='skip')
        df.columns = df.columns.str.strip()  # remove leading/trailing whitespace
        # Ensure required columns exist
        for col in ["Vehicle", "Year", "Make", "Model", "VehicleAttributes", "Trim"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

vehicle_ref_df = load_vehicle_ref(VEHICLE_REF_FILE)

# --- STREAMLIT UI ---
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
with st.expander("YMMT Filter (optional)"):
    year_filter = st.text_input("Year")
    make_filter = st.text_input("Make")
    model_filter = st.text_input("Model")
    trim_filter = st.text_input("Trim")

if st.button("Search Vehicle") and vehicle_input:
    # Try to find make/model from reference
    example_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].str.lower() == vehicle_input.lower()]
    example_make = example_row['Make'].values[0] if not example_row.empty else ""
    example_model = example_row['Model'].values[0] if not example_row.empty else ""

    matches_df, raw_matches = smart_vehicle_match(cads_df, vehicle_input, example_make, example_model)

    if matches_df.empty:
        st.warning("No matching vehicles found")
    else:
        st.write(f"Smart match found: {vehicle_input}")
        # Apply optional YMMT filters
        if year_filter:
            matches_df = matches_df[matches_df['AD_YEAR'] == year_filter]
        if make_filter:
            matches_df = matches_df[matches_df['AD_MAKE'].str.lower() == make_filter.lower()]
        if model_filter:
            matches_df = matches_df[matches_df['AD_MODEL'].str.lower() == model_filter.lower()]
        if trim_filter:
            matches_df = matches_df[matches_df['AD_TRIM'].str.lower() == trim_filter.lower()]

        # Display selectable table
        st.write("Select applicable vehicle(s):")
        selected_indices = []
        for idx, row in matches_df.iterrows():
            checkbox = st.checkbox(
                f"{row['AD_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['AD_TRIM']} | Model Code: {row['AD_MFGCODE']}",
                key=f"chk_{idx}"
            )
            if checkbox:
                selected_indices.append(idx)
        if st.button("Submit Mapping"):
            for idx in selected_indices:
                sel_row = matches_df.loc[idx]
                mappings_df = pd.concat([
                    mappings_df,
                    pd.DataFrame([{
                        "Vehicle": vehicle_input,
                        "AD_VEH_ID": sel_row["AD_VEH_ID"]
                    }])
                ], ignore_index=True)
            save_mapping(mappings_df)
            st.success(f"Vehicle '{vehicle_input}' mapped successfully!")

# Display existing mappings
if not mappings_df.empty:
    st.subheader("Existing Mappings")
    st.dataframe(mappings_df)
