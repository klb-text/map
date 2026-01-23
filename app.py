import pandas as pd
import streamlit as st
import os
import csv
import requests
from io import StringIO
from rapidfuzz import process, fuzz

# ----------------- CONFIG -----------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "mappings.csv"

# GitHub config (from secrets)
GH_TOKEN = st.secrets["github"]["token"]
GH_OWNER = st.secrets["github"]["owner"]
GH_REPO = st.secrets["github"]["repo"]
GH_BRANCH = st.secrets["github"]["branch"]

# ----------------- HELPERS -----------------
def load_csv(path: str, sep=","):
    """Load CSV or tab-delimited file."""
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep, encoding="utf-8")


def save_mappings(df: pd.DataFrame):
    """Save mappings locally and push to GitHub."""
    df.to_csv(MAPPINGS_FILE, index=False)
    # Push to GitHub via API
    try:
        import base64, json
        url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{MAPPINGS_FILE}"
        # Get SHA if file exists
        r = requests.get(url, headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"})
        sha = r.json().get("sha")
        content = base64.b64encode(df.to_csv(index=False).encode("utf-8")).decode("utf-8")
        data = {
            "message": f"Update mappings",
            "branch": GH_BRANCH,
            "content": content,
        }
        if sha:
            data["sha"] = sha
        r2 = requests.put(url, headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}, data=json.dumps(data))
        if r2.status_code in [200, 201]:
            st.success("Mappings pushed to GitHub successfully.")
        else:
            st.warning(f"Could not push mappings to GitHub: {r2.status_code}")
    except Exception as e:
        st.warning(f"Error pushing to GitHub: {e}")


def smart_match_vehicle(vehicle_name: str, ref_df: pd.DataFrame, threshold=80):
    """Return best matching vehicle from reference."""
    choices = ref_df["Vehicle"].tolist()
    match, score, idx = process.extractOne(vehicle_name, choices, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return ref_df.iloc[idx]
    return None


# ----------------- LOAD FILES -----------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE, sep="\t")

# Load mappings if exists
if os.path.exists(MAPPINGS_FILE):
    mappings_df = load_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["vehicle", "year", "make", "model", "trim", "cads_style_id"])


# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# Vehicle input
vehicle_input = st.text_input("Enter Vehicle Name (freeform)")

# Button to search CADS
if st.button("Search CADS / Map Vehicle"):
    if vehicle_input:
        match_row = smart_match_vehicle(vehicle_input, vehicle_ref_df)
        if match_row is not None:
            st.info(f"Smart match found: {match_row['Vehicle']}")
            year = match_row["Year"]
            make = match_row["Make"]
            model = match_row["Model"]
            trim = match_row.get("Trim", "")
        else:
            st.warning("No smart match found. Please enter Year, Make, Model, Trim manually:")
            year = st.text_input("Year")
            make = st.text_input("Make")
            model = st.text_input("Model")
            trim = st.text_input("Trim")

        # Filter CADS for potential matches
        filtered_cads = cads_df[
            (cads_df["MODEL_YEAR"].astype(str) == str(year)) &
            (cads_df["DIVISION_NAME"].str.lower() == make.lower()) &
            (cads_df["MODEL_NAME"].str.lower() == model.lower())
        ]
        if trim:
            filtered_cads = filtered_cads[filtered_cads["STYLE_NAME"].str.contains(trim, case=False, na=False)]

        st.subheader("CADS Matches")
        st.dataframe(filtered_cads)

        # Select CADS row to map
        if not filtered_cads.empty:
            selected_idx = st.selectbox("Select CADS style to map", filtered_cads.index.tolist())
            selected_cads = filtered_cads.loc[selected_idx]
            new_mapping = {
                "vehicle": vehicle_input,
                "year": year,
                "make": make,
                "model": model,
                "trim": trim,
                "cads_style_id": selected_cads["STYLE_ID"]
            }
            mappings_df = pd.concat([mappings_df, pd.DataFrame([new_mapping])], ignore_index=True)
            st.success(f"Vehicle mapped to CADS ID: {selected_cads['STYLE_ID']}")
            save_mappings(mappings_df)

# Show existing mappings
st.subheader("Existing Mappings")
st.dataframe(mappings_df)

# Show total counts
st.info(f"Total CADS rows: {len(cads_df)}")
st.info(f"Total Mappings: {len(mappings_df)}")
