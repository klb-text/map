# app.py
import base64, json, io, re, difflib, time
from typing import Optional, List, Dict, Tuple, Set, Any
import pandas as pd
import streamlit as st
from requests.adapters import HTTPAdapter, Retry
import requests
from rapidfuzz import fuzz
from pathlib import Path
import csv

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ---------------- Secrets / GitHub ----------------
gh_cfg = st.secrets.get("github", {})
GH_TOKEN = gh_cfg.get("token")
GH_OWNER = gh_cfg.get("owner")
GH_REPO = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

# ---------------- File Paths ----------------
CADS_FILE = "CADS.csv"
MAPPINGS_FILE = "Mappings.csv"
VEHICLE_REF_FILE = "vehicle_example.csv"  # Reference file for Make/Model/Trim

# ---------------- Utils ----------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_mappings_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

# ---------------- GitHub sync ----------------
def push_mappings_to_github(path: str):
    if not GH_TOKEN:
        st.warning("No GitHub token provided. Mappings not synced.")
        return
    import base64
    from github import Github
    g = Github(GH_TOKEN)
    repo = g.get_repo(f"{GH_OWNER}/{GH_REPO}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        file = repo.get_contents(path, ref=GH_BRANCH)
        repo.update_file(file.path, f"Update mappings {time.time()}", content, file.sha, branch=GH_BRANCH)
    except Exception as e:
        repo.create_file(path, f"Create mappings {time.time()}", content, branch=GH_BRANCH)

# ---------------- Load Data ----------------
try:
    cads_df = load_csv(CADS_FILE)
except FileNotFoundError:
    st.error(f"{CADS_FILE} not found.")
    st.stop()

# Load mappings if exists
if Path(MAPPINGS_FILE).exists():
    mappings_df = load_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["vehicle_input","matched_indices"])

# Load vehicle reference
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# ---------------- Session State ----------------
if "vehicle_input" not in st.session_state:
    st.session_state.vehicle_input = ""
if "harvested_df" not in st.session_state:
    st.session_state.harvested_df = pd.DataFrame()
if "selected_rows" not in st.session_state:
    st.session_state.selected_rows = []

# ---------------- Sidebar ----------------
st.sidebar.header("Mapping Controls")
threshold = st.sidebar.slider("Fuzzy Match Threshold", 80, 100, 90)

# ---------------- Vehicle Input ----------------
vehicle_input = st.text_input("Enter Vehicle Name", st.session_state.vehicle_input)
st.session_state.vehicle_input = vehicle_input

# ---------------- Fuzzy Matching ----------------
def fuzzy_match_vehicle(vehicle_name: str, ref_df: pd.DataFrame, threshold: int=90) -> List[int]:
    """
    Returns list of indices in ref_df that match vehicle_name above threshold
    """
    matches = []
    for idx, row in ref_df.iterrows():
        combined = " ".join([str(row.get(c,"")) for c in ["Year","Make","Model","Trim"]]).strip()
        score = fuzz.ratio(vehicle_name.lower(), combined.lower())
        if score >= threshold:
            matches.append(idx)
    return matches

# ---------------- Harvest / Selection ----------------
def harvest_vehicle(vehicle_name: str, threshold: int=90):
    # Fuzzy match against vehicle reference
    matched_indices = fuzzy_match_vehicle(vehicle_name, vehicle_ref_df, threshold)
    harvested = pd.DataFrame()
    if matched_indices:
        # Pull CADS rows matching reference
        for idx in matched_indices:
            ref_row = vehicle_ref_df.loc[idx]
            year = ref_row.get("Year")
            make = ref_row.get("Make")
            model = ref_row.get("Model")
            trim = ref_row.get("Trim")
            cad_rows = cads_df[
                (cads_df.get("MODEL_YEAR")==year) &
                (cads_df.get("AD_MAKE", "").str.lower()==str(make).lower()) &
                (cads_df.get("AD_MODEL", "").str.lower()==str(model).lower())
            ]
            if trim:
                cad_rows = cad_rows[cad_rows.get("AD_TRIM","").str.lower()==str(trim).lower()]
            harvested = pd.concat([harvested, cad_rows])
    else:
        # fallback: ask for full YMM/T input
        st.info("No close match found. Please enter Year / Make / Model / Trim below.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
        with col2:
            make = st.text_input("Make")
        with col3:
            model = st.text_input("Model")
        with col4:
            trim = st.text_input("Trim")
        cad_rows = cads_df[
            (cads_df.get("MODEL_YEAR")==year) &
            (cads_df.get("AD_MAKE","").str.lower()==make.lower()) &
            (cads_df.get("AD_MODEL","").str.lower()==model.lower())
        ]
        if trim:
            cad_rows = cad_rows[cad_rows.get("AD_TRIM","").str.lower()==trim.lower()]
        harvested = cad_rows
    harvested = harvested.drop_duplicates()
    return harvested, matched_indices

# ---------------- Selection Table ----------------
def render_selection_table(df: pd.DataFrame) -> List[int]:
    selected_indices = []
    st.write("CADS Results")
    if df.empty:
        st.write("No rows found.")
        return selected_indices
    for idx, row in df.iterrows():
        key = f"row_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        label = f"{row.get('AD_MAKE','')} {row.get('AD_MODEL','')} {row.get('AD_TRIM','')} ({row.get('MODEL_YEAR','')})"
        st.session_state[key] = st.checkbox(label, st.session_state[key])
        if st.session_state[key]:
            selected_indices.append(idx)
    return selected_indices

# ---------------- Run Harvest ----------------
if vehicle_input:
    st.subheader("Harvest Mode Output")
    harvested_df, matched_indices = harvest_vehicle(vehicle_input, threshold)
    st.session_state.harvested_df = harvested_df
    selected_indices = render_selection_table(harvested_df)
    st.session_state.selected_rows = selected_indices
    # Save mapping
    if selected_indices:
        existing = mappings_df[mappings_df.vehicle_input==vehicle_input]
        if not existing.empty:
            mappings_df = mappings_df[mappings_df.vehicle_input!=vehicle_input]
        mappings_df = pd.concat([mappings_df, pd.DataFrame([{
            "vehicle_input": vehicle_input,
            "matched_indices": json.dumps(selected_indices)
        }])], ignore_index=True)
        save_mappings_to_csv(mappings_df, MAPPINGS_FILE)
        push_mappings_to_github(MAPPINGS_FILE)

# ---------------- Display Summary ----------------
st.subheader("Summary")
st.write(f"Total CADS rows: {len(cads_df)}")
st.write(f"Total Mappings: {len(mappings_df)}")
st.write(f"Selected rows for current vehicle: {len(selected_indices)}")
