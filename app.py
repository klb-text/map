import base64, json, io, re, difflib, time
from typing import Optional, List, Dict, Tuple, Set, Any
import requests, pandas as pd, streamlit as st
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
from github import Github

# ===================== Page Config =====================
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ===================== Secrets / Config =====================
gh_cfg = st.secrets.get("github", {})
GH_TOKEN = gh_cfg.get("token")
GH_OWNER = gh_cfg.get("owner")
GH_REPO = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

# ===================== File Paths =====================
CADS_FILE = "CADS.csv"
MAPPINGS_FILE = "Mappings.csv"

# ===================== GitHub Integration =====================
def get_github_file_content(path: str) -> str:
    g = Github(GH_TOKEN)
    repo = g.get_repo(f"{GH_OWNER}/{GH_REPO}")
    try:
        file_content = repo.get_contents(path, ref=GH_BRANCH)
        return base64.b64decode(file_content.content).decode()
    except:
        return ""

def push_github_file(path: str, content: str, message: str):
    g = Github(GH_TOKEN)
    repo = g.get_repo(f"{GH_OWNER}/{GH_REPO}")
    try:
        file_content = repo.get_contents(path, ref=GH_BRANCH)
        repo.update_file(path, message, content, file_content.sha, branch=GH_BRANCH)
    except:
        repo.create_file(path, message, content, branch=GH_BRANCH)

# ===================== Load CADS =====================
@st.cache_data
def load_cads(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

# ===================== Load / Save Mappings =====================
def load_mappings() -> pd.DataFrame:
    if Path(MAPPINGS_FILE).exists():
        return pd.read_csv(MAPPINGS_FILE)
    content = get_github_file_content(MAPPINGS_FILE)
    if content:
        return pd.read_csv(io.StringIO(content))
    return pd.DataFrame(columns=["vehicle", "year", "make", "model", "trim", "STYLE_ID"])

def save_mappings(df: pd.DataFrame):
    df.to_csv(MAPPINGS_FILE, index=False)
    push_github_file(MAPPINGS_FILE, df.to_csv(index=False), "Update vehicle mappings")

# ===================== Vehicle Matching =====================
def smart_vehicle_match(vehicle: str, cads_df: pd.DataFrame) -> pd.DataFrame:
    vehicle_lower = vehicle.lower()
    # Exact / close match search
    mask = cads_df.apply(lambda row: vehicle_lower in str(row["MODEL_NAME"]).lower(), axis=1)
    matches = cads_df[mask]
    if not matches.empty:
        return matches
    # Fallback: YMM/T match
    return cads_df[cads_df.apply(lambda row: vehicle_lower in " ".join([str(row.get(col,"")).lower() for col in ["MODEL_YEAR","DIVISION_NAME","MODEL_NAME","STYLE_NAME","TRIM"]]), axis=1)]

# ===================== Session State =====================
if "selected_rows" not in st.session_state:
    st.session_state["selected_rows"] = []

# ===================== Main App =====================
cads_df = load_cads(CADS_FILE)
mappings_df = load_mappings()

vehicle_input = st.text_input("Enter Vehicle Name", "")

if vehicle_input:
    filtered_df = smart_vehicle_match(vehicle_input, cads_df)
    st.write(f"Found {len(filtered_df)} matching CADS rows for '{vehicle_input}'")

    for idx, row in filtered_df.iterrows():
        key = f"{row['STYLE_ID']}_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        st.session_state[key] = st.checkbox(
            f"{row['MODEL_YEAR']}-{row['DIVISION_NAME']}-{row['MODEL_NAME']}-{row['STYLE_NAME']}-{row['TRIM']} (STYLE_ID: {row['STYLE_ID']})",
            value=st.session_state[key]
        )
        if st.session_state[key]:
            if row['STYLE_ID'] not in st.session_state["selected_rows"]:
                st.session_state["selected_rows"].append(row['STYLE_ID'])
        else:
            if row['STYLE_ID'] in st.session_state["selected_rows"]:
                st.session_state["selected_rows"].remove(row['STYLE_ID'])

if st.session_state["selected_rows"]:
    st.subheader("Selected Mappings:")
    selected_df = cads_df[cads_df['STYLE_ID'].isin(st.session_state["selected_rows"])].copy()
    st.dataframe(selected_df)
    # Merge/update mappings
    for _, r in selected_df.iterrows():
        vehicle_name = vehicle_input
        exists = mappings_df["STYLE_ID"].isin([r["STYLE_ID"]])
        if not exists.any():
            mappings_df = pd.concat([mappings_df, pd.DataFrame([{
                "vehicle": vehicle_name,
                "year": r["MODEL_YEAR"],
                "make": r["DIVISION_NAME"],
                "model": r["MODEL_NAME"],
                "trim": r["TRIM"],
                "STYLE_ID": r["STYLE_ID"]
            }])], ignore_index=True)
    save_mappings(mappings_df)
    st.success("Mappings updated and synced to GitHub!")

# ===================== Display Harvest / Debug =====================
st.subheader("Harvest Mode Output")
st.write("Harvesting from manual inputs...")
st.write(f"Total CADS rows: {len(cads_df)}")
st.write(f"Total Mappings: {len(mappings_df)}")
