
# basic_app.py — Option A: Pull Mappings.csv from GitHub (Contents API) for Mozenda
import base64, json, io, time
from typing import Optional, List
import pandas as pd
import streamlit as st
import requests
from rapidfuzz import fuzz
from pathlib import Path

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ---------------- Secrets / GitHub ----------------
gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")
GH_PATH   = gh_cfg.get("path", "Mappings.csv")  # default path inside the repo

# ---------------- File Paths (Local) ----------------
CADS_FILE = "CADS.csv"                   # Local CADS
VEHICLE_REF_FILE = "vehicle_example.txt" # Reference file for Make/Model/Trim

# ---------------- Utils ----------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Robust CSV loader that auto-detects delimiter."""
    return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")

def normalize(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower().replace("-", " ").replace("/", " ")
    return " ".join(s.split())

# ---------------- GitHub: Read CSV helper (Contents API) ----------------
@st.cache_data(show_spinner=False)
def fetch_csv_from_github(owner: str, repo: str, path: str, ref: str = "main", token: Optional[str] = None) -> pd.DataFrame:
    """
    Uses GitHub Contents API to read a CSV file (works with private repos when token is provided).
    Returns empty DataFrame if the file doesn't exist or on error.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            content_b64 = data.get("content", "")
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
        elif resp.status_code == 404:
            # File not found on GitHub (okay on first run)
            return pd.DataFrame()
        elif resp.status_code == 401:
            st.error("GitHub 401 Unauthorized: token invalid or not SSO-authorized.")
            return pd.DataFrame()
        elif resp.status_code == 403:
            st.error("GitHub 403 Forbidden: token lacks Contents: Read for this repo/branch.")
            return pd.DataFrame()
        else:
            st.error(f"GitHub fetch error for {path}: HTTP {resp.status_code} - {resp.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching GitHub CSV {path}: {e}")
        return pd.DataFrame()

# ---------------- GitHub: Upsert CSV (Contents API) ----------------
def github_upsert_csv(owner: str, repo: str, branch: str, file_path_in_repo: str,
                      token: str, new_rows_df: pd.DataFrame) -> tuple[bool, str]:
    """
    Upsert into a CSV in GitHub (read existing -> union -> dedupe -> PUT).
    Expects columns: [year, make, model, trim, model_code, source]
    """
    if new_rows_df is None or new_rows_df.empty:
        return False, "No rows to commit."

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # 1) Read existing file
    get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path_in_repo}?ref={branch}"
    r = requests.get(get_url, headers=headers, timeout=30)

    existing_df = pd.DataFrame(columns=["year","make","model","trim","model_code","source"])
    sha = None
    if r.status_code == 200:
        js = r.json()
        sha = js.get("sha")
        decoded = base64.b64decode(js.get("content", "")).decode("utf-8", errors="replace")
        existing_df = pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
    elif r.status_code not in (404,):
        return False, f"Failed fetching existing file: HTTP {r.status_code} - {r.text}"

    # 2) Merge & dedupe
    combined = pd.concat([existing_df, new_rows_df], ignore_index=True).fillna("")
    combined.drop_duplicates(inplace=True)
    combined = combined[["year","make","model","trim","model_code","source"]]
    combined = combined.sort_values(by=["year","make","model","trim","model_code"]).reset_index(drop=True)

    # 3) Encode CSV
    buf = io.StringIO()
    combined.to_csv(buf, index=False, encoding="utf-8")
    content_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("utf-8")

    # 4) PUT new content
    put_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path_in_repo}"
    payload = {
        "message": f"chore(mappings): upsert {len(new_rows_df)} row(s) via basic_app - {time.time()}",
        "content": content_b64,
        "branch": branch,
        "committer": {"name": gh_cfg.get("author_name","AFF Bot"),
                      "email": gh_cfg.get("author_email","aff-bot@example.com")}
    }
    if sha:
        payload["sha"] = sha

    r2 = requests.put(put_url, headers=headers, data=json.dumps(payload), timeout=30)
    if r2.status_code in (200, 201):
        return True, f"Committed to {owner}/{repo}@{branch}:{file_path_in_repo}"
    if r2.status_code == 403:
        return False, "403 Forbidden on commit. Check PAT Contents: Read/Write and branch protections."
    if r2.status_code == 401:
        return False, "401 Unauthorized. Rotate PAT and authorize SSO."
    return False, f"Commit failed: HTTP {r2.status_code} - {r2.text}"

# ---------------- Load Data ----------------
try:
    cads_df = load_csv(CADS_FILE)
except FileNotFoundError:
    st.error(f"{CADS_FILE} not found.")
    st.stop()

# Normalize CADS critical cols
for col in ["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE","STYLE_ID"]:
    if col not in cads_df.columns:
        cads_df[col] = ""
cads_df = cads_df.fillna("").astype(str)

# Vehicle reference (optional; used by fuzzy match helper)
try:
    vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
except Exception:
    vehicle_ref_df = pd.DataFrame(columns=["Year","Make","Model","Trim"])

# ---------------- Load Mappings from GitHub (Option A) ----------------
# Expected columns in Mappings.csv: year,make,model,trim,model_code,source
mappings_df = fetch_csv_from_github(GH_OWNER, GH_REPO, GH_PATH, GH_BRANCH, GH_TOKEN)

# ---------------- Session State ----------------
if "vehicle_input" not in st.session_state:
    st.session_state.vehicle_input = ""
if "harvested_df" not in st.session_state:
    st.session_state.harvested_df = pd.DataFrame()
if "selected_rows" not in st.session_state:
    st.session_state.selected_rows = []

# ---------------- Sidebar ----------------
st.sidebar.header("Mapping Controls")
threshold = st.sidebar.slider("Fuzzy Match Threshold", 70, 100, 85)

# ---------------- Vehicle Input ----------------
vehicle_input = st.text_input("Enter Vehicle Name", st.session_state.vehicle_input,
                              placeholder="e.g., 2025 Land Rover Range Rover P400 SE SWB")
st.session_state.vehicle_input = vehicle_input

# ---------------- Fuzzy Matching ----------------
def fuzzy_match_vehicle(vehicle_name: str, ref_df: pd.DataFrame, threshold: int=90) -> List[int]:
    """
    Returns list of indices in ref_df that match vehicle_name above threshold
    """
    matches = []
    q = normalize(vehicle_name)
    for idx, row in ref_df.iterrows():
        combined = " ".join([str(row.get(c,"")) for c in ["Year","Make","Model","Trim"]]).strip()
        score = fuzz.token_set_ratio(q, normalize(combined))
        if score >= threshold:
            matches.append(idx)
    return matches

# ---------------- Harvest / Selection ----------------
def harvest_vehicle(vehicle_name: str, threshold: int=90):
    """
    Harvest CADS rows either by matching against vehicle_ref_df or manual YMMT fallback.
    """
    matched_indices = fuzzy_match_vehicle(vehicle_name, vehicle_ref_df, threshold) if not vehicle_ref_df.empty else []
    harvested = pd.DataFrame()

    if matched_indices:
        for idx in matched_indices:
            ref_row = vehicle_ref_df.loc[idx]
            year  = str(ref_row.get("Year", ""))
            make  = str(ref_row.get("Make", ""))
            model = str(ref_row.get("Model", ""))
            trim  = str(ref_row.get("Trim", ""))

            cand = cads_df[
                (cads_df["MODEL_YEAR"].astype(str) == str(year)) &
                (cads_df["AD_MAKE"].str.lower()  == make.lower()) &
                (cads_df["AD_MODEL"].str.lower() == model.lower())
            ]
            if trim:
                cand = cand[cand["TRIM"].str.lower() == trim.lower()]
            harvested = pd.concat([harvested, cand], ignore_index=False)
    else:
        st.info("No close match found. Please enter Year / Make / Model / Trim below.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            year = st.text_input("Year (exact)", value="")
        with col2:
            make = st.text_input("Make (exact)", value="")
        with col3:
            model = st.text_input("Model (exact)", value="")
        with col4:
            trim = st.text_input("Trim (optional)", value="")

        cand = cads_df[
            (cads_df["MODEL_YEAR"].astype(str) == str(year)) &
            (cads_df["AD_MAKE"].str.lower()  == make.lower()) &
            (cads_df["AD_MODEL"].str.lower() == model.lower())
        ]
        if trim:
            cand = cand[cand["TRIM"].str.lower() == trim.lower()]
        harvested = cand

    harvested = harvested.drop_duplicates()
    return harvested, matched_indices

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
        label = f"{row.get('AD_MAKE','')} {row.get('AD_MODEL','')} {row.get('TRIM','')} ({row.get('MODEL_YEAR','')}) | Model Code: {row.get('AD_MFGCODE','')} | Style ID: {row.get('STYLE_ID','')}"
        st.session_state[key] = st.checkbox(label, st.session_state[key])
        if st.session_state[key]:
            selected_indices.append(idx)
    return selected_indices

# ---------------- Mozenda Mode (API-like outlet) ----------------
# Returns joined data: Mappings (GitHub) + CADS (local), minimal chrome
params = st.experimental_get_query_params()
is_mozenda = params.get("mozenda", ["0"])[0] == "1"
out_format = params.get("format", ["html"])[0].lower()  # html|csv|json

@st.cache_data(show_spinner=False)
def build_mozenda_output(mappings_df: pd.DataFrame, cads_df: pd.DataFrame) -> pd.DataFrame:
    if mappings_df.empty:
        return pd.DataFrame()
    md = mappings_df.rename(columns=str).fillna("").astype(str)
    cd = cads_df.rename(columns=str).fillna("").astype(str)

    # Normalize expected header (year,make,model,trim,model_code,source)
    expected_cols = ["year","make","model","trim","model_code","source"]
    for c in expected_cols:
        if c not in md.columns:
            md[c] = ""

    merged = md.merge(
        cd,
        left_on=["year","make","model","trim","model_code"],
        right_on=["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"],
        how="left",
        suffixes=("","_cad")
    )

    # Keep a compact set that Mozenda can ingest
    cols = ["year","make","model","trim","model_code","source","STYLE_ID"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[cols].reset_index(drop=True)

if is_mozenda:
    out_df = build_mozenda_output(mappings_df, cads_df)
    if out_format == "csv":
        st.write(out_df.to_csv(index=False))
    elif out_format == "json":
        st.write(out_df.to_json(orient="records"))
    else:
        st.dataframe(out_df, hide_index=True, use_container_width=True)
    st.stop()

# ---------------- Run Harvest (UI) ----------------
if vehicle_input:
    st.subheader("Harvest Mode Output")
    harvested_df, matched_indices = harvest_vehicle(vehicle_input, threshold)
    st.session_state.harvested_df = harvested_df
    selected_indices = render_selection_table(harvested_df)
    st.session_state.selected_rows = selected_indices

    # Build canonical mappings rows from selections and (optionally) commit to GitHub
    if selected_indices:
        sel_df = harvested_df.loc[selected_indices].copy()
        to_commit = sel_df.rename(columns={
            "MODEL_YEAR": "year",
            "AD_MAKE": "make",
            "AD_MODEL": "model",
            "TRIM": "trim",
            "AD_MFGCODE": "model_code"
        })[["year","make","model","trim","model_code"]]
        to_commit["source"] = "user"
        for c in to_commit.columns:
            to_commit[c] = to_commit[c].astype(str).str.strip()
        to_commit.drop_duplicates(inplace=True)

        st.success("Rows selected. Preview of canonical mappings to commit:")
        st.dataframe(to_commit, use_container_width=True)

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Commit Selected to GitHub Mappings.csv"):
                if not GH_TOKEN or not GH_OWNER or not GH_REPO:
                    st.error("GitHub secrets not configured. Cannot commit.")
                else:
                    ok, msg = github_upsert_csv(
                        owner=GH_OWNER, repo=GH_REPO, branch=GH_BRANCH,
                        file_path_in_repo=GH_PATH, token=GH_TOKEN,
                        new_rows_df=to_commit
                    )
                    if ok:
                        st.success(msg)
                        # Refresh cached GitHub pull so Mozenda endpoint stays current
                        fetch_csv_from_github.clear()
                        _ = fetch_csv_from_github(GH_OWNER, GH_REPO, GH_PATH, GH_BRANCH, GH_TOKEN)
                    else:
                        st.error(msg)
        with c2:
            st.download_button(
                "Download Selected as CSV",
                data=to_commit.to_csv(index=False),
                file_name="Mappings_to_commit.csv",
                mime="text/csv"
            )

# ---------------- Display Summary ----------------
st.subheader("Summary")
st.write(f"Total CADS rows: {len(cads_df)}")
if not mappings_df.empty:
    st.write(f"Mappings in GitHub: {len(mappings_df)} (path: {GH_PATH})")
else:
    st.write("Mappings in GitHub: 0 (no file found yet)")

st.caption("Mozenda endpoints: add `?mozenda=1&format=csv|html|json` to this page URL to retrieve joined data.")

# --- EOF ---
