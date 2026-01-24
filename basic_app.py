
# basic_app.py — Read-only Mozenda outlet:
# - Pulls Mappings.csv from GitHub (Contents API if token ok; fallback to raw URL)
# - Joins with local CADS.csv to add STYLE_ID (and other CADS fields if you want)
# - Exposes ?mozenda=1&format=csv|html|json endpoint for Mozenda

import base64, io
from typing import Optional
import pandas as pd
import streamlit as st
import requests

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping - Read Only", layout="wide")
st.title("AFF Vehicle Mapping (Read Only)")

# ---------------- Secrets / GitHub ----------------
gh = st.secrets.get("github", {})
GH_TOKEN  = gh.get("token")           # optional if repo is public
GH_OWNER  = gh.get("owner")           # e.g., "klb-text"
GH_REPO   = gh.get("repo")            # e.g., "map"
GH_BRANCH = gh.get("branch", "main")
MAP_PATH  = gh.get("path", "Mappings.csv")  # default Mappings.csv at repo root

# ---------------- Local Files ----------------
CADS_FILE = "CADS.csv"  # local CADS.csv
# Expected columns in Mappings.csv: year,make,model,trim,model_code,source
# Expected CADS columns: MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE, STYLE_ID

# ---------------- Utils ----------------
@st.cache_data
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")

def _gh_headers(token: Optional[str]):
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def _gh_contents_url(owner: str, repo: str, path: str, ref: str):
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"

def _gh_raw_url(owner: str, repo: str, path: str, ref: str):
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

@st.cache_data(show_spinner=False)
def fetch_mappings_github(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> pd.DataFrame:
    """
    Try Contents API (supports private) with PAT; on 401/403 or missing token,
    fallback to raw.githubusercontent.com (works for public).
    Returns empty DF if not found or error.
    """
    # 1) Contents API (private or public)
    try:
        r = requests.get(_gh_contents_url(owner, repo, path, ref), headers=_gh_headers(token), timeout=20)
        if r.status_code == 200:
            content_b64 = r.json().get("content", "")
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
        elif r.status_code in (401, 403):
            # Auth issue; try raw fallback next
            pass
        elif r.status_code == 404:
            # Not found; try raw fallback next (in case of public or CDN delay)
            pass
        else:
            # Unexpected; try raw fallback
            pass
    except Exception:
        # Try raw fallback
        pass

    # 2) Raw URL fallback (public only)
    try:
        raw_url = _gh_raw_url(owner, repo, path, ref)
        r2 = requests.get(raw_url, timeout=20)
        if r2.status_code == 200:
            return pd.read_csv(io.StringIO(r2.text), dtype=str).fillna("")
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def build_mozenda_output(mappings_df: pd.DataFrame, cads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Mappings -> CADS on (year, make, model, trim, model_code) -> (MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE).
    Output minimal set Mozenda needs; extend as necessary.
    """
    if mappings_df.empty or cads_df.empty:
        return pd.DataFrame()

    md = mappings_df.rename(columns=str).fillna("").astype(str)
    cd = cads_df.rename(columns=str).fillna("").astype(str)

    # Ensure required columns exist
    for c in ["year","make","model","trim","model_code","source"]:
        if c not in md.columns:
            md[c] = ""

    for c in ["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE","STYLE_ID"]:
        if c not in cd.columns:
            cd[c] = ""

    merged = md.merge(
        cd,
        left_on=["year","make","model","trim","model_code"],
        right_on=["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"],
        how="left",
        suffixes=("","_cad")
    )

    # Keep a tight, stable schema for Mozenda
    cols = ["year","make","model","trim","model_code","source","STYLE_ID"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[cols].reset_index(drop=True)

# ---------------- Load CADS ----------------
try:
    cads_df = load_local_csv(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load {CADS_FILE}: {e}")
    st.stop()

# ---------------- Pull Mappings (read-only) ----------------
mappings_df = fetch_mappings_github(GH_OWNER, GH_REPO, MAP_PATH, GH_BRANCH, GH_TOKEN)

# ---------------- Mozenda Mode: API-like outlet ----------------
params = st.experimental_get_query_params()
is_mozenda = params.get("mozenda", ["0"])[0] == "1"
out_format = params.get("format", ["csv"])[0].lower()  # default to csv for Mozenda

if is_mozenda:
    out_df = build_mozenda_output(mappings_df, cads_df)
    if out_format == "json":
        st.write(out_df.to_json(orient="records"))
    elif out_format == "html":
        st.dataframe(out_df, hide_index=True, use_container_width=True)
    else:
        # CSV default (ideal for Mozenda)
        st.write(out_df.to_csv(index=False))
    st.stop()

# ---------------- Simple Preview UI (for humans) ----------------
st.caption("This page serves a read-only outlet for Mozenda. Use the query param ?mozenda=1&format=csv|html|json")

# Show helpful status
col1, col2 = st.columns(2)
with col1:
    st.subheader("Mappings (GitHub)")
    if mappings_df.empty:
        if GH_TOKEN:
            st.warning("Mappings.csv not found or unauthorized. If repo is private, ensure the token is valid and SSO-authorized.")
        else:
            st.warning("Mappings.csv not found via raw. If repo is private, add a token; if public, ensure the file exists.")
    else:
        st.success(f"Loaded {len(mappings_df)} mapping rows from GitHub: {GH_OWNER}/{GH_REPO}@{GH_BRANCH}:{MAP_PATH}")
        st.dataframe(mappings_df.head(50), use_container_width=True)

with col2:
    st.subheader("CADS (Local)")
    st.info(f"Rows: {len(cads_df)}")
    st.dataframe(cads_df.head(50), use_container_width=True)

st.subheader("Mozenda Endpoints")
base_url = st.request.url.split("?")[0] if hasattr(st, "request") else ""
st.code(f"{base_url}?mozenda=1&format=csv", language="text")
st.code(f"{base_url}?mozenda=1&format=html", language="text")
st.code(f"{base_url}?mozenda=1&format=json", language="text")

# --- EOF ---
