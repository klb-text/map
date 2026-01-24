
# basic_app.py — Read-only Mozenda outlet with alias-only exact lookup (no fuzzy)
# - Pulls Mappings.csv (canonical) and Aliases.csv (alias -> canonical) from GitHub
# - Joins canonical mappings to local CADS.csv to add STYLE_ID (and any CADS fields you want)
# - Search bar + Mozenda endpoints use EXACT alias match (case-insensitive, whitespace-normalized)
# - If alias isn't found, returns no results

import base64
import io
from typing import Optional
import pandas as pd
import streamlit as st
import requests

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping - Read Only (Alias Exact)", layout="wide")
st.title("AFF Vehicle Mapping (Read Only) — Alias Exact Lookup")

# ---------------- Secrets / GitHub ----------------
gh = st.secrets.get("github", {})
GH_TOKEN   = gh.get("token")             # optional if repo is public
GH_OWNER   = gh.get("owner")             # e.g., "klb-text"
GH_REPO    = gh.get("repo")              # e.g., "map"
GH_BRANCH  = gh.get("branch", "main")
MAP_PATH   = gh.get("path", "Mappings.csv")            # canonical file
ALIASES_PATH = gh.get("aliases_path", "Aliases.csv")   # alias -> canonical file

# ---------------- Local Files ----------------
CADS_FILE = "CADS.csv"  # local CADS.csv
# Expected:
#   Mappings.csv columns: year,make,model,trim,model_code,source
#   Aliases.csv  columns: alias,alias_norm,year,make,model,trim,model_code,source,created_at
#   CADS columns: MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE, STYLE_ID

# ---------------- Utils ----------------
@st.cache_data
def load_local_csv(path: str) -> pd.DataFrame:
    """Robust CSV loader that auto-detects delimiter and returns strings."""
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

def normalize(s: str) -> str:
    """Case-insensitive, collapses whitespace and common separators for exact-ish equality."""
    s = str(s or "")
    s = s.strip().lower().replace("-", " ").replace("/", " ")
    return " ".join(s.split())

def _fetch_csv_contents_api(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(_gh_contents_url(owner, repo, path, ref), headers=_gh_headers(token), timeout=20)
        if r.status_code == 200:
            content_b64 = r.json().get("content", "")
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
        return None
    except Exception:
        return None

def _fetch_csv_raw(owner: str, repo: str, path: str, ref: str) -> Optional[pd.DataFrame]:
    try:
        raw_url = _gh_raw_url(owner, repo, path, ref)
        r = requests.get(raw_url, timeout=20)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text), dtype=str).fillna("")
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fetch_csv_github(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> pd.DataFrame:
    """
    Try Contents API first (handles private repos with PAT), fallback to raw (for public).
    Returns empty DF if not found or error.
    """
    df = _fetch_csv_contents_api(owner, repo, path, ref, token)
    if df is None:
        raw_df = _fetch_csv_raw(owner, repo, path, ref)
        df = raw_df if raw_df is not None else pd.DataFrame()
    return df

@st.cache_data(show_spinner=False)
def join_mappings_to_cads(mappings_df: pd.DataFrame, cads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join canonical Mappings -> CADS on (year,make,model,trim,model_code).
    Adds a stable 'canon_key' for fast lookups.
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

    merged["canon_key"] = (
        merged["year"].astype(str) + "|" +
        merged["make"].astype(str) + "|" +
        merged["model"].astype(str) + "|" +
        merged["trim"].astype(str) + "|" +
        merged["model_code"].astype(str)
    )

    # Keep a compact, stable schema; add more CADS fields here if needed
    cols = ["year","make","model","trim","model_code","source","STYLE_ID","canon_key"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[cols].reset_index(drop=True)

def exact_alias_filter(
    aliases_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    alias_query: str
) -> pd.DataFrame:
    """
    Exact (normalized) alias-only pipeline:
      - Normalize the input alias string.
      - Match aliases_df['alias_norm'] == normalized input.
      - Collect canonical keys; constrain joined_df to those canonical keys.
      - If no alias match -> return empty.
    """
    if joined_df.empty or aliases_df.empty:
        return pd.DataFrame()

    qn = normalize(alias_query)
    if not qn:
        return pd.DataFrame()

    al = aliases_df.copy()
    # Ensure alias_norm exists and normalized
    if "alias_norm" not in al.columns:
        al["alias_norm"] = al["alias"].astype(str).map(normalize)
    else:
        al["alias_norm"] = al["alias_norm"].astype(str).map(normalize)

    # Build canon_key in aliases
    required_cols = ["year","make","model","trim","model_code"]
    for c in required_cols:
        if c not in al.columns:
            al[c] = ""
    al["canon_key"] = (
        al["year"].astype(str) + "|" +
        al["make"].astype(str) + "|" +
        al["model"].astype(str) + "|" +
        al["trim"].astype(str) + "|" +
        al["model_code"].astype(str)
    )

    # Exact normalized match only (no fuzzy)
    hits = al[al["alias_norm"] == qn]
    if hits.empty:
        return pd.DataFrame()

    keys = set(hits["canon_key"].tolist())
    out = joined_df[joined_df["canon_key"].isin(keys)].copy()
    return out.reset_index(drop=True)

# ---------------- Load CADS ----------------
try:
    cads_df = load_local_csv(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load {CADS_FILE}: {e}")
    st.stop()

# ---------------- Pull GitHub CSVs (read-only) ----------------
mappings_df = fetch_csv_github(GH_OWNER, GH_REPO, MAP_PATH, GH_BRANCH, GH_TOKEN)
aliases_df  = fetch_csv_github(GH_OWNER, GH_REPO, ALIASES_PATH, GH_BRANCH, GH_TOKEN)

# ---------------- Build joined dataset ----------------
joined_df = join_mappings_to_cads(mappings_df, cads_df)

# ---------------- Query params (for Mozenda and deep links) ----------------
params     = st.experimental_get_query_params()
q_param    = params.get("q", [""])[0]                  # the exact alias you mapped in app.py
is_mozenda = params.get("mozenda", ["0"])[0] == "1"
out_format = params.get("format", ["csv"])[0].lower()  # csv|html|json (csv default)

# ---------------- Search UI (Exact alias) ----------------
st.header("Search by Alias (Exact, case-insensitive)")
with st.form("search_form"):
    q_input = st.text_input(
        "Alias (paste exactly what you mapped in app.py)",
        value=q_param,
        placeholder="e.g., 2026 Integra FWD Continuously Variable Transmission"
    )
    do_search = st.form_submit_button("Search")

clear_clicked = st.button("Clear")

if "search_results" not in st.session_state:
    st.session_state["search_results"] = pd.DataFrame()
if "search_alias" not in st.session_state:
    st.session_state["search_alias"] = q_param

if clear_clicked:
    st.session_state["search_alias"] = ""
    st.session_state["search_results"] = pd.DataFrame()
    st.experimental_set_query_params()
elif do_search:
    st.session_state["search_alias"] = q_input
    st.session_state["search_results"] = exact_alias_filter(aliases_df, joined_df, q_input)
    st.experimental_set_query_params(q=q_input)

# ---------------- Mozenda Mode: API-like outlet ----------------
if is_mozenda:
    out_df = exact_alias_filter(aliases_df, joined_df, q_param)
    payload_df = out_df  # already minimal schema; add/remove columns as needed
    if out_format == "json":
        st.write(payload_df.to_json(orient="records"))
    elif out_format == "html":
        st.dataframe(payload_df, hide_index=True, use_container_width=True)
    else:
        st.write(payload_df.to_csv(index=False))
    st.stop()

# ---------------- Human-friendly Preview ----------------
if mappings_df.empty:
    st.warning("Mappings.csv not found (or unauthorized).")
else:
    st.success(f"Loaded {len(mappings_df)} canonical mappings from GitHub: {GH_OWNER}/{GH_REPO}@{GH_BRANCH}:{MAP_PATH}")

if aliases_df.empty:
    st.warning("Aliases.csv not found. Searches require alias entries created in app.py.")
else:
    st.caption(f"Aliases available: {len(aliases_df)}")

st.subheader("Results")
results_df = st.session_state["search_results"]
if results_df.empty and st.session_state["search_alias"]:
    st.info("No results for that alias. Ensure you pasted the exact title you mapped in app.py.")
elif not results_df.empty:
    st.dataframe(results_df, hide_index=True, use_container_width=True)

# Helpful endpoints
st.markdown("---")
st.subheader("Mozenda Endpoints")
base_url = st.request.url.split("?")[0] if hasattr(st, "request") else ""
st.code(f"{base_url}?mozenda=1&format=csv&q=2026%20Integra%20FWD%20Continuously%20Variable%20Transmission", language="text")
st.code(f"{base_url}?mozenda=1&format=json&q=2026%20Integra%20FWD%20Continuously%20Variable%20Transmission", language="text")
st.code(f"{base_url}?mozenda=1&format=html&q=2026%20Integra%20FWD%20Continuously%20Variable%20Transmission", language="text")

# --- EOF ---
