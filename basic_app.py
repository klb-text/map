
# basic_app.py — Read-only Mozenda outlet with alias-exact lookup and plain-text table output
# - Pulls Mappings.csv (canonical) and Aliases.csv (alias -> canonical) from GitHub
# - Joins canonical mappings to local CADS.csv to add STYLE_ID (and any CADS fields you want)
# - EXACT alias match (case-insensitive, whitespace-normalized)
# - If alias exists with status='no_data' but no mapped rows, returns "No Vehicle Data"
# - NEW: In Mozenda mode (?mozenda=1), prints ONLY plain-text TSV/CSV blocks and halts.
# - Human mode shows the same plain-text blocks (no table), with an optional checkbox to render a grid.

import base64
import io
from typing import Optional, Tuple, List
import pandas as pd
import streamlit as st
import requests

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping - Read Only (Alias Exact)", layout="wide")
st.title("AFF Vehicle Mapping (Read Only) — Alias Exact Lookup (Plain Text)")

# ---------------- Secrets / GitHub ----------------
gh = st.secrets.get("github", {})
GH_TOKEN     = gh.get("token")                 # optional if repo is public
GH_OWNER     = gh.get("owner")                 # e.g., "klb-text"
GH_REPO      = gh.get("repo")                  # e.g., "map"
GH_BRANCH    = gh.get("branch", "main")
MAP_PATH     = gh.get("path", "Mappings.csv")            # canonical mappings
ALIASES_PATH = gh.get("aliases_path", "Aliases.csv")     # alias -> canonical (may include 'status')

# ---------------- Local Files ----------------
CADS_FILE = "CADS.csv"  # local CADS.csv
# Expected:
#   Mappings.csv columns: year,make,model,trim,model_code,source
#   Aliases.csv  columns: alias,alias_norm,year,make,model,trim,model_code,source,status,created_at
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

# TTL so updates propagate without manual refresh; refresh button/flag also supported
@st.cache_data(show_spinner=False, ttl=60)
def fetch_csv_github(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> pd.DataFrame:
    """
    Try Contents API first (handles private repos with PAT), fallback to raw (for public).
    Returns empty DF if not found or error. Cached for ttl seconds.
    """
    df = _fetch_csv_contents_api(owner, repo, path, ref, token)
    if df is None:
        raw_df = _fetch_csv_raw(owner, repo, path, ref)
        df = raw_df if raw_df is not None else pd.DataFrame()
    return df

@st.cache_data(show_spinner=False, ttl=60)
def join_mappings_to_cads(mappings_df: pd.DataFrame, cads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join canonical Mappings -> CADS on (year, make, model, trim, model_code).
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

def exact_alias_filter_with_no_data(
    aliases_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    cads_df: pd.DataFrame,
    alias_query: str
) -> Tuple[pd.DataFrame, bool]:
    """
    Exact alias-only:
      - If alias (normalized) exists with mapped canonical -> return mapped rows from joined_df (or alias->CADS fallback).
      - If alias exists only with status='no_data' (and no mapped canonical present) -> return (empty, True).
      - Else -> return (empty, False).
    """
    if aliases_df.empty:
        return pd.DataFrame(), False

    qn = normalize(alias_query)
    if not qn:
        return pd.DataFrame(), False

    al = aliases_df.rename(columns=str).copy()
    # Ensure expected columns exist
    for c in ["alias","alias_norm","year","make","model","trim","model_code","source","status","created_at"]:
        if c not in al.columns:
            al[c] = ""
    # Normalize alias_norm
    al["alias_norm"] = al["alias_norm"].astype(str).map(normalize) if "alias_norm" in al.columns else al["alias"].astype(str).map(normalize)

    hits = al[al["alias_norm"] == qn]
    if hits.empty:
        return pd.DataFrame(), False

    # Treat rows with status missing as mapped (back-compat)
    mapped_hits = hits[(hits["status"].str.lower() == "mapped") | (hits["status"] == "")]
    if not mapped_hits.empty:
        # Build keys and try joined_df first
        mapped_hits["canon_key"] = (
            mapped_hits["year"].astype(str) + "|" +
            mapped_hits["make"].astype(str) + "|" +
            mapped_hits["model"].astype(str) + "|" +
            mapped_hits["trim"].astype(str) + "|" +
            mapped_hits["model_code"].astype(str)
        )
        keys = set(mapped_hits["canon_key"].tolist())
        out = joined_df[joined_df["canon_key"].isin(keys)].copy()
        if not out.empty:
            return out.reset_index(drop=True), False

        # Fallback: join alias canonical directly to CADS if join hasn't populated yet
        alias_canon = mapped_hits[["year","make","model","trim","model_code"]].drop_duplicates()
        if cads_df.empty:
            return pd.DataFrame(), False
        direct = alias_canon.merge(
            cads_df,
            left_on=["year","make","model","trim","model_code"],
            right_on=["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"],
            how="left"
        )
        direct["source"] = "alias"
        direct["canon_key"] = (
            direct["year"].astype(str) + "|" +
            direct["make"].astype(str) + "|" +
            direct["model"].astype(str) + "|" +
            direct["trim"].astype(str) + "|" +
            direct["model_code"].astype(str)
        )
        cols = ["year","make","model","trim","model_code","source","STYLE_ID","canon_key"]
        for c in cols:
            if c not in direct.columns:
                direct[c] = ""
        return direct[cols].reset_index(drop=True), False

    # If there are any no_data hits AND no mapped hits, show the "No Vehicle Data" state
    no_data_hits = hits[hits["status"].str.lower() == "no_data"]
    if not no_data_hits.empty:
        return pd.DataFrame(), True

    return pd.DataFrame(), False

# ---------------- Load CADS ----------------
try:
    cads_df = load_local_csv(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load {CADS_FILE}: {e}")
    st.stop()

# ---------------- Query Params ----------------
params     = st.experimental_get_query_params()
q_param    = params.get("q", [""])[0]               # alias string to look up
refresh_qp = params.get("refresh", ["0"])[0] == "1" # ?refresh=1 to force cache clear
moz_param  = params.get("mozenda", ["0"])[0].strip().lower()
is_mozenda = moz_param in ("1", "true", "yes", "y")

# ---------------- Refresh controls (human mode only) ----------------
if not is_mozenda:
    if st.button("Refresh from GitHub") or refresh_qp:
        fetch_csv_github.clear()
        join_mappings_to_cads.clear()
        st.success("Cache cleared. Latest GitHub files will be fetched on this run.")

# ---------------- Pull GitHub CSVs (read-only) ----------------
mappings_df = fetch_csv_github(GH_OWNER, GH_REPO, MAP_PATH, GH_BRANCH, GH_TOKEN)
aliases_df  = fetch_csv_github(GH_OWNER, GH_REPO, ALIASES_PATH, GH_BRANCH, GH_TOKEN)

# ---------------- Build joined dataset ----------------
joined_df = join_mappings_to_cads(mappings_df, cads_df)

# ---------------- Helper: emit plain-text blocks ----------------
def emit_plain_text_blocks(res_df: pd.DataFrame, is_no_data: bool):
    """
    Always emit:
      READY_FLAG
      NO_VEHICLE_DATA
      ROW_COUNT
      HARVEST_TSV_START ... HARVEST_TSV_END  (tab-separated)
      HARVEST_CSV_START ... HARVEST_CSV_END  (comma-separated)
    """
    preferred_cols = ["year", "make", "model", "trim", "model_code", "source", "STYLE_ID", "canon_key"]

    if res_df.empty:
        res_df = pd.DataFrame(columns=preferred_cols)
    else:
        for c in preferred_cols:
            if c not in res_df.columns:
                res_df[c] = ""
        remaining = [c for c in res_df.columns if c not in preferred_cols]
        res_df = res_df[preferred_cols + remaining]

    ready = 1
    no_data_flag = 1 if is_no_data else 0
    row_count = len(res_df)

    # Small meta header Mozenda can parse quickly
    st.code(
        f"READY_FLAG={ready}\nNO_VEHICLE_DATA={no_data_flag}\nROW_COUNT={row_count}",
        language="text"
    )

    # TSV block (plain text table; best for line-by-line parsing)
    st.code(
        "HARVEST_TSV_START\n"
        + res_df.to_csv(index=False, sep="\t")
        + "HARVEST_TSV_END",
        language="text"
    )

    # Optional CSV block (kept for convenience; Mozenda can ignore if using TSV)
    st.code(
        "HARVEST_CSV_START\n"
        + res_df.to_csv(index=False)
        + "HARVEST_CSV_END",
        language="text"
    )

# ---------------- Mozenda Mode: CSV/TSV-only outlet (short-circuit) ----------
if is_mozenda:
    res_df, is_no_data = exact_alias_filter_with_no_data(aliases_df, joined_df, cads_df, q_param)
    emit_plain_text_blocks(res_df, is_no_data)
    st.stop()  # ⛔ absolutely no grid/table below this point

# ---------------- Human-friendly UI (plain text by default; optional grid) ---
st.header("Search by Alias (Exact, case-insensitive)")
with st.form("search_form"):
    q_input = st.text_input(
        "Alias (paste exactly what you mapped in app.py)",
        value=q_param,
        placeholder="e.g., 2026 Integra or 2026 Integra FWD Continuously Variable Transmission"
    )
    do_search = st.form_submit_button("Search")

clear_clicked = st.button("Clear")
if clear_clicked:
    st.experimental_set_query_params()
    q_input = ""

if do_search or q_input:
    res_df, is_no_data = exact_alias_filter_with_no_data(aliases_df, joined_df, cads_df, q_input)
    emit_plain_text_blocks(res_df, is_no_data)

    # Optional: allow a grid only when explicitly requested
    show_grid = st.checkbox("Show grid (optional)", value=False)
    if show_grid and not res_df.empty:
        st.dataframe(res_df, hide_index=True, use_container_width=True)

# --- EOF ---
