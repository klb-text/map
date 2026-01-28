
# basic_app.py — Read-only Mozenda outlet with alias-only exact lookup
# NEW: Renders a plain HTML <table> with stable ids/data-attributes so Mozenda can parse each cell.
# - Same data sources & logic as before (Mappings.csv, Aliases.csv, CADS.csv, exact alias, "No Vehicle Data")
# - In Mozenda mode (?mozenda=1) -> outputs ONLY the HTML table + meta div, then st.stop() (no Streamlit grid)
# - In human mode -> shows the same HTML table; optional checkbox to also display Streamlit dataframe if you want

import base64
import html
import io
from typing import Optional, Tuple, List
import pandas as pd
import streamlit as st
import requests

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping - Read Only (Alias Exact)", layout="wide")
st.title("AFF Vehicle Mapping (Read Only) — Alias Exact Lookup (HTML Table)")

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

# ---------------- Pull GitHub CSVs (read-only) ----------------
if not is_mozenda and st.button("Refresh from GitHub") or (not is_mozenda and refresh_qp):
    fetch_csv_github.clear()
    join_mappings_to_cads.clear()
    st.success("Cache cleared. Latest GitHub files will be fetched on this run.")

mappings_df = fetch_csv_github(GH_OWNER, GH_REPO, MAP_PATH, GH_BRANCH, GH_TOKEN)
aliases_df  = fetch_csv_github(GH_OWNER, GH_REPO, ALIASES_PATH, GH_BRANCH, GH_TOKEN)
joined_df   = join_mappings_to_cads(mappings_df, cads_df)

# ---------------- HTML table builder (parsable) ----------------
PREFERRED_COLS = ["year", "make", "model", "trim", "model_code", "source", "STYLE_ID", "canon_key"]

def to_stable_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PREFERRED_COLS)
    out = df.copy()
    for c in PREFERRED_COLS:
        if c not in out.columns:
            out[c] = ""
    remaining = [c for c in out.columns if c not in PREFERRED_COLS]
    return out[PREFERRED_COLS + remaining]

def build_html_table(df: pd.DataFrame, table_id: str = "mozenda-table") -> str:
    """
    Returns a minimal, semantic HTML table Mozenda can parse cell-by-cell.
    Each <td> includes data-col="colname"; each <tr> includes data-row="n".
    """
    safe_cols: List[str] = list(df.columns)
    # Header
    thead_cells = "".join(f"<th scope='col' data-col='{html.escape(c)}'>{html.escape(c)}</th>" for c in safe_cols)
    thead = f"<thead><tr>{thead_cells}</tr></thead>"

    # Body
    body_rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        tds = []
        for c in safe_cols:
            val = "" if pd.isna(row[c]) else str(row[c])
            tds.append(f"<td data-col='{html.escape(c)}'>{html.escape(val)}</td>")
        body_rows.append(f"<tr data-row='{i}' data-canon-key='{html.escape(str(row.get('canon_key','')))}'>{''.join(tds)}</tr>")
    tbody = f"<tbody>{''.join(body_rows)}</tbody>"

    # Simple CSS to keep it readable; avoid complex wrappers (Mozenda likes simple DOM)
    css = """
    <style>
      table#mozenda-table { border-collapse: collapse; width: 100%; }
      #mozenda-table th, #mozenda-table td { border: 1px solid #ccc; padding: 6px 8px; font-family: system-ui, sans-serif; font-size: 14px; }
      #mozenda-table thead th { background: #f7f7f7; }
    </style>
    """

    # Meta div for easy polling
    meta = f"<div id='mozenda-meta' data-ready='1' data-row-count='{len(df)}' data-no-data='0'></div>"

    return f"<!-- MOZ_TABLE_START -->{css}{meta}<table id='{table_id}' role='table'>{thead}{tbody}</table><!-- MOZ_TABLE_END -->"

def build_meta_no_data() -> str:
    # Header-only table (same schema) + meta no_data=1
    head_cells = "".join(f"<th scope='col' data-col='{html.escape(c)}'>{html.escape(c)}</th>" for c in PREFERRED_COLS)
    thead = f"<thead><tr>{head_cells}</tr></thead>"
    meta = "<div id='mozenda-meta' data-ready='1' data-row-count='0' data-no-data='1'></div>"
    css = """
    <style>
      table#mozenda-table { border-collapse: collapse; width: 100%; }
      #mozenda-table th, #mozenda-table td { border: 1px solid #ccc; padding: 6px 8px; font-family: system-ui, sans-serif; font-size: 14px; }
      #mozenda-table thead th { background: #f7f7f7; }
    </style>
    """
    return f"<!-- MOZ_TABLE_START -->{css}{meta}<table id='mozenda-table' role='table'>{thead}<tbody></tbody></table><!-- MOZ_TABLE_END -->"

# ---------------- Branch: Mozenda mode (HTML table only, then stop) ----------
if is_mozenda:
    res_df, is_no_data = exact_alias_filter_with_no_data(aliases_df, joined_df, cads_df, q_param)
    if is_no_data:
        st.markdown(build_meta_no_data(), unsafe_allow_html=True)
        st.stop()
    table_df = to_stable_schema(res_df)
    st.markdown(build_html_table(table_df), unsafe_allow_html=True)
    st.stop()  # ⛔ nothing else renders in Mozenda mode

# ---------------- Human mode --------------------------------------------------
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

    if is_no_data:
        st.warning("No Vehicle Data")
        st.markdown(build_meta_no_data(), unsafe_allow_html=True)
    else:
        table_df = to_stable_schema(res_df)
        st.markdown(build_html_table(table_df), unsafe_allow_html=True)

    # Optional: show Streamlit grid only when explicitly enabled
    show_grid = st.checkbox("Show Streamlit grid (optional)", value=False)
    if show_grid and not res_df.empty:
        st.dataframe(res_df, hide_index=True, use_container_width=True)

# --- EOF ---
