
# basic_app.py — Read-only Mozenda outlet with alias-only exact lookup
# NEW: Renders a plain HTML <table> with stable ids/data-attributes so Mozenda can parse each cell.
# - Same data sources & logic as before (Mappings.csv, Aliases.csv, CADS.csv, exact alias, "No Vehicle Data")
# - In Mozenda mode (?mozenda=1) -> outputs ONLY the HTML table + meta div, then st.stop() (no Streamlit grid)
# - In human mode -> shows the same HTML table; optional checkbox to also display Streamlit dataframe if you want
# - Adds status message under Search: "Vehicle has Mapping" | "No Build Vehicle Data" | "Vehicle Not Mapped"
# - Always includes status text in HTML (status div + table caption), even for empty/no-data cases.

import base64
import csv
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

# Missing configuration warnings (won't stop app, but helpful)
if not GH_OWNER or not GH_REPO:
    st.warning("GitHub repository configuration is missing (owner/repo). CSV fetch may return empty.")

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
        # Modern preference is Bearer (token still works)
        h["Authorization"] = f"Bearer {token}"
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

def _sniff_then_read_csv(text: str) -> pd.DataFrame:
    """Try to auto-detect delimiter; fallback to comma."""
    try:
        sample = text[:2048]
        try:
            dialect = csv.Sniffer().sniff(sample)
            sep = dialect.delimiter
        except Exception:
            sep = None  # pandas infers with engine='python'
        return pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str).fillna("")
    except Exception:
        # Last resort
        return pd.read_csv(io.StringIO(text), dtype=str).fillna("")

def _fetch_csv_contents_api(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(_gh_contents_url(owner, repo, path, ref), headers=_gh_headers(token), timeout=20)
        if r.status_code == 200:
            content_b64 = r.json().get("content", "")
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return _sniff_then_read_csv(decoded)
        return None
    except Exception:
        return None

def _fetch_csv_raw(owner: str, repo: str, path: str, ref: str) -> Optional[pd.DataFrame]:
    try:
        raw_url = _gh_raw_url(owner, repo, path, ref)
        r = requests.get(raw_url, timeout=20)
        if r.status_code == 200:
            return _sniff_then_read_csv(r.text)
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
    if not owner or not repo:
        return pd.DataFrame()
    df = _fetch_csv_contents_api(owner, repo, path, ref, token)
    if df is None:
        raw_df = _fetch_csv_raw(owner, repo, path, ref)
        df = raw_df if raw_df is not None else pd.DataFrame()
    return df

def _normalize_join_keys(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Strip spaces and coerce to string for join stability."""
    if df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).map(lambda x: x.strip())
        else:
            out[c] = ""
    return out

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

    # Defensive normalization for join stability
    md = _normalize_join_keys(md, ["year","make","model","trim","model_code"])
    cd = _normalize_join_keys(cd, ["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE","STYLE_ID"])

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
    if "alias_norm" in al.columns:
        al["alias_norm"] = al["alias_norm"].astype(str).map(normalize)
    else:
        al["alias_norm"] = al["alias"].astype(str).map(normalize)

    hits = al[al["alias_norm"] == qn]
    if hits.empty:
        return pd.DataFrame(), False

    # Treat rows with status missing as mapped (back-compat)
    mapped_mask = (hits["status"].str.lower() == "mapped") | (hits["status"] == "")
    mapped_hits = hits.loc[mapped_mask].copy()

    if not mapped_hits.empty:
        mapped_hits = mapped_hits.assign(
            canon_key=(
                mapped_hits["year"].astype(str) + "|" +
                mapped_hits["make"].astype(str) + "|" +
                mapped_hits["model"].astype(str) + "|" +
                mapped_hits["trim"].astype(str) + "|" +
                mapped_hits["model_code"].astype(str)
            )
        )
        keys = set(mapped_hits["canon_key"].tolist())
        out = joined_df[joined_df["canon_key"].isin(keys)].copy()
        if not out.empty:
            return out.reset_index(drop=True), False

        # Fallback: join alias canonical directly to CADS if join hasn't populated yet
        alias_canon = mapped_hits[["year","make","model","trim","model_code"]].drop_duplicates()
        if cads_df.empty:
            return pd.DataFrame(), False

        alias_canon = _normalize_join_keys(alias_canon, ["year","make","model","trim","model_code"])
        cd_norm = _normalize_join_keys(cads_df, ["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE","STYLE_ID"])

        direct = alias_canon.merge(
            cd_norm,
            left_on=["year","make","model","trim","model_code"],
            right_on=["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"],
            how="left"
        )
        direct = direct.assign(
            source="alias",
            canon_key=(
                direct["year"].astype(str) + "|" +
                direct["make"].astype(str) + "|" +
                direct["model"].astype(str) + "|" +
                direct["trim"].astype(str) + "|" +
                direct["model_code"].astype(str)
            )
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

# ---------------- Status helpers ----------------
def determine_alias_status(aliases_df: pd.DataFrame, alias_query: str) -> str:
    """
    Returns one of: 'mapped', 'no_data', 'unmapped'
      - 'mapped'  -> there exists at least one alias row where status is 'mapped' or '' (back-compat)
      - 'no_data' -> there exists at least one alias row where status is 'no_data' and no mapped rows
      - 'unmapped'-> no alias rows match (or no status recognized)
    """
    if aliases_df.empty:
        return "unmapped"

    qn = normalize(alias_query)
    if not qn:
        return "unmapped"

    al = aliases_df.rename(columns=str).copy()
    for c in ["alias", "alias_norm", "status"]:
        if c not in al.columns:
            al[c] = ""
    al["alias_norm"] = al["alias_norm"].astype(str).map(normalize) if "alias_norm" in al.columns else al["alias"].astype(str).map(normalize)

    hits = al[al["alias_norm"] == qn]
    if hits.empty:
        return "unmapped"

    mapped = hits[(hits["status"].str.lower() == "mapped") | (hits["status"] == "")]
    if not mapped.empty:
        return "mapped"

    no_data = hits[hits["status"].str.lower() == "no_data"]
    if not no_data.empty:
        return "no_data"

    return "unmapped"

def _status_text_from_code(code: str) -> str:
    if code == "mapped":
        return "Vehicle has Mapping"
    if code == "no_data":
        return "No Build Vehicle Data"
    return "Vehicle Not Mapped"

def build_status_div(status_code: str, status_text: str) -> str:
    """
    A small status block Mozenda can optionally read.
    Example: <div id="mozenda-status" data-status="mapped">Vehicle has Mapping</div>
    """
    return f"<div id='mozenda-status' data-status='{html.escape(status_code)}'>{html.escape(status_text)}</div>"

# ---------------- HTML table builder (parsable) ----------------
PREFERRED_COLS = ["year", "make", "model", "trim", "model_code", "source", "STYLE_ID", "canon_key"]

def to_stable_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PREFERRED_COLS)
    out = df.copy()
    # Coerce preferred cols to string to avoid NaN emergence
    for c in PREFERRED_COLS:
        if c not in out.columns:
            out[c] = ""
        else:
            out[c] = out[c].astype(str).fillna("")
    remaining = [c for c in out.columns if c not in PREFERRED_COLS]
    return out[PREFERRED_COLS + remaining]

def build_html_table(
    df: pd.DataFrame,
    table_id: str = "mozenda-table",
    status_text: Optional[str] = None,
    status_code: Optional[str] = None
) -> str:
    """
    Returns a minimal, semantic HTML table Mozenda can parse cell-by-cell.
    Each <td> includes data-col="colname"; each <tr> includes data-row="n".
    Prepends a status <div> and includes a <caption> with the same text when provided.
    """
    safe_cols: List[str] = list(df.columns)
    caption = f"<caption>{html.escape(status_text)}</caption>" if status_text else ""
    thead_cells = "".join(f"<th scope='col' data-col='{html.escape(c)}'>{html.escape(c)}</th>" for c in safe_cols)
    thead = f"<thead><tr>{thead_cells}</tr></thead>"

    body_rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        tds = []
        for c in safe_cols:
            val = "" if pd.isna(row[c]) else str(row[c])
            tds.append(f"<td data-col='{html.escape(c)}'>{html.escape(val)}</td>")
        body_rows.append(f"<tr data-row='{i}' data-canon-key='{html.escape(str(row.get('canon_key','')))}'>{''.join(tds)}</tr>")
    tbody = f"<tbody>{''.join(body_rows)}</tbody>"

    css = """
    <style>
      table#mozenda-table { border-collapse: collapse; width: 100%; }
      #mozenda-table th, #mozenda-table td { border: 1px solid #ccc; padding: 6px 8px; font-family: system-ui, sans-serif; font-size: 14px; }
      #mozenda-table thead th { background: #f7f7f7; }
      #mozenda-table caption { caption-side: top; text-align: left; padding: 6px 0 8px; font-weight: 600; color: #333; }
    </style>
    """
    status_block = build_status_div(status_code or "unknown", status_text or "")
    # Include status in meta for quick agent checks
    meta = (
        f"<div id='mozenda-meta' data-ready='1' data-row-count='{len(df)}' "
        f"data-no-data='0' data-status='{html.escape(status_code or 'unknown')}'></div>"
    )

    return f"<!-- MOZ_TABLE_START -->{css}{status_block}{meta}<table id='{table_id}' role='table'>{caption}{thead}{tbody}</table><!-- MOZ_TABLE_END -->"

def build_meta_no_data(status_text: str, status_code: str) -> str:
    """
    Header-only table (same schema) + status block + meta with data-no-data=1.
    Includes a <caption> so there's visible text Mozenda can read even when no rows.
    """
    head_cells = "".join(f"<th scope='col' data-col='{html.escape(c)}'>{html.escape(c)}</th>" for c in PREFERRED_COLS)
    thead = f"<thead><tr>{head_cells}</tr></thead>"
    caption = f"<caption>{html.escape(status_text)}</caption>"
    status_block = build_status_div(status_code, status_text)
    meta = (
        f"<div id='mozenda-meta' data-ready='1' data-row-count='0' data-no-data='1' "
        f"data-status='{html.escape(status_code)}'></div>"
    )
    css = """
    <style>
      table#mozenda-table { border-collapse: collapse; width: 100%; }
      #mozenda-table th, #mozenda-table td { border: 1px solid #ccc; padding: 6px 8px; font-family: system-ui, sans-serif; font-size: 14px; }
      #mozenda-table thead th { background: #f7f7f7; }
      #mozenda-table caption { caption-side: top; text-align: left; padding: 6px 0 8px; font-weight: 600; color: #333; }
    </style>
    """
    return f"<!-- MOZ_TABLE_START -->{css}{status_block}{meta}<table id='mozenda-table' role='table'>{caption}{thead}<tbody></tbody></table><!-- MOZ_TABLE_END -->"

# ---------------- Load CADS ----------------
try:
    cads_df = load_local_csv(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load {CADS_FILE}: {e}")
    st.stop()

# ---------------- Query Params ----------------
# Use new API where available; fall back to experimental for older Streamlit
try:
    params = st.query_params  # type: ignore[attr-defined]
    q_param = params.get("q", "")
    refresh_qp = params.get("refresh", "0") == "1"
    moz_param = str(params.get("mozenda", "0")).strip().lower()
except Exception:
    params = st.experimental_get_query_params()
    q_param = params.get("q", [""])[0]
    refresh_qp = params.get("refresh", ["0"])[0] == "1"
    moz_param = params.get("mozenda", ["0"])[0].strip().lower()

is_mozenda = moz_param in ("1", "true", "yes", "y")

# ---------------- Pull GitHub CSVs (read-only) ----------------
refresh_clicked = False
if not is_mozenda:
    refresh_clicked = st.button("Refresh from GitHub")

if (not is_mozenda) and (refresh_clicked or refresh_qp):
    fetch_csv_github.clear()
    join_mappings_to_cads.clear()
    st.success("Cache cleared. Latest GitHub files will be fetched on this run.")

mappings_df = fetch_csv_github(GH_OWNER, GH_REPO, MAP_PATH, GH_BRANCH, GH_TOKEN)
aliases_df  = fetch_csv_github(GH_OWNER, GH_REPO, ALIASES_PATH, GH_BRANCH, GH_TOKEN)
joined_df   = join_mappings_to_cads(mappings_df, cads_df)

# ---------------- Branch: Mozenda mode (HTML table only, then stop) ----------
if is_mozenda:
    status_code = determine_alias_status(aliases_df, q_param)
    status_text = _status_text_from_code(status_code)

    res_df, is_no_data = exact_alias_filter_with_no_data(aliases_df, joined_df, cads_df, q_param)

    # If no_data or unmapped with empty result, return header-only table + status text
    if is_no_data or (status_code in ("no_data", "unmapped") and res_df.empty):
        st.markdown(build_meta_no_data(status_text=status_text, status_code=status_code), unsafe_allow_html=True)
        st.stop()

    table_df = to_stable_schema(res_df)
    st.markdown(build_html_table(table_df, status_text=status_text, status_code=status_code), unsafe_allow_html=True)
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
    try:
        st.query_params.clear()  # New API
    except Exception:
        st.experimental_set_query_params()
    q_input = ""

if do_search or q_input:
    # Compute status first (what to show under the button)
    status_code = determine_alias_status(aliases_df, q_input)
    status_text = _status_text_from_code(status_code)

    # Show the status message (right under the search button area)
    if status_code == "mapped":
        st.success(status_text)
    elif status_code == "no_data":
        st.warning(status_text)
    else:
        st.info(status_text)

    # Get data
    res_df, is_no_data = exact_alias_filter_with_no_data(aliases_df, joined_df, cads_df, q_input)

    # Always render HTML carrying the same status text (caption + status div)
    if is_no_data or (status_code in ("no_data", "unmapped") and res_df.empty):
        st.markdown(build_meta_no_data(status_text=status_text, status_code=status_code), unsafe_allow_html=True)
    else:
        table_df = to_stable_schema(res_df)
        st.markdown(build_html_table(table_df, status_text=status_text, status_code=status_code), unsafe_allow_html=True)

    # Optional: show Streamlit grid (explicit only)
    show_grid = st.checkbox("Show Streamlit grid (optional)", value=False)
    if show_grid and not res_df.empty:
        st.dataframe(res_df, hide_index=True, use_container_width=True)

# --- EOF ---
