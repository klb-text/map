# app.py — Full Streamlit CADS / Vehicle Mapping Script

import streamlit as st
import pandas as pd
import json
from typing import Optional, List, Dict, Any

# -------------------- Helper functions --------------------
def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def _first_nonempty(*vals) -> str:
    for v in vals:
        if v is not None:
            sv = str(v).strip()
            if sv: return sv
    return ""

def _get_bool(name: str, default: bool) -> bool:
    v = params.get(name, [None])[0]
    if v is None: return default
    return str(v).strip() in ("1","true","True","yes","on")

def _get_float(name: str, default: float) -> float:
    v = params.get(name, [None])[0]
    try:
        return float(v) if v is not None else default
    except:
        return default

def _get_int(name: str, default: int) -> int:
    v = params.get(name, [None])[0]
    try:
        return int(v) if v is not None else default
    except:
        return default

def _get_str(name: str, default: str = "") -> str:
    v = params.get(name, [None])[0]
    return v if v is not None else default

def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure object columns are stripped of whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df

# -------------------- HTML Table Renderer --------------------
HARVEST_PREF_ORDER = ["AD_YEAR","AD_MAKE","AD_MODEL","MODEL_NAME","STYLE_NAME","AD_SERIES","Trim","AD_TRIM","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"]

def render_harvest_table(
    df: pd.DataFrame,
    table_id: str = "cads_harvest_table",
    preferred_order: Optional[List[str]] = None,
    visible_only_cols: Optional[List[str]] = None,
    include_attr_cols: Optional[List[str]] = None,
    caption: Optional[str] = None,
    plain: bool = False,
):
    if df is None or len(df) == 0:
        st.markdown("<p id='harvest-empty'>No rows</p>", unsafe_allow_html=True)
        return
    cols = list(df.columns)
    if visible_only_cols:
        cols = [c for c in cols if c in visible_only_cols]
    if preferred_order:
        front = [c for c in preferred_order if c in cols]
        back  = [c for c in cols if c not in front]
        cols = front + back
    style_key = "STYLE_ID" if "STYLE_ID" in df.columns else None
    veh_key   = "AD_VEH_ID" if "AD_VEH_ID" in df.columns else None
    attr_cols = include_attr_cols or []

    parts = []
    parts.append(f"<table id='{_html_escape(table_id)}' class='cads-harvest' data-source='harvest'>")
    if caption:
        parts.append(f"<caption>{_html_escape(caption)}</caption>")
    parts.append("<thead><tr>")
    for c in cols:
        parts.append(f"<th scope='col' data-col-key='{_html_escape(c)}'>{_html_escape(c)}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for idx, row in df.iterrows():
        row_key = _first_nonempty(row.get(style_key), row.get(veh_key), idx)
        tr_attrs = []
        if row_key != "":
            tr_attrs.append(f"data-row-key='{_html_escape(row_key)}'")
        for c in attr_cols:
            if c in df.columns:
                val = _first_nonempty(row.get(c))
                if val:
                    tr_attrs.append(f"data-{_html_escape(c).lower().replace(' ','_').replace('/','-') }='{_html_escape(val)}'")
        parts.append(f"<tr {' '.join(tr_attrs)}>")
        for c in cols:
            val = row.get(c, "")
            parts.append(f"<td data-col-key='{_html_escape(c)}'>{_html_escape(val)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    if not plain:
        css = """
        <style>
        table.cads-harvest { border-collapse: collapse; width: 100%; font: 13px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Arial; }
        table.cads-harvest th, table.cads-harvest td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
        table.cads-harvest thead th { position: sticky; top: 0; background: #f8f8f8; z-index: 2; }
        table.cads-harvest caption { text-align:left; font-weight:600; margin: 6px 0; }
        .harvest-note { margin: 6px 0 14px; color: #444; font-size: 12px; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    st.markdown("\n".join(parts), unsafe_allow_html=True)

# -------------------- Query Params & Loaders --------------------
params = st.experimental_get_query_params()
HARVEST_MODE   = (params.get("harvest", ["0"]) [0] == "1")
HARVEST_SOURCE = (params.get("source",  ["mapped"]) [0])  # mapped | inputs | quick_ymmt | quick_vehicle | unmapped | catalog

# -------------------- CADS Loaders --------------------
@st.cache_data(ttl=600)
def _load_cads_df(cads_path: Optional[str] = None, cads_is_excel: Optional[bool] = None, sheet_name: Optional[str] = None, ref: Optional[str] = None):
    path = cads_path if cads_path is not None else CADS_PATH
    is_xlsx = cads_is_excel if cads_is_excel is not None else CADS_IS_EXCEL
    ref = ref or GH_BRANCH
    if is_xlsx:
        sn = sheet_name if sheet_name is not None else CADS_SHEET_NAME_DEFAULT
        try: sn = int(sn)
        except Exception: pass
        return load_cads_from_github_excel(GH_OWNER, GH_REPO, path, GH_TOKEN, ref=ref, sheet_name=sn)
    return load_cads_from_github_csv(GH_OWNER, GH_REPO, path, GH_TOKEN, ref=ref)

# -------------------- Harvest Mode --------------------
def _run_harvest():
    st.markdown("<p>Harvest mode not fully implemented in this simplified version.</p>")

# -------------------- Main Streamlit UI --------------------
if HARVEST_MODE:
    _run_harvest()
    st.stop()

# Sidebar: CADS Settings
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value="data/CADS.csv")
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=False)
CADS_SHEET_NAME_DEFAULT = st.sidebar.text_input("Excel sheet name/index", value="Sheet1")
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

# Sidebar: Vehicle Catalog
st.sidebar.subheader("Vehicle Catalog")
VEH_CATALOG_PATH = st.sidebar.text_input("Vehicle Catalog path in repo", value="data/AFF Vehicles YMMT.csv")
st.session_state["veh_catalog_path"] = VEH_CATALOG_PATH

# Sidebar: Matching Controls
st.sidebar.subheader("Matching Controls")
TRIM_AS_HINT = st.sidebar.checkbox("Use Trim as hint", value=True)
TRIM_EXACT_ONLY = st.sidebar.checkbox("Trim must be exact", value=False)
MODEL_EXACT_WHEN_FULL = st.sidebar.checkbox("Model exact when multi-word", value=False)
STRICT_AND = st.sidebar.checkbox("Require strict AND across filters", value=True)
YEAR_REQUIRE_EXACT = st.sidebar.checkbox("Require exact year match", value=True)
STOPWORD_THRESHOLD = st.sidebar.slider("Stopword threshold", 0.1, 0.9, 0.6, 0.05)
TOKEN_MIN_LEN = st.sidebar.slider("Token minimum length", 1, 5, 2, 1)

# Effective model columns override
EFFECTIVE_MODEL_COLS_OVERRIDE = st.sidebar.text_input("Effective model columns (comma-separated)", value="AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES")
OVERRIDE_COLS = [c.strip() for c in EFFECTIVE_MODEL_COLS_OVERRIDE.split(",") if c.strip()] or None

# -------------------- Helper to load CADS --------------------
def _load_cads_df_ui():
    if cads_upload is not None:
        if cads_upload.name.lower().endswith(".xlsx"):
            return pd.read_excel(cads_upload, engine="openpyxl")
        return pd.read_csv(cads_upload)
    if CADS_IS_EXCEL:
        sheet_arg = CADS_SHEET_NAME_DEFAULT
        try: sheet_arg = int(sheet_arg)
        except Exception: pass
        return load_cads_from_github_excel("GH_OWNER", "GH_REPO", CADS_PATH, "GH_TOKEN", ref="main", sheet_name=sheet_arg)
    return load_cads_from_github_csv("GH_OWNER", "GH_REPO", CADS_PATH, "GH_TOKEN", ref="main")

# -------------------- Vehicle Quick Lookup --------------------
st.header("Vehicle Quick Lookup (catalog → CADS)")
veh_catalog_txt = st.text_input("Vehicle (Year Make Model [Trim])", placeholder="e.g., 2026 Lexus RX 350 Premium AWD")

if st.button("Find by Vehicle"):
    if not (veh_catalog_txt or "").strip():
        st.warning("Enter a vehicle string first.")
    else:
        try:
            df_cat = load_vehicle_catalog("GH_OWNER","GH_REPO", VEH_CATALOG_PATH, "GH_TOKEN", ref="main")
            cat_idx = build_catalog_index(df_cat)
            parsed = parse_vehicle_against_catalog(veh_catalog_txt, cat_idx)
            if not parsed:
                st.info("Could not parse Vehicle deterministically.")
            else:
                y_s, mk_s, md_s, tr_s = parsed["year"], parsed["make"], parsed["model"], parsed["trim"]
                st.caption(f"Catalog parsed → Y={y_s}, Make={mk_s}, Model={md_s}, Trim={tr_s}")
                df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
                results, diag = filter_cads_generic(
                    df_cads, y_s, mk_s, md_s, tr_s,
                    exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                    trim_exact_only=False,
                    strict_and=STRICT_AND,
                    stopword_threshold=STOPWORD_THRESHOLD,
                    token_min_len=TOKEN_MIN_LEN,
                    effective_model_cols_override=OVERRIDE_COLS,
                    trim_as_hint=True,
                    year_require_exact=YEAR_REQUIRE_EXACT,
                )
                if len(results)==0:
                    st.warning("No CADS rows matched.")
                else:
                    render_harvest_table(results, table_id="quick_vehicle_results", caption=f"{len(results)} CADS row(s) matched.")
        except Exception as e:
            st.error(f"Vehicle search failed: {e}")

# -------------------- Legacy Quick Vehicle Text Search --------------------
st.header("Legacy Quick Vehicle Text Search")
veh_legacy = st.text_input("Vehicle text (legacy contains search)", placeholder="e.g., 2026 Pacifica Select AWD")
if st.button("Find by Vehicle (legacy contains)"):
    df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
    res = find_rows_by_vehicle_text(df_cads, veh_legacy)
    if res is None or len(res)==0:
        st.info("No rows matched vehicle text.")
    else:
        render_harvest_table(res, table_id="legacy_search_results", caption=f"{len(res)} CADS row(s) matched.")
