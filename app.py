import base64, json, io, re, difflib, time
from typing import Optional, List, Dict, Tuple, Set, Any
import requests, pandas as pd, streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ===================== Page Config =====================
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ===================== Secrets / Config =====================
gh_cfg = st.secrets.get("github", {})
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")
GH_TOKEN  = gh_cfg.get("token")
CADS_PATH = "data/CADS.csv"
CADS_IS_EXCEL = False
CADS_SHEET_NAME_DEFAULT = 0
MAPPINGS_PATH = "data/mappings.json"

# ===================== Utility Functions =====================
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

HARVEST_PREF_ORDER = ["AD_YEAR","AD_MAKE","AD_MODEL","MODEL_NAME","STYLE_NAME","AD_SERIES","Trim","AD_TRIM","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"]

# ===================== Render CADS Tables =====================
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

# ===================== Query Params =====================
params = st.experimental_get_query_params()
HARVEST_MODE   = (params.get("harvest", ["0"]) [0] == "1")
HARVEST_SOURCE = (params.get("source",  ["mapped"]) [0])  # mapped | inputs | quick_ymmt | quick_vehicle | unmapped | catalog

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

# ===================== CADS Loaders =====================
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

# ===================== HARVEST MODE =====================
def _run_harvest():
    trim_as_hint         = _get_bool("trim_as_hint", True)
    trim_exact_only      = _get_bool("trim_exact_only", False)
    strict_and           = _get_bool("strict_and", True)
    model_exact_when_full= _get_bool("model_exact_when_full", False)
    year_require_exact   = _get_bool("year_require_exact", True)
    stopword_threshold   = _get_float("stopword_threshold", 0.60)
    token_min_len        = _get_int("token_min_len", 2)
    plain                = _get_bool("plain", False)

    cads_path  = _get_str("cads_path", CADS_PATH)
    cads_is_xl = _get_bool("cads_is_excel", CADS_IS_EXCEL)
    cads_sheet = _get_str("cads_sheet", CADS_SHEET_NAME_DEFAULT)
    ref_branch = _get_str("ref", GH_BRANCH)

    oc = _get_str("override_cols", "AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES")
    override_cols = [c.strip() for c in oc.split(",") if c.strip()] or None

    mappings = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref_branch)
    df_cads  = _load_cads_df(cads_path, cads_is_xl, cads_sheet, ref=ref_branch)
    df_cads  = _strip_object_columns(df_cads)

    source = HARVEST_SOURCE

    # --- Input-driven / Quick YMMT ---
    if source in ("inputs", "quick_ymmt"):
        year  = _get_str("year", "")
        make  = _get_str("make", "")
        model = _get_str("model", "")
        trim  = _get_str("trim",  "")
        results, _ = filter_cads_generic(
            df_cads, year, make, model, trim,
            exact_model_when_full=model_exact_when_full,
            trim_exact_only=trim_exact_only, strict_and=strict_and,
            stopword_threshold=stopword_threshold, token_min_len=token_min_len,
            effective_model_cols_override=override_cols,
            trim_as_hint=trim_as_hint, year_require_exact=year_require_exact,
        )
        render_harvest_table(
            results,
            table_id="cads_inputs_results" if source=="inputs" else "cads_mapped_quick_ymmt",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Input-driven results" if source=="inputs" else "CADS – Quick YMM(/T) mapped results",
            plain=plain,
        ); st.stop()

    # --- Mapped search ---
    elif source == "mapped":
        year = _get_str("year", ""); make = _get_str("make", ""); model = _get_str("model", ""); trim = _get_str("trim", "")
        ymmt_list = find_mappings_by_ymmt_all(mappings, year, make, model, trim if canon_text(trim, True) else None)
        hits = []
        for _, mp in ymmt_list:
            df_hit, diag = match_cads_rows_for_mapping(
                df_cads, mp,
                exact_model_when_full=model_exact_when_full, trim_exact_only=trim_exact_only, strict_and=strict_and,
                stopword_threshold=stopword_threshold, token_min_len=token_min_len,
                effective_model_cols_override=override_cols, trim_as_hint=True, year_require_exact=year_require_exact,
            )
            if len(df_hit) > 0:
                df_hit = df_hit.copy()
                df_hit["__mapped_key__"] = f"{mp.get('make','')},{mp.get('model','')},{mp.get('trim','')},{mp.get('year','')}"
                df_hit["__tier__"] = diag.get("tier_used")
                hits.append(df_hit)
        df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True) if hits else df_cads.iloc[0:0]
        render_harvest_table(
            df_union,
            table_id="cads_mapped_results",
            preferred_order=HARVEST_PREF_ORDER + ["__mapped_key__","__tier__"],
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE","__mapped_key__","__tier__"],
            caption="CADS – Mapped search results",
            plain=plain,
        ); st.stop()

    # --- Quick vehicle ---
    elif source == "quick_vehicle":
        veh_txt = _get_str("vehicle", "")
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_mapped_quick_vehicle", caption="No vehicle provided", plain=plain); st.stop()
        veh_hits = find_rows_by_vehicle_text(df_cads, veh_txt)
        if veh_hits is None: veh_hits = df_cads.iloc[0:0]
        render_harvest_table(
            veh_hits,
            table_id="cads_mapped_quick_vehicle",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Quick Vehicle text results",
            plain=plain,
        ); st.stop()

    # --- Unmapped search ---
    elif source == "unmapped":
        veh_txt = canon_text(_get_str("vehicle", ""))
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_unmapped_results", caption="No vehicle provided", plain=plain); st.stop()
        hits = []
        for col in ["Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series","STYLE_NAME","AD_MODEL","MODEL_NAME"]:
            if col in df_cads.columns:
                ser = df_cads[col].astype(str).str.lower(); mask = ser.str.contains(veh_txt, na=False)
                if mask.any(): hits.append(df_cads[mask])
        df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True) if hits else df_cads.iloc[0:0]
        render_harvest_table(
            df_union,
            table_id="cads_unmapped_results",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Unmapped search results",
            plain=plain,
        ); st.stop()

    # --- Catalog search ---
    elif source == "catalog":
        veh_txt = _get_str("vehicle", "")
        cat_path = _get_str("catalog_path", "data/AFF Vehicles YMMT.csv")
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_catalog_results", caption="No vehicle provided", plain=plain); st.stop()
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, cat_path, GH_TOKEN, ref=ref_branch)
            cat_idx = build_catalog_index(df_cat)
            parsed = parse_vehicle_against_catalog(veh_txt, cat_idx)
            if not parsed:
                render_harvest_table(df_cads.iloc[0:0], table_id="cads_catalog_results", caption="Catalog did not find a close match", plain=plain); st.stop()
            y_s, mk_s, md_s, tr_s = parsed["year"], parsed["make"], parsed["model"], parsed["trim"]
            results, _ = filter_cads_generic(
                df_cads, y_s, mk_s, md_s, tr_s,
                exact_model_when_full=model_exact_when_full,
                trim_exact_only=False, strict_and=strict_and,
                stopword_threshold=stopword_threshold, token_min_len=token_min_len,
                effective_model_cols_override=override_cols, trim_as_hint=True, year_require_exact=year_require_exact,
            )
            render_harvest_table(
                results,
                table_id="cads_catalog_results",
                preferred_order=HARVEST_PREF_ORDER,
                include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
                caption="CADS – Catalog-accelerated results",
                plain=plain,
            ); st.stop()
        except Exception as e:
            st.markdown(f"<p id='harvest-error'>Catalog harvest failed: {e}</p>", unsafe_allow_html=True); st.stop()

    st.markdown("<p id='harvest-empty'>No harvest source matched or insufficient parameters.</p>", unsafe_allow_html=True)
    st.stop()

if HARVEST_MODE:
    _run_harvest()

# ===================== Sidebar UI =====================
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

st.sidebar.subheader("Vehicle Catalog")
VEH_CATALOG_PATH = st.sidebar.text_input("Vehicle Catalog path in repo", value="data/AFF Vehicles YMMT.csv")
st.session_state["veh_catalog_path"] = VEH_CATALOG_PATH

# ===================== Actions & Mappings =====================
st.sidebar.header("Actions")
load_branch = st.sidebar.text_input("Branch to load mappings from", value=st.session_state.get("load_branch", GH_BRANCH), help="Branch we read mappings.json from.")
st.session_state["load_branch"] = load_branch

if st.sidebar.button("🔄 Reload mappings from GitHub"):
    try:
        fetched = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, st.session_state["load_branch"])
        st.session_state["mappings"] = fetched
        st.session_state["local_mappings_modified"] = False
        st.sidebar.success(f"Reloaded {len(fetched)} mapping(s) from {st.session_state['load_branch']}.")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

commit_msg = st.sidebar.text_input("Commit message", value="chore(app): update AFF vehicle mappings via Streamlit")
use_feature_branch = st.sidebar.checkbox("Use feature branch (aff-mapping-app)", value=False)

# ===================== Matching Controls =====================
TRIM_AS_HINT = st.sidebar.checkbox("Use Trim as hint (do not filter)", value=True)
TRIM_EXACT_ONLY = st.sidebar.checkbox("Trim must be exact (no token-subset)", value=False)
MODEL_EXACT_WHEN_FULL = st.sidebar.checkbox("Model exact when input is multi-word", value=False)
STRICT_AND = st.sidebar.checkbox("Require strict AND across provided filters", value=True)
YEAR_REQUIRE_EXACT = st.sidebar.checkbox("Year must match exactly", value=True)

# ===================== Vehicle Mapping =====================
st.subheader("Vehicle Mapping")
vehicle_input = st.text_input("Enter Vehicle", value="", placeholder="e.g., 2025 Genesis GV70 2.5T AWD")
mapped_vehicle = None
if vehicle_input:
    mapped_vehicle = find_rows_by_vehicle_text(_load_cads_df(), vehicle_input)
    if mapped_vehicle is not None and len(mapped_vehicle) > 0:
        st.markdown(f"Found {len(mapped_vehicle)} matching CADS row(s). You can select one to map manually or refine with YMM/T.")
        render_harvest_table(mapped_vehicle, table_id="vehicle_quick_results", caption="Vehicle Quick Search Results")

# ===================== Mapping Actions =====================
st.subheader("Manual Mapping")
st.markdown(
    """
    - Select the vehicle row from results above.
    - Adjust or add Year/Make/Model/Trim if needed.
    - Save mapping to mappings.json with commit message.
    """
)
# Placeholder for mapping selection UI; in real use, you would implement row selection + mapping fields
# st.data_editor(mapped_vehicle)  # Optional interactive editor

# ===================== EOF =====================
st.markdown("<hr><p style='text-align:center;color:#888;font-size:12px;'>AFF Vehicle Mapping App - fully rebuilt with vehicle-centered search logic.</p>", unsafe_allow_html=True)
