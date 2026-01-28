
# app.py — Alias → Canonical mapping with Aliases.csv (mapped) and no-data logging
import io
import json
import base64
import time
import re
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
from thefuzz import fuzz  # Tip: add thefuzz[speedup] to requirements for python-Levenshtein

# =========================
# --- Configuration ---
# =========================
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"  # optional hinting file

st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# =========================
# --- GitHub secrets ---
# =========================
gh = st.secrets.get("github", {})
GH_TOKEN   = gh.get("token")
GH_OWNER   = gh.get("owner")
GH_REPO    = gh.get("repo")
GH_BRANCH  = gh.get("branch", "main")
CANON_PATH = gh.get("path", "Mappings.csv")                # canonical mappings
ALIASES_PATH = gh.get("aliases_path", "Aliases.csv")       # alias -> canonical (includes no_data)
AUTHOR_NAME  = gh.get("author_name", "AFF Bot")
AUTHOR_EMAIL = gh.get("author_email", "aff-bot@example.com")

# =========================
# --- Data Loading ---
# =========================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV/TXT using python engine to auto-detect delimiter.
    Coerce to string and fill NaNs to stabilize .str ops.
    """
    try:
        df = pd.read_csv(path, sep=None, engine='python', dtype=str)
        return df.fillna('')
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# Normalize CADS columns that we rely on
for col in ['MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID']:
    if col not in cads_df.columns:
        cads_df[col] = ''
cads_df = cads_df.fillna('').astype(str)

# =========================
# --- Helpers ---
# =========================
def normalize(s: str) -> str:
    """Lowercase, trim, collapse whitespace, replace common separators."""
    s = str(s or "")
    s = s.strip().lower()
    s = s.replace("-", " ").replace("/", " ")
    s = " ".join(s.split())
    return s

def infer_from_alias(alias: str, ref_df: pd.DataFrame, fallback_make: str = "", fallback_model: str = ""):
    """
    Infer (year, make, model, trim_guess) from the freeform alias.
    - Year: first 4-digit 19xx/20xx in text
    - Make/Model: from vehicle_example.txt exact/fuzzy row if available; else fallbacks (form inputs)
    - Trim guess: alias minus inferred tokens (best-effort; not used for strictness unless user filled Trim)
    """
    year = ""
    m = re.search(r"\b(19|20)\d{2}\b", str(alias))
    if m:
        year = m.group(0)

    make = fallback_make or ""
    model = fallback_model or ""
    trim_guess = ""

    if not ref_df.empty and 'Vehicle' in ref_df.columns:
        if '_vnorm' not in ref_df.columns:
            ref_df['_vnorm'] = ref_df['Vehicle'].astype(str).map(normalize)
        vn = normalize(alias)
        exact = ref_df[ref_df['_vnorm'] == vn]
        if not exact.empty:
            make = make or exact['Make'].astype(str).fillna('').values[0] if 'Make' in exact.columns else make
            model = model or exact['Model'].astype(str).fillna('').values[0] if 'Model' in exact.columns else model
        else:
            # fuzzy fallback to suggest make/model
            try:
                scores = ref_df['_vnorm'].map(lambda x: fuzz.token_set_ratio(vn, x))
                top_idx = scores.idxmax()
                if pd.notna(top_idx) and scores.loc[top_idx] >= 80:
                    row = ref_df.loc[top_idx]
                    make = make or str(row.get('Make', "")) or make
                    model = model or str(row.get('Model', "")) or model
            except Exception:
                pass

    # Remove found tokens from alias to guess trim (not strictly needed)
    tokens = [year, make, model]
    rem = normalize(alias)
    for t in tokens:
        if t:
            rem = rem.replace(normalize(t), " ")
    trim_guess = " ".join(rem.split()).strip()

    return str(year), make, model, trim_guess

def get_example_make_model(vehicle_name: str):
    """Optional hint from a reference file (exact or fuzzy)."""
    if vehicle_ref_df.empty or 'Vehicle' not in vehicle_ref_df.columns:
        return None, None

    # Build normalized column once
    if '_vnorm' not in vehicle_ref_df.columns:
        vehicle_ref_df['_vnorm'] = vehicle_ref_df['Vehicle'].astype(str).map(normalize)

    vn = normalize(vehicle_name)
    ref_row = vehicle_ref_df[vehicle_ref_df['_vnorm'] == vn]
    if not ref_row.empty:
        make = ref_row['Make'].values[0] if 'Make' in ref_row.columns else None
        model = ref_row['Model'].values[0] if 'Model' in ref_row.columns else None
        return make, model

    # Fuzzy fallback
    try:
        scores = vehicle_ref_df['_vnorm'].map(lambda x: fuzz.token_set_ratio(vn, x))
        top_idx = scores.idxmax()
        if pd.notna(top_idx) and scores.loc[top_idx] >= 80:
            row = vehicle_ref_df.loc[top_idx]
            return row.get('Make', None), row.get('Model', None)
    except Exception:
        pass
    return None, None

# ---- NEW: Attach visible vehicle attribute columns for the table ----
ATTR_CANDIDATES = {
    "Cab":    ["CAB", "AD_CAB", "CAB_STYLE", "CABTYPE", "CAB_TYPE"],
    "Drive":  ["DRIVE", "DRIVETRAIN", "DRIVE_TRAIN", "AD_DRIVE", "DRIVE_TYPE"],
    "Body":   ["BODY", "BODY_STYLE", "AD_BODY"],
    "Engine": ["ENGINE", "AD_ENGINE", "ENGINE_DESC", "ENGINE_DESCRIPTION"],
    "Box":    ["BOX", "BED", "BED_LENGTH", "BOX_SIZE", "AD_BOX"],
}

def _first_series_or_blank(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].astype(str).fillna('')
    return pd.Series([''] * len(df), index=df.index)

def attach_vehicle_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'Cab','Drive','Body','Engine','Box' exist (derived from best-available CADS columns)."""
    out = df.copy()
    for out_col, cand in ATTR_CANDIDATES.items():
        out[out_col] = _first_series_or_blank(out, cand)
    return out

def smart_vehicle_match(
    df: pd.DataFrame,
    vehicle_q: str,
    year: str = "",
    make: str = "",
    model: str = "",
    trim: str = "",
    top_n: int = 20,
    score_cutoff: int = 60
) -> pd.DataFrame:
    """
    Score candidates row-by-row using token_set_ratio.
    Returns a dataframe including 'score' and a stable 'map_key'.
    """
    if df.empty or not vehicle_q:
        return pd.DataFrame()

    work = df.copy()
    for col in ['MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM']:
        work[col] = work[col].astype(str).fillna('')

    # Explicit filters (narrow candidate set first)
    if year:
        work = work[work['MODEL_YEAR'] == str(year)]
    if make:
        work = work[work['AD_MAKE'].str.lower() == make.lower()]
    if model:
        work = work[work['AD_MODEL'].str.lower() == model.lower()]
    if trim:
        work = work[work['TRIM'].str.lower().str.contains(trim.lower())]

    if work.empty:
        return pd.DataFrame()

    # Combined search string
    work = work.copy()
    work['vehicle_search'] = (
        work['MODEL_YEAR'].str.strip() + ' ' +
        work['AD_MAKE'].str.strip() + ' ' +
        work['AD_MODEL'].str.strip() + ' ' +
        work['TRIM'].str.strip()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    q_norm = normalize(vehicle_q)
    work['score'] = work['vehicle_search'].map(lambda s: fuzz.token_set_ratio(q_norm, normalize(s)))

    # Keep top results above cutoff
    work = (
        work[work['score'] >= score_cutoff]
        .sort_values(['score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM'],
                     ascending=[False, False, True, True, True])
        .head(top_n)
        .copy()
    )

    # Stable key per row for persistent selection across reruns
    work['map_key'] = (
        work['MODEL_YEAR'] + '|' +
        work['AD_MAKE'] + '|' +
        work['AD_MODEL'] + '|' +
        work['TRIM'] + '|' +
        work['AD_MFGCODE'] + '|' +
        work['STYLE_ID']
    )

    # Attach attrs for display
    work = attach_vehicle_attrs(work)

    cols = [
        'map_key', 'score',
        'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM',
        'AD_MFGCODE', 'STYLE_ID',
        'Cab', 'Drive', 'Body', 'Engine', 'Box',
        'vehicle_search'
    ]
    for c in cols:
        if c not in work.columns:
            work[c] = ""
    return work[cols]

def strict_filter(df: pd.DataFrame, year: str = "", make: str = "", model: str = "", trim: str = "") -> pd.DataFrame:
    """
    Apply strict equality on only the fields provided (non-empty).
    """
    if df.empty:
        return df
    work = df.copy()
    if year:
        work = work[work['MODEL_YEAR'].astype(str) == str(year)]
    if make:
        work = work[work['AD_MAKE'].str.lower() == make.lower()]
    if model:
        work = work[work['AD_MODEL'].str.lower() == model.lower()]
    if trim:
        work = work[work['TRIM'].str.lower() == trim.lower()]
    return work

# ---- NEW: strict Year + Model Code lookup (no fuzzy) ----
def lookup_by_year_modelcode(df: pd.DataFrame, year: str, model_code: str) -> pd.DataFrame:
    """
    Return *all* matching rows for exact Year + Model Code.
    Sets score=100; attaches attrs.
    """
    if df.empty or not year or not model_code:
        return pd.DataFrame()

    work = df.copy()
    work = work[
        (work['MODEL_YEAR'].astype(str) == str(year)) &
        (work['AD_MFGCODE'].astype(str).str.casefold() == str(model_code).casefold())
    ].copy()

    if work.empty:
        return pd.DataFrame()

    work['vehicle_search'] = (
        work['MODEL_YEAR'].str.strip() + ' ' +
        work['AD_MAKE'].str.strip() + ' ' +
        work['AD_MODEL'].str.strip() + ' ' +
        work['TRIM'].str.strip()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()
    work['score'] = 100

    work['map_key'] = (
        work['MODEL_YEAR'] + '|' +
        work['AD_MAKE'] + '|' +
        work['AD_MODEL'] + '|' +
        work['TRIM'] + '|' +
        work['AD_MFGCODE'] + '|' +
        work['STYLE_ID']
    )

    work = attach_vehicle_attrs(work)

    cols = [
        'map_key', 'score',
        'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM',
        'AD_MFGCODE', 'STYLE_ID',
        'Cab', 'Drive', 'Body', 'Engine', 'Box',
        'vehicle_search'
    ]
    for c in cols:
        if c not in work.columns:
            work[c] = ""
    return work[cols].sort_values(['MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM']).reset_index(drop=True)

# =========================
# --- GitHub helpers ---
# =========================
def _gh_headers():
    if not GH_TOKEN:
        return {"Accept": "application/vnd.github+json"}
    return {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}

def _gh_contents_url(path: str, ref: str):
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={ref}"

def github_upsert_csv_with_keys(
    file_path_in_repo: str,
    new_rows_df: pd.DataFrame,
    dedup_keys: list[str],
    sort_keys: list[str]
) -> tuple[bool, str]:
    """
    Generic upsert: read existing CSV from GitHub, union with new_rows_df,
    drop duplicates using dedup_keys, sort by sort_keys, commit via Contents API.
    """
    if new_rows_df is None or new_rows_df.empty:
        return False, "No rows to commit."

    headers = _gh_headers()

    # Read existing
    get_url = _gh_contents_url(file_path_in_repo, GH_BRANCH)
    r = requests.get(get_url, headers=headers, timeout=30)

    existing_df = pd.DataFrame(columns=new_rows_df.columns)
    sha = None
    if r.status_code == 200:
        js = r.json()
        sha = js.get("sha")
        decoded = base64.b64decode(js.get("content", "")).decode("utf-8", errors="replace")
        try:
            existing_df = pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
        except Exception:
            existing_df = pd.DataFrame(columns=new_rows_df.columns)
    elif r.status_code not in (404,):
        return False, f"Failed fetching existing file: HTTP {r.status_code} - {r.text}"

    # Merge & dedupe
    combined = pd.concat([existing_df, new_rows_df], ignore_index=True).fillna("")
    if dedup_keys:
        for k in dedup_keys:
            if k not in combined.columns:
                combined[k] = ""
        combined = combined.drop_duplicates(subset=dedup_keys)
    else:
        combined = combined.drop_duplicates()
    if sort_keys:
        for k in sort_keys:
            if k not in combined.columns:
                combined[k] = ""
        combined = combined.sort_values(by=sort_keys).reset_index(drop=True)

    # Encode CSV
    buf = io.StringIO()
    combined.to_csv(buf, index=False, encoding="utf-8")
    content_b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("utf-8")

    # PUT
    put_url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{file_path_in_repo}"
    payload = {
        "message": f"chore(mappings): upsert {len(new_rows_df)} row(s) via AFF UI - {time.time()}",
        "content": content_b64,
        "branch": GH_BRANCH,
        "committer": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL}
    }
    if sha:
        payload["sha"] = sha

    r2 = requests.put(put_url, headers=headers, data=json.dumps(payload), timeout=30)
    if r2.status_code in (200, 201):
        return True, f"Committed to {GH_OWNER}/{GH_REPO}@{GH_BRANCH}:{file_path_in_repo}"
    if r2.status_code == 403:
        return False, "403 Forbidden on commit. Check PAT Contents: Read/Write and branch protections."
    if r2.status_code == 401:
        return False, "401 Unauthorized. Rotate PAT and authorize SSO."
    return False, f"Commit failed: HTTP {r2.status_code} - {r2.text}"

def commit_alias_and_canonical(
    alias_input: str,
    selected_canonical_df: pd.DataFrame,  # expects cols: MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE
    source: str = "user"
) -> tuple[bool, str]:
    """
    Commits:
      1) Aliases.csv rows with status='mapped'
      2) Ensures canonical rows exist in Mappings.csv
    """
    if selected_canonical_df.empty:
        return False, "No rows selected."

    alias_norm = normalize(alias_input)

    # Build alias rows
    alias_rows = selected_canonical_df.rename(columns={
        "MODEL_YEAR":"year", "AD_MAKE":"make", "AD_MODEL":"model",
        "TRIM":"trim", "AD_MFGCODE":"model_code"
    })[["year","make","model","trim","model_code"]].copy()

    alias_rows["alias"] = alias_input
    alias_rows["alias_norm"] = alias_norm
    alias_rows["source"] = source
    alias_rows["status"] = "mapped"  # explicitly tagged
    alias_rows["created_at"] = datetime.utcnow().isoformat() + "Z"
    for c in alias_rows.columns:
        alias_rows[c] = alias_rows[c].astype(str).str.strip()

    ok1, msg1 = github_upsert_csv_with_keys(
        file_path_in_repo=ALIASES_PATH,
        new_rows_df=alias_rows,
        dedup_keys=["alias_norm","year","make","model","trim","model_code","status"],
        sort_keys=["alias_norm","year","make","model","trim","model_code","status","created_at"]
    )
    if not ok1:
        return False, f"Aliases commit failed: {msg1}"

    # Ensure canonical rows in Mappings.csv
    canonical_rows = alias_rows[["year","make","model","trim","model_code","source"]].drop_duplicates(
        subset=["year","make","model","trim","model_code"]
    )
    ok2, msg2 = github_upsert_csv_with_keys(
        file_path_in_repo=CANON_PATH,
        new_rows_df=canonical_rows,
        dedup_keys=["year","make","model","trim","model_code"],
        sort_keys=["year","make","model","trim","model_code"]
    )
    if not ok2:
        return False, f"Canonical commit failed: {msg2}"

    return True, f"{msg1}; {msg2}"

def commit_alias_no_data(
    alias_input: str,
    year: str = "",
    make: str = "",
    model: str = "",
    trim: str = "",
    source: str = "user"
) -> tuple[bool, str]:
    """
    Writes an alias-only row with status='no_data' to Aliases.csv so basic_app.py
    can show 'No Vehicle Data' for this alias until CADS is available.
    """
    alias_norm = normalize(alias_input)
    row = {
        "alias": alias_input,
        "alias_norm": alias_norm,
        "year": str(year or ""),
        "make": make or "",
        "model": model or "",
        "trim": trim or "",
        "model_code": "",          # unknown until CADS arrives
        "source": source,
        "status": "no_data",       # flag for basic_app.py
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    df = pd.DataFrame([row])
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Dedup by alias_norm + provided Y/M/M/T + status
    ok, msg = github_upsert_csv_with_keys(
        file_path_in_repo=ALIASES_PATH,
        new_rows_df=df,
        dedup_keys=["alias_norm","year","make","model","trim","status"],
        sort_keys=["alias_norm","year","make","model","trim","status","created_at"]
    )
    return ok, msg

# ---- NEW: Load existing Aliases for "Mapped?" detection ----
@st.cache_data
def load_aliases_df() -> pd.DataFrame:
    """
    Try to load Aliases.csv from GitHub (if secrets present). If that fails,
    try local path. Returns empty DF if not found.
    """
    # Try GitHub
    if GH_OWNER and GH_REPO and ALIASES_PATH:
        try:
            url = _gh_contents_url(ALIASES_PATH, GH_BRANCH)
            r = requests.get(url, headers=_gh_headers(), timeout=30)
            if r.status_code == 200:
                js = r.json()
                decoded = base64.b64decode(js.get("content", "")).decode("utf-8", errors="replace")
                df = pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
                return df
        except Exception:
            pass
    # Fallback local
    try:
        return load_csv(ALIASES_PATH)
    except Exception:
        return pd.DataFrame()

aliases_df = load_aliases_df()

def is_row_already_mapped_for_alias(row: pd.Series, alias_text: str) -> bool:
    """
    Returns True if Aliases.csv already has a row with status='mapped' for this alias_norm
    and same (year, make, model, trim, model_code).
    """
    if aliases_df.empty or not alias_text:
        return False
    alias_norm = normalize(alias_text)
    y = str(row.get("MODEL_YEAR", ""))
    mk = str(row.get("AD_MAKE", ""))
    md = str(row.get("AD_MODEL", ""))
    tr = str(row.get("TRIM", ""))
    mc = str(row.get("AD_MFGCODE", ""))
    df = aliases_df
    needed_cols = {"alias_norm","status","year","make","model","trim","model_code"}
    if not needed_cols.issubset(set(df.columns)):
        return False
    hit = df[
        (df["alias_norm"].astype(str).map(normalize) == alias_norm) &
        (df["status"] == "mapped") &
        (df["year"].astype(str) == y) &
        (df["make"].astype(str) == mk) &
        (df["model"].astype(str) == md) &
        (df["trim"].astype(str) == tr) &
        (df["model_code"].astype(str) == mc)
    ]
    return not hit.empty

# =========================
# --- Session State ---
# =========================
if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False
if 'matches_df' not in st.session_state:
    st.session_state['matches_df'] = pd.DataFrame()
if 'selection' not in st.session_state:
    st.session_state['selection'] = {}  # map_key -> bool
if 'current_query' not in st.session_state:
    st.session_state['current_query'] = ""
if 'inferred' not in st.session_state:
    st.session_state['inferred'] = {"year":"", "make":"", "model":"", "trim":""}
# Track which search mode populated the table
if 'result_mode' not in st.session_state:
    st.session_state['result_mode'] = "alias"  # alias | year_model_code

# =========================
# --- Main Page: Search Form ---
# =========================
st.title("AFF Vehicle Mapping")
st.caption("Map a freeform vehicle (alias) to one or more CADS canonical rows. Or log the alias as 'No Vehicle Data' if CADS is not available yet.")

with st.form("search_form_main"):
    st.subheader("Search")
    vehicle_input = st.text_input("Vehicle (alias, freeform)", placeholder="e.g., 2027 Integra")

    st.markdown("**YMMT (optional, useful if CADS not available yet)**")
    c1, c2, c3, c4 = st.columns(4)
    year_input  = c1.text_input("Year")
    make_input  = c2.text_input("Make")
    model_input = c3.text_input("Model")
    trim_input  = c4.text_input("Trim")

    top_n = st.slider("How many matches to show", min_value=5, max_value=50, value=20, step=5)
    score_cutoff = st.slider("Minimum match score", min_value=0, max_value=100, value=60, step=5)

    submitted = st.form_submit_button("Search Vehicles")

# Alternate search — Year + Model Code
with st.expander("Alternate search: Year + Model Code", expanded=False):
    with st.form("search_form_mc"):
        cmc1, cmc2 = st.columns(2)
        mc_year = cmc1.text_input("Year (exact)", value="")
        mc_code = cmc2.text_input("Model Code (exact)", value="")
        submitted_mc = st.form_submit_button("Search by Year + Model Code")
    if submitted_mc:
        mc_df = lookup_by_year_modelcode(cads_df, year=mc_year, model_code=mc_code)
        st.session_state['matches_df'] = mc_df
        st.session_state['show_results'] = True
        st.session_state['current_query'] = vehicle_input  # keep alias typed above
        st.session_state['result_mode'] = "year_model_code"
        prev = st.session_state['selection']
        st.session_state['selection'] = {k: prev.get(k, False) for k in mc_df['map_key']} if not mc_df.empty else {}

# Optional hint from reference file
example_make, example_model = (None, None)
if vehicle_input:
    example_make, example_model = get_example_make_model(vehicle_input)
    if example_make or example_model:
        st.caption(f"Ref hint → Make: {example_make or '—'}, Model: {example_model or '—'}")

# Handle Search Submit
if submitted:
    inf_year, inf_make, inf_model, inf_trim_guess = infer_from_alias(
        vehicle_input, vehicle_ref_df,
        fallback_make=make_input or (example_make or ""),
        fallback_model=model_input or (example_model or "")
    )
    eff_year  = year_input or inf_year
    eff_make  = make_input or (example_make or inf_make)
    eff_model = model_input or (example_model or inf_model)
    eff_trim  = trim_input  # deliberate: do not force trim guess

    st.session_state['inferred'] = {"year":eff_year, "make":eff_make, "model":eff_model, "trim":eff_trim}

    matches_df = smart_vehicle_match(
        cads_df,
        vehicle_input,
        year=eff_year,
        make=eff_make,
        model=eff_model,
        trim="",            # keep fuzzy trim out
        top_n=top_n,
        score_cutoff=score_cutoff
    )

    st.session_state['matches_df'] = matches_df
    st.session_state['show_results'] = True
    st.session_state['current_query'] = vehicle_input
    st.session_state['result_mode'] = "alias"

    prev = st.session_state['selection']
    st.session_state['selection'] = {k: prev.get(k, False) for k in matches_df['map_key']} if not matches_df.empty else {}

# =========================
# --- Results / No Data path (with inference) ---
# =========================
if st.session_state['show_results']:
    matches_df = st.session_state['matches_df']
    alias_text = st.session_state.get('current_query', '')
    eff = st.session_state.get('inferred', {"year":"", "make":"", "model":"", "trim":""})
    eff_year, eff_make, eff_model, eff_trim = eff["year"], eff["make"], eff["model"], eff["trim"]

    # ---- NEW: add "Mapped?" column per row for current alias ----
    if not matches_df.empty:
        mapped_flags = []
        for _, r in matches_df.iterrows():
            mapped_flags.append(is_row_already_mapped_for_alias(r, alias_text))
        matches_df = matches_df.copy()
        matches_df["Mapped?"] = mapped_flags

    # No-data flow only for alias-based search
    if st.session_state.get('result_mode', 'alias') == "alias":
        strict_subset = strict_filter(cads_df, year=eff_year, make=eff_make, model=eff_model, trim=eff_trim)
        specified_any_filter = any([eff_year, eff_make, eff_model, eff_trim])
        no_data_candidate = specified_any_filter and strict_subset.empty

        if no_data_candidate:
            st.warning(
                "No exact CADS rows found for the specific Y/M/M/T you entered or I inferred "
                f"(Year={eff_year or '—'}, Make={eff_make or '—'}, Model={eff_model or '—'}"
                f"{', Trim='+eff_trim if eff_trim else ''})."
            )
            st.info("If you expect this vehicle to arrive in CADS later, log it as **'No Vehicle Data'** so the Basic app shows a clear message for this alias.")
            if st.button("Vehicle Data Not Received"):
                ok, msg = commit_alias_no_data(
                    alias_input=alias_text,
                    year=eff_year,
                    make=eff_make,
                    model=eff_model,
                    trim=eff_trim,
                    source="user"
                )
                if ok:
                    st.success("Logged as 'No Vehicle Data'. When you search this alias in the Basic app, it will show a 'No Vehicle Data' message.")
                else:
                    st.error(msg)

    if not matches_df.empty:
        st.subheader("Matching Vehicles")
        view = matches_df.copy()
        view['Select'] = view['map_key'].map(st.session_state['selection']).fillna(False).astype(bool)

        a1, a2, a3 = st.columns([1, 1, 1])
        with a1:
            select_all_clicked = st.checkbox("Select all shown", value=all(view['Select']) and len(view) > 0, key="select_all_shown")
        with a2:
            if st.button("Clear selections"):
                view['Select'] = False
        with a3:
            if st.button("Close results"):
                st.session_state['show_results'] = False

        if select_all_clicked:
            view['Select'] = True

        # ---- IMPORTANT: Disable selection if alias is empty ----
        alias_required_msg = ""
        if not (alias_text and alias_text.strip()):
            alias_required_msg = "Enter Vehicle (alias) above to enable mapping."
            # Force all Select checkboxes false in the editor to prevent mapping without alias
            view['Select'] = False

        display_cols = [
            'Select', 'score',
            'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM',
            'AD_MFGCODE', 'STYLE_ID',
            'Cab', 'Drive', 'Body', 'Engine', 'Box',
            'Mapped?',           # NEW
            'vehicle_search'
        ]
        for c in display_cols:
            if c not in view.columns:
                view[c] = ""

        # Column config (lock read-only fields)
        disabled_cols = [
            'score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM',
            'AD_MFGCODE', 'STYLE_ID', 'Cab', 'Drive', 'Body', 'Engine', 'Box', 'Mapped?', 'vehicle_search'
        ]
        # If alias not provided, also disable "Select"
        if not (alias_text and alias_text.strip()):
            disabled_cols = ['Select'] + disabled_cols

        edited = st.data_editor(
            view[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(help="Include this row in the mapping"),
                "score": st.column_config.NumberColumn(format="%d", help="Fuzzy match score (0–100)"),
                "AD_MFGCODE": st.column_config.TextColumn(label="Model Code"),
                "STYLE_ID": st.column_config.TextColumn(label="STYLE_ID"),
                "Cab": st.column_config.TextColumn(help="Cab / cab style"),
                "Drive": st.column_config.TextColumn(help="Drivetrain (e.g., 2WD, 4WD, AWD)"),
                "Body": st.column_config.TextColumn(help="Body style"),
                "Engine": st.column_config.TextColumn(help="Engine description"),
                "Box": st.column_config.TextColumn(help="Bed/box length or size"),
                "Mapped?": st.column_config.TextColumn(help="Already mapped for this alias"),
            },
            disabled=disabled_cols
        )

        # Persist selection only if alias exists
        if alias_text and alias_text.strip():
            for i, row in edited.iterrows():
                mk = matches_df.loc[i, 'map_key']
                st.session_state['selection'][mk] = bool(row['Select'])

        if alias_required_msg:
            st.warning(alias_required_msg)

        selected_keys = [k for k, v in st.session_state['selection'].items() if v]
        final_df = matches_df[matches_df['map_key'].isin(selected_keys)].copy()

        st.markdown("---")
        st.write(f"Selected: **{len(final_df)}** row(s)")

        if not final_df.empty:
            canonical_preview = final_df.rename(columns={
                "MODEL_YEAR": "year",
                "AD_MAKE": "make",
                "AD_MODEL": "model",
                "TRIM": "trim",
                "AD_MFGCODE": "model_code"
            })[["year","make","model","trim","model_code"]].copy()
            canonical_preview["source"] = "user"

            alias_preview = canonical_preview.copy()
            alias_preview.insert(0, "alias", alias_text)
            alias_preview.insert(1, "alias_norm", normalize(alias_text))
            alias_preview["status"] = "mapped"
            alias_preview["created_at"] = datetime.utcnow().isoformat() + "Z"

            # Show a small "already mapped" badge inline for any duplicates
            if "Mapped?" in matches_df.columns:
                # decor: add an indicator column to previews for UX
                alias_preview["already_mapped_for_alias"] = alias_preview.apply(
                    lambda r: "Yes" if not aliases_df.empty and not aliases_df[
                        (aliases_df["alias_norm"].astype(str).map(normalize) == normalize(alias_text)) &
                        (aliases_df["status"] == "mapped") &
                        (aliases_df["year"].astype(str) == str(r["year"])) &
                        (aliases_df["make"].astype(str) == str(r["make"])) &
                        (aliases_df["model"].astype(str) == str(r["model"])) &
                        (aliases_df["trim"].astype(str) == str(r["trim"])) &
                        (aliases_df["model_code"].astype(str) == str(r["model_code"]))
                    ].empty else "No",
                    axis=1
                )

            st.caption("Alias rows to append (status='mapped') → Aliases.csv")
            st.dataframe(alias_preview, use_container_width=True)
            st.caption("Canonical rows ensured → Mappings.csv")
            st.dataframe(canonical_preview, use_container_width=True)

        cA, cB = st.columns([1, 1])
        with cA:
            # Disable commit if alias missing
            commit_disabled = not (alias_text and alias_text.strip())
            if commit_disabled:
                st.button("Commit to GitHub (Aliases + Mappings)", disabled=True)
                st.info("Enter Vehicle (alias) to enable committing selected mapping(s).")
            else:
                if st.button("Commit to GitHub (Aliases + Mappings)"):
                    if final_df.empty:
                        st.warning("No rows selected to commit.")
                    elif not GH_TOKEN or not GH_OWNER or not GH_REPO:
                        st.error("GitHub secrets not configured. Cannot commit.")
                    else:
                        ok, msg = commit_alias_and_canonical(
                            alias_input=alias_text,
                            selected_canonical_df=final_df[["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"]],
                            source="user"
                        )
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

        with cB:
            if not final_df.empty:
                csv_blob = final_df[["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE","STYLE_ID"]].to_csv(index=False)
                st.download_button(
                    "Download Selected (CSV)",
                    data=csv_blob,
                    file_name="vehicle_mapping_selection.csv",
                    mime="text/csv"
                )

    # If there were no matches either way, still provide a no-data pathway (only meaningful for alias flow)
    if st.session_state.get('result_mode', 'alias') == "alias" and st.session_state['matches_df'].empty:
        st.warning("No CADS matches found for this alias.")
        st.info("Optionally log it as **'No Vehicle Data'** if you expect CADS to arrive later.")
        if st.button("Vehicle Data Not Received"):
            ok, msg = commit_alias_no_data(
                alias_input=alias_text,
                year=eff_year,
                make=eff_make,
                model=eff_model,
                trim=eff_trim,
                source="user"
            )
            if ok:
                st.success("Logged as 'No Vehicle Data'. When you search this alias in the Basic app, it will show a 'No Vehicle Data' message.")
            else:
                st.error(msg)

# --- EOF ---
