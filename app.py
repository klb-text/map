
# app.py — Alias → Canonical mapping, committing Mappings.csv (canonical) and Aliases.csv (aliases)
import os
import io
import json
import base64
import time
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
CANON_PATH = gh.get("path", "Mappings.csv")        # canonical file
ALIASES_PATH = gh.get("aliases_path", "Aliases.csv")  # alias file
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
    scores = vehicle_ref_df['_vnorm'].map(lambda x: fuzz.token_set_ratio(vn, x))
    top_idx = scores.idxmax()
    if pd.notna(top_idx) and scores.loc[top_idx] >= 80:
        row = vehicle_ref_df.loc[top_idx]
        return row.get('Make', None), row.get('Model', None)

    return None, None

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

    # Explicit filters
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

    # Combined string and scoring
    work = work.copy()
    work['vehicle_search'] = (
        work['MODEL_YEAR'].str.strip() + ' ' +
        work['AD_MAKE'].str.strip() + ' ' +
        work['AD_MODEL'].str.strip() + ' ' +
        work['TRIM'].str.strip()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    q_norm = normalize(vehicle_q)
    work['score'] = work['vehicle_search'].map(lambda s: fuzz.token_set_ratio(q_norm, normalize(s)))

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

    cols = ['map_key', 'score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID', 'vehicle_search']
    for c in cols:
        if c not in work.columns:
            work[c] = ""
    return work[cols]

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
        combined = combined.drop_duplicates(subset=dedup_keys)
    else:
        combined = combined.drop_duplicates()
    if sort_keys:
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
    selected_canonical_df: pd.DataFrame,  # expects columns: MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE
    source: str = "user"
) -> tuple[bool, str]:
    """
    1) Append alias rows to Aliases.csv (dedup on alias_norm + canonical)
    2) Ensure canonical rows exist in Mappings.csv (dedup on canonical only)
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
    alias_rows["created_at"] = datetime.utcnow().isoformat() + "Z"
    for c in alias_rows.columns:
        alias_rows[c] = alias_rows[c].astype(str).str.strip()

    # 1) Upsert alias rows
    ok1, msg1 = github_upsert_csv_with_keys(
        file_path_in_repo=ALIASES_PATH,
        new_rows_df=alias_rows,
        dedup_keys=["alias_norm","year","make","model","trim","model_code"],
        sort_keys=["alias_norm","year","make","model","trim","model_code","created_at"]
    )
    if not ok1:
        return False, f"Aliases commit failed: {msg1}"

    # 2) Ensure canonical rows in Mappings.csv
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

# =========================
# --- Main Page: Search Form ---
# =========================
st.title("AFF Vehicle Mapping")
st.caption("Map a freeform vehicle (alias) to one or more CADS canonical rows. Commits both Aliases.csv and Mappings.csv.")

with st.form("search_form_main"):
    st.subheader("Search")
    vehicle_input = st.text_input("Vehicle (alias, freeform)", placeholder="e.g., 2026 Integra FWD Continuously Variable Transmission")

    st.markdown("**YMMT Filter (optional)**")
    c1, c2, c3, c4 = st.columns(4)
    year_input = c1.text_input("Year")
    make_input = c2.text_input("Make")
    model_input = c3.text_input("Model")
    trim_input = c4.text_input("Trim")

    top_n = st.slider("How many matches to show", min_value=5, max_value=50, value=20, step=5)
    score_cutoff = st.slider("Minimum match score", min_value=0, max_value=100, value=60, step=5)

    submitted = st.form_submit_button("Search Vehicles")

# Optional hint from reference file
example_make, example_model = (None, None)
if vehicle_input:
    example_make, example_model = get_example_make_model(vehicle_input)
    if example_make or example_model:
        st.caption(f"Ref hint → Make: {example_make or '—'}, Model: {example_model or '—'}")

# =========================
# --- Handle Search Submit ---
# =========================
if submitted:
    matches_df = smart_vehicle_match(
        cads_df,
        vehicle_input,
        year=year_input,
        make=make_input,
        model=model_input,
        trim=trim_input,
        top_n=top_n,
        score_cutoff=score_cutoff
    )

    st.session_state['matches_df'] = matches_df
    st.session_state['show_results'] = not matches_df.empty
    st.session_state['current_query'] = vehicle_input

    # Reset/preserve selection only for the current results
    prev = st.session_state['selection']
    st.session_state['selection'] = {k: prev.get(k, False) for k in matches_df['map_key']} if not matches_df.empty else {}

# =========================
# --- Results ---
# =========================
if st.session_state['show_results']:
    matches_df = st.session_state['matches_df']

    if matches_df.empty:
        st.warning(f"No matching vehicles found for: {st.session_state.get('current_query','(blank)')}")
    else:
        st.subheader("Matching Vehicles")

        # Build a view with a Select checkbox column bound to session_state
        view = matches_df.copy()
        view['Select'] = view['map_key'].map(st.session_state['selection']).fillna(False).astype(bool)

        # Bulk actions row
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

        # Single editable table with the checkbox inside
        display_cols = ['Select', 'score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID', 'vehicle_search']
        edited = st.data_editor(
            view[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(help="Include this row in the mapping"),
                "score": st.column_config.NumberColumn(format="%d", help="Fuzzy match score (0–100)"),
                "AD_MFGCODE": st.column_config.TextColumn(label="Model Code"),
            },
            disabled=['score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID', 'vehicle_search']
        )

        # Persist checkbox edits back to session_state using index alignment
        for i, row in edited.iterrows():
            mk = matches_df.loc[i, 'map_key']
            st.session_state['selection'][mk] = bool(row['Select'])

        # Build final selection
        selected_keys = [k for k, v in st.session_state['selection'].items() if v]
        final_df = matches_df[matches_df['map_key'].isin(selected_keys)].copy()

        st.markdown("---")
        st.write(f"Selected: **{len(final_df)}** row(s)")

        # Build canonical preview (for clarity)
        canonical_preview = final_df.rename(columns={
            "MODEL_YEAR": "year",
            "AD_MAKE": "make",
            "AD_MODEL": "model",
            "TRIM": "trim",
            "AD_MFGCODE": "model_code"
        })[["year","make","model","trim","model_code"]].copy()
        canonical_preview["source"] = "user"

        # Build alias preview (alias → canonical)
        alias_preview = canonical_preview.copy()
        alias_preview.insert(0, "alias", st.session_state.get("current_query", ""))
        alias_preview.insert(1, "alias_norm", normalize(st.session_state.get("current_query", "")))
        alias_preview["created_at"] = datetime.utcnow().isoformat() + "Z"

        if not final_df.empty:
            st.caption("Alias rows to be appended to Aliases.csv")
            st.dataframe(alias_preview, use_container_width=True)
            st.caption("Canonical rows ensured in Mappings.csv")
            st.dataframe(canonical_preview, use_container_width=True)

        cA, cB = st.columns([1, 1])
        with cA:
            if st.button("Commit to GitHub (Aliases + Mappings)"):
                if final_df.empty:
                    st.warning("No rows selected to commit.")
                elif not GH_TOKEN or not GH_OWNER or not GH_REPO:
                    st.error("GitHub secrets not configured. Cannot commit.")
                else:
                    ok, msg = commit_alias_and_canonical(
                        alias_input=st.session_state.get("current_query", ""),
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

# =========================
# --- Footer / Tips ---
# =========================
with st.expander("Tips & Notes", expanded=False):
    st.markdown(
        """
- This UI supports **many aliases → one canonical** mapping.  
  - **Aliases.csv** stores every freeform input you commit (dedup on alias_norm + canonical).  
  - **Mappings.csv** remains canonical, deduped by (year, make, model, trim, model_code).
- Mozenda should continue reading **Mappings.csv** (canonical). When joined to CADS, it may fan out 1→many per canonical—this is expected and correct.
- Ensure your GitHub token has **Contents: Read/Write** permissions and is **SSO-authorized** if your org enforces SSO.
        """
    )

# --- EOF ---
