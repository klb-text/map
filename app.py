
# app.py
import os
import io
import json
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
from thefuzz import fuzz  # Tip: use thefuzz[speedup] in requirements for faster scoring

# =========================
# --- Configuration ---
# =========================
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"

st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

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
    """
    Try to find Make/Model hints in the VEHICLE_REF_FILE.
    First exact normalized match, then fuzzy fallback (>=80).
    """
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
    example_make: str | None = None,
    example_model: str | None = None,
    top_n: int = 20,
    score_cutoff: int = 60
) -> pd.DataFrame:
    """
    Score candidates row-by-row using token_set_ratio.
    Returns a dataframe including 'score' and a stable 'map_key'.
    """
    if df.empty or not vehicle_q:
        return pd.DataFrame()

    needed = ['MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID']
    base = df.copy()
    for col in needed:
        if col not in base.columns:
            base[col] = ""

    # Ensure strings
    for col in ['MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID']:
        base[col] = base[col].astype(str).fillna('')

    # Explicit filters (narrow candidate set first)
    work = base
    if year:
        work = work[work['MODEL_YEAR'] == str(year)]
    if make:
        work = work[work['AD_MAKE'].str.lower() == make.lower()]
    if model:
        work = work[work['AD_MODEL'].str.lower() == model.lower()]
    if trim:
        work = work[work['TRIM'].str.lower().str.contains(trim.lower())]

    # Fallback to example make/model if nothing matches the filters
    if work.empty and example_make and example_model:
        work = base[
            (base['AD_MAKE'].str.lower() == example_make.lower()) &
            (base['AD_MODEL'].str.lower() == example_model.lower())
        ]

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
    work['vs_norm'] = work['vehicle_search'].map(normalize)
    work['score'] = work['vs_norm'].map(lambda s: fuzz.token_set_ratio(q_norm, s))

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

    cols = ['map_key', 'score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID', 'vehicle_search']
    return work[cols]

# =========================
# --- Build Mappings.csv rows from selection ---
# =========================
def build_mappings_from_selection(final_df: pd.DataFrame, source: str = "user") -> pd.DataFrame:
    """
    Convert selected matches to the canonical Mappings.csv schema and dedupe.
    Output columns: year, make, model, trim, model_code, source
    """
    if final_df is None or final_df.empty:
        return pd.DataFrame(columns=["year","make","model","trim","model_code","source"])

    out = final_df.rename(columns={
        "MODEL_YEAR": "year",
        "AD_MAKE": "make",
        "AD_MODEL": "model",
        "TRIM": "trim",
        "AD_MFGCODE": "model_code"
    })[["year","make","model","trim","model_code"]].copy()

    out["source"] = source
    for c in ["year","make","model","trim","model_code","source"]:
        out[c] = out[c].astype(str).str.strip()

    out.drop_duplicates(inplace=True)
    return out

# =========================
# --- GitHub API Commit (uses st.secrets['github']) ---
# =========================
def github_api_commit_mappings(new_rows_df: pd.DataFrame) -> tuple[bool, str]:
    """
    Upsert into Mappings.csv in GitHub using the Contents API:
    - Read existing file (if present)
    - Append new rows, dedupe, sort
    - PUT updated file with commit message
    Secrets required:
      [github]
      token, owner, repo, branch
      (optional) path, author_name, author_email
    """
    import requests

    gh = st.secrets.get("github", {})
    token    = gh.get("token")
    owner    = gh.get("owner")
    repo     = gh.get("repo")
    branch   = gh.get("branch", "main")
    file_path= gh.get("path", "Mappings.csv")  # default to repo root
    author_n = gh.get("author_name", "AFF Bot")
    author_e = gh.get("author_email", "aff-bot@example.com")

    if not token or not owner or not repo:
        return False, "GitHub token/owner/repo not configured in st.secrets['github']."

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    # 1) Get current file (if exists) to obtain sha and content
    get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    r = requests.get(get_url, headers=headers)
    sha = None
    existing_df = pd.DataFrame(columns=["year","make","model","trim","model_code","source"])

    if r.status_code == 200:
        content_b64 = r.json().get("content", "")
        sha = r.json().get("sha")
        decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        existing_df = pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
    elif r.status_code == 404:
        # file doesn't exist yet; that's OK
        pass
    else:
        return False, f"Failed to read existing file: HTTP {r.status_code} - {r.text}"

    # 2) Merge, dedupe, sort (optional)
    combined = pd.concat([existing_df, new_rows_df], ignore_index=True).fillna("")
    combined.drop_duplicates(inplace=True)
    combined = combined[["year","make","model","trim","model_code","source"]]
    combined = combined.sort_values(by=["year","make","model","trim","model_code"]).reset_index(drop=True)

    # 3) Re-encode as CSV (utf-8)
    csv_buffer = io.StringIO()
    combined.to_csv(csv_buffer, index=False, encoding="utf-8")
    content_b64 = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode("utf-8")

    # 4) Commit via PUT
    put_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    commit_message = f"chore(mappings): upsert {len(new_rows_df)} row(s) via AFF UI - {datetime.utcnow().isoformat()}Z"
    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch,
        "committer": {"name": author_n, "email": author_e}
    }
    if sha:
        payload["sha"] = sha

    r2 = requests.put(put_url, headers=headers, data=json.dumps(payload))
    if r2.status_code in (200, 201):
        return True, f"Committed to {owner}/{repo}@{branch}:{file_path}"
    else:
        return False, f"Commit failed: HTTP {r2.status_code} - {r2.text}"

# =========================
# --- Session State ---
# =========================
if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False
if 'matches_df' not in st.session_state:
    st.session_state['matches_df'] = pd.DataFrame()
if 'selection' not in st.session_state:
    # map_key -> bool
    st.session_state['selection'] = {}
if 'current_query' not in st.session_state:
    st.session_state['current_query'] = ""

# =========================
# --- Main Page: Search Form ---
# =========================
st.title("AFF Vehicle Mapping")
st.caption("Select one or more CADS rows that match your freeform vehicle input. Supports 1→many mapping.")

with st.form("search_form_main"):
    st.subheader("Search")
    vehicle_input = st.text_input("Vehicle (freeform)", placeholder="e.g., 2025 Land Rover Range Rover P400 SE SWB")

    st.markdown("**YMMT Filter (optional)**")
    c1, c2, c3, c4 = st.columns(4)
    year_input = c1.text_input("Year")
    make_input = c2.text_input("Make")
    model_input = c3.text_input("Model")
    trim_input = c4.text_input("Trim")

    top_n = st.slider("How many matches to show", min_value=5, max_value=50, value=20, step=5)
    score_cutoff = st.slider("Minimum match score", min_value=0, max_value=100, value=60, step=5)

    submitted = st.form_submit_button("Search Vehicles")

# Optional hint from reference file (still on main page)
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
        example_make=example_make,
        example_model=example_model,
        top_n=top_n,
        score_cutoff=score_cutoff
    )

    st.session_state['matches_df'] = matches_df
    st.session_state['show_results'] = not matches_df.empty
    st.session_state['current_query'] = vehicle_input

    # Reset or preserve selection only for keys in current results
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
                "MODEL_YEAR": st.column_config.TextColumn(),
                "AD_MAKE": st.column_config.TextColumn(),
                "AD_MODEL": st.column_config.TextColumn(),
                "TRIM": st.column_config.TextColumn(),
                "AD_MFGCODE": st.column_config.TextColumn(help="Model Code"),
                "STYLE_ID": st.column_config.TextColumn(),
                "vehicle_search": st.column_config.TextColumn(help="Combined search string")
            },
            disabled=['score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID', 'vehicle_search']
        )

        # Persist checkbox edits back to session_state (align by index)
        for i, row in edited.iterrows():
            mk = matches_df.loc[i, 'map_key']
            st.session_state['selection'][mk] = bool(row['Select'])

        # Build final selection
        selected_keys = [k for k, v in st.session_state['selection'].items() if v]
        final_df = matches_df[matches_df['map_key'].isin(selected_keys)].copy()

        st.markdown("---")
        st.write(f"Selected: **{len(final_df)}** row(s)")

        # Preview mappings to be committed
        mappings_df = build_mappings_from_selection(final_df, source="user")
        if not mappings_df.empty:
            st.caption("Preview of rows to commit to Mappings.csv")
            st.dataframe(mappings_df, use_container_width=True)

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            if st.button("Submit Mapping"):
                if final_df.empty:
                    st.warning("No rows selected.")
                else:
                    st.success("Mapping submitted!")
                    st.dataframe(final_df[['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']].reset_index(drop=True))

        with cB:
            if not final_df.empty:
                csv = final_df[['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']].to_csv(index=False)
                st.download_button(
                    "Download mapping as CSV",
                    data=csv,
                    file_name="vehicle_mapping.csv",
                    mime="text/csv"
                )

        with cC:
            commit_btn = st.button("Commit to GitHub")
            if commit_btn:
                if mappings_df.empty:
                    st.warning("No rows selected to commit.")
                else:
                    ok, msg = github_api_commit_mappings(mappings_df)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

# =========================
# --- Footer / Tips ---
# =========================
with st.expander("Tips & Notes", expanded=False):
    st.markdown(
        """
- Enter a freeform vehicle (e.g., `2025 Land Rover Range Rover P400 SE SWB`) and optional YMMT filters, then click **Search Vehicles**.
- The results table stays open while you click the **Select** checkboxes (supports 1→many).
- **Commit to GitHub** appends/dedupes into `Mappings.csv` on your configured branch.
- For performance on large CADS, install `thefuzz[speedup]` to enable `python-Levenshtein`.
        """
    )

# --- EOF ---
