
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection + attribute-based existing mapping detection
# Repo: klb-text/map, Branch: main

import base64
import json
import time
import io
from typing import Optional, List, Dict, Tuple
import requests
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# ---------------------------------------------------------------------
# Secrets / Config
# ---------------------------------------------------------------------
gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

# Paths in your repo
MAPPINGS_PATH   = "data/mappings.json"
AUDIT_LOG_PATH  = "data/mappings_log.jsonl"
CADS_PATH       = "CADS.csv"         # default root-level CADS.csv
CADS_IS_EXCEL   = False
CADS_CODE_PREFS = ["STYLE_ID", "AD_MFGCODE", "AD_VEH_ID"]  # preferred code column order

# ---------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------
def gh_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

def gh_contents_url(owner, repo, path):
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

def gh_ref_heads(owner, repo, branch):
    return f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}"

def get_file(owner, repo, path, token, ref=None):
    params = {"ref": ref} if ref else {}
    return requests.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params)

def get_file_sha(owner, repo, path, token, ref=None):
    r = get_file(owner, repo, path, token, ref)
    if r.status_code == 200:
        return r.json()["sha"]
    if r.status_code == 404:
        return None
    raise RuntimeError(f"Failed to fetch SHA ({r.status_code}): {r.text}")

def load_json_from_github(owner, repo, path, token, ref=None):
    r = get_file(owner, repo, path, token, ref)
    if r.status_code == 200:
        j = r.json()
        decoded = base64.b64decode(j["content"]).decode("utf-8")
        return json.loads(decoded)
    if r.status_code == 404:
        return None
    raise RuntimeError(f"Failed to load file ({r.status_code}): {r.text}")

def get_branch_head_sha(owner, repo, branch, token):
    r = requests.get(gh_ref_heads(owner, repo, branch), headers=gh_headers(token))
    if r.status_code == 200:
        return r.json()["object"]["sha"]
    if r.status_code == 404:
        return None
    raise RuntimeError(f"Failed to read branch {branch} head ({r.status_code}): {r.text}")

def ensure_feature_branch(owner, repo, token, source_branch, feature_branch):
    base_sha = get_branch_head_sha(owner, repo, source_branch, token)
    if not base_sha:
        return False
    r_feat = requests.get(gh_ref_heads(owner, repo, feature_branch), headers=gh_headers(token))
    if r_feat.status_code == 200:
        return True
    if r_feat.status_code != 404:
        raise RuntimeError(f"Failed checking feature branch ({r_feat.status_code}): {r_feat.text}")
    r_create = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs",
        headers=gh_headers(token),
        json={"ref": f"refs/heads/{feature_branch}", "sha": base_sha},
    )
    return r_create.status_code in (201, 422)

def save_json_to_github(
    owner, repo, path, token, branch, payload_dict,
    commit_message, author_name=None, author_email=None,
    use_feature_branch=False, feature_branch_name="aff-mapping-app"
):
    content = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    target_branch = branch
    if use_feature_branch and ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
        target_branch = feature_branch_name
    sha = get_file_sha(owner, repo, path, token, ref=target_branch)
    data = {"message": commit_message, "content": content_b64, "branch": target_branch}
    if sha:
        data["sha"] = sha
    if author_name and author_email:
        data["committer"] = {"name": author_name, "email": author_email}
    r = requests.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
    if r.status_code in (200, 201):
        return r.json()
    if r.status_code == 409:
        latest_sha = get_file_sha(owner, repo, path, token, ref=target_branch)
        if latest_sha and not data.get("sha"):
            data["sha"] = latest_sha
            r2 = requests.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
            if r2.status_code in (200, 201):
                return r2.json()
    raise RuntimeError(f"Failed to save file ({r.status_code}): {r.text}")

def append_jsonl_to_github(
    owner, repo, path, token, branch, record, commit_message,
    use_feature_branch=False, feature_branch_name="aff-mapping-app"
):
    target_branch = branch
    if use_feature_branch and ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
        target_branch = feature_branch_name
    r = get_file(owner, repo, path, token, ref=target_branch)
    lines, sha = "", None
    if r.status_code == 200:
        sha = r.json()["sha"]
        existing = base64.b64decode(r.json()["content"]).decode("utf-8")
        if existing:
            lines = existing if existing.endswith("\n") else existing + "\n"
    elif r.status_code != 404:
        raise RuntimeError(f"Failed to read log file ({r.status_code}): {r.text}")
    lines += json.dumps(record, ensure_ascii=False) + "\n"
    content_b64 = base64.b64encode(lines.encode("utf-8")).decode("utf-8")
    data = {"message": commit_message, "content": content_b64, "branch": target_branch}
    if sha:
        data["sha"] = sha
    r2 = requests.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
    if r2.status_code in (200, 201):
        return r2.json()
    raise RuntimeError(f"Failed to append log ({r2.status_code}): {r2.text}")

# ---------------------------------------------------------------------
# CADS loaders (CSV/Excel) + caching
# ---------------------------------------------------------------------
@st.cache_data(ttl=600)
def _decode_bytes_to_text(raw: bytes) -> tuple[str, str]:
    if not raw or raw.strip() == b"":
        return ("", "empty")
    encoding = "utf-8"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        encoding = "utf-16"
    elif raw.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    text = raw.decode(encoding, errors="replace")
    return (text, encoding)

@st.cache_data(ttl=600)
def load_cads_from_github_csv(owner, repo, path, token, ref=None) -> pd.DataFrame:
    import csv
    params = {"ref": ref} if ref else {}
    r = requests.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params)
    if r.status_code == 200:
        j = r.json()
        raw = None
        if "content" in j and j["content"]:
            try:
                raw = base64.b64decode(j["content"])
            except Exception:
                raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = requests.get(j["download_url"])
            if r2.status_code == 200:
                raw = r2.content
        if raw is None or raw.strip() == b"":
            raise ValueError(f"CADS file `{path}` appears to be empty or unavailable via API.")
        text, _enc = _decode_bytes_to_text(raw)
        sample = text[:4096]
        delimiter = None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            delimiter = dialect.delimiter
        except Exception:
            for cand in [",", "\t", ";", "|"]:
                if cand in sample:
                    delimiter = cand
                    break
        if delimiter is None:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, on_bad_lines="skip")
        else:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, on_bad_lines="skip", engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")
        if df.empty or len(df.columns) == 0:
            raise ValueError("CADS CSV parsed but produced no columns or rows. Check delimiter and headers.")
        df = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else "")
        return df
    if r.status_code == 404:
        raise FileNotFoundError(f"CADS file not found at {path} in {owner}/{repo}@{ref or 'default'}")
    raise RuntimeError(f"Failed to load CADS CSV ({r.status_code}): {r.text}")

@st.cache_data(ttl=600)
def load_cads_from_github_excel(owner, repo, path, token, ref=None, sheet_name=0) -> pd.DataFrame:
    params = {"ref": ref} if ref else {}
    r = requests.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params)
    if r.status_code == 200:
        j = r.json()
        raw = None
        if "content" in j and j["content"]:
            try:
                raw = base64.b64decode(j["content"])
            except Exception:
                raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = requests.get(j["download_url"])
            if r2.status_code == 200:
                raw = r2.content
        if raw is None or raw.strip() == b"":
            raise ValueError(f"CADS file `{path}` appears to be empty or unavailable via API.")
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, engine="openpyxl")
        df = df.astype(str).applymap(lambda x: str(x).strip())
        return df
    if r.status_code == 404:
        raise FileNotFoundError(f"CADS file not found at {path} in {owner}/{repo}@{ref or 'default'}")
    raise RuntimeError(f"Failed to load CADS Excel ({r.status_code}): {r.text}")

# ---------------------------------------------------------------------
# Filtering helpers (column-aware + exact/contains)
# ---------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None

def _normalize(val: str) -> str:
    return (val or "").strip()

def _to_int_or_str(val: str):
    s = _normalize(val)
    try:
        return int(s)
    except Exception:
        return s

def filter_cads(
    df: pd.DataFrame,
    year: str, make: str, model: str, trim: str, vehicle: str,
    exact_year: bool = True, exact_mmt: bool = False, case_sensitive: bool = False,
) -> pd.DataFrame:
    y, mk, md, tr, vh = map(_normalize, (year, make, model, trim, vehicle))
    df2 = df.copy()
    YEAR_CANDS    = ["AD_YEAR", "Year", "MY", "ModelYear", "Model Year"]
    MAKE_CANDS    = ["AD_MAKE", "Make", "MakeName", "Manufacturer"]
    MODEL_CANDS   = ["AD_MODEL", "Model", "Line", "Carline", "Series"]
    TRIM_CANDS    = ["AD_TRIM", "Trim", "Grade", "Variant", "Submodel"]
    VEHICLE_CANDS = ["Vehicle", "Description", "ModelTrim", "ModelName", "AD_SERIES", "Series"]
    year_col    = _find_col(df2, YEAR_CANDS)
    make_col    = _find_col(df2, MAKE_CANDS)
    model_col   = _find_col(df2, MODEL_CANDS)
    trim_col    = _find_col(df2, TRIM_CANDS)
    vehicle_col = _find_col(df2, VEHICLE_CANDS)
    for col in [year_col, make_col, model_col, trim_col, vehicle_col]:
        if col and col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip()

    def col_contains(col, val):
        if not col or val == "":
            return None
        series = df2[col].astype(str)
        return series.str.contains(val, case=case_sensitive, na=False)

    def col_equals(col, val):
        if not col or val == "":
            return None
        series = df2[col].astype(str)
        return (series == val) if case_sensitive else (series.str.lower() == val.lower())

    masks = []
    if y and year_col:
        y_parsed = _to_int_or_str(y)
        try:
            df_year_int = df2[year_col].astype(int)
            masks.append(df_year_int == y_parsed if isinstance(y_parsed, int) else (col_equals(year_col, y) if exact_year else col_contains(year_col, y)))
        except Exception:
            masks.append(col_equals(year_col, y) if exact_year else col_contains(year_col, y))
    if make_col and mk:
        masks.append(col_equals(make_col, mk) if exact_mmt else col_contains(make_col, mk))
    if model_col and md:
        masks.append(col_equals(model_col, md) if exact_mmt else col_contains(model_col, md))
    if trim_col and tr:
        masks.append(col_equals(trim_col, tr) if exact_mmt else col_contains(trim_col, tr))

    primary = None
    for m in masks:
        if m is not None:
            primary = m if primary is None else (primary & m)

    if primary is not None:
        res = df2[primary]
        if len(res) > 0:
            return res

    if make_col and model_col and mk and md:
        mm = col_equals(make_col, mk) if exact_mmt else col_contains(make_col, mk)
        mo = col_equals(model_col, md) if exact_mmt else col_contains(model_col, md)
        if mm is not None and mo is not None:
            res_mm = df2[mm & mo]
            if len(res_mm) > 0:
                return res_mm

    if make_col and vehicle_col and mk and vh:
        mv = col_equals(make_col, mk) if exact_mmt else col_contains(make_col, mk)
        vv = col_equals(vehicle_col, vh) if exact_mmt else col_contains(vehicle_col, vh)
        if mv is not None and vv is not None:
            res_mv = df2[mv & vv]
            if len(res_mv) > 0:
                return res_mv

    return df2.iloc[0:0]

# ---------------------------------------------------------------------
# Mapping utilities
# ---------------------------------------------------------------------
def secrets_status():
    missing = []
    if not GH_TOKEN:
        missing.append("github.token")
    if not GH_OWNER:
        missing.append("github.owner")
    if not GH_REPO:
        missing.append("github.repo")
    if not GH_BRANCH:
        missing.append("github.branch")
    return missing

def build_key(year: str, make: str, model: str, trim: str, vehicle: str) -> str:
    y, mk, md, tr, vh = map(_normalize, (year, make, model, trim, vehicle))
    if mk and (y or md or tr):
        return f"{y}-{mk}-{md}-{tr}".strip("-")
    elif mk and vh:
        return f"{mk}:{vh}"
    elif mk and md:
        return f"{mk}:{md}"
    else:
        return mk or vh or "UNSPECIFIED"

def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    prefs = [c for c in CADS_CODE_PREFS if c in df.columns]
    return prefs if prefs else list(df.columns)

def _eq(a: str, b: str, case_sensitive: bool = False) -> bool:
    a = _normalize(a); b = _normalize(b)
    return (a == b) if case_sensitive else (a.lower() == b.lower())

def find_existing_mappings(
    mappings: Dict[str, Dict[str, str]],
    year: str, make: str, model: str, trim: str, vehicle: str,
    exact_year: bool = True, case_sensitive: bool = False
) -> List[Tuple[str, Dict[str, str]]]:
    y, mk, md, tr, vh = map(_normalize, (year, make, model, trim, vehicle))
    out = []
    for k, v in mappings.items():
        vy  = _normalize(v.get("year", ""))
        vmk = _normalize(v.get("make", ""))
        vmd = _normalize(v.get("model", ""))
        vtr = _normalize(v.get("trim", ""))
        vvh = _normalize(v.get("vehicle", ""))

        if y:
            if exact_year:
                try:
                    if int(vy) != int(y):
                        continue
                except Exception:
                    if not _eq(vy, y, case_sensitive):
                        continue
            else:
                if not _eq(vy, y, case_sensitive):
                    continue
        if mk and not _eq(vmk, mk, case_sensitive):
            continue
        if md and not _eq(vmd, md, case_sensitive):
            continue
        if tr and not _eq(vtr, tr, case_sensitive):
            continue
        if vh and not _eq(vvh, vh, case_sensitive):
            continue

        out.append((k, v))
    return out

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("AFF Vehicle Mapping")

# Load mappings on first run
if "mappings" not in st.session_state:
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
    except Exception as e:
        st.session_state.mappings = {}
        st.warning(f"Starting with empty mappings (load error): {e}")

# Sidebar: actions
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Reload from GitHub"):
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
        st.sidebar.success("Reloaded from GitHub.")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

commit_msg = st.sidebar.text_input("Commit message", value="chore(app): update AFF vehicle mappings via Streamlit", key="commit_message_input")
use_feature_branch = st.sidebar.checkbox("Use feature branch (aff-mapping-app)", value=False, key="feature_branch_checkbox")

miss = secrets_status()
secrets_ok = (len(miss) == 0)
if not secrets_ok:
    st.sidebar.warning("Missing secrets: " + ", ".join(miss))

if st.sidebar.button("💾 Commit mappings to GitHub", key="commit_button"):
    if not secrets_ok:
        st.sidebar.error("Cannot commit: fix missing secrets first.")
    else:
        try:
            resp = save_json_to_github(
                GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH,
                st.session_state.mappings,
                commit_message=commit_msg,
                author_name="AFF Mapping App",
                author_email="aff-app@coxautoinc.com",
                use_feature_branch=use_feature_branch,
            )
            st.sidebar.success("Mappings committed ✅")
            st.sidebar.caption(f"Commit: {resp['commit']['sha'][:7]}")
            try:
                append_jsonl_to_github(
                    GH_OWNER, GH_REPO, AUDIT_LOG_PATH, GH_TOKEN, GH_BRANCH,
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "user": "streamlit-app",
                        "action": "commit",
                        "count": len(st.session_state.mappings),
                        "path": MAPPINGS_PATH,
                        "branch": GH_BRANCH if not use_feature_branch else "aff-mapping-app",
                    },
                    commit_message="chore(app): append audit commit entry",
                    use_feature_branch=use_feature_branch,
                )
            except Exception as log_err:
                st.sidebar.warning(f"Audit log append failed (non-blocking): {log_err}")
        except Exception as e:
            st.sidebar.error(f"Commit failed: {e}")
            st.sidebar.info("If main is protected, enable feature branch and merge via PR.")

# Backup / Restore
st.sidebar.subheader("Backup / Restore")
backup = json.dumps(st.session_state.mappings, indent=2, ensure_ascii=False)
st.sidebar.download_button("⬇️ Download mappings.json", data=backup, file_name="mappings.json", mime="application/json", key="download_button")
uploaded = st.sidebar.file_uploader("⬆️ Upload mappings.json (local restore)", type=["json"], key="upload_file")
if uploaded:
    try:
        st.session_state.mappings = json.load(uploaded)
        st.sidebar.success("Local restore complete. Remember to Commit to GitHub.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded JSON: {e}")

# Sidebar: CADS settings & matching controls
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH, key="cads_path_input")
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL, key="cads_is_excel_checkbox")
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value="0", key="cads_sheet_input")
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv", "xlsx"], key="cads_upload")

st.sidebar.subheader("Matching Controls")
EXACT_YEAR = st.sidebar.checkbox("Exact Year match", value=True, key="exact_year_checkbox")
EXACT_MMT  = st.sidebar.checkbox("Exact Make/Model/Trim match", value=False, key="exact_mmt_checkbox")
CASE_SENSITIVE = st.sidebar.checkbox("Case sensitive matching", value=False, key="case_sensitive_checkbox")
BLOCK_SEARCH_IF_MAPPED = st.sidebar.checkbox("Block CADS search if mapping exists", value=True, key="block_search_checkbox")

# ---------------------------------------------------------------------
# Mapping editor inputs
# ---------------------------------------------------------------------
st.subheader("Edit / Add Mapping")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: year = st.text_input("Year", key="year_input", placeholder="e.g., 2025")
with c2: make = st.text_input("Make", key="make_input", placeholder="e.g., Acura")
with c3: model = st.text_input("Model", key="model_input", placeholder="e.g., MDX")
with c4: trim = st.text_input("Trim", key="trim_input", placeholder="e.g., Base")
with c5: vehicle = st.text_input("Vehicle (alt)", key="vehicle_input", placeholder="e.g., MDX 3.5L")
with c6: mapped_code = st.text_input("Mapped Code", key="code_input", placeholder="e.g., ACU-MDX-BASE")

# Existing mapping detection (attribute-based)
matches = find_existing_mappings(st.session_state.mappings, year, make, model, trim, vehicle, exact_year=EXACT_YEAR, case_sensitive=CASE_SENSITIVE)
st.subheader("Existing Mapping (for current inputs)")
if matches:
    st.success(f"Already mapped: {len(matches)} match(es) found.")
    rows = []
    for k, v in matches:
        rows.append({
            "Key": k, "Year": v.get("year", ""), "Make": v.get("make", ""),
            "Model": v.get("model", ""), "Trim": v.get("trim", ""),
            "Vehicle": v.get("vehicle", ""), "Code": v.get("code", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No existing mapping for current inputs.")

# Add/Update/Delete local mapping
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("Add/Update (local)", key="add_update_local"):
        k = build_key(year, make, model, trim, vehicle)
        if (not _normalize(make)) and (not _normalize(vehicle)):
            st.error("Provide at least Make or Vehicle, and optionally Year/Model/Trim.")
        else:
            st.session_state.mappings[k] = {
                "year": _normalize(year),
                "make": _normalize(make),
                "model": _normalize(model),
                "trim": _normalize(trim),
                "vehicle": _normalize(vehicle),
                "code": _normalize(mapped_code),
            }
            st.success(f"Updated local mapping for `{k}`.")
with b2:
    if st.button("Delete (local)", key="delete_local"):
        k = build_key(year, make, model, trim, vehicle)
        if k in st.session_state.mappings:
            st.session_state.mappings.pop(k)
            st.success(f"Deleted local mapping `{k}`.")
        else:
            st.warning("Mapping not found.")
with b3:
    if st.button("🔎 Search CADS", key="search_cads"):
        try:
            if BLOCK_SEARCH_IF_MAPPED and matches:
                st.info("Search blocked: this input is already mapped. Toggle 'Block CADS search if mapping exists' OFF to search anyway.")
            else:
                if cads_upload is not None:
                    if cads_upload.name.lower().endswith(".xlsx"):
                        df_cads = pd.read_excel(cads_upload, engine="openpyxl")
                    else:
                        df_cads = pd.read_csv(cads_upload)
                else:
                    if CADS_IS_EXCEL:
                        sheet_arg = CADS_SHEET_NAME
                        try:
                            sheet_arg = int(sheet_arg)
                        except Exception:
                            pass
                        df_cads = load_cads_from_github_excel(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH, sheet_name=sheet_arg)
                    else:
                        df_cads = load_cads_from_github_csv(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH)

                results = filter_cads(
                    df_cads, year, make, model, trim, vehicle,
                    exact_year=EXACT_YEAR, exact_mmt=EXACT_MMT, case_sensitive=CASE_SENSITIVE,
                )
                if len(results) == 0:
                    st.warning("No CADS rows matched your input. Try relaxing the filters (e.g., omit Trim).")
                else:
                    selectable = results.copy()
                    if "Select" not in selectable.columns:
                        selectable.insert(0, "Select", False)
                    st.session_state["results_df"] = selectable
                    st.session_state["code_candidates"] = get_cads_code_candidates(results)
                    st.session_state["code_column"] = (
                        st.session_state["code_candidates"][0] if st.session_state["code_candidates"] else None
                    )
                    st.success(f"Found {len(selectable)} CADS rows. Use checkboxes to select one or more.")
        except FileNotFoundError as fnf:
            st.error(str(fnf))
            st.info(f"Ensure the CADS file exists at `{CADS_PATH}` in `{GH_OWNER}/{GH_REPO}` on branch `{GH_BRANCH}`.")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

st.caption("Local changes persist while you navigate pages. Use **Commit mappings to GitHub** (sidebar) to save permanently.")

# ---------------------------------------------------------------------
# Select vehicles from CADS results and add to mappings
# ---------------------------------------------------------------------
if "results_df" in st.session_state:
    st.subheader("Select vehicles from CADS results")

    code_candidates = st.session_state.get("code_candidates", [])
    st.session_state["code_column"] = st.selectbox(
        "Mapped Code column (from CADS results)",
        options=code_candidates if code_candidates else list(st.session_state["results_df"].columns),
        index=0 if code_candidates else 0,
        key="code_column_select",
    )

    csel1, csel2 = st.columns(2)
    with csel1:
        if st.button("✅ Select All", key="select_all_btn"):
            df_tmp = st.session_state["results_df"].copy()
            df_tmp["Select"] = True
            st.session_state["results_df"] = df_tmp
    with csel2:
        if st.button("🧹 Clear Selection", key="clear_selection_btn"):
            df_tmp = st.session_state["results_df"].copy()
            df_tmp["Select"] = False
            st.session_state["results_df"] = df_tmp

    edited = st.data_editor(
        st.session_state["results_df"],
        key="results_editor",
        use_container_width=True,
        num_rows="dynamic",
    )
    st.session_state["results_df"] = edited

    selected_rows = edited[edited["Select"] == True]
    st.caption(f"Selected {len(selected_rows)} vehicle(s).")

    if st.button("➕ Add selected vehicle(s) to mappings", key="add_selected_to_mappings"):
        if selected_rows.empty:
            st.warning("No rows selected. Tick the 'Select' checkbox for one or more rows.")
        else:
            df2 = selected_rows.copy()
            year_col    = _find_col(df2, ["AD_YEAR", "Year", "MY", "ModelYear", "Model Year"])
            make_col    = _find_col(df2, ["AD_MAKE", "Make", "MakeName", "Manufacturer"])
            model_col   = _find_col(df2, ["AD_MODEL", "Model", "Line", "Carline", "Series"])
            trim_col    = _find_col(df2, ["AD_TRIM", "Trim", "Grade", "Variant", "Submodel"])
            vehicle_col = _find_col(df2, ["Vehicle", "Description", "ModelTrim", "ModelName", "AD_SERIES", "Series"])
            code_col    = st.session_state.get("code_column")

            added = 0
            for _, row in df2.iterrows():
                yv  = _normalize(row.get(year_col, ""))    if year_col else ""
                mkv = _normalize(row.get(make_col, ""))    if make_col else ""
                mdv = _normalize(row.get(model_col, ""))   if model_col else ""
                trv = _normalize(row.get(trim_col, ""))    if trim_col else ""
                vhv = _normalize(row.get(vehicle_col, "")) if vehicle_col else ""
                key = build_key(yv, mkv, mdv, trv, vhv)
                code_val = _normalize(str(row.get(code_col, ""))) if code_col else ""

                st.session_state.mappings[key] = {
                    "year": yv, "make": mkv, "model": mdv, "trim": trv,
                    "vehicle": vhv, "code": code_val,
                }
                added += 1
            st.success(f"Added/updated {added} mapping(s). You can commit them in the sidebar.")

# ---------------------------------------------------------------------
# Current Mappings table
# ---------------------------------------------------------------------
st.subheader("Current Mappings (session)")
if st.session_state.mappings:
    rows = []
    for k, v in st.session_state.mappings.items():
        rows.append({
            "Key": k,
            "Year": v.get("year", ""),
            "Make": v.get("make", ""),
            "Model": v.get("model", ""),
            "Trim": v.get("trim", ""),
            "Vehicle": v.get("vehicle", ""),
            "Code": v.get("code", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No mappings yet. Add one above or select CADS rows to add mappings.")

# --- EOF ---
