
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search
# Repo: klb-text/map, Branch: main

import base64
import json
import time
import io
from typing import Optional
import requests
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# ---------------------------------------------------------------------
# Secrets / Config (Streamlit Cloud → Settings → Secrets)
# ---------------------------------------------------------------------
# [github]
# token  = "ghp_XXXXXXXXXXXXXXXXXXXXXXXX"
# owner  = "klb-text"
# repo   = "map"
# branch = "main"

gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

# Paths in your repo
MAPPINGS_PATH  = "data/mappings.json"       # JSON file to store mappings (created by app)
AUDIT_LOG_PATH = "data/mappings_log.jsonl"  # optional append-only audit log (JSONL)
CADS_PATH      = "CADS.csv"                 # default to root-level CADS.csv per your repo screenshot
CADS_IS_EXCEL  = False                      # set True if CADS is Excel

# ---------------------------------------------------------------------
# GitHub helpers (Contents API + refs)
# ---------------------------------------------------------------------
def gh_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

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
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"Failed to fetch SHA ({r.status_code}): {r.text}")

def load_json_from_github(owner, repo, path, token, ref=None):
    r = get_file(owner, repo, path, token, ref)
    if r.status_code == 200:
        j = r.json()
        decoded = base64.b64decode(j["content"]).decode("utf-8")
        return json.loads(decoded)
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"Failed to load file ({r.status_code}): {r.text}")

def get_branch_head_sha(owner, repo, branch, token):
    r = requests.get(gh_ref_heads(owner, repo, branch), headers=gh_headers(token))
    if r.status_code == 200:
        return r.json()["object"]["sha"]
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"Failed to read branch {branch} head ({r.status_code}): {r.text}")

def ensure_feature_branch(owner, repo, token, source_branch, feature_branch):
    base_sha = get_branch_head_sha(owner, repo, source_branch, token)
    if not base_sha:
        return False

    r_feat = requests.get(gh_ref_heads(owner, repo, feature_branch), headers=gh_headers(token))
    if r_feat.status_code == 200:
        return True
    elif r_feat.status_code != 404:
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
    if use_feature_branch:
        if ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
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
    if use_feature_branch:
        if ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
            target_branch = feature_branch_name

    r = get_file(owner, repo, path, token, ref=target_branch)
    lines = ""
    sha = None
    if r.status_code == 200:
        sha = r.json()["sha"]
        existing = base64.b64decode(r.json()["content"]).decode("utf-8")
        lines = existing
        if not lines.endswith("\n") and lines != "":
            lines += "\n"
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
    else:
        raise RuntimeError(f"Failed to append log ({r2.status_code}): {r2.text}")

# ---------------------------------------------------------------------
# CADS loaders (CSV or Excel) + caching
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

    elif r.status_code == 404:
        raise FileNotFoundError(f"CADS file not found at {path} in {owner}/{repo}@{ref or 'default'}")
    else:
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

    elif r.status_code == 404:
        raise FileNotFoundError(f"CADS file not found at {path} in {owner}/{repo}@{ref or 'default'}")
    else:
        raise RuntimeError(f"Failed to load CADS Excel ({r.status_code}): {r.text}")

# ---------------------------------------------------------------------
# CADS filter (robust across column name variants)
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

def filter_cads(df: pd.DataFrame, year: str, make: str, model: str, trim: str, vehicle: str) -> pd.DataFrame:
    y  = (year or "").strip()
    mk = (make or "").strip()
    md = (model or "").strip()
    tr = (trim or "").strip()
    vh = (vehicle or "").strip()

    df2 = df.copy()

    year_col    = _find_col(df2, ["Year", "MY", "ModelYear"])
    make_col    = _find_col(df2, ["Make", "MakeName", "Manufacturer"])
    model_col   = _find_col(df2, ["Model", "Line", "Carline", "Series"])
    trim_col    = _find_col(df2, ["Trim", "Grade", "Variant", "Submodel"])
    vehicle_col = _find_col(df2, ["Vehicle", "Description", "ModelTrim", "ModelName"])

    for col in [year_col, make_col, model_col, trim_col, vehicle_col]:
        if col and col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip()

    conds = []
    if y and year_col:
        conds.append(df2[year_col].astype(str).str.contains(y, case=False, na=False))
    if mk and make_col:
        conds.append(df2[make_col].astype(str).str.contains(mk, case=False, na=False))
    if md and model_col:
        conds.append(df2[model_col].astype(str).str.contains(md, case=False, na=False))
    if tr and trim_col:
        conds.append(df2[trim_col].astype(str).str.contains(tr, case=False, na=False))

    if conds:
        mask = conds[0]
        for c in conds[1:]:
            mask = mask & c
        res = df2[mask]
        if len(res) > 0:
            return res

    if mk and md and make_col and model_col:
        mm = (df2[make_col].astype(str).str.contains(mk, case=False, na=False)) & \
             (df2[model_col].astype(str).str.contains(md, case=False, na=False))
        res_mm = df2[mm]
        if len(res_mm) > 0:
            return res_mm

    if mk and vh and make_col and vehicle_col:
        mv = (df2[make_col].astype(str).str.contains(mk, case=False, na=False)) & \
             (df2[vehicle_col].astype(str).str.contains(vh, case=False, na=False))
        res_mv = df2[mv]
        if len(res_mv) > 0:
            return res_mv

    return df2.iloc[0:0]

# ---------------------------------------------------------------------
# Utility
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

def build_key(year: str, make: str, model: str, trim: str, vehicle: str):
    y  = (year or "").strip()
    mk = (make or "").strip()
    md = (model or "").strip()
    tr = (trim or "").strip()
    vh = (vehicle or "").strip()
    if mk and (y or md or tr):
        return f"{y}-{mk}-{md}-{tr}".strip("-")
    elif mk and vh:
        return f"{mk}:{vh}"
    elif mk and md:
        return f"{mk}:{md}"
    else:
        return mk or vh or "UNSPECIFIED"

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("AFF Vehicle Mapping")

# Load mappings into session state on first run
if "mappings" not in st.session_state:
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
    except Exception as e:
        st.session_state.mappings = {}
        st.warning(f"Starting with empty mappings (load error): {e}")

# Sidebar: actions
st.sidebar.header("Actions")

# Reload
if st.sidebar.button("🔄 Reload from GitHub"):
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
        st.sidebar.success("Reloaded from GitHub.")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

# Commit controls
commit_msg = st.sidebar.text_input(
    "Commit message",
    value="chore(app): update AFF vehicle mappings via Streamlit",
    key="commit_message_input",
)
use_feature_branch = st.sidebar.checkbox(
    "Use feature branch (aff-mapping-app)",
    value=False,
    key="feature_branch_checkbox",
)

missing = secrets_status()
secrets_ok = (len(missing) == 0)
if not secrets_ok:
    st.sidebar.warning("Missing secrets: " + ", ".join(missing))

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
st.sidebar.download_button(
    "⬇️ Download mappings.json",
    data=backup,
    file_name="mappings.json",
    mime="application/json",
    key="download_button",
)

uploaded = st.sidebar.file_uploader(
    "⬆️ Upload mappings.json (local restore)",
    type=["json"],
    key="upload_file",
)
if uploaded:
    try:
        st.session_state.mappings = json.load(uploaded)
        st.sidebar.success("Local restore complete. Remember to Commit to GitHub.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded JSON: {e}")

# Sidebar: CADS settings
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH, key="cads_path_input")
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL, key="cads_is_excel_checkbox")
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value="0", key="cads_sheet_input")
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv", "xlsx"], key="cads_upload")

# ---------------------------------------------------------------------
# Mapping editor (YMMT or Make+Vehicle)
# ---------------------------------------------------------------------
st.subheader("Edit / Add Mapping")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    year = st.text_input("Year", key="year_input", placeholder="e.g., 2025")
with c2:
    make = st.text_input("Make", key="make_input", placeholder="e.g., Acura")
with c3:
    model = st.text_input("Model", key="model_input", placeholder="e.g., MDX")
with c4:
    trim = st.text_input("Trim", key="trim_input", placeholder="e.g., Base")
with c5:
    vehicle = st.text_input("Vehicle (alt)", key="vehicle_input", placeholder="e.g., MDX 3.5L")
with c6:
    mapped_code = st.text_input("Mapped Code", key="code_input", placeholder="e.g., ACU-MDX-BASE")

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("Add/Update (local)", key="add_update_local"):
        k = build_key(year, make, model, trim, vehicle)
        if (not make.strip()) and (not vehicle.strip()):
            st.error("Provide at least Make or Vehicle, and optionally Year/Model/Trim.")
        else:
            st.session_state.mappings[k] = {
                "year": (year or "").strip(),
                "make": (make or "").strip(),
                "model": (model or "").strip(),
                "trim": (trim or "").strip(),
                "vehicle": (vehicle or "").strip(),
                "code": (mapped_code or "").strip(),
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
                    df_cads = load_cads_from_github_excel(
                        GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH, sheet_name=sheet_arg
                    )
                else:
                    df_cads = load_cads_from_github_csv(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH)

            results = filter_cads(df_cads, year, make, model, trim, vehicle)
            if len(results) == 0:
                st.warning("No CADS rows matched your input. Try relaxing the filters (e.g., omit Trim).")
            else:
                st.success(f"Found {len(results)} CADS rows.")
                st.dataframe(results, use_container_width=True)
        except FileNotFoundError as fnf:
            st.error(str(fnf))
            st.info(f"Ensure the CADS file exists at `{CADS_PATH}` in `{GH_OWNER}/{GH_REPO}` on branch `{GH_BRANCH}`.")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

st.caption("Local changes persist while you navigate pages. Use **Commit mappings to GitHub** (sidebar) to save permanently.")

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
    st.dataframe(rows, use_container_width=True)
else:
    st.info("No mappings yet. Add one above.")

# --- EOF ---
