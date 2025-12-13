
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence
# Repo: klb/aff-vehicle-mapping, Branch: main

import base64
import json
import time
import requests
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# ---------------------------------------------------------------------
# Secrets / Config (Streamlit Cloud → Settings → Secrets)
# ---------------------------------------------------------------------
# [github]
# token = "ghp_XXXXXXXXXXXXXXXXXXXXXXXX"
# owner = "klb"
# repo  = "aff-vehicle-mapping"
# branch = "main"

gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

MAPPINGS_PATH  = "data/mappings.json"       # JSON file in repo to store mappings
AUDIT_LOG_PATH = "data/mappings_log.jsonl"  # optional append-only audit log (JSONL)

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
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}",
        headers=gh_headers(token),
    )
    if r.status_code == 200:
        return r.json()["object"]["sha"]
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"Failed to read branch {branch} head ({r.status_code}): {r.text}")

def ensure_feature_branch(owner, repo, token, source_branch, feature_branch):
    base_sha = get_branch_head_sha(owner, repo, source_branch, token)
    if not base_sha:
        return False  # source branch missing; caller may fallback to source_branch

    r_feat = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{feature_branch}",
        headers=gh_headers(token),
    )
    if r_feat.status_code == 200:
        return True  # already exists
    elif r_feat.status_code != 404:
        raise RuntimeError(f"Failed checking feature branch ({r_feat.status_code}): {r_feat.text}")

    r_create = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs",
        headers=gh_headers(token),
        json={"ref": f"refs/heads/{feature_branch}", "sha": base_sha},
    )
    # 201 created; 422 unprocessable if race/exists—treat as success
    return r_create.status_code in (201, 422)

def save_json_to_github(
    owner, repo, path, token, branch, payload_dict,
    commit_message, author_name=None, author_email=None,
    use_feature_branch=False, feature_branch_name="aff-mapping-app"
):
    """
    Create/update a JSON file via the GitHub Contents API with robust handling:
    - Optional write to a feature branch (to avoid main protection).
    - Retry once on 409 by re-fetching the latest SHA.
    """
    content = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    target_branch = branch
    if use_feature_branch:
        if ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
            target_branch = feature_branch_name

    sha = get_file_sha(owner, repo, path, token, ref=target_branch)
    data = {"message": commit_message, "content": content_b64, "branch": target_branch}
    if sha:
        data["sha"] = sha  # update

    if author_name and author_email:
        data["committer"] = {"name": author_name, "email": author_email}

    r = requests.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
    if r.status_code in (200, 201):
        return r.json()

    # Retry if conflict: refresh SHA and attempt update
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
    """
    Append a JSON line to an audit file, writing to main or feature branch.
    """
    target_branch = branch
    if use_feature_branch:
        if ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
            target_branch = feature_branch_name

    # Read existing
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

    # Append record
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
# Utility
# ---------------------------------------------------------------------
def secrets_status():
    missing = []
    if not GH_TOKEN:  missing.append("github.token")
    if not GH_OWNER:  missing.append("github.owner")
    if not GH_REPO:   missing.append("github.repo")
    if not GH_BRANCH: missing.append("github.branch")
    return missing

def build_key(year: str, make: str, model: str, trim: str, vehicle: str):
    # Prefer YMMT if present; otherwise use Make+Vehicle or Make+Model
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
            # Optional audit line (non-blocking)
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

# ---------------------------------------------------------------------
# Mapping editor (YMMT or Make+Vehicle)
# ---------------------------------------------------------------------
st.subheader("Edit / Add Mapping")

# Inputs (unique keys so they never collide)
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

b1, b2 = st.columns([1, 1])
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

st.caption("Local changes persist while you navigate pages. Use **Commit mappings to GitHub** (sidebar) to save permanently.")

# ---------------------------------------------------------------------
# Current mappings table
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
