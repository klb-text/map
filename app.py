
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence
# Owner = klb, Repo = aff-vehicle-mapping, Branch = main

import base64
import json
import time
import requests
import streamlit as st

# ------------------------------
# 0) Page config
# ------------------------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# ------------------------------
# 1) Secrets / Config
# ------------------------------
# In Streamlit Cloud: Settings → Secrets
# [github]
# token = "ghp_XXXXXXXXXXXXXXXXXXXXXXXX"
# owner = "klb"
# repo = "aff-vehicle-mapping"
# branch = "main"

gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

MAPPINGS_PATH   = "data/mappings.json"          # JSON file in repo to store mappings
AUDIT_LOG_PATH  = "data/mappings_log.jsonl"     # optional append-only audit log (JSONL)

# ------------------------------
# 2) GitHub Contents API helpers
# ------------------------------
def gh_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

def gh_contents_url(owner, repo, path):
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

def get_file(owner, repo, path, token, ref=None):
    """Return Response from GET /contents/{path}."""
    params = {"ref": ref} if ref else {}
    return requests.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params)

def get_file_sha(owner, repo, path, token, ref=None):
    """Return sha if file exists, None if 404; raise on other errors."""
    r = get_file(owner, repo, path, token, ref)
    if r.status_code == 200:
        return r.json()["sha"]
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"Failed to fetch SHA ({r.status_code}): {r.text}")

def load_json_from_github(owner, repo, path, token, ref=None):
    """Return dict from JSON file; None if not found; raise on other errors."""
    r = get_file(owner, repo, path, token, ref)
    if r.status_code == 200:
        j = r.json()
        decoded = base64.b64decode(j["content"]).decode("utf-8")
        return json.loads(decoded)
    elif r.status_code == 404:
        return None
    else:
        raise RuntimeError(f"Failed to load file ({r.status_code}): {r.text}")

def save_json_to_github(owner, repo, path, token, branch, payload_dict, commit_message, author_name=None, author_email=None):
    """
    Create or update a JSON file via the GitHub Contents API.
    Returns the response JSON (contains commit info).
    """
    content = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    sha = get_file_sha(owner, repo, path, token, ref=branch)
    data = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha  # required for updates

    if author_name and author_email:
        data["committer"] = {"name": author_name, "email": author_email}

    r = requests.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
    if r.status_code in (200, 201):
        return r.json()
    elif r.status_code == 409:
        # Conflict: branch protection or stale SHA
        raise RuntimeError(f"Conflict (409): {r.text}")
    else:
        raise RuntimeError(f"Failed to save file ({r.status_code}): {r.text}")

def append_jsonl_to_github(owner, repo, path, token, branch, record, commit_message):
    """
    Append a line to a JSONL file.
    This reads existing content (if any), appends one newline + record, and commits.
    """
    # Read existing
    r = get_file(owner, repo, path, token, ref=branch)
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

    data = {"message": commit_message, "content": content_b64, "branch": branch}
    if sha:
        data["sha"] = sha

    r2 = requests.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
    if r2.status_code in (200, 201):
        return r2.json()
    else:
        raise RuntimeError(f"Failed to append log ({r2.status_code}): {r2.text}")

# ------------------------------
# 3) Utility
# ------------------------------
def normalize_key(make: str, model: str):
    return f"{(make or '').strip()}:{(model or '').strip()}".strip()

def require_secrets():
    missing = []
    if not GH_TOKEN:  missing.append("github.token")
    if not GH_OWNER:  missing.append("github.owner")
    if not GH_REPO:   missing.append("github.repo")
    if not GH_BRANCH: missing.append("github.branch")
    return missing

# ------------------------------
# 4) UI
# ------------------------------
st.title("AFF Vehicle Mapping")

# Secrets sanity-check (collapsible)
with st.expander("🔐 Secrets sanity check"):
    st.write(f"Owner/Repo: {GH_OWNER}/{GH_REPO}")
    st.write(f"Branch: {GH_BRANCH}")
    st.write(f"Token present: {bool(GH_TOKEN)}")
    miss = require_secrets()
    if miss:
        st.error("Missing secrets: " + ", ".join(miss))
        st.stop()

# Load mappings into session state on first run
if "mappings" not in st.session_state:
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
    except Exception as e:
        st.session_state.mappings = {}
        st.warning(f"Could not load existing mappings (starting empty): {e}")

# Sidebar actions
st.sidebar.header("Actions")

if st.sidebar.button("🔄 Reload from GitHub"):
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
        st.sidebar.success("Reloaded from GitHub.")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

commit_msg = st.sidebar.text_input("Commit message", value="chore(app): update AFF vehicle mappings via Streamlit")
if st.sidebar.button("💾 Commit mappings to GitHub"):
    try:
        resp = save_json_to_github(
            GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH,
            st.session_state.mappings,
            commit_message=commit_msg,
            author_name="AFF Mapping App",
            author_email="aff-app@coxautoinc.com",
        )
        st.sidebar.success("Mappings committed ✅")
        st.sidebar.caption(f"Commit: {resp['commit']['sha'][:7]}")
        # Optional audit line
        try:
            append_jsonl_to_github(
                GH_OWNER, GH_REPO, AUDIT_LOG_PATH, GH_TOKEN, GH_BRANCH,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "user": "streamlit-app",
                    "action": "commit",
                    "count": len(st.session_state.mappings),
                    "path": MAPPINGS_PATH,
                },
                commit_message="chore(app): append audit commit entry",
            )
        except Exception as log_err:
            st.sidebar.warning(f"Audit log append failed (non-blocking): {log_err}")
    except Exception as e:
        st.sidebar.error(f"Commit failed: {e}")
        st.sidebar.info("If main is protected, target a feature branch or use a PR workflow.")

# Local backup/restore
st.sidebar.subheader("Backup / Restore")
backup = json.dumps(st.session_state.mappings, indent=2, ensure_ascii=False)
st.sidebar.download_button("⬇️ Download mappings.json", data=backup, file_name="mappings.json", mime="application/json")

uploaded = st.sidebar.file_uploader("⬆️ Upload mappings.json (local restore)", type=["json"])
if uploaded:
    try:
        st.session_state.mappings = json.load(uploaded)
        st.sidebar.success("Local restore complete. Remember to Commit to GitHub.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded JSON: {e}")

# ------------------------------
# 5) Mapping editor UI
# ------------------------------
st.subheader("Edit / Add Mapping")

col1, col2, col3 = st.columns(3)
with col1:
    make = st.text_input("Make", value="")
with col2:
    model = st.text_input("Model", value="")
with col3:
    mapped_code = st.text_input("Mapped Code", value="")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("Add/Update (local)"):
        key = normalize_key(make, model)
        if key in (":", "") or not make.strip() or not model.strip():
            st.error("Make and Model are required.")
        else:
            st.session_state.mappings[key] = {
                "make": make.strip(),
                "model": model.strip(),
                "code": mapped_code.strip(),
            }
            st.success(f"Updated local mapping for `{key}`.")

with c2:
    if st.button("Delete (local)"):
        key = normalize_key(make, model)
        if key in st.session_state.mappings:
            st.session_state.mappings.pop(key)
            st.success(f"Deleted local mapping `{key}`.")
        else:
            st.warning("Mapping not found.")

with c3:
    st.caption("Local changes persist during navigation; click **Commit** (sidebar) to write to GitHub.")

# ------------------------------
# 6) Current mappings table
# ------------------------------
st.subheader("Current Mappings (session)")
if st.session_state.mappings:
    rows = [{"Key": k, "Make": v.get("make", ""), "Model": v.get("model", ""), "Code": v.get("code", "")}
            for k, v in st.session_state.mappings.items()]
    st.dataframe(rows, use_container_width=True)
else:
    st.info("No mappings yet. Add one above.")

# ------------------------------
# 7) Optional: GitHub smoke test
# ------------------------------
with st.expander("🧪 Optional: GitHub read/write smoke test"):
    test_path = "data/streamlit_secrets_smoketest.json"

    # Read test
    r_read = requests.get(
        gh_contents_url(GH_OWNER, GH_REPO, test_path),
        headers=gh_headers(GH_TOKEN),
        params={"ref": GH_BRANCH},
    )
    if r_read.status_code == 200:
        st.success("Read OK: test file exists.")
        st.caption(f"SHA: {r_read.json().get('sha', '')[:7]}")
    else:
        st.info(f"Read returned {r_read.status_code} (404 expected if not present).")

    # Write test
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "ok": True}
    content = json.dumps(payload, indent=2)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    sha = r_read.json().get("sha") if r_read.status_code == 200 else None
    req_body = {"message": "test(streamlit): smoke test write from app", "content": content_b64, "branch": GH_BRANCH}
    if sha:
        req_body["sha"] = sha

    r_write = requests.put(
        gh_contents_url(GH_OWNER, GH_REPO, test_path),
        headers=gh_headers(GH_TOKEN),
        json=req_body,
    )

    if r_write.status_code in (200, 201):
        st.success(f"Write OK: {test_path} committed.")
        st.caption(f"Commit: {r_write.json()['commit']['sha'][:7]}")
    else:
        st.error(f"Write failed ({r_write.status_code}): {r_write.text}")
        st.info("If branch protection is enabled, use a feature branch or PR flow.")

# --- EOF ---
