
# basic_app.py
# AFF Vehicle Mapping – Minimal Harvest App
# Purpose: Deterministic, read-only vehicle lookup for Mozenda
# Full Build (2026-01-22)

import base64
import json
import re
import requests
import streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Harvest", layout="wide")

# ---------------- GitHub Config ----------------
gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

MAPPINGS_PATH    = "data/mappings.json"
UNBUILDABLE_PATH = "data/unbuildable_vehicles.json"

# ---------------- HTTP Session ----------------
_session = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)

def gh_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

def gh_contents_url(owner, repo, path):
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

# ---------------- Helpers ----------------
def canon_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# ---------------- GitHub Loaders ----------------
@st.cache_data(ttl=60)
def fetch_mappings_from_github(owner, repo, path, token, ref):
    r = _session.get(
        gh_contents_url(owner, repo, path),
        headers=gh_headers(token),
        params={"ref": ref},
        timeout=15,
    )
    if r.status_code == 200:
        decoded = base64.b64decode(r.json()["content"]).decode("utf-8")
        try:
            data = json.loads(decoded)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    elif r.status_code == 404:
        return {}
    else:
        raise RuntimeError(f"Failed to load mappings ({r.status_code}): {r.text}")

@st.cache_data(ttl=60)
def fetch_unbuildable_from_github(owner, repo, path, token, ref):
    r = _session.get(
        gh_contents_url(owner, repo, path),
        headers=gh_headers(token),
        params={"ref": ref},
        timeout=15,
    )
    if r.status_code == 200:
        decoded = base64.b64decode(r.json()["content"]).decode("utf-8")
        try:
            data = json.loads(decoded)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    elif r.status_code == 404:
        return {}
    else:
        raise RuntimeError(f"Failed to load unbuildable list ({r.status_code}): {r.text}")

# ---------------- Start App ----------------
st.title("AFF Vehicle Harvest")

params = st.experimental_get_query_params()
HARVEST_MODE = params.get("harvest", ["0"])[0] == "1"
vehicle = params.get("vehicle", [""])[0]

vehicle_text = (vehicle or "").strip()

# ---------------- Missing Vehicle Data Check (CRITICAL) ----------------
if vehicle_text:
    unbuildable = fetch_unbuildable_from_github(
        GH_OWNER, GH_REPO, UNBUILDABLE_PATH, GH_TOKEN, GH_BRANCH
    )

    if canon_text(vehicle_text) in {canon_text(k) for k in unbuildable.keys()}:
        st.markdown(
            "<p id='vehicle-status' data-status='missing'>Missing Vehicle Data</p>",
            unsafe_allow_html=True
        )
        st.stop()

# ---------------- Load Mappings ----------------
mappings = fetch_mappings_from_github(
    GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH
)

# ---------------- Render Results ----------------

def render_mapping_table(rows):
    if not rows:
        if not vehicle_text:
            st.markdown(
                "<p id='vehicle-status' data-status='empty'>No vehicle provided</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<p id='vehicle-status' data-status='unmapped'>"
                f"No mapped entries found for: {vehicle_text}"
                f"</p>",
                unsafe_allow_html=True
            )
        return


    parts = [
        "<table id='mapped_harvest' data-source='mapped'>",
        "<thead><tr>",
        "<th>Vehicle</th><th>Program Type</th><th>Code</th><th>Model Code</th><th>Scope</th>",
        "</tr></thead><tbody>"
    ]

    for r in rows:
        parts.append(
            "<tr>"
            f"<td>{r.get('vehicle','')}</td>"
            f"<td>{r.get('program_type','')}</td>"
            f"<td>{r.get('code','')}</td>"
            f"<td>{r.get('model_code','')}</td>"
            f"<td>{r.get('scope','')}</td>"
            "</tr>"
        )

    parts.append("</tbody></table>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)

# ---------------- Vehicle Lookup ----------------
matched = []

if vehicle_text:
    cv = canon_text(vehicle_text)
    for _, v in mappings.items():
        if canon_text(v.get("vehicle", "")) == cv:
            matched.append(v)

render_mapping_table(matched)

if HARVEST_MODE:
    st.stop()

# --- EOF ---
