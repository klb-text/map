
# basic_app.py — Read-only Mozenda outlet with top search bar
# - Pulls Mappings.csv from GitHub (Contents API; falls back to raw if needed)
# - Joins with local CADS.csv to add STYLE_ID (and more if you want)
# - Exposes search bar for humans AND ?q=... for Mozenda
# - Mozenda mode: ?mozenda=1&format=csv|html|json[&q=...][&score=...][&limit=...]

import base64, io
from typing import Optional
import pandas as pd
import streamlit as st
import requests
from rapidfuzz import fuzz

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping - Read Only", layout="wide")

# ---------------- Secrets / GitHub ----------------
gh = st.secrets.get("github", {})
GH_TOKEN  = gh.get("token")            # optional if repo is public
GH_OWNER  = gh.get("owner")            # e.g., "klb-text"
GH_REPO   = gh.get("repo")             # e.g., "map"
GH_BRANCH = gh.get("branch", "main")
MAP_PATH  = gh.get("path", "Mappings.csv")  # default Mappings.csv at repo root

# ---------------- Local Files ----------------
CADS_FILE = "CADS.csv"  # local CADS.csv
# Expected columns in Mappings.csv: year,make,model,trim,model_code,source
# Expected CADS columns: MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE, STYLE_ID

