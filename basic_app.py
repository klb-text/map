
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

# ---------------- Utils ----------------
@st.cache_data
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")

def _gh_headers(token: Optional[str]):
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def _gh_contents_url(owner: str, repo: str, path: str, ref: str):
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"

def _gh_raw_url(owner: str, repo: str, path: str, ref: str):
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

def normalize(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower().replace("-", " ").replace("/", " ")
    return " ".join(s.split())

@st.cache_data(show_spinner=False)
def fetch_mappings_github(owner: str, repo: str, path: str, ref: str, token: Optional[str]) -> pd.DataFrame:
    """
    Try Contents API (supports private) with PAT; on 401/403 or missing token,
    fallback to raw.githubusercontent.com (works for public repos).
    Returns empty DF if not found or error.
    """
    # 1) Contents API attempt
    try:
        r = requests.get(_gh_contents_url(owner, repo, path, ref), headers=_gh_headers(token), timeout=20)
        if r.status_code == 200:
            content_b64 = r.json().get("content", "")
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(decoded), dtype=str).fillna("")
        # fallback conditions
    except Exception:
        pass

    # 2) Raw fallback (public only)
    try:
        raw_url = _gh_raw_url(owner, repo, path, ref)
        r2 = requests.get(raw_url, timeout=20)
        if r2.status_code == 200:
            return pd.read_csv(io.StringIO(r2.text), dtype=str).fillna("")
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def join_mappings_to_cads(mappings_df: pd.DataFrame, cads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Mappings -> CADS on (year, make, model, trim, model_code) -> (MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE).
    Returns a joined DataFrame with a compact schema useful for Mozenda and search.
    """
    if mappings_df.empty or cads_df.empty:
        return pd.DataFrame()

    md = mappings_df.rename(columns=str).fillna("").astype(str)
    cd = cads_df.rename(columns=str).fillna("").astype(str)

    # Ensure required columns exist
    for c in ["year","make","model","trim","model_code","source"]:
        if c not in md.columns:
            md[c] = ""

    for c in ["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE","STYLE_ID"]:
        if c not in cd.columns:
            cd[c] = ""

    merged = md.merge(
        cd,
        left_on=["year","make","model","trim","model_code"],
        right_on=["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"],
        how="left",
        suffixes=("","_cad")
    )

    # Build a combined search string (for the top search bar and q=)
    merged["search_text"] = (
        merged["year"].astype(str).str.strip() + " " +
        merged["make"].astype(str).str.strip() + " " +
        merged["model"].astype(str).str.strip() + " " +
        merged["trim"].astype(str).str.strip() + " " +
        merged["model_code"].astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Keep a tight, stable schema for Mozenda & display
    cols = ["year","make","model","trim","model_code","source","STYLE_ID","search_text"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[cols].reset_index(drop=True)

def fuzzy_filter(df: pd.DataFrame, query: str, score_cutoff: int = 70, limit: int = 50) -> pd.DataFrame:
    """
    Fuzzy filter on 'search_text' using token_set_ratio against the freeform query.
    """
    if df.empty or not query:
        return df.head(limit).copy()
    qn = normalize(query)
    tmp = df.copy()
    tmp["score"] = tmp["search_text"].map(lambda s: fuzz.token_set_ratio(qn, normalize(s)))
    tmp = tmp[tmp["score"] >= score_cutoff].sort_values(["score","year","make","model","trim"], ascending=[False, False, True, True, True])
    return tmp.head(limit)

# ---------------- Load CADS ----------------
try:
    cads_df = load_local_csv(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load {CADS_FILE}: {e}")
    st.stop()

# ---------------- Pull Mappings (read-only) ----------------
mappings_df = fetch_mappings_github(GH_OWNER, GH_REPO, MAP_PATH, GH_BRANCH, GH_TOKEN)

# ---------------- Build joined dataset ----------------
joined_df = join_mappings_to_cads(mappings_df, cads_df)

# ---------------- Query params (for Mozenda and deep links) ----------------
params = st.experimental_get_query_params()
q_param      = params.get("q", [""])[0]                 # freeform q
score_param  = int(params.get("score", [70])[0])        # min score
limit_param  = int(params.get("limit", [50])[0])        # top N
is_mozenda   = params.get("mozenda", ["0"])[0] == "1"
out_format   = params.get("format", ["csv"])[0].lower() # csv|html|json (csv default)

# ---------------- Top Search Bar (for humans) ----------------
st.header("Search Already-Mapped Vehicles")
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    q_input = st.text_input("Vehicle (freeform)", value=q_param, placeholder="e.g., 2025 Land Rover Range Rover P400 SE SWB")
with colB:
    score_input = st.slider("Min score", 0, 100, score_param, 5)
with colC:
    limit_input = st.number_input("Max rows", min_value=1, max_value=200, value=limit_param, step=1)

# Apply fuzzy filter to **already-mapped** vehicles (from Mappings.csv)
filtered_df = fuzzy_filter(joined_df, q_input, score_cutoff=score_input, limit=int(limit_input))

# ---------------- Mozenda Mode: API-like outlet ----------------
if is_mozenda:
    # Use q/score/limit from query params
    out_df = fuzzy_filter(joined_df, q_param, score_cutoff=score_param, limit=limit_param)
    # Output minimal chrome for Mozenda
    if out_format == "json":
        st.write(out_df.drop(columns=["search_text"], errors="ignore").to_json(orient="records"))
    elif out_format == "html":
        st.dataframe(out_df.drop(columns=["search_text"], errors="ignore"), hide_index=True, use_container_width=True)
    else:
        st.write(out_df.drop(columns=["search_text"], errors="ignore").to_csv(index=False))
    st.stop()

# ---------------- Human-friendly Preview ----------------
if mappings_df.empty:
    st.warning("Mappings.csv not found (or unauthorized). This page lists only vehicles that have already been mapped. "
               "If your repo is private, make sure your token has Contents: Read and is SSO-authorized, or make the repo/file public.")
else:
    st.success(f"Loaded {len(mappings_df)} mapped rows from GitHub: {GH_OWNER}/{GH_REPO}@{GH_BRANCH}:{MAP_PATH}")

st.subheader("Results")
if filtered_df.empty:
    st.info("No mapped rows matched your search. Try lowering the score or changing the query.")
else:
    # Show the table without the internal search_text
    st.dataframe(
        filtered_df.drop(columns=["search_text"], errors="ignore"),
        hide_index=True,
        use_container_width=True
    )

# Helpful links / endpoints
st.markdown("---")
st.subheader("Mozenda Endpoints (copy & use in your agent)")
base_url = st.request.url.split("?")[0] if hasattr(st, "request") else ""
st.code(f"{base_url}?mozenda=1&format=csv&q=2025%20Land%20Rover%20Range%20Rover%20P400%20SE%20SWB&score=70&limit=50", language="text")
st.code(f"{base_url}?mozenda=1&format=json&q=2025%20Land%20Rover%20Range%20Rover%20P400%20SE%20SWB&score=70&limit=50", language="text")
st.code(f"{base_url}?mozenda=1&format=html&q=2025%20Land%20Rover%20Range%20Rover%20P400%20SE%20SWB&score=70&limit=50", language="text")

# --- EOF ---
