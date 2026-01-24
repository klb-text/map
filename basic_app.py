
# basic_app.py — Read-only Mozenda outlet with Search + Clear and tiered matching
# - Pulls Mappings.csv from GitHub (Contents API; falls back to raw if needed)
# - Joins with local CADS.csv to add STYLE_ID (and optional attributes if present)
# - Top search bar with "Search" and "Clear" buttons
# - Tiered fuzzy: FULL (YMMT + model_code + attrs) then LENIENT (ignores drivetrain/trans tokens)
# - Mozenda mode: ?mozenda=1&format=csv|html|json[&q=...][&score=...][&limit=...][&lenient=0|1]

import base64, io
from typing import Optional, List
import pandas as pd
import streamlit as st
import requests
from rapidfuzz import fuzz

# ---------------- Page Config ----------------
st.set_page_config(page_title="AFF Vehicle Mapping - Read Only", layout="wide")
st.title("AFF Vehicle Mapping (Read Only)")

# ---------------- Secrets / GitHub ----------------
gh = st.secrets.get("github", {})
GH_TOKEN  = gh.get("token")            # optional if repo is public
GH_OWNER  = gh.get("owner")            # e.g., "klb-text"
GH_REPO   = gh.get("repo")             # e.g., "map"
GH_BRANCH = gh.get("branch", "main")
MAP_PATH  = gh.get("path", "Mappings.csv")  # default Mappings.csv at repo root

# ---------------- Local Files ----------------
CADS_FILE = "CADS.csv"  # local CADS.csv
# Expected Mappings.csv columns: year,make,model,trim,model_code,source
# Expected CADS columns: MODEL_YEAR, AD_MAKE, AD_MODEL, TRIM, AD_MFGCODE, STYLE_ID
# Optional CADS columns (if available): DRIVETRAIN/DRIVE_TYPE, TRANSMISSION/TRANS_DESC

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

# domain tokens we may ignore for lenient match
DRIVETRAIN_TOKENS = {
    "fwd","front wheel drive","front-wheel drive","front wheel-drive",
    "awd","all wheel drive","all-wheel drive","all wheel-drive",
    "4wd","4x4","four wheel drive","four-wheel drive",
    "rwd","rear wheel drive","rear-wheel drive","rear wheel-drive",
    "2wd","two wheel drive","two-wheel drive"
}
TRANSMISSION_TOKENS = {
    "cvt","continuously variable transmission",
    "automatic","auto","at","a/t",
    "manual","mt","m/t","stick",
    "dct","dual clutch","dual-clutch"
}

def strip_attr_tokens(text: str) -> str:
    """Remove drivetrain/transmission tokens for lenient matching."""
    t = " " + normalize(text) + " "
    # remove multi-word first to avoid partial overlaps
    multi = sorted([x for x in DRIVETRAIN_TOKENS | TRANSMISSION_TOKENS if " " in x], key=len, reverse=True)
    single = sorted([x for x in DRIVETRAIN_TOKENS | TRANSMISSION_TOKENS if " " not in x], key=len, reverse=True)
    for w in multi:
        t = t.replace(f" {w} ", " ")
    for w in single:
        t = t.replace(f" {w} ", " ")
    return " ".join(t.split())

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
        # else: fall back
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
    Add search_text_full (includes attrs) and search_text_lenient (attrs removed) per row.
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

    # Optional CADS attributes that may help match variants
    drive_cols = [c for c in cd.columns if c.upper() in {"DRIVETRAIN","DRIVE_TRAIN","DRIVE TYPE","DRIVE_TYPE"}]
    trans_cols = [c for c in cd.columns if c.upper() in {"TRANS","TRANSMISSION","TRANS_DESC","TRANS_DESCRIPTION","TRANSMISSION_DESCRIPTION"}]
    attr_cols = drive_cols + trans_cols

    merged = md.merge(
        cd,
        left_on=["year","make","model","trim","model_code"],
        right_on=["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"],
        how="left",
        suffixes=("","_cad")
    )

    # Build search strings
    base_text = (
        merged["year"].astype(str).str.strip() + " " +
        merged["make"].astype(str).str.strip() + " " +
        merged["model"].astype(str).str.strip() + " " +
        merged["trim"].astype(str).str.strip() + " " +
        merged["model_code"].astype(str).str.strip()
    )

    if attr_cols:
        attrs_text = merged[attr_cols].apply(lambda r: " ".join([str(x) for x in r if str(x).strip() != ""]), axis=1)
    else:
        attrs_text = ""

    merged["search_text_full"] = (
        (base_text + " " + attrs_text).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    merged["search_text_lenient"] = merged["search_text_full"].map(strip_attr_tokens)

    # Keep stable schema for Mozenda & display
    cols = ["year","make","model","trim","model_code","source","STYLE_ID","search_text_full","search_text_lenient"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[cols].reset_index(drop=True)

def fuzzy_filter_tiered(df: pd.DataFrame, query: str, score_cutoff: int = 70, limit: int = 50, force_lenient: bool = False) -> pd.DataFrame:
    """
    Tiered matching:
      - FULL score: token_set_ratio(query, search_text_full)
      - LENIENT score: token_set_ratio(query, search_text_lenient)  (ignores drivetrain/trans tokens)
    Final score = max(FULL, LENIENT).
    If force_lenient=True, use only LENIENT for matching (useful when you *know* input has extra attrs).
    """
    if df.empty:
        return df
    if not query:
        out = df.copy()
        out["score_full"] = 0
        out["score_lenient"] = 0
        out["score"] = 0
        out["matched_mode"] = ""
        return out.head(limit)

    qn = normalize(query)
    tmp = df.copy()

    if force_lenient:
        tmp["score_full"] = 0
        tmp["score_lenient"] = tmp["search_text_lenient"].map(lambda s: fuzz.token_set_ratio(qn, normalize(s)))
        tmp["score"] = tmp["score_lenient"]
        tmp["matched_mode"] = "lenient"
    else:
        tmp["score_full"] = tmp["search_text_full"].map(lambda s: fuzz.token_set_ratio(qn, normalize(s)))
        tmp["score_lenient"] = tmp["search_text_lenient"].map(lambda s: fuzz.token_set_ratio(qn, normalize(s)))
        tmp["score"] = tmp[["score_full","score_lenient"]].max(axis=1)
        tmp["matched_mode"] = tmp.apply(lambda r: "full" if r["score_full"] >= r["score_lenient"] else "lenient", axis=1)

    tmp = tmp[tmp["score"] >= score_cutoff].sort_values(
        ["score","matched_mode","year","make","model","trim"], ascending=[False, True, False, True, True, True]
    )
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
params      = st.experimental_get_query_params()
q_param     = params.get("q", [""])[0]                  # freeform q
score_param = int(params.get("score", [70])[0])         # min score
limit_param = int(params.get("limit", [50])[0])         # top N
lenient_qp  = params.get("lenient", ["0"])[0] == "1"    # force lenient matching
is_mozenda  = params.get("mozenda", ["0"])[0] == "1"
out_format  = params.get("format", ["csv"])[0].lower()  # csv|html|json (csv default)

# ---------------- Session state for interactive search ----------------
if "search_query" not in st.session_state:
    st.session_state["search_query"] = q_param
if "search_score" not in st.session_state:
    st.session_state["search_score"] = score_param
if "search_limit" not in st.session_state:
    st.session_state["search_limit"] = limit_param
if "force_lenient" not in st.session_state:
    st.session_state["force_lenient"] = lenient_qp
if "search_results" not in st.session_state:
    st.session_state["search_results"] = pd.DataFrame()

# ---------------- Top Search + Buttons ----------------
st.header("Search Already-Mapped Vehicles")

with st.form("search_form"):
    colA, colB, colC, colD = st.columns([2, 1, 1, 1])
    with colA:
        q_input = st.text_input(
            "Vehicle (freeform)",
            value=st.session_state["search_query"],
            placeholder="e.g., 2026 Integra FWD Continuously Variable Transmission"
        )
    with colB:
        score_input = st.slider("Min score", 0, 100, st.session_state["search_score"], 5)
    with colC:
        limit_input = st.number_input("Max rows", min_value=1, max_value=200, value=st.session_state["search_limit"], step=1)
    with colD:
        force_lenient = st.checkbox("Force lenient (ignore drive/trans)", value=st.session_state["force_lenient"],
                                    help="When on, matching ignores drivetrain & transmission tokens (FWD/AWD/RWD, CVT/AT/MT/DCT).")

    do_search = st.form_submit_button("Search")

# Clear button outside the form so it fires immediately
clear_clicked = st.button("Clear")

if clear_clicked:
    st.session_state["search_query"] = ""
    st.session_state["search_score"] = 70
    st.session_state["search_limit"] = 50
    st.session_state["force_lenient"] = False
    st.session_state["search_results"] = pd.DataFrame()
    # Clear query params
    st.experimental_set_query_params()
elif do_search:
    # Persist inputs
    st.session_state["search_query"] = q_input
    st.session_state["search_score"] = int(score_input)
    st.session_state["search_limit"] = int(limit_input)
    st.session_state["force_lenient"] = bool(force_lenient)

    # Compute results with tiered matching
    st.session_state["search_results"] = fuzzy_filter_tiered(
        joined_df,
        query=q_input,
        score_cutoff=int(score_input),
        limit=int(limit_input),
        force_lenient=bool(force_lenient)
    )

    # Set query params for shareable link / Mozenda parity
    st.experimental_set_query_params(
        q=q_input,
        score=int(score_input),
        limit=int(limit_input),
        lenient=int(bool(force_lenient))
    )

# ---------------- Mozenda Mode: API-like outlet ----------------
if is_mozenda:
    out_df = fuzzy_filter_tiered(
        joined_df,
        query=q_param,
        score_cutoff=score_param,
        limit=limit_param,
        force_lenient=lenient_qp
    )
    payload_df = out_df.drop(columns=["search_text_full","search_text_lenient","score_full","score_lenient","score","matched_mode"], errors="ignore")
    if out_format == "json":
        st.write(payload_df.to_json(orient="records"))
    elif out_format == "html":
        st.dataframe(payload_df, hide_index=True, use_container_width=True)
    else:
        st.write(payload_df.to_csv(index=False))
    st.stop()

# ---------------- Human-friendly Preview ----------------
if mappings_df.empty:
    st.warning(
        "Mappings.csv not found (or unauthorized). This page lists only vehicles that have already been mapped. "
        "If your repo is private, make sure your token has Contents: Read and is SSO-authorized, or make the repo/file public."
    )
else:
    st.success(f"Loaded {len(mappings_df)} mapped rows from GitHub: {GH_OWNER}/{GH_REPO}@{GH_BRANCH}:{MAP_PATH}")

st.subheader("Results")
results_df = st.session_state["search_results"]
if results_df.empty and st.session_state["search_query"]:
    st.info("No mapped rows matched your search. Try lowering the score, increasing the limit, or turning on 'Force lenient'.")
elif not results_df.empty:
    # Show the table without internal search columns
    show_cols = [c for c in results_df.columns if c not in {"search_text_full","search_text_lenient","score_full","score_lenient"}]
    st.dataframe(results_df[show_cols], hide_index=True, use_container_width=True)

# Helpful endpoints
st.markdown("---")
st.subheader("Mozenda Endpoints (copy & use in your agent)")
base_url = st.request.url.split("?")[0] if hasattr(st, "request") else ""
st.code(f"{base_url}?mozenda=1&format=csv&q=2026%20Integra%20FWD%20Continuously%20Variable%20Transmission&score=70&limit=50&lenient=1", language="text")
st.code(f"{base_url}?mozenda=1&format=json&q=2026%20Integra%20FWD%20Continuously%20Variable%20Transmission&score=70&limit=50&lenient=1", language="text")
st.code(f"{base_url}?mozenda=1&format=html&q=2026%20Integra%20FWD%20Continuously%20Variable%20Transmission&score=70&limit=50&lenient=1", language="text")

# --- EOF ---
