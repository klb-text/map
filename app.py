
import os
import unicodedata
import base64
import requests
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# ======================================
# Config
# ======================================
st.set_page_config(page_title="Simple Vehicle Mapper (POC)", layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CADS_FILE_DEFAULT = "CADS.csv"
REQUIRED_CADS_COLS = {"ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"}

# GitHub settings from Streamlit Secrets
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO = st.secrets["REPO"]          # e.g., "klb-text/map"
BRANCH = st.secrets["BRANCH"]      # e.g., "main"
FILE_PATH = st.secrets["FILE_PATH"]  # e.g., "Mappings.csv"

# ======================================
# Helpers
# ======================================
def ascii_fold(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")

def norm(s: str) -> str:
    s = ascii_fold(s).lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split())

def srckey_strict(year, make, model, trim) -> str:
    return "\n".join([norm(year), norm(make), norm(model), norm(trim)])

@st.cache_data
def load_cads(source):
    if hasattr(source, "name"):
        ext = os.path.splitext(source.name)[1].lower()
    elif isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
    else:
        ext = ".csv"

    if ext == ".xlsx":
        df = pd.read_excel(source, dtype=str, engine="openpyxl")
    else:
        df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8")

    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_CADS_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CADS file missing columns: {sorted(missing)}")
    return df

# ======================================
# GitHub Persistence
# ======================================
def read_maps() -> pd.DataFrame:
    url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        content = base64.b64decode(r.json()["content"]).decode("utf-8")
        return pd.read_csv(pd.compat.StringIO(content), dtype=str)
    return pd.DataFrame(columns=["src_year","src_make","src_model","src_trim","cad_year","cad_make","cad_model","cad_trim","ad_mfgcode","saved_by","created_utc"])

def write_maps(df: pd.DataFrame):
    csv_data = df.to_csv(index=False)
    url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    # Get current file SHA
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha")
    payload = {
        "message": "Update mappings",
        "content": base64.b64encode(csv_data.encode("utf-8")).decode("utf-8"),
        "branch": BRANCH,
        "sha": sha
    }
    resp = requests.put(url, headers=headers, json=payload)
    if resp.status_code not in [200, 201]:
        st.error(f"GitHub save failed: {resp.status_code} {resp.text}")

# ======================================
# Mapping logic
# ======================================
def parse_vehicle_text(vehicle_text: str, cads_df: pd.DataFrame):
    vt = (vehicle_text or "").strip()
    if not vt:
        return "", "", "", ""
    toks = vt.split()
    year = toks[0] if toks and toks[0].isdigit() and len(toks[0]) == 4 else ""
    if year:
        toks = toks[1:]
    seq = " ".join(toks).lower()

    makes = sorted(pd.Series(cads_df["ad_make"]).dropna().unique().tolist(), key=len, reverse=True)
    models = sorted(pd.Series(cads_df["ad_model"]).dropna().unique().tolist(), key=len, reverse=True)
    makes_l = [" ".join(str(m).lower().split()) for m in makes]
    models_l = [" ".join(str(m).lower().split()) for m in models]

    make_l = ""
    for m in makes_l:
        if seq.startswith(m + " ") or seq == m:
            make_l = m
            break
    if make_l:
        rest = seq[len(make_l):].strip()
    else:
        parts = seq.split()
        make_l = parts[0] if parts else ""
        rest = " ".join(parts[1:]) if len(parts) > 1 else ""

    model_l = ""
    for mdl in models_l:
        if rest.startswith(mdl + " ") or rest == mdl or seq == mdl:
            model_l = mdl
            break
    if model_l:
        trim_l = rest[len(model_l):].strip()
    else:
        rem = rest.split()
        model_l = rem[0] if rem else ""
        trim_l = " ".join(rem[1:]) if len(rem) > 1 else ""

    make_human = next((m for m in makes if norm(m) == make_l), make_l)
    model_human = next((m for m in models if norm(m) == model_l), model_l)
    return year, make_human, model_human, trim_l

def find_existing_mapping(maps_df: pd.DataFrame, year, make, model, trim) -> pd.Series | None:
    if maps_df.empty:
        return None
    maps_df["_srckey"] = maps_df.apply(
        lambda r: srckey_strict(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
    )
    key = srckey_strict(year, make, model, trim)
    hit = maps_df[maps_df["_srckey"] == key]
    return None if hit.empty else hit.iloc[0]

def candidates_by_ymmt(cads_df: pd.DataFrame, year, make, model, trim=""):
    if not (year and make and model):
        return pd.DataFrame(columns=cads_df.columns)
    base = cads_df[
        (cads_df["ad_year"].apply(norm) == norm(year)) &
        (cads_df["ad_make"].apply(norm) == norm(make)) &
        (cads_df["ad_model"].apply(norm) == norm(model))
    ].copy()
    if trim:
        base["trim_score"] = base["ad_trim"].apply(lambda s: fuzz.token_set_ratio(norm(trim), norm(s)))
        base = base.sort_values("trim_score", ascending=False)
    return base

def save_mapping(maps_df: pd.DataFrame, src_year, src_make, src_model, src_trim, cad_row: pd.Series):
    if not maps_df.empty:
        maps_df["_srckey"] = maps_df.apply(lambda r: srckey_strict(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1)
        current_key = srckey_strict(src_year, src_make, src_model, src_trim)
        maps_df = maps_df[maps_df["_srckey"] != current_key].drop(columns=["_srckey"], errors="ignore")
    new_row = {
        "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
        "cad_year": cad_row.get("ad_year", ""), "cad_make": cad_row.get("ad_make", ""),
        "cad_model": cad_row.get("ad_model", ""), "cad_trim": cad_row.get("ad_trim", ""),
        "ad_mfgcode": cad_row.get("ad_mfgcode", ""),
        "saved_by": os.getenv("APP_USER", "user"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)

# ======================================
# UI
# ======================================
st.title("Simple Vehicle Mapper (POC with GitHub persistence)")

# CADS source
st.subheader("CADS source")
cads_choice = st.radio("Load CADS from:", ["Local file (CADS.csv)", "Upload file (CSV/XLSX)"], horizontal=True)
if cads_choice.startswith("Local"):
    cads_df = load_cads(CADS_FILE_DEFAULT)
else:
    up = st.file_uploader("Upload CADS file", type=["csv", "xlsx"])
    if up is None:
        st.stop()
    cads_df = load_cads(up)

st.caption(f"Loaded CADS with {len(cads_df)} rows.")

# Load mappings from GitHub
maps_df = read_maps()

# Search input
st.subheader("Search")
mode = st.radio("Search by:", ["Vehicle string", "Y/M/M/T"], horizontal=True)
if mode == "Vehicle string":
    vehicle_text = st.text_input("Vehicle (e.g., '2025 Land Rover Range Rover Sport P360 SE')")
    src_year, src_make, src_model, src_trim = parse_vehicle_text(vehicle_text, cads_df)
else:
    c1, c2, c3, c4 = st.columns(4)
    with c1: src_year = st.text_input("Year")
    with c2: src_make = st.text_input("Make")
    with c3: src_model = st.text_input("Model")
    with c4: src_trim = st.text_input("Trim (optional)")

if not (src_year or src_make or src_model or src_trim):
    st.stop()

hit = find_existing_mapping(maps_df, src_year, src_make, src_model, src_trim)
if hit is not None:
    st.success("✅ Vehicle mapped already")
    st.write(hit)
    st.stop()

st.warning("⚠️ Needs mapping")

cands = candidates_by_ymmt(cads_df, src_year, src_make, src_model, src_trim)
if cands.empty:
    st.error("No CADS candidates found.")
    st.stop()

view_cols = ["ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"]
st.dataframe(cands[view_cols], use_container_width=True, height=260)

labels = [
    f"{r['ad_year']} {r['ad_make']} {r['ad_model']} {r['ad_trim']} | code={r['ad_mfgcode']}"
    for _, r in cands[view_cols].reset_index(drop=True).iterrows()
]

# Guard against empty labels just in case
selected_pos = st.radio(
    "Choose a candidate",
    options=list(range(len(labels))),
    format_func=lambda i: labels[i] if i < len(labels) else "—",
    index=0
)

if st.button("💾 Save Mapping", type="primary"):
    # Validate selection index (defensive)
    if len(cands) == 0:
        st.error("No CADS candidates available to save.")
        st.stop()

    if selected_pos < 0 or selected_pos >= len(cands):
        st.error(f"Invalid selection index: {selected_pos}")
        st.stop()

    cad_row = cands.iloc[selected_pos]

    # ✅ Single, complete call (the duplicated call was the syntax error)
    new_maps = save_mapping(
        maps_df,
        src_year,
        src_make,
        src_model,
        src_trim,
        cad_row,
    )

    write_maps(new_maps)
    st.success("✅ Mapping saved to GitHub")
    st.toast("Vehicle mapped.", icon="✅")
