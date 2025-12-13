
# app.py
import os
import io
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
LOCAL_MAPS_PATH = os.path.join(SCRIPT_DIR, "Mappings.csv")

# CADS (catalog) must have these columns
REQUIRED_CADS_COLS = {"ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"}

# --- Secrets/env with graceful fallback (GitHub optional) ---
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
REPO = st.secrets.get("REPO") or os.environ.get("REPO")            # e.g., "owner/repo"
BRANCH = st.secrets.get("BRANCH") or os.environ.get("BRANCH", "main")
FILE_PATH = st.secrets.get("FILE_PATH") or os.environ.get("FILE_PATH")  # e.g., "Mappings.csv"
GITHUB_ENABLED = bool(GITHUB_TOKEN and REPO and FILE_PATH)

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

def get_query_params() -> dict:
    """Unified get for query params across Streamlit versions."""
    try:
        return dict(st.query_params)  # Streamlit >=1.32
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def qp_get(qp: dict, key: str, default: str = "") -> str:
    """Extract a scalar query param (handle list or str)."""
    val = qp.get(key, default)
    if isinstance(val, list):
        return val[0] if val else default
    return str(val) if val is not None else default

@st.cache_data
def load_cads(source):
    """Load CADS CSV/XLSX and normalize columns."""
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

def _normalize_maps_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize mappings DataFrame to the expected schema:
    year, make, model, trim, model_code, source
    Also supports older schema: src_* and ad_mfgcode/saved_by.
    """
    expected_cols = ["year", "make", "model", "trim", "model_code", "source"]

    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "src_year": "year",
        "src_make": "make",
        "src_model": "model",
        "src_trim": "trim",
        "ad_mfgcode": "model_code",
        "saved_by": "source",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[expected_cols].astype(str)
    return df

# ======================================
# GitHub / Local Persistence
# ======================================
def read_maps() -> pd.DataFrame:
    """Read mappings from GitHub if enabled; otherwise from local CSV; otherwise return empty DataFrame."""
    if GITHUB_ENABLED:
        url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        r = requests.get(url, headers=headers, params={"ref": BRANCH})
        if r.status_code == 200:
            content_b64 = r.json()["content"]
            content = base64.b64decode(content_b64).decode("utf-8")
            df = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False, encoding="utf-8")
            return _normalize_maps_columns(df)
        return pd.DataFrame(columns=["year", "make", "model", "trim", "model_code", "source"])

    if os.path.exists(LOCAL_MAPS_PATH):
        try:
            df = pd.read_csv(LOCAL_MAPS_PATH, dtype=str, keep_default_na=False, encoding="utf-8")
            return _normalize_maps_columns(df)
        except Exception as e:
            st.warning(f"Could not read local mappings: {e}. Starting empty.")

    return pd.DataFrame(columns=["year", "make", "model", "trim", "model_code", "source"])

def write_maps(df: pd.DataFrame) -> bool:
    """
    Write mappings to GitHub if enabled; otherwise save to local CSV.
    Returns True on success, False on failure.
    """
    df = _normalize_maps_columns(df)

    if GITHUB_ENABLED:
        csv_data = df.to_csv(index=False)
        url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }

        r = requests.get(url, headers=headers, params={"ref": BRANCH})
        sha = r.json().get("sha") if r.status_code == 200 else None

        payload = {
            "message": "Update mappings",
            "content": base64.b64encode(csv_data.encode("utf-8")).decode("utf-8"),
            "branch": BRANCH,
            "committer": {
                "name": os.getenv("APP_USER", "Vehicle Mapper"),
                "email": os.getenv("APP_USER_EMAIL", "noreply@example.com"),
            },
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, headers=headers, json=payload)
        if resp.status_code in (200, 201):
            return True
        else:
            st.error(f"GitHub save failed: {resp.status_code} {resp.text}")
            return False

    try:
        df.to_csv(LOCAL_MAPS_PATH, index=False, encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"Local save failed: {e}")
        return False

# ======================================
# Mapping logic
# ======================================
def parse_vehicle_text(vehicle_text: str, cads_df: pd.DataFrame):
    """Parse a vehicle free-text into year/make/model/trim using CADS vocab."""
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
    """Return existing mapping row for the given src YMMT or None."""
    if maps_df.empty:
        return None

    required = {"year", "make", "model", "trim"}
    if not required.issubset(set(maps_df.columns)):
        return None

    try:
        maps_df["_srckey"] = maps_df.apply(
            lambda r: srckey_strict(r.get("year", ""), r.get("make", ""), r.get("model", ""), r.get("trim", "")),
            axis=1,
        )
    except Exception as e:
        st.warning(f"Could not compute existing mapping keys: {e}")
        return None

    key = srckey_strict(year, make, model, trim)
    hit = maps_df[maps_df["_srckey"] == key]
    return None if hit.empty else hit.iloc[0]

def candidates_by_ymmt(cads_df: pd.DataFrame, year, make, model, trim=""):
    """Return CADS candidates filtered by Y/M/M, sorted by trim similarity if trim provided."""
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

def choose_best_candidate(cands: pd.DataFrame, trim: str, strict: bool = False) -> tuple[pd.Series | None, str, int]:
    """
    Choose the best candidate row:
      - If strict and exact trim match exists, use it.
      - Else use the top by trim_score or the first row if no trim_score.
    Returns (row, match_type, rank_index)
    """
    if cands.empty:
        return None, "none", -1

    if strict:
        exact = cands[cands["ad_trim"].apply(norm) == norm(trim)]
        if not exact.empty:
            return exact.iloc[0], "strict", 1

    if "trim_score" in cands.columns and len(cands) > 0:
        return cands.iloc[0], "best", 1

    return cands.iloc[0], "first", 1

def save_mapping(maps_df: pd.DataFrame, src_year, src_make, src_model, src_trim, cad_row: pd.Series):
    """Append/update mapping for the src YMMT to the selected CADS row using target schema."""
    maps_df = _normalize_maps_columns(maps_df)

    if not maps_df.empty:
        try:
            maps_df["_srckey"] = maps_df.apply(
                lambda r: srckey_strict(r.get("year", ""), r.get("make", ""), r.get("model", ""), r.get("trim", "")),
                axis=1,
            )
            current_key = srckey_strict(src_year, src_make, src_model, src_trim)
            maps_df = maps_df[maps_df["_srckey"] != current_key]
        finally:
            maps_df = maps_df.drop(columns=["_srckey"], errors="ignore")

    new_row = {
        "year": str(src_year),
        "make": str(src_make),
        "model": str(src_model),
        "trim": str(src_trim),
        "model_code": str(cad_row.get("ad_mfgcode", "")),
        "source": os.getenv("APP_USER", "user"),
    }

    maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)
    maps_df = maps_df[["year", "make", "model", "trim", "model_code", "source"]]
    return maps_df

# ======================================
# Mozenda mode (query-parameter driven)
# ======================================
def run_mozenda_mode(cads_df: pd.DataFrame, maps_df: pd.DataFrame):
    qp = get_query_params()
    mozenda = qp_get(qp, "mozenda", "0") == "1"
    if not mozenda:
        return False  # continue to interactive UI

    # ⭐ NEW: allow a single 'vehicle' param plus optional 'make' override
    vehicle = qp_get(qp, "vehicle", "")
    if vehicle:
        y, m, mdl, tr = parse_vehicle_text(vehicle, cads_df)
        make_override = qp_get(qp, "make", "")
        src_year = qp_get(qp, "year", y)
        src_make = make_override or m
        src_model = qp_get(qp, "model", mdl)
        src_trim = qp_get(qp, "trim", tr)
    else:
        src_year = qp_get(qp, "year", "")
        src_make = qp_get(qp, "make", "")
        src_model = qp_get(qp, "model", "")
        src_trim = qp_get(qp, "trim", "")

    fmt = qp_get(qp, "format", "json").lower()
    autosave = qp_get(qp, "autosave", "0") == "1"
    strict = qp_get(qp, "strict", "0") == "1"

    if not (src_year and src_make and src_model):
        st.json({"status": "error", "message": "Missing required params: year, make, model", "saved": False})
        return True

    hit = find_existing_mapping(maps_df, src_year, src_make, src_model, src_trim)
    if hit is not None:
        payload = {
            "status": "ok",
            "saved": False,
            "year": hit.get("year", ""),
            "make": hit.get("make", ""),
            "model": hit.get("model", ""),
            "trim": hit.get("trim", ""),
            "model_code": hit.get("model_code", ""),
            "match_type": "existing",
            "candidate_rank": 1,
            "message": "Already mapped",
        }
        return _emit_payload(payload, fmt)

    cands = candidates_by_ymmt(cads_df, src_year, src_make, src_model, src_trim)
    if cands.empty:
        payload = {"status": "error", "message": "No CADS candidates found", "saved": False}
        return _emit_payload(payload, fmt)

    cad_row, match_type, rank = choose_best_candidate(cands, src_trim, strict=strict)
    if cad_row is None:
        payload = {"status": "error", "message": "No valid candidate", "saved": False}
        return _emit_payload(payload, fmt)

    saved = False
    message = "Candidate selected (not saved)"
    if autosave:
        new_maps = save_mapping(maps_df, src_year, src_make, src_model, src_trim, cad_row)
        saved = write_maps(new_maps)
        message = "Mapped and saved" if saved else "Save failed"

    payload = {
        "status": "ok" if (saved or not autosave) else "error",
        "saved": saved,
        "year": str(src_year),
        "make": str(src_make),
        "model": str(src_model),
        "trim": str(src_trim),
        "model_code": str(cad_row.get("ad_mfgcode", "")),
        "match_type": match_type,
        "candidate_rank": rank,
        "message": message,
    }
    return _emit_payload(payload, fmt)

def _emit_payload(payload: dict, fmt: str) -> bool:
    if fmt == "csv":
        df = pd.DataFrame([payload])
        st.write(df.to_csv(index=False))
    else:
        st.json(payload)
    return True

# ======================================
# UI
# ======================================
st.title("Simple Vehicle Mapper (POC with GitHub persistence)")

# Load CADS
cads_choice = st.radio("Load CADS from:", ["Local file (CADS.csv)", "Upload file (CSV/XLSX)"], horizontal=True)
if cads_choice.startswith("Local"):
    cads_df = load_cads(CADS_FILE_DEFAULT)
else:
    up = st.file_uploader("Upload CADS file", type=["csv", "xlsx"])
    if up is None:
        st.stop()
    cads_df = load_cads(up)

st.caption(f"Loaded CADS with {len(cads_df)} rows.")

# Load mappings
maps_df = read_maps()

# 🔁 Mozenda mode: Short-circuit if query params request headless output
if run_mozenda_mode(cads_df, maps_df):
    st.stop()

# Interactive Search
st.subheader("Search")
mode = st.radio("Search by:", ["Vehicle string", "Y/M/M/T"], horizontal=True)

if mode == "Vehicle string":
    # ⭐ NEW: Make override next to vehicle text
    col_v, col_m = st.columns([3, 2])
    with col_v:
        vehicle_text = st.text_input("Vehicle (e.g., '2025 Range Rover Sport P360 SE')")  # Make may be omitted
    with col_m:
        make_override = st.text_input("Make (optional override)", placeholder="e.g., Land Rover")

    parsed_year, parsed_make, src_model, src_trim = parse_vehicle_text(vehicle_text, cads_df)
    src_year = parsed_year
    src_make = make_override or parsed_make

    # Small diagnostic
    st.caption(
        f"Parsed → Year: {src_year or '—'}, Make: {src_make or '—'}, "
        f"Model: {src_model or '—'}, Trim: {src_trim or '—'}"
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        src_year = st.text_input("Year")
    with c2:
        src_make = st.text_input("Make")
    with c3:
        src_model = st.text_input("Model")
    with c4:
        src_trim = st.text_input("Trim (optional)")

# Stop if nothing entered
if not (src_year or src_make or src_model or src_trim):
    st.stop()

# Existing mapping?
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
selected_pos = st.radio("Choose a candidate", options=list(range(len(labels))), format_func=lambda i: labels[i], index=0)

if st.button("💾 Save Mapping", type="primary"):
    cad_row = cands.iloc[selected_pos]
    new_maps = save_mapping(maps_df, src_year, src_make, src_model, src_trim, cad_row)

    saved = write_maps(new_maps)
    if saved:
        st.success("✅ Mapping saved")
