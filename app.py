
import os
import unicodedata
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# ======================================
# Config
# ======================================
st.set_page_config(page_title="Simple Vehicle Mapper", layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CADS_FILE_DEFAULT = "CADS.csv"         # local CADS default name (in working dir)
DEFAULT_MAPS_FILE = os.path.join(SCRIPT_DIR, "Mappings.csv")  # pinned next to script by default

REQUIRED_CADS_COLS = {"ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"}

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
    """Strict source key: normalized Year/Make/Model/Trim (no synonyms)."""
    return "\n".join([norm(year), norm(make), norm(model), norm(trim)])

@st.cache_data
def load_cads(source):
    """
    Load CADS and return FULL dataframe (column names lower-cased).
    Asserts required columns exist.
    """
    # Infer file type from path or uploaded file
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

def ensure_maps_path(path_text=None) -> str:
    """
    Determine the maps file path:
    - If user provided a path, use it (absolute or relative to working dir).
    - Otherwise, use DEFAULT_MAPS_FILE (pinned next to the script).
    Create parent dirs if needed.
    """
    if path_text:
        maps_path = os.path.abspath(path_text.strip())
    else:
        maps_path = DEFAULT_MAPS_FILE

    parent = os.path.dirname(maps_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return maps_path

def read_maps(path: str) -> pd.DataFrame:
    """
    Read mappings; if file is absent or missing columns, create/patch in-memory.
    """
    cols = [
        "src_year", "src_make", "src_model", "src_trim",   # source YMMT
        "cad_year", "cad_make", "cad_model", "cad_trim",   # snapshot of mapped CADS YMMT
        "ad_mfgcode",                                      # primary CADS link (optional)
        "saved_by", "created_utc"                          # audit
    ]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def write_maps(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8")
    if not os.path.exists(path):
        raise IOError(f"File not found after write: {os.path.abspath(path)}")
    if os.path.getsize(path) == 0:
        raise IOError(f"File size is 0 after write: {os.path.abspath(path)}")

def parse_vehicle_text(vehicle_text: str, cads_df: pd.DataFrame):
    """
    Simple parser: YEAR + MAKE + MODEL from CADS vocab; remainder is TRIM.
    Supports single-string inputs like "2025 Land Rover Range Rover Sport P360 SE".
    """
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
    """Check if a mapping already exists for the strict source key."""
    if maps_df.empty:
        return None
    maps_df["_srckey"] = maps_df.apply(
        lambda r: srckey_strict(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
    )
    key = srckey_strict(year, make, model, trim)
    hit = maps_df[maps_df["_srckey"] == key]
    return None if hit.empty else hit.iloc[0]

def link_cads_row(cads_df: pd.DataFrame, mapped_row: pd.Series) -> pd.DataFrame:
    """
    Re-link the mapped CADS row in the current CADS upload:
      1) Try exact match by ad_mfgcode
      2) If code changed, fall back to stored CADS YMMT snapshot
    """
    code = str(mapped_row.get("ad_mfgcode", "")).strip()
    if code:
        by_code = cads_df[cads_df["ad_mfgcode"].astype(str) == code]
        if not by_code.empty:
            return by_code

    cond = (
        (cads_df["ad_year"].apply(norm) == norm(mapped_row["cad_year"])) &
        (cads_df["ad_make"].apply(norm) == norm(mapped_row["cad_make"])) &
        (cads_df["ad_model"].apply(norm) == norm(mapped_row["cad_model"])) &
        (cads_df["ad_trim"].apply(norm) == norm(mapped_row["cad_trim"]))
    )
    return cads_df[cond]

def candidates_by_ymmt(cads_df: pd.DataFrame, year, make, model, trim=""):
    """
    Candidate list: exact Year/Make/Model (required); optional trim similarity score.
    Returns a dataframe sorted by similarity when trim is provided.
    """
    if not (year and make and model):
        return pd.DataFrame(columns=cads_df.columns)

    base = cads_df[
        (cads_df["ad_year"].apply(norm) == norm(year)) &
        (cads_df["ad_make"].apply(norm) == norm(make)) &
        (cads_df["ad_model"].apply(norm) == norm(model))
    ].copy()

    if trim:
        base["trim_score"] = base["ad_trim"].apply(
            lambda s: fuzz.token_set_ratio(norm(trim), norm(s))
        )
        base = base.sort_values("trim_score", ascending=False)
    return base

def save_mapping(maps_df: pd.DataFrame, src_year, src_make, src_model, src_trim,
                 cad_row: pd.Series, saved_by="user") -> pd.DataFrame:
    """
    Replace any existing mapping for the same strict source key; then append the new one.
    """
    if not maps_df.empty:
        maps_df["_srckey"] = maps_df.apply(
            lambda r: srckey_strict(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
        )
        current_key = srckey_strict(src_year, src_make, src_model, src_trim)
        maps_df = maps_df[maps_df["_srckey"] != current_key].drop(columns=["_srckey"], errors="ignore")

    new_row = {
        "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
        "cad_year": cad_row.get("ad_year", ""), "cad_make": cad_row.get("ad_make", ""),
        "cad_model": cad_row.get("ad_model", ""), "cad_trim": cad_row.get("ad_trim", ""),
        "ad_mfgcode": cad_row.get("ad_mfgcode", ""),
        "saved_by": saved_by,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)
    return maps_df

def verify_srckey(path: str, src_year, src_make, src_model, src_trim) -> dict:
    """
    Re-load Mappings.csv and confirm the new row exists by strict source key ONLY.
    Returns a dict with details for diagnostics.
    """
    details = {
        "path": os.path.abspath(path),
        "exists": os.path.exists(path),
        "size": os.path.getsize(path) if os.path.exists(path) else 0,
        "rows": 0,
        "matched_rows_by_srckey": 0,
        "srckey": srckey_strict(src_year, src_make, src_model, src_trim),
    }
    if not details["exists"]:
        return details

    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]
    details["rows"] = len(df)

    df["_srckey"] = df.apply(
        lambda r: srckey_strict(r.get("src_year", ""), r.get("src_make", ""), r.get("src_model", ""), r.get("src_trim", "")),
        axis=1
    )
    details["matched_rows_by_srckey"] = int((df["_srckey"] == details["srckey"]).sum())
    return details

# ======================================
# UI
# ======================================
st.title("Simple Vehicle Mapper")

# --- Mappings file location (pinned by default; override if you want another folder) ---
st.subheader("Mappings file")
use_custom_path = st.checkbox("Use a custom Mappings.csv path (override default next to script)", value=False)
maps_path_text = ""
if use_custom_path:
    maps_path_text = st.text_input("Mappings.csv path (absolute or relative)", value=DEFAULT_MAPS_FILE)
MAPS_FILE = ensure_maps_path(maps_path_text if use_custom_path else DEFAULT_MAPS_FILE)

debug_mode = st.checkbox("Debug mode (show keys and diagnostics)", value=False)

# CADS source
st.subheader("CADS source")
cads_choice = st.radio("Load CADS from:", ["Local file (CADS.csv)", "Upload file (CSV/XLSX)"], horizontal=True)

if cads_choice.startswith("Local"):
    try:
        cads_df = load_cads(CADS_FILE_DEFAULT)
    except Exception as e:
        st.error(f"Failed to load local CADS file '{CADS_FILE_DEFAULT}': {e}")
        st.stop()
else:
    up = st.file_uploader("Upload CADS file", type=["csv", "xlsx"])
    if up is None:
        st.info("Upload a CADS file to continue.")
        st.stop()
    try:
        cads_df = load_cads(up)
    except Exception as e:
        st.error(f"Failed to load uploaded CADS: {e}")
        st.stop()

st.caption(f"Loaded CADS with {len(cads_df)} rows and {len(cads_df.columns)} columns.")

# Load mappings (persisted)
maps_df = read_maps(MAPS_FILE)

# Search input
st.subheader("Search")
mode = st.radio("Search by:", ["Vehicle string", "Y/M/M/T"], horizontal=True)

if mode == "Vehicle string":
    vehicle_text = st.text_input("Vehicle (e.g., '2025 Land Rover Range Rover Sport P360 SE')")
    src_year, src_make, src_model, src_trim = parse_vehicle_text(vehicle_text, cads_df)
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

# Require enough input to proceed
if mode == "Y/M/M/T" and not (src_year and src_make and src_model):
    st.info("Enter Year, Make, and Model (Trim optional).")
    st.stop()
elif mode == "Vehicle string" and not (src_year or src_make or src_model or src_trim):
    st.stop()

if debug_mode:
    st.code(f"SRCKEY: {srckey_strict(src_year, src_make, src_model, src_trim)}")

# Check mapping memory first
hit = find_existing_mapping(maps_df, src_year, src_make, src_model, src_trim)

if hit is not None:
    st.success("✅ Vehicle mapped already")
    st.caption(f"Source: {hit['src_year']} {hit['src_make']} {hit['src_model']} {hit['src_trim']}")
    # Show CADS attributes of the mapped row in the current CADS upload
    linked = link_cads_row(cads_df, hit)
    if linked.empty:
        st.warning("Mapping exists, but the CADS row was not found in the current upload (code or trim may have changed).")
        st.write(pd.DataFrame([{
            "cad_year": hit["cad_year"], "cad_make": hit["cad_make"], "cad_model": hit["cad_model"],
            "cad_trim": hit["cad_trim"], "ad_mfgcode (saved)": hit["ad_mfgcode"]
        }]))
    else:
        st.write("Mapped CADS row:")
        st.dataframe(linked, use_container_width=True)
    st.stop()

# Not mapped yet → candidates
st.warning("⚠️ Needs mapping")
cands = candidates_by_ymmt(cads_df, src_year, src_make, src_model, src_trim)

if cands.empty:
    st.error("No CADS candidates found for this Year/Make/Model (check spelling or try different inputs).")
    st.stop()

view_cols = ["ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"]
st.write(f"Found {len(cands)} candidate(s). Pick the correct vehicle below.")

# Show candidate table
try:
    st.dataframe(cands[view_cols], use_container_width=True, height=260)
except Exception:
    safe_cols = [c for c in view_cols if c in cands.columns]
    st.dataframe(cands[safe_cols], use_container_width=True, height=260)

# --- Robust selection block (position-based) ---
cands_safe = cands.copy()
for col in view_cols:
    if col in cands_safe.columns:
        cands_safe[col] = cands_safe[col].astype(str).fillna("")

cands_view = cands_safe[view_cols].reset_index(drop=True)

labels = [
    f"{r['ad_year']} {r['ad_make']} {r['ad_model']} {r['ad_trim']}  | code={r['ad_mfgcode']}"
    for _, r in cands_view.iterrows()
]

selected_pos = st.radio(
    "Choose a candidate to map",
    options=list(range(len(labels))),
    format_func=lambda i: labels[i],
    index=0 if labels else None
)

if labels and selected_pos is not None:
    cad_row = cands_view.iloc[int(selected_pos)]
    if st.button("💾 Save Mapping", type="primary"):
        try:
            # Replace any prior mapping for the same strict source key, then append the new row
            new_maps = save_mapping(
                maps_df,
                src_year, src_make, src_model, src_trim,
                cad_row,
                saved_by=os.getenv("APP_USER", "user")
            )
            # Persist to disk
            write_maps(new_maps, MAPS_FILE)
        except Exception as e:
            st.exception(e)  # show full traceback to help diagnose
            st.stop()

        # --- Verify saved row exists by srckey only ---
        ver = verify_srckey(MAPS_FILE, src_year, src_make, src_model, src_trim)
        if ver.get("matched_rows_by_srckey", 0) < 1:
            st.error("Save attempted, but verification failed. Check file path/permissions below.")
            st.write(ver)  # diagnostics: path, size, rows, srckey, matched_rows_by_srckey
            st.stop()

        # Refresh caches so the next query shows "mapped already"
        st.cache_data.clear()
        st.success("✅ Mapping saved")
        st.toast("Vehicle mapped.", icon="✅")
        st.rerun()
else:
    st.info("Select one candidate above to enable saving.")

# Footer: paths (helps verify you're writing to the right files)
st.caption(f"Script directory: {SCRIPT_DIR}")
st.caption(f"Working directory (cwd): {os.getcwd()}")
st.caption(f"CADS path: {os.path.abspath(CADS_FILE_DEFAULT) if cads_choice.startswith('Local') else '(uploaded)'}")
st
