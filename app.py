# External → CADS Vehicle Mapper (POC) — v2.2
# Build: 2025-11-30 4:41 PM ET

import os
import re
import unicodedata
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# ---------------------------
# Config & constants
# ---------------------------
st.set_page_config(page_title="External → CADS Vehicle Mapper (POC)", layout="wide")
st.info("Build: 2025-11-30 4:41 PM ET — External→CADS mapping v2.2 (uploads, better fuzzy, manage & delete)")

APP_PASSWORD = os.getenv("APP_PASSWORD", "mypassword")
APP_USER = os.getenv("APP_USER", "anonymous")
API_TOKEN = os.getenv("API_TOKEN", "mozenda-token")

CADS_FILE_DEFAULT = "CADS.csv"
SRC_MAP_FILE = "SourceMappings.csv"
SRC_MAP_LOG = "SourceMappings.log.csv"  # optional audit log

REQUIRED_CADS_COLS = {"ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"}

# Synonyms / token replacements commonly seen in external trims
SYNONYMS = {
    "swb": "", "short wheelbase": "",
    "lwb": "7 seat", "long wheelbase": "7 seat", "7-seater": "7 seat",
    "awdx": "awd", "awd": "all wheel drive", "4wd": "four wheel drive",
    "mhev": "mild hybrid", "phev": "plug-in hybrid", "bev": "electric",
    "tm": "", "®": "", "℠": "", "(tm)": "", "(r)": "", "(sm)": "",
}

# ---------------------------
# Normalization helpers
# ---------------------------
def ascii_fold(s: str) -> str:
    # Remove accents/symbols reliably (e.g., ®, ™ issues)
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s

def norm(s: str) -> str:
    s = ascii_fold(s or "")
    s = s.lower().strip()
    # unify separators
    s = s.replace("_", " ").replace("-", " ")
    # collapse whitespace
    s = " ".join(s.split())
    return s

def normalize_external_trim(t: str) -> str:
    s = norm(t)
    for k, v in SYNONYMS.items():
        k_norm = norm(k)
        if k_norm and k_norm in s:
            repl = norm(v)
            s = s.replace(k_norm, repl).strip()
    return " ".join(s.split())

def normalize_source_key_ymmt(year, make, model, trim) -> str:
    return "\n".join([norm(year), norm(make), norm(model), norm(normalize_external_trim(trim))])

def normalize_source_key_ymm(year, make, model) -> str:
    return "\n".join([norm(year), norm(make), norm(model)])

# ---------------------------
# Data IO
# ---------------------------
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

@st.cache_data
def load_table(path_or_buf, file_hint: str = "") -> pd.DataFrame:
    """
    Supports CSV and XLSX (openpyxl). If 'file_hint' is provided, uses it to choose parser.
    """
    try:
        if isinstance(path_or_buf, str):
            ext = (os.path.splitext(path_or_buf)[1] or "").lower()
        else:
            # Uploaded file-like: rely on hint or try CSV first
            ext = (file_hint or "").lower()

        if ext in [".xlsx", "xlsx"]:
            # Excel reader requires openpyxl
            df = pd.read_excel(path_or_buf, dtype=str, engine="openpyxl")
        else:
            df = pd.read_csv(path_or_buf, dtype=str, keep_default_na=False)
        return _lower_cols(df)
    except Exception as e:
        raise RuntimeError(f"Failed to load table ({file_hint or path_or_buf}): {e}")

@st.cache_data
def load_cads(source) -> pd.DataFrame:
    df = load_table(source, file_hint=os.path.splitext(CADS_FILE_DEFAULT)[1])
    missing = REQUIRED_CADS_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CADS file missing columns: {sorted(missing)}")
    df = df[list(REQUIRED_CADS_COLS)]
    st.caption("Loaded CADS columns: " + ", ".join(df.columns))
    return df

def load_source_mappings(path: str) -> pd.DataFrame:
    cols = [
        "scope",
        "src_year", "src_make", "src_model", "src_trim",
        "cad_year", "cad_make", "cad_model", "cad_trim",
        "model_code", "source"
    ]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = load_table(path, file_hint=os.path.splitext(path)[1])
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c != "scope" else "ymmt"
    return df[cols]

def save_source_mappings(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def append_log_row(row: dict):
    try:
        log_df = pd.DataFrame([row])
        if os.path.exists(SRC_MAP_LOG):
            log_df.to_csv(SRC_MAP_LOG, mode="a", header=False, index=False)
        else:
            log_df.to_csv(SRC_MAP_LOG, index=False)
    except Exception:
        # Non-blocking
        pass

# ---------------------------
# Matching logic
# ---------------------------
def fuzzy_filter_cads(
    df: pd.DataFrame,
    year: str, make: str, model: str, trim: str,
    threshold: int = 80,
    weights = (0.35, 0.40, 0.25)  # (make, model, trim)
) -> pd.DataFrame:
    """
    Weighted fuzzy: partial_ratio for make, model; token_set_ratio for trim.
    Returns rows with weighted score >= threshold (0-100).
    """
    filtered = df.copy()
    if year:
        filtered = filtered[filtered["ad_year"].apply(norm) == norm(year)]

    candidates = []
    for idx, row in filtered.iterrows():
        # base scores
        s_make = fuzz.partial_ratio(norm(make), norm(row["ad_make"])) if make else 0
        s_model = fuzz.partial_ratio(norm(model), norm(row["ad_model"])) if model else 0
        # trim uses token_set to be order-insensitive
        s_trim = fuzz.token_set_ratio(norm(trim), norm(row["ad_trim"])) if trim else 0

        w_make, w_model, w_trim = weights
        score = s_make * w_make + s_model * w_model + s_trim * w_trim
        if score >= threshold:
            candidates.append((idx, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered.loc[[c[0] for c in candidates]]

def parse_vehicle(vehicle_text: str, cads_df: pd.DataFrame):
    vt = str(vehicle_text or "").strip()
    if not vt:
        return "", "", "", ""
    tokens = vt.split()
    year = tokens[0] if tokens and tokens[0].isdigit() and len(tokens[0]) == 4 else ""
    if year:
        tokens = tokens[1:]
    seq = " ".join(tokens).lower()

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
        if rest.startswith(mdl + " ") or seq == mdl or rest == mdl:
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

# ---------------------------
# UI — Auth
# ---------------------------
st.title("🔒 External → CADS Vehicle Mapper (POC)")

pw = st.text_input("Enter password", type="password")
if pw != APP_PASSWORD:
    st.stop()
st.success("Authenticated ✅")

# ---------------------------
# UI — CADS source
# ---------------------------
st.subheader("CADS source")
cads_choice = st.radio(
    "Load CADS from:",
    ["Local file (CADS.csv)", "Upload file (CSV/XLSX)"],
    horizontal=True
)
cads_df = None

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
        # infer hint by file name extension
        fname = getattr(up, "name", "")
        hint = (os.path.splitext(fname)[1] or "").lower().replace(".", "")
        cads_df = load_cads(up if hint != "xlsx" else up)
    except Exception as e:
        st.error(f"Failed to load uploaded CADS: {e}")
        st.stop()

# ---------------------------
# UI — External input
# ---------------------------
st.subheader("External input")
vehicle_text = st.text_input("External Vehicle (e.g., '2025 Land Rover Range Rover Sport P360 SE')")
py, pmake, pmodel, ptrim = parse_vehicle(vehicle_text, cads_df)

c1, c2, c3, c4 = st.columns(4)
with c1: src_year = st.text_input("External Year", py)
with c2: src_make = st.text_input("External Make", pmake)
with c3: src_model = st.text_input("External Model", pmodel)
with c4: src_trim = st.text_input("External Trim", ptrim)

scope = st.radio("Mapping scope", ["Exact (Y+Make+Model+Trim)", "Vehicle line (Y+Make+Model)"], index=0, horizontal=True)
scope_val = "ymmt" if scope.startswith("Exact") else "ymm"

col_thr, col_wm, col_wmd, col_wt = st.columns([1, 1, 1, 1])
with col_thr:
    threshold = st.slider("Fuzzy threshold", 60, 95, 80)
with col_wm:
    w_make = st.slider("Weight: Make", 0.0, 1.0, 0.35)
with col_wmd:
    w_model = st.slider("Weight: Model", 0.0, 1.0, 0.40)
with col_wt:
    w_trim = st.slider("Weight: Trim", 0.0, 1.0, 0.25)

# ---------------------------
# Search / Resolve
# ---------------------------
if st.button("Search / Resolve"):
    src_maps_df = load_source_mappings(SRC_MAP_FILE)

    srckey_ymmt = normalize_source_key_ymmt(src_year, src_make, src_model, src_trim)
    src_maps_df["__srckey_ymmt__"] = src_maps_df.apply(
        lambda r: normalize_source_key_ymmt(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
    )
    hit_exact = src_maps_df[
        (src_maps_df["scope"].str.lower() == "ymmt") & (src_maps_df["__srckey_ymmt__"] == srckey_ymmt)
    ]

    srckey_ymm = normalize_source_key_ymm(src_year, src_make, src_model)
    src_maps_df["__srckey_ymm__"] = src_maps_df.apply(
        lambda r: normalize_source_key_ymm(r["src_year"], r["src_make"], r["src_model"]), axis=1
    )
    hit_line = src_maps_df[
        (src_maps_df["scope"].str.lower() == "ymm") & (src_maps_df["__srckey_ymm__"] == srckey_ymm)
    ]

    hit = hit_exact if not hit_exact.empty else hit_line
    if not hit.empty:
        r = hit.iloc[0]
        st.success(f"Found saved Source Mapping ✅ (scope: {r['scope']})")
        st.write({
            "model_code": r["model_code"],
            "mapped_to": f"{r['cad_year']} {r['cad_make']} {r['cad_model']} {r['cad_trim']}"
        })
    else:
        sT_norm = normalize_external_trim(src_trim)
        exact = cads_df.copy()
        if src_year:  exact = exact[exact["ad_year"].apply(norm) == norm(src_year)]
        if src_make:  exact = exact[exact["ad_make"].apply(norm) == norm(src_make)]
        if src_model: exact = exact[exact["ad_model"].apply(norm) == norm(src_model)]
        if sT_norm and scope_val == "ymmt":
            exact = exact[exact["ad_trim"].apply(norm) == norm(sT_norm)]

        results = exact if not exact.empty else fuzzy_filter_cads(
            cads_df, src_year, src_make, src_model,
            sT_norm if scope_val == "ymmt" else "",
            threshold=threshold, weights=(w_make, w_model, w_trim)
        )
        base_df = results.copy()

        if results.empty:
            st.error("No CADS candidates found.")
        else:
            st.write(f"Found {len(results)} CADS candidate(s). Use filters to narrow, then tick a row to save.")

            # defaults
            defaults = {
                "only_model_eq": bool(src_model),
                "apply_trim_tokens": (scope_val == "ymmt") and bool(src_trim),
                "apply_quick_filter": False,
                "quick_filter_text": "",
                "auto_save": True,
            }
            for k, v in defaults.items():
                if k not in st.session_state:
                    st.session_state[k] = v

            rcol1, rcol2 = st.columns([1, 3])
            with rcol1:
                if st.button("♻️ Reset filters"):
                    st.session_state["only_model_eq"] = bool(src_model)
                    st.session_state["apply_trim_tokens"] = (scope_val == "ymmt") and bool(src_trim)
                    st.session_state["apply_quick_filter"] = False
                    st.session_state["quick_filter_text"] = ""

            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                st.session_state["only_model_eq"] = st.checkbox(
                    "Model must equal external model",
                    value=st.session_state["only_model_eq"]
                )
                if st.session_state["only_model_eq"] and src_model:
                    results = results[results["ad_model"].apply(norm) == norm(src_model)]

            ext_tokens = [t for t in norm(src_trim).split()
                          if t not in {"swb", "short", "wheelbase", "lwb", "long", "seat", "7-seater"}]

            with colB:
                st.session_state["apply_trim_tokens"] = st.checkbox(
                    "Filter by external trim tokens",
                    value=st.session_state["apply_trim_tokens"],
                    disabled=(scope_val != "ymmt")
                )
                if st.session_state["apply_trim_tokens"] and ext_tokens and scope_val == "ymmt":
                    results = results[results["ad_trim"].apply(
                        lambda s: all(tok in norm(s) for tok in ext_tokens)
                    )]

            with colC:
                st.session_state["apply_quick_filter"] = st.checkbox(
                    "Apply quick filter", value=st.session_state["apply_quick_filter"]
                )
                st.session_state["quick_filter_text"] = st.text_input(
                    "Quick filter (model/trim/code)",
                    value=st.session_state["quick_filter_text"],
                    placeholder="type any text and tick 'Apply quick filter'"
                )
                if st.session_state["apply_quick_filter"]:
                    q = norm(st.session_state["quick_filter_text"])
                    if q:
                        results = results[results.apply(
                            lambda r: q in norm(r["ad_model"]) or q in norm(r["ad_trim"]) or q in norm(r["ad_mfgcode"]),
                            axis=1
                        )]

            if results.empty:
                st.warning("All candidates were filtered out. Uncheck filters or clear quick filter text.")

            st.dataframe(base_df[["ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"]].head(20))

            if not results.empty:
                st.write(f"Filtered down to {len(results)} row(s). Tick exactly one row to select.")
                view_cols = ["ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"]

                # Keep a stable row id to map back reliably
                display_df = results.reset_index().rename(columns={"index": "__row_id__"})
                display_df = display_df[["__row_id__"] + view_cols].copy()
                display_df.insert(0, "Select", pd.Series([False] * len(display_df), index=display_df.index))

                edited = st.data_editor(
                    display_df,
                    use_container_width=True,
                    num_rows="fixed",
                    disabled=view_cols + ["__row_id__"],
                    hide_index=True,
                    key="cads_editor"
                )

                # Robust selection detection (handles None/strings)
                sel_mask = edited["Select"].fillna(False)
                if sel_mask.dtype != bool:
                    sel_mask = sel_mask.astype(str).str.lower().isin(["true", "1", "yes"])  # normalize

                selected_rows = edited[sel_mask]
                if selected_rows.empty:
                    st.info("Tick a checkbox in the leftmost column to select a row.")
                elif len(selected_rows) > 1:
                    st.warning("Please select exactly one row.")
                else:
                    sel_row = selected_rows.iloc[0]
                    original_idx = int(sel_row["__row_id__"])  # maps back to results
                    cad_row = results.loc[original_idx]

                    code = st.text_input(
                        "Model Code (override optional)",
                        value=str(cad_row["ad_mfgcode"]),
                        key="override_code"
                    )
                    st.session_state["auto_save"] = st.checkbox(
                        "Save immediately when a single row is selected",
                        value=True
                    )

                    def _save_mapping(selected_code: str):
                        src_maps_df_fresh = load_source_mappings(SRC_MAP_FILE)
                        if scope_val == "ymmt":
                            srckey = normalize_source_key_ymmt(src_year, src_make, src_model, src_trim)
                            src_maps_df_fresh["__srckey_ymmt__"] = src_maps_df_fresh.apply(
                                lambda r: normalize_source_key_ymmt(
                                    r["src_year"], r["src_make"], r["src_model"], r["src_trim"]),
                                axis=1
                            )
                            src_maps_df_fresh = src_maps_df_fresh[
                                ~((src_maps_df_fresh["scope"].str.lower() == "ymmt") &
                                  (src_maps_df_fresh["__srckey_ymmt__"] == srckey))
                            ].drop(columns=["__srckey_ymmt__"], errors="ignore")
                        else:
                            srckey = normalize_source_key_ymm(src_year, src_make, src_model)
                            src_maps_df_fresh["__srckey_ymm__"] = src_maps_df_fresh.apply(
                                lambda r: normalize_source_key_ymm(r["src_year"], r["src_make"], r["src_model"]),
                                axis=1
                            )
                            src_maps_df_fresh = src_maps_df_fresh[
                                ~((src_maps_df_fresh["scope"].str.lower() == "ymm") &
                                  (src_maps_df_fresh["__srckey_ymm__"] == srckey))
                            ].drop(columns=["__srckey_ymm__"], errors="ignore")

                        new_row = {
                            "scope": scope_val,
                            "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
                            "cad_year": cad_row["ad_year"], "cad_make": cad_row["ad_make"],
                            "cad_model": cad_row["ad_model"], "cad_trim": cad_row["ad_trim"],
                            "model_code": selected_code, "source": "ui"
                        }
                        src_maps_df_fresh = pd.concat(
                            [src_maps_df_fresh, pd.DataFrame([new_row])],
                            ignore_index=True
                        )
                        save_source_mappings(src_maps_df_fresh, SRC_MAP_FILE)

                        # append audit log (best-effort)
                        append_log_row({
                            **new_row,
                            "saved_by": APP_USER,
                            "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")
                        })

                        st.success("Saved ✅ This external input will now resolve to the mapped CADS code.")
                        st.write({
                            "external": f"{src_year} {src_make} {src_model} {src_trim} (scope: {scope_val})",
                            "mapped_to": f"{cad_row['ad_year']} {cad_row['ad_make']} {cad_row['ad_model']} {cad_row['ad_trim']}",
                            "model_code": selected_code,
                            "srckey": srckey
                        })
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()

                    if st.session_state["auto_save"]:
                        _save_mapping(str(cad_row["ad_mfgcode"]))
                    else:
                        if st.button("💾 Save Source → CADS Mapping"):
                            _save_mapping(code)

# ---------------------------
# Mappings management
# ---------------------------
st.divider()
st.subheader("Source Mappings (external → CADS)")

maps_df = load_source_mappings(SRC_MAP_FILE)
c_filter1, c_filter2, c_filter3 = st.columns(3)
with c_filter1:
    scope_filter = st.selectbox("Scope filter", ["All", "ymmt", "ymm"], index=0)
with c_filter2:
    src_filter = st.text_input("Quick filter (src make/model/trim)")
with c_filter3:
    cad_filter = st.text_input("Quick filter (CADS model/trim/code)")

filtered = maps_df.copy()
if scope_filter in {"ymmt", "ymm"}:
    filtered = filtered[filtered["scope"].str.lower() == scope_filter]
q_src = norm(src_filter)
q_cad = norm(cad_filter)
if q_src:
    filtered = filtered[filtered.apply(
        lambda r: q_src in norm(r["src_make"]) or q_src in norm(r["src_model"]) or q_src in norm(r["src_trim"]),
        axis=1
    )]
if q_cad:
    filtered = filtered[filtered.apply(
        lambda r: q_cad in norm(r["cad_model"]) or q_cad in norm(r["cad_trim"]) or q_cad in norm(r["model_code"]),
        axis=1
    )]

display_cols = [
    "scope",
    "src_year", "src_make", "src_model", "src_trim",
    "cad_year", "cad_make", "cad_model", "cad_trim",
    "model_code", "source"
]
if filtered.empty:
    st.info("No mappings yet.")
else:
    # add helper columns for deletion
    show_df = filtered.reset_index().rename(columns={"index": "__row_id__"})
    show_df.insert(0, "Delete", pd.Series([False] * len(show_df), index=show_df.index))

    edited_maps = st.data_editor(
        show_df[["Delete", "__row_id__"] + display_cols],
        use_container_width=True,
        num_rows="fixed",
        disabled=display_cols + ["__row_id__"],
        hide_index=True,
        key="maps_editor"
    )

    # Delete workflow
    del_mask = edited_maps["Delete"].fillna(False)
    if del_mask.dtype != bool:
        del_mask = del_mask.astype(str).str.lower().isin(["true", "1", "yes"])

    to_delete = edited_maps[del_mask]
    if not to_delete.empty:
        dcount = len(to_delete)
        if st.button(f"🗑️ Delete selected ({dcount})"):
            # Remove by row index from original 'filtered' mapping
            keep_idx = set(filtered.index) - set(to_delete["__row_id__"].astype(int).tolist())
            new_df = filtered.loc[sorted(keep_idx)]
            # Merge back with rows excluded due filters (preserve others)
            survivors = pd.concat([new_df, maps_df.loc[set(maps_df.index) - set(filtered.index)]], ignore_index=True)
            save_source_mappings(survivors, SRC_MAP_FILE)
            st.success(f"Deleted {dcount} mapping(s).")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

st.download_button("Download SourceMappings.csv", load_source_mappings(SRC_MAP_FILE).to_csv(index=False),
                   "SourceMappings.csv", "text/csv")

# Optional: download audit log (if exists)
if os.path.exists(SRC_MAP_LOG):
    log_bytes = open(SRC_MAP_LOG, "rb").read()
    st.download_button("Download SourceMappings.log.csv", log_bytes, "SourceMappings.log.csv", "text/csv")
