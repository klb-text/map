# External → CADS Vehicle Mapper (POC) — v2.2.2
# Build: 2025-11-30 5:25 PM ET

import os
import unicodedata
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# ---------------------------
# Config & constants
# ---------------------------
st.set_page_config(page_title="External → CADS Vehicle Mapper (POC)", layout="wide")
st.info("Build: 2025-11-30 5:25 PM ET — v2.2.2 (strict source key, replace-or-append, explicit save)")

APP_PASSWORD = os.getenv("APP_PASSWORD", "mypassword")
APP_USER = os.getenv("APP_USER", "anonymous")
API_TOKEN = os.getenv("API_TOKEN", "mozenda-token")

CADS_FILE_DEFAULT = "CADS.csv"
SRC_MAP_FILE = "SourceMappings.csv"
SRC_MAP_LOG = "SourceMappings.log.csv"  # optional audit log

REQUIRED_CADS_COLS = {"ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"}

# Session keys
RESOLVE_FLAG = "resolve_triggered"
LAST_SAVE_KEY = "last_save_info"

# Synonyms used ONLY for matching (not for source key)
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
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s

def norm(s: str) -> str:
    s = ascii_fold(s or "")
    s = s.lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s

def normalize_external_trim_for_match(t: str) -> str:
    """Matching-only normalization: apply synonyms for better fuzzy search."""
    s = norm(t)
    for k, v in SYNONYMS.items():
        k_norm = norm(k)
        if k_norm and k_norm in s:
            s = s.replace(k_norm, norm(v)).strip()
    return " ".join(s.split())

def source_key_ymmt_strict(year, make, model, trim) -> str:
    """
    STRICT source key for de-dupe:
    - Year/Make/Model: norm()
    - Trim: norm() ONLY (NO synonyms)
    This avoids collisions like 'P400 SE SWB' → 'p400 se' overwriting 'P400 SE'.
    """
    return "\n".join([norm(year), norm(make), norm(model), norm(trim)])

def source_key_ymm_strict(year, make, model) -> str:
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
    try:
        if isinstance(path_or_buf, str):
            ext = (os.path.splitext(path_or_buf)[1] or "").lower()
        else:
            ext = (file_hint or "").lower()
        if ext in [".xlsx", "xlsx"]:
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
        pass  # non-blocking

# ---------------------------
# Matching logic
# ---------------------------
def fuzzy_filter_cads(
    df: pd.DataFrame,
    year: str, make: str, model: str, trim: str,
    threshold: int = 80,
    weights = (0.35, 0.40, 0.25)
) -> pd.DataFrame:
    filtered = df.copy()
    if year:
        filtered = filtered[filtered["ad_year"].apply(norm) == norm(year)]
    candidates = []
    for idx, row in filtered.iterrows():
        s_make = fuzz.partial_ratio(norm(make), norm(row["ad_make"])) if make else 0
        s_model = fuzz.partial_ratio(norm(model), norm(row["ad_model"])) if model else 0
        # matching uses synonym-normalized trim
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
            make_l = m; break
    if make_l:
        rest = seq[len(make_l):].strip()
    else:
        parts = seq.split()
        make_l = parts[0] if parts else ""
        rest = " ".join(parts[1:]) if len(parts) > 1 else ""
    model_l = ""
    for mdl in models_l:
        if rest.startswith(mdl + " ") or seq == mdl or rest == mdl:
            model_l = mdl; break
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

# Show last save banner after rerun
if LAST_SAVE_KEY in st.session_state:
    info = st.session_state[LAST_SAVE_KEY]
    st.success(
        f"Saved ✅  {info['external']} → {info['mapped_to']}  (code: {info['model_code']}, scope: {info['scope']})"
    )
    del st.session_state[LAST_SAVE_KEY]

# ---------------------------
# UI — CADS source
# ---------------------------
st.subheader("CADS source")
cads_choice = st.radio("Load CADS from:", ["Local file (CADS.csv)", "Upload file (CSV/XLSX)"], horizontal=True)
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

# Trigger persistent resolve
if st.button("Search / Resolve", type="primary"):
    st.session_state[RESOLVE_FLAG] = True

# ---------------------------
# Resolve block (persistent)
# ---------------------------
if st.session_state.get(RESOLVE_FLAG, False):

    src_maps_df = load_source_mappings(SRC_MAP_FILE)

    # Hit existing mapping?
    strict_srckey_ymmt = source_key_ymmt_strict(src_year, src_make, src_model, src_trim)
    src_maps_df["__srckey_ymmt__strict"] = src_maps_df.apply(
        lambda r: source_key_ymmt_strict(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
    )
    hit_exact = src_maps_df[
        (src_maps_df["scope"].str.lower() == "ymmt") & (src_maps_df["__srckey_ymmt__strict"] == strict_srckey_ymmt)
    ]

    strict_srckey_ymm = source_key_ymm_strict(src_year, src_make, src_model)
    src_maps_df["__srckey_ymm__strict"] = src_maps_df.apply(
        lambda r: source_key_ymm_strict(r["src_year"], r["src_make"], r["src_model"]), axis=1
    )
    hit_line = src_maps_df[
        (src_maps_df["scope"].str.lower() == "ymm") & (src_maps_df["__srckey_ymm__strict"] == strict_srckey_ymm)
    ]

    hit = hit_exact if not hit_exact.empty else hit_line
    if not hit.empty:
        r = hit.iloc[0]
        st.success(f"Found saved Source Mapping ✅ (scope: {r['scope']})")
        st.write({
            "model_code": r["model_code"],
            "mapped_to": f"{r['cad_year']} {r['cad_make']} {r['cad_model']} {r['cad_trim']}"
        })

    # Candidate search
    sT_match = normalize_external_trim_for_match(src_trim)
    exact = cads_df.copy()
    if src_year:  exact = exact[exact["ad_year"].apply(norm) == norm(src_year)]
    if src_make:  exact = exact[exact["ad_make"].apply(norm) == norm(src_make)]
    if src_model: exact = exact[exact["ad_model"].apply(norm) == norm(src_model)]
    if sT_match and scope_val == "ymmt":
        exact = exact[exact["ad_trim"].apply(norm) == norm(sT_match)]

    results = exact if not exact.empty else fuzzy_filter_cads(
        cads_df, src_year, src_make, src_model,
        sT_match if scope_val == "ymmt" else "",
        threshold=threshold, weights=(w_make, w_model, w_trim)
    )

    if results.empty:
        st.error("No CADS candidates found.")
    else:
        st.write(f"Found {len(results)} CADS candidate(s). Choose one below to save mapping.")

        # Optional quick filters
        defaults = {
            "only_model_eq": bool(src_model),
            "apply_trim_tokens": (scope_val == "ymmt") and bool(src_trim),
            "apply_quick_filter": False,
            "quick_filter_text": "",
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

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
                results = results[results["ad_trim"].apply(lambda s: all(tok in norm(s) for tok in ext_tokens))]

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
        else:
            # --- Radio-based selection (robust) ---
            view_cols = ["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]
            display_df = results.reset_index().rename(columns={"index": "__row_id__"})[["__row_id__"] + view_cols].copy()

            st.dataframe(display_df[view_cols], use_container_width=True, height=240)

            labels = [
                f"{r['ad_year']} {r['ad_make']} {r['ad_model']} | {r['ad_trim']} | code={r['ad_mfgcode']}"
                for _, r in display_df.iterrows()
            ]
            idx_to_label = dict(zip(display_df["__row_id__"], labels))
            label_to_idx = {v: k for k, v in idx_to_label.items()}

            selected_label = st.radio("Choose a candidate to map", options=labels, index=0 if labels else None)
            selected_idx = label_to_idx.get(selected_label, None)

            # --- Save controls ---
            # Source key preview & existing count under strict key
            srckey_preview = (source_key_ymmt_strict(src_year, src_make, src_model, src_trim)
                              if scope_val == "ymmt"
                              else source_key_ymm_strict(src_year, src_make, src_model))
            st.caption(f"Source key (strict, used for de-dupe): {srckey_preview}")

            # Count existing rows that would match this key+scope
            existing_count = 0
            if scope_val == "ymmt":
                existing_count = int(((src_maps_df["scope"].str.lower() == "ymmt") &
                                      (src_maps_df["__srckey_ymmt__strict"] == srckey_preview)).sum())
            else:
                existing_count = int(((src_maps_df["scope"].str.lower() == "ymm") &
                                      (src_maps_df["__srckey_ymm__strict"] == srckey_preview)).sum())
            st.caption(f"Existing mappings with this key & scope: {existing_count}")

            auto_save = st.checkbox("Save immediately after choosing", value=False)
            replace_existing = st.checkbox("Replace existing mapping for this key (otherwise append)", value=False)

            if selected_idx is None:
                st.info("Select one candidate above.")
            else:
                cad_row = results.loc[selected_idx]
                code = st.text_input(
                    "Model Code (override optional)",
                    value=str(cad_row["ad_mfgcode"]),
                    key="override_code"
                )

                def _save_mapping(selected_code: str):
                    # fresh load for write operations
                    src_maps_df_fresh = load_source_mappings(SRC_MAP_FILE)

                    # remove only if 'replace_existing' is ON
                    if scope_val == "ymmt":
                        srckey = source_key_ymmt_strict(src_year, src_make, src_model, src_trim)
                        src_maps_df_fresh["__srckey_ymmt__strict"] = src_maps_df_fresh.apply(
                            lambda r: source_key_ymmt_strict(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
                        )
                        if replace_existing:
                            src_maps_df_fresh = src_maps_df_fresh[
                                ~((src_maps_df_fresh["scope"].str.lower() == "ymmt") &
                                  (src_maps_df_fresh["__srckey_ymmt__strict"] == srckey))
                            ]
                        src_maps_df_fresh = src_maps_df_fresh.drop(columns=["__srckey_ymmt__strict"], errors="ignore")
                    else:
                        srckey = source_key_ymm_strict(src_year, src_make, src_model)
                        src_maps_df_fresh["__srckey_ymm__strict"] = src_maps_df_fresh.apply(
                            lambda r: source_key_ymm_strict(r["src_year"], r["src_make"], r["src_model"]), axis=1
                        )
                        if replace_existing:
                            src_maps_df_fresh = src_maps_df_fresh[
                                ~((src_maps_df_fresh["scope"].str.lower() == "ymm") &
                                  (src_maps_df_fresh["__srckey_ymm__strict"] == srckey))
                            ]
                        src_maps_df_fresh = src_maps_df_fresh.drop(columns=["__srckey_ymm__strict"], errors="ignore")

                    new_row = {
                        "scope": scope_val,
                        "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
                        "cad_year": cad_row["ad_year"], "cad_make": cad_row["ad_make"],
                        "cad_model": cad_row["ad_model"], "cad_trim": cad_row["ad_trim"],
                        "model_code": selected_code, "source": "ui"
                    }
                    src_maps_df_fresh = pd.concat([src_maps_df_fresh, pd.DataFrame([new_row])], ignore_index=True)

                    # Write & verify
                    try:
                        save_source_mappings(src_maps_df_fresh, SRC_MAP_FILE)
                    except Exception as e:
                        st.error(f"Failed to write SourceMappings.csv: {e}")
                        return

                    # Verify the new row exists (by srckey+scope and model_code)
                    verify_df = load_source_mappings(SRC_MAP_FILE)
                    if scope_val == "ymmt":
                        verify_df["__srckey_ymmt__strict"] = verify_df.apply(
                            lambda r: source_key_ymmt_strict(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
                        )
                        saved_ok = not verify_df[
                            (verify_df["scope"].str.lower() == "ymmt") &
                            (verify_df["__srckey_ymmt__strict"] == srckey) &
                            (verify_df["model_code"].astype(str) == str(selected_code))
                        ].empty
                        verify_df = verify_df.drop(columns=["__srckey_ymmt__strict"], errors="ignore")
                    else:
                        verify_df["__srckey_ymm__strict"] = verify_df.apply(
                            lambda r: source_key_ymm_strict(r["src_year"], r["src_make"], r["src_model"]), axis=1
                        )
                        saved_ok = not verify_df[
                            (verify_df["scope"].str.lower() == "ymm") &
                            (verify_df["__srckey_ymm__strict"] == srckey) &
                            (verify_df["model_code"].astype(str) == str(selected_code))
                        ].empty
                        verify_df = verify_df.drop(columns=["__srckey_ymm__strict"], errors="ignore")

                    if not saved_ok:
                        st.error("Save attempted, but verification failed. Check write permissions/path.")
                        return

                    # audit log (best-effort)
                    append_log_row({
                        **new_row,
                        "saved_by": APP_USER,
                        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "action": "replace" if replace_existing else "append",
                        "srckey_strict": srckey,
                    })

                    # persistent banner after rerun
                    st.session_state[LAST_SAVE_KEY] = {
                        "external": f"{src_year} {src_make} {src_model} {src_trim} (scope: {scope_val})",
                        "mapped_to": f"{cad_row['ad_year']} {cad_row['ad_make']} {cad_row['ad_model']} {cad_row['ad_trim']}",
                        "model_code": selected_code,
                        "scope": scope_val,
                    }
                    try:
                        st.toast("Saved mapping.", icon="✅")
                    except Exception:
                        pass
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()

                if auto_save:
                    _save_mapping(str(cad_row["ad_mfgcode"]))
                else:
                    if st.button("💾 Save Source → CADS Mapping", type="primary"):
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
    del_mask = edited_maps["Delete"].fillna(False)
    if del_mask.dtype != bool:
        del_mask = del_mask.astype(str).str.lower().isin(["true", "1", "yes"])
    to_delete = edited_maps[del_mask]
    if not to_delete.empty:
        dcount = len(to_delete)
        if st.button(f"🗑️ Delete selected ({dcount})"):
            keep_idx = set(filtered.index) - set(to_delete["__row_id__"].astype(int).tolist())
            new_df = filtered.loc[sorted(keep_idx)]
            survivors = pd.concat([new_df, maps_df.loc[set(maps_df.index) - set(filtered.index)]], ignore_index=True)
            save_source_mappings(survivors, SRC_MAP_FILE)
            st.success(f"Deleted {dcount} mapping(s).")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

st.download_button("Download SourceMappings.csv", load_source_mappings(SRC_MAP_FILE).to_csv(index=False),
                   "SourceMappings.csv", "text/csv")

if os.path.exists(SRC_MAP_LOG):
    log_bytes = open(SRC_MAP_LOG, "rb").read()
    st.download_button("Download SourceMappings.log.csv", log_bytes, "SourceMappings.log.csv", "text/csv")
