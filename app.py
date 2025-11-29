
import os
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# -----------------------------------
# Page config FIRST
# -----------------------------------
st.set_page_config(page_title="External → CADS Vehicle Mapper (POC)", layout="wide")
st.info("Build: 2025-11-22 7:10 PM ET — External→CADS mapping v1.6 (synonym-aware keys, no cache on saves)")

# -----------------------------------
# POC CONFIG (no secrets needed for now)
# -----------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD", "mypassword")   # password gate
CADS_FILE    = "CADS.csv"                                 # must exist in repo root
SRC_MAP_FILE = "SourceMappings.csv"                       # external→CADS crosswalk
API_TOKEN    = os.getenv("API_TOKEN", "mozenda-token")    # token for API mode

# -----------------------------------
# Helpers
# -----------------------------------
def norm(s: str) -> str:
    """Lower + strip + collapse internal whitespace."""
    s = str(s or "").lower().strip()
    return " ".join(s.split())

def normalize_key(year, make, model, trim) -> str:
    """Normalized composite key (Y|Make|Model|Trim) — used for CADS-side keys."""
    return "|".join([norm(year), norm(make), norm(model), norm(trim)])

# External trim synonyms: adjust/extend as your source vocabulary evolves
SYNONYMS = {
    "swb": "",                # treat SWB as standard (non-7-seat) CADS trim
    "short wheelbase": "",
    "lwb": "7 seat",          # map LWB to CADS "7 Seat"
    "long wheelbase": "7 seat",
    "7-seater": "7 seat",
}
def normalize_external_trim(t: str) -> str:
    s = norm(t)
    for k, v in SYNONYMS.items():
        if k in s:
            s = s.replace(k, v).strip()
    return " ".join(s.split())

def normalize_source_key(year, make, model, trim) -> str:
    """
    Build a key for EXTERNAL input that also collapses wheelbase synonyms:
    SWB → '' (standard); LWB/Long Wheelbase/7-Seater → '7 seat'.
    """
    return "|".join([norm(year), norm(make), norm(model), norm(normalize_external_trim(trim))])

# -----------------------------------
# Case-insensitive CSV loaders
# -----------------------------------
@st.cache_data
def load_cads(path: str) -> pd.DataFrame:
    """
    Loads CADS and returns only the five needed columns in lowercase:
    ad_year, ad_make, ad_model, ad_trim, ad_mfgcode
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Show raw header columns once for debugging
    st.caption("Loaded CADS columns: " + ", ".join(list(df.columns)))
    # Normalize headers to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"]]

def load_source_mappings(path: str) -> pd.DataFrame:
    """
    Loads SourceMappings (external→CADS crosswalk) FRESH every time
    (uncached so new saves are picked up immediately).
    """
    cols = [
        "src_year","src_make","src_model","src_trim",
        "cad_year","cad_make","cad_model","cad_trim",
        "model_code","source"
    ]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def save_source_mappings(df: pd.DataFrame, path: str):
    """Saves SourceMappings.csv (no caching)."""
    df.to_csv(path, index=False)

# -----------------------------------
# Fuzzy matching (CADS)
# -----------------------------------
def fuzzy_filter_cads(df: pd.DataFrame, year, make, model, trim, threshold=80) -> pd.DataFrame:
    """
    Fuzzy match Make/Model/Trim within the requested Year subset.
    Uses partial_ratio to tolerate substring differences.
    """
    filtered = df.copy()
    if year:
        filtered = filtered[filtered["ad_year"].apply(norm) == norm(year)]
    candidates = []
    for idx, row in filtered.iterrows():
        score = 0; parts = 0
        if make:
            score += fuzz.partial_ratio(norm(make),  norm(row["ad_make"]));   parts += 1
        if model:
            score += fuzz.partial_ratio(norm(model), norm(row["ad_model"]));  parts += 1
        if trim:
            score += fuzz.partial_ratio(norm(trim),  norm(row["ad_trim"]));   parts += 1
        avg = score / (parts or 1)
        if avg >= threshold:
            candidates.append((idx, avg))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered.loc[[c[0] for c in candidates]]

# -----------------------------------
# Parse external vehicle string (CADS-aware make/model, multi-word)
# -----------------------------------
def parse_vehicle(vehicle_text: str, cads_df: pd.DataFrame):
    vt = str(vehicle_text or "").strip()
    if not vt: return "", "", "", ""
    tokens = vt.split()
    year = tokens[0] if tokens and tokens[0].isdigit() and len(tokens[0]) == 4 else ""
    if year:
        tokens = tokens[1:]
    seq = " ".join(tokens).lower()

    makes = sorted(pd.Series(cads_df["ad_make"]).dropna().unique().tolist(), key=len, reverse=True)
    models = sorted(pd.Series(cads_df["ad_model"]).dropna().unique().tolist(), key=len, reverse=True)
    makes_l  = [" ".join(str(m).lower().split()) for m in makes]
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
        rest   = " ".join(parts[1:]) if len(parts) > 1 else ""

    model_l = ""
    for mdl in models_l:
        if rest.startswith(mdl + " ") or seq == mdl or rest == mdl:
            model_l = mdl; break
    if model_l:
        trim_l = rest[len(model_l):].strip()
    else:
        rem = rest.split()
        model_l = rem[0] if rem else ""
        trim_l  = " ".join(rem[1:]) if len(rem) > 1 else ""

    make_human  = next((m for m in makes  if norm(m) == make_l), make_l)
    model_human = next((m for m in models if norm(m) == model_l), model_l)
    return year, make_human, model_human, trim_l

# -----------------------------------
# API MODE (external → CADS)
# -----------------------------------
params = st.experimental_get_query_params()
if params.get("api_token", [""])[0] == API_TOKEN:
    # Load CADS + SourceMappings (fresh mappings)
    try:
        cads_df = load_cads(CADS_FILE)
    except Exception as e:
        st.json({"error": f"Failed to load CADS.csv: {e}"}); st.stop()

    src_maps_df = load_source_mappings(SRC_MAP_FILE)

    # Resolve external → CADS code
    if "get_model_code" in params:
        sy  = params.get("src_year",  [""])[0]
        sm  = params.get("src_make",  [""])[0]
        sMo = params.get("src_model", [""])[0]
        sT  = params.get("src_trim",  [""])[0]

        # 1) Try saved source mapping first (synonym-aware)
        src_key = normalize_source_key(sy, sm, sMo, sT)
        src_maps_df["__srckey__"] = src_maps_df.apply(
            lambda r: normalize_source_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
        )
        hit = src_maps_df[src_maps_df["__srckey__"] == src_key]
        if not hit.empty:
            st.json({"model_code": hit.iloc[0]["model_code"], "source": "source_mapping"}); st.stop()

        # 2) CADS fallback: normalize external trim synonyms; exact → fuzzy
        sT_norm = normalize_external_trim(sT)
        exact = cads_df.copy()
        if sy:   exact = exact[exact["ad_year"].apply(norm)  == norm(sy)]
        if sm:   exact = exact[exact["ad_make"].apply(norm)  == norm(sm)]
        if sMo:  exact = exact[exact["ad_model"].apply(norm) == norm(sMo)]
        if sT_norm: exact = exact[exact["ad_trim"].apply(norm) == norm(sT_norm)]
        if not exact.empty:
            st.json({"model_code": exact.iloc[0]["ad_mfgcode"], "source": "cads_exact"}); st.stop()
        fuzzy = fuzzy_filter_cads(cads_df, sy, sm, sMo, sT_norm)
        if not fuzzy.empty:
            st.json({"model_code": fuzzy.iloc[0]["ad_mfgcode"], "source": "cads_fuzzy"}); st.stop()
        st.json({"model_code": "", "source": "none"}); st.stop()

    # Save a source mapping programmatically (synonym-aware upsert)
    if "save_source_mapping" in params:
        sy  = params.get("src_year",  [""])[0]
        sm  = params.get("src_make",  [""])[0]
        sMo = params.get("src_model", [""])[0]
        sT  = params.get("src_trim",  [""])[0]
        cy  = params.get("cad_year",  [""])[0]
        cm  = params.get("cad_make",  [""])[0]
        cMo = params.get("cad_model", [""])[0]
        cT  = params.get("cad_trim",  [""])[0]
        code= params.get("code",      [""])[0]

        if not all([sy, sm, sMo, sT, cy, cm, cMo, cT, code]):
            st.json({"ok": False, "error": "missing params"}); st.stop()

        src_maps_df = load_source_mappings(SRC_MAP_FILE)  # fresh
        src_key = normalize_source_key(sy, sm, sMo, sT)
        if not src_maps_df.empty:
            src_maps_df["__srckey__"] = src_maps_df.apply(
                lambda r: normalize_source_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
            )
            src_maps_df = src_maps_df[src_maps_df["__srckey__"] != src_key].drop(columns=["__srckey__"], errors="ignore")

        new_row = {
            "src_year": sy, "src_make": sm, "src_model": sMo, "src_trim": sT,
            "cad_year": cy, "cad_make": cm, "cad_model": cMo, "cad_trim": cT,
            "model_code": code, "source": "api"
        }
        src_maps_df = pd.concat([src_maps_df, pd.DataFrame([new_row])], ignore_index=True)
        save_source_mappings(src_maps_df, SRC_MAP_FILE)
        st.json({"ok": True, "message": "saved"}); st.stop()

    st.json({"error": "unknown api call"}); st.stop()

# -----------------------------------
# UI MODE
# -----------------------------------
st.title("🔒 External → CADS Vehicle Mapper (POC)")

# Password gate
pw = st.text_input("Enter password", type="password")
if pw != APP_PASSWORD:
    st.stop()
st.success("Authenticated ✅")

# Load CADS + SourceMappings (UI — mappings fresh)
try:
    cads_df = load_cads(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load CADS.csv: {e}")
    st.stop()

src_maps_df = load_source_mappings(SRC_MAP_FILE)

st.subheader("External input")
vehicle_text = st.text_input("External Vehicle (e.g., '2025 Land Rover Range Rover Sport P360 SE')")

py, pmake, pmodel, ptrim = parse_vehicle(vehicle_text, cads_df)

c1, c2, c3, c4 = st.columns(4)
with c1: src_year  = st.text_input("External Year",  py)
with c2: src_make  = st.text_input("External Make",  pmake)
with c3: src_model = st.text_input("External Model", pmodel)
with c4: src_trim  = st.text_input("External Trim",  ptrim)

threshold = st.slider("Fuzzy threshold (CADS fallback)", 60, 95, 80)

if st.button("Search / Resolve"):
    # 1) Check SourceMappings first (instant recall — synonym-aware)
    src_key = normalize_source_key(src_year, src_make, src_model, src_trim)
    src_maps_df = load_source_mappings(SRC_MAP_FILE)  # ensure fresh
    src_maps_df["__srckey__"] = src_maps_df.apply(
        lambda r: normalize_source_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
    )
    hit = src_maps_df[src_maps_df["__srckey__"] == src_key]
    if not hit.empty:
        r = hit.iloc[0]
        st.success("Found saved Source Mapping ✅")
        st.write({
            "model_code": r["model_code"],
            "mapped_to": f"{r['cad_year']} {r['cad_make']} {r['cad_model']} {r['cad_trim']}"
        })
    else:
        # 2) CADS fallback: normalize external trim; exact → fuzzy
        sT_norm = normalize_external_trim(src_trim)
        exact = cads_df.copy()
        if src_year:  exact = exact[exact["ad_year"].apply(norm)  == norm(src_year)]
        if src_make:  exact = exact[exact["ad_make"].apply(norm)  == norm(src_make)]
        if src_model: exact = exact[exact["ad_model"].apply(norm) == norm(src_model)]
        if sT_norm:   exact = exact[exact["ad_trim"].apply(norm)  == norm(sT_norm)]
        results = exact if not exact.empty else fuzzy_filter_cads(cads_df, src_year, src_make, src_model, sT_norm, threshold)

        # Keep an unfiltered baseline for recovery display if filters empty the set
        base_df = results.copy()

        if results.empty:
            st.error("No CADS candidates found.")
        else:
            st.write(f"Found {len(results)} CADS candidate(s). Use the filters below to narrow, then pick the exact row.")

            # --------------------------
            # FILTER WIDGET STATE (session_state)
            # --------------------------
            # Initialize keys once
            for k, v in {
                "only_model_eq": bool(src_model),
                "apply_trim_tokens": bool(src_trim),
                "apply_quick_filter": False,
                "quick_filter_text": "",
                "auto_save": False,
                "prev_sel_idx": None,
            }.items():
                if k not in st.session_state:
                    st.session_state[k] = v

            # Reset filters button
            rcol1, rcol2 = st.columns([1, 3])
            with rcol1:
                if st.button("♻️ Reset filters"):
                    st.session_state["only_model_eq"]     = bool(src_model)
                    st.session_state["apply_trim_tokens"] = bool(src_trim)
                    st.session_state["apply_quick_filter"] = False
                    st.session_state["quick_filter_text"] = ""
                    st.session_state["prev_sel_idx"] = None
                    # No early return; we rebuild results below

            # --------------------------
            # NARROWING FILTERS
            # --------------------------
            colA, colB, colC = st.columns([1, 1, 2])

            # 1) Only rows where CADS Model == external Model (e.g., 'Range Rover Sport')
            with colA:
                st.session_state["only_model_eq"] = st.checkbox(
                    "Model must equal external model",
                    value=st.session_state["only_model_eq"]
                )
            if st.session_state["only_model_eq"] and src_model:
                results = results[results["ad_model"].apply(norm) == norm(src_model)]

            # 2) Filter by external trim tokens (e.g., "P360 SE")
            ext_tokens = [t for t in norm(src_trim).split() if t not in {"swb", "short", "wheelbase", "lwb", "long", "seat", "7-seater"}]
            with colB:
                st.session_state["apply_trim_tokens"] = st.checkbox(
                    "Filter by external trim tokens",
                    value=st.session_state["apply_trim_tokens"]
                )
            if st.session_state["apply_trim_tokens"] and ext_tokens:
                results = results[results["ad_trim"].apply(lambda s: all(tok in norm(s) for tok in ext_tokens))]

            # 3) Quick filter (user-controlled and empty by default)
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
                    results = results[
                        results.apply(
                            lambda r: q in norm(r["ad_model"])
                                   or q in norm(r["ad_trim"])
                                   or q in norm(r["ad_mfgcode"]),
                            axis=1
                        )
                    ]

            # If filters removed everything, show guidance but keep UI alive
            if results.empty:
                st.warning("All candidates were filtered out. Uncheck filters or clear quick filter text.")
                base_cols = ["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]
                st.dataframe(base_df[base_cols].head(20))
            else:
                st.write(f"Filtered down to {len(results)} row(s).")
                view_cols = ["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]
                st.dataframe(results[view_cols], use_container_width=True)

                # --------------------------
                # ERGONOMIC SELECTORS (robust)
                # --------------------------
                # Default selection is first result, so the selector always shows
                sel_idx_default = results.index.tolist()[0]

                # Optional: quick pick by model code
                codes = results["ad_mfgcode"].dropna().unique().tolist()
                if len(results) > 1 and len(codes) > 0:
                    sel_code = st.selectbox("Pick by model code (optional)", [""] + codes, index=0, key="sel_code_key")
                    if sel_code:
                        ix = results.index[results["ad_mfgcode"] == sel_code]
                        if len(ix) > 0:
                            sel_idx_default = ix[0]

                # Always-rendered descriptor selectbox
                options = results.index.tolist()
                def fmt(i):
                    r = results.loc[i]
                    return f"{r['ad_model']} • {r['ad_trim']} • {r['ad_mfgcode']} ({r['ad_year']} {r['ad_make']})"
                default_pos = options.index(sel_idx_default) if sel_idx_default in options else 0

                sel_idx = st.selectbox("Pick CADS row", options, format_func=fmt, index=default_pos, key="sel_idx_key")

                # Allow override of code (optional)
                cad_row = results.loc[sel_idx]
                code = st.text_input("Model Code (override optional)", value=cad_row["ad_mfgcode"], key="override_code")

                # Save-on-selection convenience
                st.session_state["auto_save"] = st.checkbox("Save on selection (no extra click)", value=st.session_state["auto_save"])
                if st.session_state["auto_save"] and st.session_state.get("prev_sel_idx") != sel_idx:
                    # Auto-save with the row's current code
                    src_maps_df_fresh = load_source_mappings(SRC_MAP_FILE)
                    src_key_auto = normalize_source_key(src_year, src_make, src_model, src_trim)
                    if not src_maps_df_fresh.empty:
                        src_maps_df_fresh["__srckey__"] = src_maps_df_fresh.apply(
                            lambda r: normalize_source_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
                        )
                        src_maps_df_fresh = src_maps_df_fresh[src_maps_df_fresh["__srckey__"] != src_key_auto] \
                            .drop(columns=["__srckey__"], errors="ignore")
                    new_row = {
                        "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
                        "cad_year": cad_row["ad_year"], "cad_make": cad_row["ad_make"],
                        "cad_model": cad_row["ad_model"], "cad_trim": cad_row["ad_trim"],
                        "model_code": cad_row["ad_mfgcode"], "source": "ui-auto"
                    }
                    src_maps_df_fresh = pd.concat([src_maps_df_fresh, pd.DataFrame([new_row])], ignore_index=True)
                    save_source_mappings(src_maps_df_fresh, SRC_MAP_FILE)
                    st.session_state["prev_sel_idx"] = sel_idx
                    st.success("Saved on selection ✅")
                    # Re-resolve immediately so the source-mapping path takes effect
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()

                # Manual save
                if st.button("💾 Save Source → CADS Mapping"):
                    src_maps_df_fresh = load_source_mappings(SRC_MAP_FILE)  # fresh
                    src_key_save = normalize_source_key(src_year, src_make, src_model, src_trim)
                    if not src_maps_df_fresh.empty:
                        src_maps_df_fresh["__srckey__"] = src_maps_df_fresh.apply(
                            lambda r: normalize_source_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
                        )
                        src_maps_df_fresh = src_maps_df_fresh[src_maps_df_fresh["__srckey__"] != src_key_save] \
                            .drop(columns=["__srckey__"], errors="ignore")

                    new_row = {
                        "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
                        "cad_year": cad_row["ad_year"], "cad_make": cad_row["ad_make"],
                        "cad_model": cad_row["ad_model"], "cad_trim": cad_row["ad_trim"],
                        "model_code": code, "source": "ui"
                    }
                    src_maps_df_fresh = pd.concat([src_maps_df_fresh, pd.DataFrame([new_row])], ignore_index=True)
                    save_source_mappings(src_maps_df_fresh, SRC_MAP_FILE)

                    st.success("Saved ✅ This external Y/M/M/T will now resolve to the mapped CADS code.")
                    st.write({
                        "external": f"{src_year} {src_make} {src_model} {src_trim}",
                        "mapped_to": f"{cad_row['ad_year']} {cad_row['ad_make']} {cad_row['ad_model']} {cad_row['ad_trim']}",
                        "model_code": code
                    })
                    # Re-resolve immediately so the source-mapping path takes effect
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()

# -----------------------------------
# Download current source mappings (persist manually to GitHub)
# -----------------------------------
st.divider()
st.subheader("Source Mappings (external → CADS)")
st.download_button(
    label="Download SourceMappings.csv",
    data=load_source_mappings(SRC_MAP_FILE).to_csv(index=False),
    file_name="SourceMappings.csv",
    mime="text/csv"
)
