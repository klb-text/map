import os
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

st.set_page_config(page_title="External → CADS Vehicle Mapper (POC)", layout="wide")
st.info("Build: 2025-11-22 8:10 PM ET — External→CADS mapping v1.9 (auto-save default ON)")

APP_PASSWORD = os.getenv("APP_PASSWORD", "mypassword")
CADS_FILE    = "CADS.csv"
SRC_MAP_FILE = "SourceMappings.csv"
API_TOKEN    = os.getenv("API_TOKEN", "mozenda-token")

def norm(s: str) -> str:
    s = str(s or "").lower().strip()
    return " ".join(s.split())

def normalize_key(year, make, model, trim) -> str:
    return "|".join([norm(year), norm(make), norm(model), norm(trim)])

SYNONYMS = {
    "swb": "",
    "short wheelbase": "",
    "lwb": "7 seat",
    "long wheelbase": "7 seat",
    "7-seater": "7 seat",
}

def normalize_external_trim(t: str) -> str:
    s = norm(t)
    for k, v in SYNONYMS.items():
        if k in s:
            s = s.replace(k, v).strip()
    return " ".join(s.split())

def normalize_source_key_ymmt(year, make, model, trim) -> str:
    return "|".join([norm(year), norm(make), norm(model), norm(normalize_external_trim(trim))])

def normalize_source_key_ymm(year, make, model) -> str:
    return "|".join([norm(year), norm(make), norm(model)])

@st.cache_data
def load_cads(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    st.caption("Loaded CADS columns: " + ", ".join(list(df.columns)))
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"]]

def load_source_mappings(path: str) -> pd.DataFrame:
    cols = [
        "scope",
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
            df[c] = "" if c != "scope" else "ymmt"
    return df[cols]

def save_source_mappings(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def fuzzy_filter_cads(df: pd.DataFrame, year, make, model, trim, threshold=80) -> pd.DataFrame:
    filtered = df.copy()
    if year:
        filtered = filtered[filtered["ad_year"].apply(norm) == norm(year)]
    candidates = []
    for idx, row in filtered.iterrows():
        score = 0; parts = 0
        if make:
            score += fuzz.partial_ratio(norm(make),  norm(row["ad_make"]))
            parts += 1
        if model:
            score += fuzz.partial_ratio(norm(model), norm(row["ad_model"]))
            parts += 1
        if trim:
            score += fuzz.partial_ratio(norm(trim),  norm(row["ad_trim"]))
            parts += 1
        avg = score / (parts or 1)
        if avg >= threshold:
            candidates.append((idx, avg))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered.loc[[c[0] for c in candidates]]

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

# ===== UI MODE =====
st.title("🔒 External → CADS Vehicle Mapper (POC)")
pw = st.text_input("Enter password", type="password")
if pw != APP_PASSWORD:
    st.stop()
st.success("Authenticated ✅")

try:
    cads_df = load_cads(CADS_FILE)
except Exception as e:
    st.error(f"Failed to load CADS.csv: {e}")
    st.stop()

st.subheader("External input")
vehicle_text = st.text_input("External Vehicle (e.g., '2025 Land Rover Range Rover Sport P360 SE')")
py, pmake, pmodel, ptrim = parse_vehicle(vehicle_text, cads_df)

c1, c2, c3, c4 = st.columns(4)
with c1: src_year  = st.text_input("External Year",  py)
with c2: src_make  = st.text_input("External Make",  pmake)
with c3: src_model = st.text_input("External Model", pmodel)
with c4: src_trim  = st.text_input("External Trim",  ptrim)

scope = st.radio("Mapping scope", ["Exact (Y+Make+Model+Trim)", "Vehicle line (Y+Make+Model)"], index=0)
scope_val = "ymmt" if scope.startswith("Exact") else "ymm"

threshold = st.slider("Fuzzy threshold (CADS fallback)", 60, 95, 80)

if st.button("Search / Resolve"):
    src_maps_df = load_source_mappings(SRC_MAP_FILE)

    srckey_ymmt = normalize_source_key_ymmt(src_year, src_make, src_model, src_trim)
    src_maps_df["__srckey_ymmt__"] = src_maps_df.apply(
        lambda r: normalize_source_key_ymmt(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
    )
    hit_exact = src_maps_df[(src_maps_df["scope"].str.lower() == "ymmt") & (src_maps_df["__srckey_ymmt__"] == srckey_ymmt)]

    srckey_ymm = normalize_source_key_ymm(src_year, src_make, src_model)
    src_maps_df["__srckey_ymm__"] = src_maps_df.apply(
        lambda r: normalize_source_key_ymm(r["src_year"], r["src_make"], r["src_model"]), axis=1
    )
    hit_line = src_maps_df[(src_maps_df["scope"].str.lower() == "ymm") & (src_maps_df["__srckey_ymm__"] == srckey_ymm)]

    hit = hit_exact if not hit_exact.empty else hit_line
    if not hit.empty:
        r = hit.iloc[0]
        st.success(f"Found saved Source Mapping ✅ (scope: {r['scope']})")
        st.write({"model_code": r["model_code"], "mapped_to": f"{r['cad_year']} {r['cad_make']} {r['cad_model']} {r['cad_trim']}"})
    else:
        sT_norm = normalize_external_trim(src_trim)
        exact = cads_df.copy()
        if src_year:  exact = exact[exact["ad_year"].apply(norm)  == norm(src_year)]
        if src_make:  exact = exact[exact["ad_make"].apply(norm)  == norm(src_make)]
        if src_model: exact = exact[exact["ad_model"].apply(norm) == norm(src_model)]
        if sT_norm and scope_val == "ymmt":
            exact = exact[exact["ad_trim"].apply(norm)  == norm(sT_norm)]
        results = exact if not exact.empty else fuzzy_filter_cads(cads_df, src_year, src_make, src_model, sT_norm if scope_val=="ymmt" else "", threshold)

        base_df = results.copy()
        if results.empty:
            st.error("No CADS candidates found.")
        else:
            st.write(f"Found {len(results)} CADS candidate(s). Use filters to narrow, then tick a row to save.")

            for k, v in {
                "only_model_eq": bool(src_model),
                "apply_trim_tokens": (scope_val == "ymmt") and bool(src_trim),
                "apply_quick_filter": False,
                "quick_filter_text": "",
                "auto_save": True,  # default ON
            }.items():
                if k not in st.session_state:
                    st.session_state[k] = v

            rcol1, rcol2 = st.columns([1, 3])
            with rcol1:
                if st.button("♻️ Reset filters"):
                    st.session_state["only_model_eq"]      = bool(src_model)
                    st.session_state["apply_trim_tokens"]  = (scope_val == "ymmt") and bool(src_trim)
                    st.session_state["apply_quick_filter"] = False
                    st.session_state["quick_filter_text"]  = ""

            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                st.session_state["only_model_eq"] = st.checkbox("Model must equal external model", value=st.session_state["only_model_eq"])
            if st.session_state["only_model_eq"] and src_model:
                results = results[results["ad_model"].apply(norm) == norm(src_model)]

            ext_tokens = [t for t in norm(src_trim).split() if t not in {"swb", "short", "wheelbase", "lwb", "long", "seat", "7-seater"}]
            with colB:
                st.session_state["apply_trim_tokens"] = st.checkbox("Filter by external trim tokens", value=st.session_state["apply_trim_tokens"], disabled=(scope_val != "ymmt"))
            if st.session_state["apply_trim_tokens"] and ext_tokens and scope_val == "ymmt":
                results = results[results["ad_trim"].apply(lambda s: all(tok in norm(s) for tok in ext_tokens))]

            with colC:
                st.session_state["apply_quick_filter"] = st.checkbox("Apply quick filter", value=st.session_state["apply_quick_filter"])
                st.session_state["quick_filter_text"]  = st.text_input("Quick filter (model/trim/code)", value=st.session_state["quick_filter_text"], placeholder="type any text and tick 'Apply quick filter'")
            if st.session_state["apply_quick_filter"]:
                q = norm(st.session_state["quick_filter_text"])
                if q:
                    results = results[results.apply(lambda r: q in norm(r["ad_model"]) or q in norm(r["ad_trim"]) or q in norm(r["ad_mfgcode"]), axis=1)]

            if results.empty:
                st.warning("All candidates were filtered out. Uncheck filters or clear quick filter text.")
                st.dataframe(base_df[["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]].head(20))
            else:
                st.write(f"Filtered down to {len(results)} row(s). Tick exactly one row to select.")
                view_cols = ["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]
                display_df = results[view_cols].copy()
                display_df.insert(0, "Select", pd.Series([False]*len(display_df), index=display_df.index))
                edited = st.data_editor(
                    display_df,
                    use_container_width=True,
                    num_rows="fixed",
                    disabled=view_cols,
                    hide_index=False,
                    key="cads_editor"
                )
                selected_idxs = edited.index[edited["Select"] == True].tolist()

                if len(selected_idxs) == 0:
                    st.info("Tick a checkbox in the leftmost column to select a row.")
                elif len(selected_idxs) > 1:
                    st.warning("Please select exactly one row.")
                else:
                    sel_idx = selected_idxs[0]
                    cad_row = results.loc[sel_idx]
                    code = st.text_input("Model Code (override optional)", value=cad_row["ad_mfgcode"], key="override_code")

                    st.session_state["auto_save"] = st.checkbox("Save immediately when a single row is selected", value=st.session_state["auto_save"])  # default True

                    def _save_mapping(selected_code: str):
                        src_maps_df_fresh = load_source_mappings(SRC_MAP_FILE)
                        if scope_val == "ymmt":
                            srckey = normalize_source_key_ymmt(src_year, src_make, src_model, src_trim)
                            src_maps_df_fresh["__srckey_ymmt__"] = src_maps_df_fresh.apply(lambda r: normalize_source_key_ymmt(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1)
                            src_maps_df_fresh = src_maps_df_fresh[~((src_maps_df_fresh["scope"].str.lower()=="ymmt") & (src_maps_df_fresh["__srckey_ymmt__"]==srckey))].drop(columns=["__srckey_ymmt__"], errors="ignore")
                        else:
                            srckey = normalize_source_key_ymm(src_year, src_make, src_model)
                            src_maps_df_fresh["__srckey_ymm__"] = src_maps_df_fresh.apply(lambda r: normalize_source_key_ymm(r["src_year"], r["src_make"], r["src_model"]), axis=1)
                            src_maps_df_fresh = src_maps_df_fresh[~((src_maps_df_fresh["scope"].str.lower()=="ymm") & (src_maps_df_fresh["__srckey_ymm__"]==srckey))].drop(columns=["__srckey_ymm__"], errors="ignore")

                        new_row = {
                            "scope": scope_val,
                            "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
                            "cad_year": cad_row["ad_year"], "cad_make": cad_row["ad_make"],
                            "cad_model": cad_row["ad_model"], "cad_trim": cad_row["ad_trim"],
                            "model_code": selected_code, "source": "ui"
                        }
                        src_maps_df_fresh = pd.concat([src_maps_df_fresh, pd.DataFrame([new_row])], ignore_index=True)
                        save_source_mappings(src_maps_df_fresh, SRC_MAP_FILE)
                        st.success("Saved ✅ This external input will now resolve to the mapped CADS code.")
                        st.write({"external": f"{src_year} {src_make} {src_model} {src_trim} (scope: {scope_val})", "mapped_to": f"{cad_row['ad_year']} {cad_row['ad_make']} {cad_row['ad_model']} {cad_row['ad_trim']}", "model_code": selected_code})
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()

                    # Auto-save when selected and toggle is ON
                    if st.session_state["auto_save"]:
                        _save_mapping(cad_row["ad_mfgcode"])
                    else:
                        if st.button("💾 Save Source → CADS Mapping"):
                            _save_mapping(code)

st.divider()
st.subheader("Source Mappings (external → CADS)")
st.download_button("Download SourceMappings.csv", load_source_mappings(SRC_MAP_FILE).to_csv(index=False), "SourceMappings.csv", "text/csv")
