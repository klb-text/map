import os
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# -------------------------------
# POC config (no secrets needed)
# -------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD", "mypassword")
CADS_FILE    = "CADS.csv"
SRC_MAP_FILE = "SourceMappings.csv"  # external->CADS crosswalk

# -------------------------------
# Helpers
# -------------------------------
def load_csv(path, required=None, usecols=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)
    df.columns = [c.strip().lower() for c in df.columns]
    if required:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df

def norm(s: str) -> str:
    s = str(s or "").lower().strip()
    return " ".join(s.split())

def normalize_key(year, make, model, trim):
    return "|".join([norm(year), norm(make), norm(model), norm(trim)])

def fuzzy_filter_cads(df, year, make, model, trim, threshold=80):
    """Fuzzy match on CADS within the selected year (if provided)."""
    filtered = df.copy()
    if year:
        filtered = filtered[filtered["ad_year"].apply(norm) == norm(year)]
    candidates = []
    for idx, row in filtered.iterrows():
        score = 0; parts = 0
        if make:
            score += fuzz.partial_ratio(norm(make), norm(row["ad_make"]));   parts += 1
        if model:
            score += fuzz.partial_ratio(norm(model), norm(row["ad_model"])); parts += 1
        if trim:
            score += fuzz.partial_ratio(norm(trim), norm(row["ad_trim"]));   parts += 1
        avg = score / (parts or 1)
        if avg >= threshold:
            candidates.append((idx, avg))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered.loc[[c[0] for c in candidates]]

# External trim normalizer (example rules; adjust as you learn your source)
SYNONYMS = {
    # wheelbase words → CADS language
    "swb": "",           # treat SWB as the 'standard' (non-7 seat) trim
    "lwb": "7 seat",     # map LWB to "7 Seat" variant in CADS
}
def normalize_external_trim(t: str) -> str:
    s = norm(t)
    for k, v in SYNONYMS.items():
        if k in s:
            s = s.replace(k, v).strip()
    return " ".join(s.split())

def parse_vehicle(vehicle_text, cads_df):
    """CADS-aware parse for external strings (handles multi-word make/model via CADS lists)."""
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
        parts = seq.split(); make_l = parts[0] if parts else ""; rest = " ".join(parts[1:]) if len(parts) > 1 else ""

    model_l = ""
    for mdl in models_l:
        if rest.startswith(mdl + " ") or rest == mdl:
            model_l = mdl; break
    if model_l:
        trim_l = rest[len(model_l):].strip()
    else:
        rem = rest.split()
        model_l = rem[0] if rem else ""; trim_l = " ".join(rem[1:]) if len(rem) > 1 else ""

    make_human  = next((m for m in makes  if norm(m) == make_l), make_l)
    model_human = next((m for m in models if norm(m) == model_l), model_l)
    return year, make_human, model_human, trim_l

# -------------------------------
# Load CADS
# -------------------------------
cads_df = load_csv(
    CADS_FILE,
    required={"ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"},
    usecols=["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"],
)
if cads_df is None:
    st.error("CADS.csv not found in repo root with required headers.")
    st.stop()

# -------------------------------
# Load/create SourceMappings (external→CADS)
# -------------------------------
src_required = {"src_year","src_make","src_model","src_trim","cad_year","cad_make","cad_model","cad_trim","model_code","source"}
src_maps_df = load_csv(SRC_MAP_FILE)
if src_maps_df is None:
    # Auto-create with headers
    src_maps_df = pd.DataFrame(columns=list(src_required))
    src_maps_df.to_csv(SRC_MAP_FILE, index=False)
else:
    # Ensure all columns exist
    for col in src_required:
        if col not in src_maps_df.columns:
            src_maps_df[col] = ""

# -------------------------------
# API MODE (for Mozenda)
# -------------------------------
params = st.experimental_get_query_params()
APP_TOKEN = os.getenv("API_TOKEN", "mozenda-token")  # optional; for now hardcoded default

if params.get("api_token", [""])[0] == APP_TOKEN:
    # /?api_token=...&get_model_code=true&src_year=...&src_make=...&src_model=...&src_trim=...
    if "get_model_code" in params:
        sy = params.get("src_year",  [""])[0]
        sm = params.get("src_make",  [""])[0]
        sMo = params.get("src_model", [""])[0]
        sT = params.get("src_trim",  [""])[0]

        # 1) Try saved SourceMapping first
        src_key = normalize_key(sy, sm, sMo, sT)
        src_maps_df["__key__"] = src_maps_df.apply(lambda r: normalize_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1)
        hit = src_maps_df[src_maps_df["__key__"] == src_key]
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

    # /?api_token=...&save_source_mapping=true&src_year=...&src_make=...&src_model=...&src_trim=...&cad_year=...&cad_make=...&cad_model=...&cad_trim=...&code=...
    if "save_source_mapping" in params:
        sy   = params.get("src_year",  [""])[0]
        sm   = params.get("src_make",  [""])[0]
        sMo  = params.get("src_model", [""])[0]
        sT   = params.get("src_trim",  [""])[0]
        cy   = params.get("cad_year",  [""])[0]
        cm   = params.get("cad_make",  [""])[0]
        cMo  = params.get("cad_model", [""])[0]
        cT   = params.get("cad_trim",  [""])[0]
        code = params.get("code",       [""])[0]
        if not all([sy, sm, sMo, sT, cy, cm, cMo, cT, code]):
            st.json({"ok": False, "error": "missing params"}); st.stop()

        # Upsert by normalized source key
        src_key = normalize_key(sy, sm, sMo, sT)
        if not src_maps_df.empty:
            src_maps_df["__key__"] = src_maps_df.apply(lambda r: normalize_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1)
            src_maps_df = src_maps_df[src_maps_df["__key__"] != src_key].drop(columns=["__key__"], errors="ignore")

        new_row = {
            "src_year": sy, "src_make": sm, "src_model": sMo, "src_trim": sT,
            "cad_year": cy, "cad_make": cm, "cad_model": cMo, "cad_trim": cT,
            "model_code": code, "source": "api"
        }
        src_maps_df = pd.concat([src_maps_df, pd.DataFrame([new_row])], ignore_index=True)
        src_maps_df.to_csv(SRC_MAP_FILE, index=False)
        st.json({"ok": True, "message": "saved"}); st.stop()

    st.json({"error": "unknown api call"}); st.stop()

# -------------------------------
# UI MODE
# -------------------------------
st.set_page_config(page_title="External → CADS Vehicle Mapper (POC)", layout="wide")
st.title("🔒 External → CADS Vehicle Mapper (POC)")

pw = st.text_input("Enter password", type="password")
if pw != APP_PASSWORD:
    st.stop()
st.success("Authenticated ✅")

# External inputs
st.subheader("Search by External Vehicle string or External Y/M/M/T")
vehicle_text = st.text_input("External Vehicle (e.g., '2025 Land Rover Range Rover P400 SE SWB')")

py, pmake, pmodel, ptrim = parse_vehicle(vehicle_text, cads_df)

c1, c2, c3, c4 = st.columns(4)
with c1: src_year  = st.text_input("External Year",  py)
with c2: src_make  = st.text_input("External Make",  pmake)
with c3: src_model = st.text_input("External Model", pmodel)
with c4: src_trim  = st.text_input("External Trim",  ptrim)

threshold = st.slider("Fuzzy threshold (CADS fallback)", 60, 95, 80)

if st.button("Search / Resolve"):
    # 1) Try SourceMappings first
    src_key = normalize_key(src_year, src_make, src_model, src_trim)
    src_maps_df["__key__"] = src_maps_df.apply(lambda r: normalize_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1)
    hit = src_maps_df[src_maps_df["__key__"] == src_key]
    if not hit.empty:
        r = hit.iloc[0]
        st.success("Found saved Source Mapping ✅")
        st.write({
            "model_code": r["model_code"],
            "mapped_to": f"{r['cad_year']} {r['cad_make']} {r['cad_model']} {r['cad_trim']}"
        })
    else:
        # 2) CADS fallback: normalize external trim synonyms; exact → fuzzy
        sT_norm = normalize_external_trim(src_trim)
        exact = cads_df.copy()
        if src_year:  exact = exact[exact["ad_year"].apply(norm)  == norm(src_year)]
        if src_make:  exact = exact[exact["ad_make"].apply(norm)  == norm(src_make)]
        if src_model: exact = exact[exact["ad_model"].apply(norm) == norm(src_model)]
        if sT_norm:   exact = exact[exact["ad_trim"].apply(norm)  == norm(sT_norm)]

        results = exact if not exact.empty else fuzzy_filter_cads(cads_df, src_year, src_make, src_model, sT_norm, threshold)

        if results.empty:
            st.error("No CADS candidates found.")
        else:
            st.write(f"Found {len(results)} CADS candidate(s). Select the correct one, set Model Code if needed, and Save Mapping.")

            # Candidate viewer + selector
            view_cols = ["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]
            st.dataframe(results[view_cols], use_container_width=True)

            # Simple selector by index
            options = results.index.tolist()
            def fmt(i): 
                row = results.loc[i]
                return f"{row['ad_year']} / {row['ad_make']} / {row['ad_model']} / {row['ad_trim']} ({row['ad_mfgcode']})"
            sel_idx = st.selectbox("Pick CADS row", options, format_func=fmt)

            if sel_idx is not None:
                cad_row = results.loc[sel_idx]
                # Allow overriding code (optional)
                code = st.text_input("Model Code", value=cad_row["ad_mfgcode"])

                if st.button("💾 Save Source → CADS Mapping"):
                    # Upsert by normalized source key
                    if not src_maps_df.empty:
                        src_maps_df["__key__"] = src_maps_df.apply(
                            lambda r: normalize_key(r["src_year"], r["src_make"], r["src_model"], r["src_trim"]), axis=1
                        )
                        src_maps_df = src_maps_df[src_maps_df["__key__"] != src_key].drop(columns=["__key__"], errors="ignore")

                    new_row = {
                        "src_year": src_year, "src_make": src_make, "src_model": src_model, "src_trim": src_trim,
                        "cad_year": cad_row["ad_year"], "cad_make": cad_row["ad_make"],
                        "cad_model": cad_row["ad_model"], "cad_trim": cad_row["ad_trim"],
                        "model_code": code, "source": "ui"
                    }
                    src_maps_df = pd.concat([src_maps_df, pd.DataFrame([new_row])], ignore_index=True)
                    src_maps_df.to_csv(SRC_MAP_FILE, index=False)

                    st.success("Saved. This external Y/M/M/T will now return the mapped CADS code on the next search.")
                    # Show immediate confirmation
                    st.write({
                        "external": f"{src_year} {src_make} {src_model} {src_trim}",
                        "mapped_to": f"{cad_row['ad_year']} {cad_row['ad_make']} {cad_row['ad_model']} {cad_row['ad_trim']}",
                        "model_code": code
                    })

# Download current source mappings (persist manually by uploading to GitHub)
st.divider()
st.subheader("Source Mappings (external → CADS)")
st.download_button("Download SourceMappings.csv", src_maps_df.to_csv(index=False), "SourceMappings.csv", "text/csv")
