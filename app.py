
import os
import json
import base64
import requests
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# -------------------------------
# Config (from secrets/env)
# -------------------------------
PASSWORD       = os.getenv("APP_PASSWORD", "changeme")
API_TOKEN      = os.getenv("API_TOKEN", "secret")
CADS_FILE      = "CADS.csv"
MAP_FILE       = "Mappings.csv"          # path inside repo
ADJ_FILE       = "Adjustments.csv"       # optional adjustments file
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO    = os.getenv("GITHUB_REPO", "")     # e.g., "kevinblamer/map"
GITHUB_BRANCH  = os.getenv("GITHUB_BRANCH", "main")

# -------------------------------
# Loaders & Utilities
# -------------------------------
def load_csv(path, required=None):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if required:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df

def normalize_key(year, make, model, trim):
    return "|".join([str(x or "").strip().lower() for x in [year, make, model, trim]])

def apply_adjustments(cads_df, adj_df):
    """Reapply your historical cleanup (names/codes) every time CADS updates."""
    if adj_df is None or adj_df.empty:
        return cads_df
    adj_df["key"] = adj_df.apply(lambda r: normalize_key(r.get("year"), r.get("make"), r.get("model"), r.get("trim")), axis=1)
    cads_df["key"] = cads_df.apply(lambda r: normalize_key(r.get("ad_year"), r.get("ad_make"), r.get("ad_model"), r.get("ad_trim")), axis=1)
    merged = cads_df.merge(adj_df[["key","new_trim","new_model","new_make","model_code"]], on="key", how="left")
    merged["ad_trim"]    = merged["new_trim"].fillna(merged["ad_trim"])
    merged["ad_model"]   = merged["new_model"].fillna(merged["ad_model"])
    merged["ad_make"]    = merged["new_make"].fillna(merged["ad_make"])
    merged["ad_mfgcode"] = merged["model_code"].fillna(merged["ad_mfgcode"])
    return merged.drop(columns=["key","new_trim","new_model","new_make","model_code"], errors="ignore")

def apply_mappings_to_cads(cads_df, maps_df):
    """Overlay saved mappings onto CADS so Model Codes appear in searches."""
    if maps_df is None or maps_df.empty:
        return cads_df
    for _, row in maps_df.iterrows():
        mask = (
            (cads_df["ad_year"] == row.get("year", "")) &
            (cads_df["ad_make"].str.lower()  == row.get("make", "").lower()) &
            (cads_df["ad_model"].str.lower() == row.get("model", "").lower()) &
            (cads_df["ad_trim"].str.lower()  == row.get("trim", "").lower())
        )
        cads_df.loc[mask, "ad_mfgcode"] = row.get("model_code", "")
    return cads_df

def fuzzy_filter(df, year, make, model, trim, threshold=80):
    """Return fuzzy-matched rows when exact match fails."""
    filtered = df.copy()
    if year:
        filtered = filtered[filtered["ad_year"] == year]
    candidates = []
    for idx, row in filtered.iterrows():
        score = 0
        parts = 0
        if make:
            score += fuzz.partial_ratio(make.lower(), row["ad_make"].lower()); parts += 1
        if model:
            score += fuzz.partial_ratio(model.lower(), row["ad_model"].lower()); parts += 1
        if trim:
            score += fuzz.partial_ratio(trim.lower(), row["ad_trim"].lower()); parts += 1
        avg = score / (parts or 1)
        if avg >= threshold:
            candidates.append((idx, avg))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered.loc[[c[0] for c in candidates]]

# -------------------------------
# GitHub persistence (commit Mappings.csv)
# -------------------------------
def github_get_file_sha(path):
    """Fetch current SHA of a file in repo (needed to update). Returns None if not found."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return None
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    params = {"ref": GITHUB_BRANCH}
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def github_put_file(path, content_bytes, message):
    """Create/update a file in the repo on the configured branch."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return False, "Missing GITHUB_TOKEN or GITHUB_REPO"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    sha = github_get_file_sha(path)
    data = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": GITHUB_BRANCH
    }
    if sha:  # update
        data["sha"] = sha
    r = requests.put(url, headers=headers, data=json.dumps(data))
    if r.status_code in (200, 201):
        return True, "Committed to GitHub."
    return False, f"GitHub commit failed: {r.status_code} {r.text}"

# -------------------------------
# Load CADS, Adjustments, Mappings
# -------------------------------
cads_df = load_csv(CADS_FILE, required={"ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"})
if cads_df is None:
    st.error("CADS.csv not found. Add it to the repo root and rerun.")
    st.stop()

adj_df  = load_csv(ADJ_FILE)  # optional
cads_df = apply_adjustments(cads_df, adj_df)

maps_df = load_csv(MAP_FILE)
if maps_df is None:
    maps_df = pd.DataFrame(columns=["year","make","model","trim","model_code","source"])

# Apply mappings on top of CADS so remembered codes show up
cads_df = apply_mappings_to_cads(cads_df, maps_df)

# -------------------------------
# API Mode for Mozenda (JSON)
# -------------------------------
params = st.experimental_get_query_params()
if params.get("api_token", [""])[0] == API_TOKEN:
    # /?api_token=...&options=true
    if "options" in params:
        years  = sorted(cads_df["ad_year"].unique().tolist())
        makes  = sorted(cads_df["ad_make"].unique().tolist())
        models = sorted(cads_df["ad_model"].unique().tolist())
        trims  = sorted(cads_df["ad_trim"].unique().tolist())
        st.json({"years": years, "makes": makes, "models": models, "trims": trims}); st.stop()

    # /?api_token=...&find=true&year=...&make=...&model=...&trim=...
    if "find" in params:
        year  = params.get("year",  [""])[0]
        make  = params.get("make",  [""])[0]
        model = params.get("model", [""])[0]
        trim  = params.get("trim",  [""])[0]
        threshold = int(params.get("threshold", ["80"])[0])
        # exact, then fuzzy
        exact = cads_df.copy()
        if year:  exact = exact[exact["ad_year"] == year]
        if make:  exact = exact[exact["ad_make"].str.lower()  == make.lower()]
        if model: exact = exact[exact["ad_model"].str.lower() == model.lower()]
        if trim:  exact = exact[exact["ad_trim"].str.lower()  == trim.lower()]
        result = exact if not exact.empty else fuzzy_filter(cads_df, year, make, model, trim, threshold)
        st.json(result.to_dict(orient="records")); st.stop()

    # /?api_token=...&get_model_code=true&year=...&make=...&model=...&trim=...
    if "get_model_code" in params:
        year  = params.get("year",  [""])[0]
        make  = params.get("make",  [""])[0]
        model = params.get("model", [""])[0]
        trim  = params.get("trim",  [""])[0]
        exact = cads_df.copy()
        if year:  exact = exact[exact["ad_year"] == year]
        if make:  exact = exact[exact["ad_make"].str.lower()  == make.lower()]
        if model: exact = exact[exact["ad_model"].str.lower() == model.lower()]
        if trim:  exact = exact[exact["ad_trim"].str.lower()  == trim.lower()]
        if not exact.empty:
            code = exact.iloc[0]["ad_mfgcode"]
            st.json({"model_code": code, "source": "cads+mapping"}); st.stop()
        # fallback fuzzy
        fuzzy = fuzzy_filter(cads_df, year, make, model, trim)
        if not fuzzy.empty:
            code = fuzzy.iloc[0]["ad_mfgcode"]
            st.json({"model_code": code, "source": "fuzzy"}); st.stop()
        st.json({"model_code": "", "source": "none"}); st.stop()

    # /?api_token=...&save_mapping=true&year=...&make=...&model=...&trim=...&code=...
    if "save_mapping" in params:
        year  = params.get("year",  [""])[0]
        make  = params.get("make",  [""])[0]
        model = params.get("model", [""])[0]
        trim  = params.get("trim",  [""])[0]
        code  = params.get("code",  [""])[0]
        if not all([year, make, model, trim, code]):
            st.json({"ok": False, "error": "missing parameters"}); st.stop()
        # upsert in memory
        maps_df = maps_df[
            ~((maps_df["year"] == year) &
              (maps_df["make"].str.lower()  == make.lower()) &
              (maps_df["model"].str.lower() == model.lower()) &
              (maps_df["trim"].str.lower()  == trim.lower()))
        ]
        new_row = {"year": year, "make": make, "model": model, "trim": trim, "model_code": code, "source": "api"}
        maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)

        # persist to GitHub
        csv_bytes = maps_df.to_csv(index=False).encode("utf-8")
        ok, msg = github_put_file(MAP_FILE, csv_bytes, f"API save mapping {year}/{make}/{model}/{trim}")
        st.json({"ok": ok, "message": msg}); st.stop()

# -------------------------------
# UI Mode (Human dashboard)
# -------------------------------
st.set_page_config(page_title="Private CADS Mapper", layout="wide")
st.title("🔒 CADS Vehicle Mapper (POC)")

pw = st.text_input("Enter password", type="password")
if pw != PASSWORD:
    st.stop()
st.success("Authenticated ✅")

st.subheader("Search by Vehicle string or Y/M/M/T")
vehicle_input = st.text_input("Vehicle (e.g., '2025 Land Rover Range Rover P400 SE SWB')")

# naive parse + manual fields (you can improve this later with multi-word detection)
year_guess, make_guess, model_guess, trim_guess = "", "", "", ""
parts = vehicle_input.split()
if parts and parts[0].isdigit():
    year_guess = parts[0]; parts = parts[1:]
if parts:
    make_guess  = parts[0]
if len(parts) > 1:
    model_guess = parts[1]
if len(parts) > 2:
    trim_guess  = " ".join(parts[2:])

c1, c2, c3, c4 = st.columns(4)
with c1: sel_year  = st.text_input("Year",  year_guess)
with c2: sel_make  = st.text_input("Make",  make_guess)
with c3: sel_model = st.text_input("Model", model_guess)
with c4: sel_trim  = st.text_input("Trim",  trim_guess)

threshold = st.slider("Fuzzy threshold", 60, 95, 80)

if st.button("Search"):
    # exact match first
    filtered = cads_df.copy()
    if sel_year:  filtered = filtered[filtered["ad_year"] == sel_year]
    if sel_make:  filtered = filtered[filtered["ad_make"].str.lower()  == sel_make.lower()]
    if sel_model: filtered = filtered[filtered["ad_model"].str.lower() == sel_model.lower()]
    if sel_trim:  filtered = filtered[filtered["ad_trim"].str.lower()  == sel_trim.lower()]

    if filtered.empty:
        st.warning("No exact matches found. Trying fuzzy…")
        filtered = fuzzy_filter(cads_df, sel_year, sel_make, sel_model, sel_trim, threshold)

    if filtered.empty:
        st.error("No matches found.")
    else:
        st.write(f"Found {len(filtered)} match(es): edit Model Code and save.")
        editable_df = filtered[["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]].copy()
        editable_df.rename(columns={"ad_mfgcode": "model_code"}, inplace=True)

        edited = st.data_editor(
            editable_df,
            num_rows="dynamic",
            use_container_width=True,
            key="editor"
        )

        if st.button("💾 Save All Changes"):
            # upsert each edited row into mappings
            for _, row in edited.iterrows():
                y, m, mo, t, code = row["ad_year"], row["ad_make"], row["ad_model"], row["ad_trim"], row["model_code"]
                maps_df = maps_df[
                    ~((maps_df["year"] == y) &
                      (maps_df["make"].str.lower()  == m.lower()) &
                      (maps_df["model"].str.lower() == mo.lower()) &
                      (maps_df["trim"].str.lower()  == t.lower()))
                ]
                new_row = {"year": y, "make": m, "model": mo, "trim": t, "model_code": code, "source": "ui"}
                maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)

            # persist to GitHub
            csv_bytes = maps_df.to_csv(index=False).encode("utf-8")
            ok, msg = github_put_file(MAP_FILE, csv_bytes, f"UI save mappings ({len(edited)})")
            if ok:
                st.success("Mappings persisted to GitHub. They will be remembered on next run.")
                # Immediately apply them in-session so a second search shows updated codes
                cads_df[:] = apply_mappings_to_cads(cads_df, maps_df)
            else:
                st.error(f"Failed to persist mappings to GitHub: {msg}")

# Always show a download of currently loaded mappings (from memory)
st.divider()
st.subheader("Current Mappings (loaded)")
st.download_button("Download Mappings.csv", maps_df.to_csv(index=False), "Mappings.csv", "text/csv")
st.caption("Note: For persistence on Streamlit Cloud, we commit Mappings.csv back to the GitHub repo via your token.")
