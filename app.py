
import os
import io
import json
import base64
import requests
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# -------------------------------
# Config from Streamlit Secrets
# -------------------------------
APP_PASSWORD  = os.getenv("APP_PASSWORD", "changeme")
API_TOKEN     = os.getenv("API_TOKEN", "secret")
CADS_FILE     = "CADS.csv"
MAP_FILE      = "Mappings.csv"
ADJ_FILE      = "Adjustments.csv"

GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO   = os.getenv("GITHUB_REPO", "")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

# -------------------------------
# Utilities
# -------------------------------
def load_csv_local(path, required=None):
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
    if adj_df is None or adj_df.empty:
        return cads_df
    adj_df["key"]  = adj_df.apply(lambda r: normalize_key(r.get("year"), r.get("make"), r.get("model"), r.get("trim")), axis=1)
    cads_df["key"] = cads_df.apply(lambda r: normalize_key(r.get("ad_year"), r.get("ad_make"), r.get("ad_model"), r.get("ad_trim")), axis=1)
    merged = cads_df.merge(adj_df[["key","new_trim","new_model","new_make","model_code"]], on="key", how="left")
    merged["ad_trim"]    = merged["new_trim"].fillna(merged["ad_trim"])
    merged["ad_model"]   = merged["new_model"].fillna(merged["ad_model"])
    merged["ad_make"]    = merged["new_make"].fillna(merged["ad_make"])
    merged["ad_mfgcode"] = merged["model_code"].fillna(merged["ad_mfgcode"])
    return merged.drop(columns=["key","new_trim","new_model","new_make","model_code"], errors="ignore")

def apply_mappings_to_cads(cads_df, maps_df):
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
    filtered = df.copy()
    if year:
        filtered = filtered[filtered["ad_year"] == year]
    candidates = []
    for idx, row in filtered.iterrows():
        score = 0; parts = 0
        if make:  score += fuzz.partial_ratio(make.lower(),  row["ad_make"].lower());  parts += 1
        if model: score += fuzz.partial_ratio(model.lower(), row["ad_model"].lower()); parts += 1
        if trim:  score += fuzz.partial_ratio(trim.lower(),  row["ad_trim"].lower());  parts += 1
        avg = score / (parts or 1)
        if avg >= threshold:
            candidates.append((idx, avg))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered.loc[[c[0] for c in candidates]]

# -------------------------------
# GitHub helpers
# -------------------------------
def gh_headers():
    return {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

def github_get_file_content(path):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    params = {"ref": GITHUB_BRANCH}
    r = requests.get(url, headers=gh_headers(), params=params)
    if r.status_code == 200:
        b64 = r.json().get("content", "")
        return base64.b64decode(b64)
    return None

def github_get_file_sha(path):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    params = {"ref": GITHUB_BRANCH}
    r = requests.get(url, headers=gh_headers(), params=params)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def github_put_file(path, content_bytes, message):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    sha = github_get_file_sha(path)
    data = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": GITHUB_BRANCH
    }
    if sha:
        data["sha"] = sha
    r = requests.put(url, headers=gh_headers(), data=json.dumps(data))
    return r.status_code in (200, 201), r.text

def github_load_csv(path, required=None):
    content = github_get_file_content(path)
    if content:
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]
        if required:
            missing = set(required) - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
        return df
    return None

# -------------------------------
# Load CADS, Adjustments, Mappings
# -------------------------------
cads_df = load_csv_local(CADS_FILE, required={"ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"})
if cads_df is None:
    st.error("CADS.csv not found."); st.stop()

adj_df = github_load_csv(ADJ_FILE)
cads_df = apply_adjustments(cads_df, adj_df)

maps_df = github_load_csv(MAP_FILE)
if maps_df is None:
    maps_df = pd.DataFrame(columns=["year","make","model","trim","model_code","source"])

cads_df = apply_mappings_to_cads(cads_df, maps_df)

# -------------------------------
# API Mode for Mozenda
# -------------------------------
params = st.experimental_get_query_params()
if params.get("api_token", [""])[0] == API_TOKEN:
    if "get_model_code" in params:
        year = params.get("year", [""])[0]
        make = params.get("make", [""])[0]
        model = params.get("model", [""])[0]
        trim = params.get("trim", [""])[0]
        match = fuzzy_filter(cads_df, year, make, model, trim)
        if not match.empty:
            st.json({"model_code": match.iloc[0]["ad_mfgcode"]})
        else:
            st.json({"model_code": ""})
        st.stop()

    if "save_mapping" in params:
        year = params.get("year", [""])[0]
        make = params.get("make", [""])[0]
        model = params.get("model", [""])[0]
        trim = params.get("trim", [""])[0]
        code = params.get("code", [""])[0]
        if not all([year, make, model, trim, code]):
            st.json({"ok": False, "error": "missing params"}); st.stop()
        maps_df = maps_df[
            ~((maps_df["year"] == year) &
              (maps_df["make"].str.lower() == make.lower()) &
              (maps_df["model"].str.lower() == model.lower()) &
              (maps_df["trim"].str.lower() == trim.lower()))
        ]
        maps_df = pd.concat([maps_df, pd.DataFrame([{
            "year": year, "make": make, "model": model, "trim": trim, "model_code": code, "source": "api"
        }])], ignore_index=True)
        ok, msg = github_put_file(MAP_FILE, maps_df.to_csv(index=False).encode("utf-8"), f"API save mapping")
        st.json({"ok": ok, "message": msg}); st.stop()

# -------------------------------
# UI Mode
# -------------------------------
st.set_page_config(page_title="CADS Mapper", layout="wide")
st.title("🔒 CADS Vehicle Mapper")

pw = st.text_input("Enter password", type="password")
if pw != APP_PASSWORD:
    st.stop()

st.success("Authenticated ✅")

st.subheader("Search by Vehicle or Y/M/M/T")
vehicle_input = st.text_input("Vehicle (e.g., '2025 Land Rover Range Rover P400 SE SWB')")

yr, mk, mdl, trm = "", "", "", ""
parts = vehicle_input.split()
if parts and parts[0].isdigit():
    yr = parts[0]; parts = parts[1:]
if parts: mk = parts[0]
if len(parts) > 1: mdl = parts[1]
if len(parts) > 2: trm = " ".join(parts[2:])

c1, c2, c3, c4 = st.columns(4)
with c1: sel_year = st.text_input("Year", yr)
with c2: sel_make = st.text_input("Make", mk)
with c3: sel_model = st.text_input("Model", mdl)
with c4: sel_trim = st.text_input("Trim", trm)

threshold = st.slider("Fuzzy threshold", 60, 95, 80)

if st.button("Search"):
    exact = cads_df.copy()
    if sel_year: exact = exact[exact["ad_year"] == sel_year]
    if sel_make: exact = exact[exact["ad_make"].str.lower() == sel_make.lower()]
    if sel_model: exact = exact[exact["ad_model"].str.lower() == sel_model.lower()]
    if sel_trim: exact = exact[exact["ad_trim"].str.lower() == sel_trim.lower()]
    results = exact if not exact.empty else fuzzy_filter(cads_df, sel_year, sel_make, sel_model, sel_trim, threshold)

    if results.empty:
        st.error("No matches found.")
    else:
        st.write(f"Found {len(results)} match(es). Edit Model Code and click Save:")
        editable = results[["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]].copy()
        editable.rename(columns={"ad_mfgcode": "model_code"}, inplace=True)
        edited = st.data_editor(editable, num_rows="dynamic", use_container_width=True)

        if st.button("💾 Save All Changes"):
            for _, row in edited.iterrows():
                y, m, mo, t, code = row["ad_year"], row["ad_make"], row["ad_model"], row["ad_trim"], row["model_code"]
                maps_df = maps_df[
                    ~((maps_df["year"] == y) &
                      (maps_df["make"].str.lower() == m.lower()) &
                      (maps_df["model"].str.lower() == mo.lower()) &
                      (maps_df["trim"].str.lower() == t.lower()))
                ]
                maps_df = pd.concat([maps_df, pd.DataFrame([{
                    "year": y, "make": m, "model": mo, "trim": t, "model_code": code, "source": "ui"
                }])], ignore_index=True)
            ok, msg = github_put_file(MAP_FILE, maps_df.to_csv(index=False).encode("utf-8"), f"UI save mappings")
            if ok:
                st.success("Mappings committed to GitHub. They will be remembered.")
                cads_df[:] = apply_mappings_to_cads(cads_df, maps_df)
            else:
                st.error(f"Failed to persist mappings: {msg}")

st.divider()
st.subheader("Current mappings (loaded)")
st.download_button("Download Mappings.csv", maps_df.to_csv(index=False), "Mappings.csv", "text/csv")
