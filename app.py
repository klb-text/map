
import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
import os

# -------------------------------
# Hardcoded Config for POC
# -------------------------------
APP_PASSWORD = "mypassword"       # UI login password
API_TOKEN = "mozenda-token"       # Token for Mozenda API calls
CADS_FILE = "CADS.csv"
MAP_FILE = "Mappings.csv"

# -------------------------------
# Helpers
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

def fuzzy_filter(df, year, make, model, trim, threshold=80):
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
# Load CADS and Mappings
# -------------------------------
cads_df = load_csv(CADS_FILE, required={"ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"})
if cads_df is None:
    st.error("CADS.csv not found. Upload it to the repo and rerun.")
    st.stop()

maps_df = load_csv(MAP_FILE)
if maps_df is None:
    maps_df = pd.DataFrame(columns=["year","make","model","trim","model_code"])

# Apply mappings to CADS
for _, row in maps_df.iterrows():
    mask = (
        (cads_df["ad_year"] == row["year"]) &
        (cads_df["ad_make"].str.lower() == row["make"].lower()) &
        (cads_df["ad_model"].str.lower() == row["model"].lower()) &
        (cads_df["ad_trim"].str.lower() == row["trim"].lower())
    )
    cads_df.loc[mask, "ad_mfgcode"] = row["model_code"]

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

# -------------------------------
# UI Mode
# -------------------------------
st.set_page_config(page_title="CADS Mapper", layout="wide")
st.title("🔒 CADS Vehicle Mapper")

pw = st.text_input("Enter password", type="password")
if pw != APP_PASSWORD:
    st.stop()

st.success("Authenticated ✅")

# -------------------------------
# Input Fields
# -------------------------------
st.subheader("Search by Vehicle or Y/M/M/T")
vehicle_input = st.text_input("Vehicle (e.g., '2025 Ford F-150 XL')")

year_guess, make_guess, model_guess, trim_guess = "", "", "", ""
parts = vehicle_input.split()
if parts and parts[0].isdigit():
    year_guess = parts[0]; parts = parts[1:]
if parts:
    make_guess = parts[0]
if len(parts) > 1:
    model_guess = parts[1]
if len(parts) > 2:
    trim_guess = " ".join(parts[2:])

c1, c2, c3, c4 = st.columns(4)
with c1: sel_year = st.text_input("Year", year_guess)
with c2: sel_make = st.text_input("Make", make_guess)
with c3: sel_model = st.text_input("Model", model_guess)
with c4: sel_trim = st.text_input("Trim", trim_guess)

threshold = st.slider("Fuzzy threshold", 60, 95, 80)

if st.button("Search"):
    filtered = cads_df.copy()
    if sel_year: filtered = filtered[filtered["ad_year"] == sel_year]
    if sel_make: filtered = filtered[filtered["ad_make"].str.lower() == sel_make.lower()]
    if sel_model: filtered = filtered[filtered["ad_model"].str.lower() == sel_model.lower()]
    if sel_trim: filtered = filtered[filtered["ad_trim"].str.lower() == sel_trim.lower()]

    if filtered.empty:
        st.warning("No exact matches found. Trying fuzzy…")
        filtered = fuzzy_filter(cads_df, sel_year, sel_make, sel_model, sel_trim, threshold)

    if filtered.empty:
        st.error("No matches found.")
    else:
        st.write(f"Found {len(filtered)} match(es): edit Model Code and save.")
        editable_df = filtered[["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]].copy()
        editable_df.rename(columns={"ad_mfgcode": "model_code"}, inplace=True)

        edited = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

        if st.button("💾 Save All Changes"):
            for _, row in edited.iterrows():
                maps_df = maps_df[
                    ~((maps_df["year"] == row["ad_year"]) &
                      (maps_df["make"].str.lower() == row["ad_make"].lower()) &
                      (maps_df["model"].str.lower() == row["ad_model"].lower()) &
                      (maps_df["trim"].str.lower() == row["ad_trim"].lower()))
                ]
                new_row = {
                    "year": row["ad_year"],
                    "make": row["ad_make"],
                    "model": row["ad_model"],
                    "trim": row["ad_trim"],
                    "model_code": row["model_code"]
                }
                maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)
            maps_df.to_csv(MAP_FILE, index=False)
            st.success("Mappings saved locally. Download below.")

# Always show download button
st.download_button("Download Current Mappings.csv", maps_df.to_csv(index=False), "Mappings.csv", "text/csv")
