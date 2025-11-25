
import streamlit as st
import pandas as pd
import os
from rapidfuzz import fuzz

# -------------------------------
# Config
# -------------------------------
PASSWORD = os.getenv("APP_PASSWORD", "changeme")
API_TOKEN = os.getenv("API_TOKEN", "secret")
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
        if make:
            score += fuzz.partial_ratio(make.lower(), row["ad_make"].lower())
        if model:
            score += fuzz.partial_ratio(model.lower(), row["ad_model"].lower())
        if trim:
            score += fuzz.partial_ratio(trim.lower(), row["ad_trim"].lower())
        avg_score = score / (sum(bool(x) for x in [make, model, trim]) or 1)
        if avg_score >= threshold:
            candidates.append((idx, avg_score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return filtered.loc[[c[0] for c in candidates]]

# -------------------------------
# Load CADS and Mappings
# -------------------------------
cads_df = load_csv(CADS_FILE, required={"ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"})
if cads_df is None:
    st.error("Upload CADS.csv to the repo and refresh.")
    st.stop()

maps_df = load_csv(MAP_FILE)
if maps_df is None:
    maps_df = pd.DataFrame(columns=["year","make","model","trim","model_code"])

# Apply previous mappings to CADS
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
    if "mapping" in params:
        year = params.get("year", [""])[0]
        make = params.get("make", [""])[0]
        model = params.get("model", [""])[0]
        trim = params.get("trim", [""])[0]
        match = fuzzy_filter(cads_df, year, make, model, trim)
        st.json(match.to_dict(orient="records"))
        st.stop()

# -------------------------------
# UI Mode
# -------------------------------
st.set_page_config(page_title="Private CADS Mapper", layout="wide")
st.title("🔒 Private CADS Mapper")

pw = st.text_input("Enter password", type="password")
if pw != PASSWORD:
    st.stop()

st.success("Authenticated ✅")

# -------------------------------
# Input Fields
# -------------------------------
st.subheader("Search by Vehicle or Y/M/M/T")
vehicle_input = st.text_input("Enter Vehicle (e.g., '2025 Ford F-150 XL')")

parsed_year, parsed_make, parsed_model, parsed_trim = "", "", "", ""
if vehicle_input.strip():
    parts = vehicle_input.split()
    parsed_year = parts[0] if parts and parts[0].isdigit() else ""
    if len(parts) > 1: parsed_make = parts[1]
    if len(parts) > 2: parsed_model = parts[2]
    if len(parts) > 3: parsed_trim = parts[3]

col1, col2, col3, col4 = st.columns(4)
with col1:
    sel_year = st.text_input("Year", parsed_year)
with col2:
    sel_make = st.text_input("Make", parsed_make)
with col3:
    sel_model = st.text_input("Model", parsed_model)
with col4:
    sel_trim = st.text_input("Trim", parsed_trim)

# -------------------------------
# Search Logic
# -------------------------------
if st.button("Search"):
    filtered = cads_df.copy()
    if sel_year: filtered = filtered[filtered["ad_year"] == sel_year]
    if sel_make: filtered = filtered[filtered["ad_make"].str.lower() == sel_make.lower()]
    if sel_model: filtered = filtered[filtered["ad_model"].str.lower() == sel_model.lower()]
    if sel_trim: filtered = filtered[filtered["ad_trim"].str.lower() == sel_trim.lower()]

    if filtered.empty:
        st.warning("No exact matches found. Trying fuzzy match...")
        filtered = fuzzy_filter(cads_df, sel_year, sel_make, sel_model, sel_trim)

    if filtered.empty:
        st.error("No matches found even with fuzzy matching.")
    else:
        st.write(f"Found {len(filtered)} match(es):")
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
            st.success("All changes saved to Mappings.csv.")
            st.download_button("Download Mappings.csv", maps_df.to_csv(index=False), "Mappings.csv", "text/csv")
