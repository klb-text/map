
import streamlit as st
import pandas as pd
import os

# -------------------------------
# Config
# -------------------------------
PASSWORD = os.getenv("APP_PASSWORD", "changeme")
API_TOKEN = os.getenv("API_TOKEN", "secret")
CADS_FILE = "CADS.csv"
MAP_FILE = "Mappings.csv"
ADJ_FILE = "Adjustments.csv"

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

def normalize_key(year, make, model, trim):
    return "|".join([str(x or "").strip().lower() for x in [year, make, model, trim]])

def apply_adjustments(cads_df, adj_df):
    if adj_df is None or adj_df.empty:
        return cads_df
    adj_df["key"] = adj_df.apply(lambda r: normalize_key(r["year"], r["make"], r["model"], r["trim"]), axis=1)
    cads_df["key"] = cads_df.apply(lambda r: normalize_key(r["ad_year"], r["ad_make"], r["ad_model"], r["ad_trim"]), axis=1)
    merged = cads_df.merge(adj_df[["key","new_trim","new_model","new_make","model_code"]], on="key", how="left")
    merged["ad_trim"] = merged["new_trim"].fillna(merged["ad_trim"])
    merged["ad_model"] = merged["new_model"].fillna(merged["ad_model"])
    merged["ad_make"] = merged["new_make"].fillna(merged["ad_make"])
    merged["ad_mfgcode"] = merged["model_code"].fillna(merged["ad_mfgcode"])
    return merged.drop(columns=["key","new_trim","new_model","new_make","model_code"], errors="ignore")

# -------------------------------
# Load Data
# -------------------------------
cads_df = load_csv(CADS_FILE, required={"ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"})
if cads_df is None:
    st.error("Upload CADS.csv to the repo and refresh.")
    st.stop()

adj_df = load_csv(ADJ_FILE)
cads_df = apply_adjustments(cads_df, adj_df)

maps_df = load_csv(MAP_FILE)
if maps_df is None:
    maps_df = pd.DataFrame(columns=["year","make","model","trim","model_code","source"])

# -------------------------------
# API Mode for Mozenda
# -------------------------------
params = st.experimental_get_query_params()
if params.get("api_token", [""])[0] == API_TOKEN:
    if "options" in params:
        years = sorted(cads_df["ad_year"].unique().tolist())
        makes = sorted(cads_df["ad_make"].unique().tolist())
        models = sorted(cads_df["ad_model"].unique().tolist())
        trims = sorted(cads_df["ad_trim"].unique().tolist())
        st.json({"years": years, "makes": makes, "models": models, "trims": trims})
        st.stop()
    elif "mapping" in params:
        year = params.get("year", [""])[0]
        make = params.get("make", [""])[0]
        model = params.get("model", [""])[0]
        trim = params.get("trim", [""])[0]
        match = cads_df[
            (cads_df["ad_year"] == year) &
            (cads_df["ad_make"].str.lower() == make.lower()) &
            (cads_df["ad_model"].str.lower() == model.lower()) &
            (cads_df["ad_trim"].str.lower() == trim.lower())
        ]
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
# Vehicle Text Input
# -------------------------------
st.subheader("Search by Vehicle String or Y/M/M/T")
vehicle_input = st.text_input("Enter Vehicle (e.g., '2025 Ford F-150 XL')")

parsed_year, parsed_make, parsed_model, parsed_trim = "", "", "", ""
if vehicle_input.strip():
    parts = vehicle_input.split()
    parsed_year = parts[0] if parts and parts[0].isdigit() else ""
    if len(parts) > 1: parsed_make = parts[1]
    if len(parts) > 2: parsed_model = parts[2]
    if len(parts) > 3: parsed_trim = parts[3]

# -------------------------------
# Dropdowns for fallback
# -------------------------------
years = sorted(cads_df["ad_year"].unique().tolist())
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
        st.warning("No matches found.")
    else:
        st.write(f"Found {len(filtered)} match(es):")
        st.dataframe(filtered[["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]])

        # Allow mapping update if single match
        if len(filtered) == 1:
            existing_code = filtered.iloc[0]["ad_mfgcode"]
            new_code = st.text_input("Model Code", value=existing_code)
            if st.button("💾 Save Mapping"):
                maps_df = maps_df[
                    ~((maps_df["year"] == sel_year) & (maps_df["make"].str.lower() == sel_make.lower()) &
                      (maps_df["model"].str.lower() == sel_model.lower()) & (maps_df["trim"].str.lower() == sel_trim.lower()))
                ]
                new_row = {"year": sel_year, "make": sel_make, "model": sel_model,
                           "trim": sel_trim, "model_code": new_code.strip(), "source": "user"}
                maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)
                maps_df.to_csv(MAP_FILE, index=False)
                st.success("Mapping saved to Mappings.csv.")
                st.download_button("Download Mappings.csv", maps_df.to_csv(index=False), "Mappings.csv", "text/csv")
