
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
# Load Data
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
    # API endpoint logic
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
            (cads_df["ad_make"] == make) &
            (cads_df["ad_model"] == model) &
            (cads_df["ad_trim"] == trim)
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

st.subheader("Select Vehicle")
years = sorted(cads_df["ad_year"].unique().tolist())
col1, col2, col3, col4 = st.columns(4)
with col1:
    sel_year = st.selectbox("Year", [""] + years)
filtered = cads_df[cads_df["ad_year"] == sel_year] if sel_year else cads_df
with col2:
    makes = sorted(filtered["ad_make"].unique().tolist())
    sel_make = st.selectbox("Make", [""] + makes)
filtered = filtered[filtered["ad_make"] == sel_make] if sel_make else filtered
with col3:
    models = sorted(filtered["ad_model"].unique().tolist())
    sel_model = st.selectbox("Model", [""] + models)
filtered = filtered[filtered["ad_model"] == sel_model] if sel_model else filtered
with col4:
    trims = sorted(filtered["ad_trim"].unique().tolist())
    sel_trim = st.selectbox("Trim", [""] + trims)

if not all([sel_year, sel_make, sel_model, sel_trim]):
    st.info("Select all fields to proceed.")
    st.stop()

target = cads_df[
    (cads_df["ad_year"] == sel_year) &
    (cads_df["ad_make"] == sel_make) &
    (cads_df["ad_model"] == sel_model) &
    (cads_df["ad_trim"] == sel_trim)
]

st.write("Matching CADS rows:")
st.dataframe(target[["ad_year","ad_make","ad_model","ad_trim","ad_mfgcode"]])

existing_code = next((c for c in target["ad_mfgcode"].tolist() if c.strip()), "")
new_code = st.text_input("Model Code", value=existing_code)

if st.button("💾 Save Mapping"):
    if not new_code.strip():
        st.error("Enter a model code.")
    else:
        maps_df = maps_df[
            ~((maps_df["year"] == sel_year) & (maps_df["make"] == sel_make) &
              (maps_df["model"] == sel_model) & (maps_df["trim"] == sel_trim))
        ]
        new_row = {"year": sel_year, "make": sel_make, "model": sel_model,
                   "trim": sel_trim, "model_code": new_code.strip(), "source": "user"}
        maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)
        maps_df.to_csv(MAP_FILE, index=False)
        st.success("Mapping saved to Mappings.csv.")
        st.download_button("Download Mappings.csv", maps_df.to_csv(index=False), "Mappings.csv", "text/csv")

