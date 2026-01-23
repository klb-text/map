import streamlit as st
import pandas as pd
import os
from rapidfuzz import process, fuzz

# ---------------------- File Paths ----------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# ---------------------- Load CSV / TXT ----------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        pd.DataFrame().to_csv(path, index=False)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, delimiter="\t")

# ---------------------- Load Data ----------------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)
if os.path.exists(MAPPINGS_FILE):
    mappings_df = pd.read_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["Vehicle", "Year", "Make", "Model", "Trim", "STYLE_ID"])

# ---------------------- Streamlit UI ----------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle Name (freeform)")

selected_vehicle = None
manual_ymmt = {}

# ---------------------- Smart Match ----------------------
if vehicle_input:
    choices = vehicle_ref_df["Vehicle"].tolist()
    match, score, idx = process.extractOne(vehicle_input, choices, scorer=fuzz.token_sort_ratio)
    st.write(f"Smart match found: **{match}** (score {score})")
    confirm = st.button("Confirm Smart Match")
    if confirm:
        selected_vehicle = match
        ref_row = vehicle_ref_df.iloc[idx]
        manual_ymmt = {
            "Year": ref_row.get("Year", ""),
            "Make": ref_row.get("Make", ""),
            "Model": ref_row.get("Model", ""),
            "Trim": ref_row.get("Trim", "")
        }

# ---------------------- Manual YMMT (horizontal) ----------------------
st.subheader("Manual YMMT Entry (if smart match not correct)")
with st.form("manual_ymmt_form"):
    col1, col2, col3, col4 = st.columns(4)
    manual_year = col1.text_input("Year", value=manual_ymmt.get("Year", ""))
    manual_make = col2.text_input("Make", value=manual_ymmt.get("Make", ""))
    manual_model = col3.text_input("Model", value=manual_ymmt.get("Model", ""))
    manual_trim = col4.text_input("Trim", value=manual_ymmt.get("Trim", ""))
    apply_ymmt = st.form_submit_button("Apply Manual YMMT")
    if apply_ymmt:
        selected_vehicle = vehicle_input
        manual_ymmt = {
            "Year": manual_year.strip(),
            "Make": manual_make.strip(),
            "Model": manual_model.strip(),
            "Trim": manual_trim.strip()
        }

# ---------------------- Filter CADS ----------------------
filtered_cads = cads_df.copy()
if selected_vehicle:
    if manual_ymmt.get("Year"):
        filtered_cads = filtered_cads[filtered_cads["MODEL_YEAR"].astype(str).str.contains(str(manual_ymmt["Year"]), na=False)]
    if manual_ymmt.get("Make"):
        filtered_cads = filtered_cads[filtered_cads["DIVISION_NAME"].astype(str).str.contains(str(manual_ymmt["Make"]), na=False)]
    if manual_ymmt.get("Model"):
        filtered_cads = filtered_cads[filtered_cads["MODEL_NAME"].astype(str).str.contains(str(manual_ymmt["Model"]), na=False)]
    if manual_ymmt.get("Trim"):
        trim = str(manual_ymmt["Trim"])
        filtered_cads = filtered_cads[filtered_cads["STYLE_NAME"].astype(str).str.contains(trim, na=False)]

# ---------------------- Display table with checkboxes ----------------------
st.subheader("Filtered CADS Rows")
if not filtered_cads.empty:
    # Add checkbox column
    filtered_cads["Select"] = False
    for idx, row in filtered_cads.iterrows():
        key = f"select_{idx}"
        filtered_cads.at[idx, "Select"] = st.checkbox(
            f"{row['DIVISION_NAME']} {row['MODEL_NAME']} {row['STYLE_NAME']}",
            key=key
        )
    st.dataframe(filtered_cads.drop(columns=["Select"]))
else:
    st.write("No CADS rows matched your selection.")

# ---------------------- Save Mapping ----------------------
selected_rows = filtered_cads[filtered_cads["Select"]]
if st.button("Save Mapping"):
    if selected_rows.empty:
        st.warning("No rows selected to save.")
    else:
        new_mappings = selected_rows.copy()
        new_mappings["Vehicle"] = selected_vehicle
        new_mappings["Year"] = manual_ymmt.get("Year", "")
        new_mappings["Make"] = manual_ymmt.get("Make", "")
        new_mappings["Model"] = manual_ymmt.get("Model", "")
        new_mappings["Trim"] = manual_ymmt.get("Trim", "")
        new_mappings = new_mappings[["Vehicle", "Year", "Make", "Model", "Trim", "STYLE_ID"]]

        mappings_df = pd.concat([mappings_df, new_mappings], ignore_index=True)
        mappings_df.drop_duplicates(subset=["Vehicle", "STYLE_ID"], inplace=True)
        mappings_df.to_csv(MAPPINGS_FILE, index=False)
        st.success(f"Saved {len(new_mappings)} mappings!")

st.subheader("Total Mappings")
st.write(len(mappings_df))
