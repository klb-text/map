import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# ---------------------- File paths ----------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# ---------------------- Safe loader ----------------------
@st.cache_data
def load_vehicle_reference(path):
    """
    Loads vehicle reference from TXT or CSV safely.
    TXT = one vehicle per line
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            vehicles = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"Vehicle": vehicles})

    # CSV fallback
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df.columns = ["Vehicle"]
    return df

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# ---------------------- Load data ----------------------
vehicle_ref_df = load_vehicle_reference(VEHICLE_REF_FILE)
cads_df = load_csv(CADS_FILE)

if os.path.exists(MAPPINGS_FILE):
    mappings_df = load_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(
        columns=["Vehicle","Year","Make","Model","Trim","Model Code","STYLE_ID"]
    )

# ---------------------- UI ----------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")
search_clicked = st.button("Search")

matched_vehicle = None
matches_df = pd.DataFrame()

# ---------------------- Search logic ----------------------
if search_clicked and vehicle_input:
    choices = vehicle_ref_df["Vehicle"].tolist()
    result = process.extractOne(
        vehicle_input, choices, scorer=fuzz.WRatio, score_cutoff=70
    )

    if result:
        matched_vehicle, score, _ = result
        st.success(f"Smart match found: {matched_vehicle} (score {score:.1f})")
    else:
        st.warning("No close vehicle match found.")

# ---------------------- YMMT (left → right) ----------------------
st.subheader("YMMT Filter (optional)")

c1, c2, c3, c4 = st.columns(4)
year = c1.text_input("Year")
make = c2.text_input("Make")
model = c3.text_input("Model")
trim = c4.text_input("Trim")

# ---------------------- Filter CADS ----------------------
if search_clicked:
    matches_df = cads_df.copy()

    if year:
        matches_df = matches_df[matches_df["MODEL_YEAR"].astype(str) == year]
    if make:
        matches_df = matches_df[matches_df["MAKE"].str.contains(make, case=False, na=False)]
    if model:
        matches_df = matches_df[matches_df["MODEL_NAME"].str.contains(model, case=False, na=False)]
    if trim:
        matches_df = matches_df["TRIM"].str.contains(trim, case=False, na=False)

    if matches_df.empty:
        st.warning("No CADS matches found.")
    else:
        st.subheader("Applicable Vehicle Lines")

        display_df = matches_df[
            ["MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"]
        ].copy()

        display_df.rename(columns={
            "MODEL_YEAR": "Year",
            "MODEL_NAME": "Model",
            "AD_MFGCODE": "Model Code"
        }, inplace=True)

        display_df.insert(0, "Select", False)

        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        selected = edited_df[edited_df["Select"]]

        if not selected.empty:
            if st.button("Save Selected Vehicles"):
                out = selected.copy()
                out["Vehicle"] = matched_vehicle or vehicle_input

                out = out[
                    ["Vehicle","Year","MAKE","Model","TRIM","Model Code","STYLE_ID"]
                ]
                out.rename(columns={"MAKE":"Make","TRIM":"Trim"}, inplace=True)

                mappings_df = pd.concat([mappings_df, out], ignore_index=True)
                mappings_df.drop_duplicates(
                    subset=["Vehicle","STYLE_ID"], inplace=True
                )
                mappings_df.to_csv(MAPPINGS_FILE, index=False)

                st.success(f"Saved {len(out)} vehicle mapping(s)")

# ---------------------- Existing mappings ----------------------
st.subheader("Saved Mappings")
st.dataframe(mappings_df, use_container_width=True)
