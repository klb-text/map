import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os
from github import Github

# ------------------------
# CONFIG
# ------------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# GitHub settings
GITHUB_TOKEN = "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"
GITHUB_OWNER = "klb-text"
GITHUB_REPO = "map"
GITHUB_BRANCH = "main"

# ------------------------
# FUNCTIONS
# ------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, sep="\t")  # tab-separated

def save_mappings_to_github(df):
    g = Github(GITHUB_TOKEN)
    repo = g.get_user(GITHUB_OWNER).get_repo(GITHUB_REPO)
    content_file = MAPPINGS_FILE
    try:
        contents = repo.get_contents(content_file, ref=GITHUB_BRANCH)
        repo.update_file(contents.path, "Update mappings", df.to_csv(index=False), contents.sha, branch=GITHUB_BRANCH)
    except Exception as e:
        # If file doesn't exist, create it
        repo.create_file(content_file, "Create mappings", df.to_csv(index=False), branch=GITHUB_BRANCH)

def smart_vehicle_match(df, vehicle_input, limit=10):
    choices = df["STYLE_NAME"].astype(str).tolist()
    results = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=limit)
    matched_values = [r[0] for r in results if r[1] > 60]
    return df[df["STYLE_NAME"].isin(matched_values)]

# ------------------------
# LOAD DATA
# ------------------------
st.title("AFF Vehicle Mapping")

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

if os.path.exists(MAPPINGS_FILE):
    mappings_df = load_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["VehicleInput","MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"])

# ------------------------
# USER INPUT
# ------------------------
vehicle_input = st.text_input("Enter Vehicle (freeform)")

st.subheader("YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_input = col1.text_input("Year")
make_input = col2.text_input("Make")
model_input = col3.text_input("Model")
trim_input = col4.text_input("Trim")

search_clicked = st.button("Search Vehicles")

# ------------------------
# SEARCH & DISPLAY TABLE
# ------------------------
if search_clicked and vehicle_input.strip() != "":
    # Try smart match first
    matches_df = smart_vehicle_match(cads_df, vehicle_input)

    # Apply optional YMMT filters
    if year_input:
        matches_df = matches_df[matches_df["MODEL_YEAR"].astype(str) == year_input]
    if make_input:
        matches_df = matches_df[matches_df["MAKE"].str.contains(make_input, case=False, na=False)]
    if model_input:
        matches_df = matches_df[matches_df["MODEL_NAME"].str.contains(model_input, case=False, na=False)]
    if trim_input:
        matches_df = matches_df[matches_df["TRIM"].str.contains(trim_input, case=False, na=False)]

    if not matches_df.empty:
        st.subheader("Applicable Vehicle Lines")
        # Create a selection dictionary in session state
        if "selected_vehicles" not in st.session_state:
            st.session_state.selected_vehicles = {}

        for idx, row in matches_df.iterrows():
            key = f"{row['STYLE_ID']}_{row['AD_MFGCODE']}"
            if key not in st.session_state.selected_vehicles:
                st.session_state.selected_vehicles[key] = False
            st.session_state.selected_vehicles[key] = st.checkbox(
                f"{row['MODEL_YEAR']} {row['MAKE']} {row['MODEL_NAME']} {row['TRIM']} (Model Code: {row['AD_MFGCODE']})",
                st.session_state.selected_vehicles[key]
            )

        # ------------------------
        # SUBMIT MAPPING BUTTON
        # ------------------------
        if st.button("Submit Mapping"):
            to_save = []
            for key, selected in st.session_state.selected_vehicles.items():
                if selected:
                    style_id, mfgcode = key.split("_")
                    row = matches_df[matches_df["STYLE_ID"] == int(style_id)].iloc[0]
                    to_save.append({
                        "VehicleInput": vehicle_input,
                        "MODEL_YEAR": row["MODEL_YEAR"],
                        "MAKE": row["MAKE"],
                        "MODEL_NAME": row["MODEL_NAME"],
                        "TRIM": row["TRIM"],
                        "AD_MFGCODE": row["AD_MFGCODE"],
                        "STYLE_ID": row["STYLE_ID"]
                    })
            if to_save:
                new_df = pd.DataFrame(to_save)
                mappings_df = pd.concat([mappings_df, new_df], ignore_index=True)
                mappings_df.drop_duplicates(subset=["VehicleInput","STYLE_ID"], inplace=True)
                mappings_df.to_csv(MAPPINGS_FILE, index=False)
                save_mappings_to_github(mappings_df)
                st.success(f"{len(to_save)} mapping(s) saved!")
                # Reset selections
                st.session_state.selected_vehicles = {}
    else:
        st.warning("No matching vehicles found.")
