import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# ---------------------- File paths ----------------------
CADS_FILE = "CADS.csv"
MAPPINGS_FILE = "Mappings.csv"
VEHICLE_REF_FILE = "vehicle_example.csv"  # or .txt

# ---------------------- Load CSV helper ----------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# Load source files
cads_df = load_csv(CADS_FILE)
if os.path.exists(MAPPINGS_FILE):
    mappings_df = load_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["Vehicle","Year","Make","Model","Trim","AD_MFGCODE","STYLE_ID"])

vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# ---------------------- Streamlit UI ----------------------
st.title("AFF Vehicle Mapping")
vehicle_input = st.text_input("Enter Vehicle Name (freeform)")

search_clicked = st.button("Search")

# Placeholder for filtered matches
filtered_matches = pd.DataFrame()

manual_ymmt = {}

if search_clicked and vehicle_input:
    # ------------------ Smart match ------------------
    choices = vehicle_ref_df["Vehicle"].tolist()
    match, score, idx = process.extractOne(vehicle_input, choices, scorer=fuzz.WRatio, score_cutoff=60)
    
    if match:
        st.success(f"Smart match found: {match} (score {score:.1f})")
        filtered_matches = vehicle_ref_df[vehicle_ref_df["Vehicle"] == match]
    else:
        st.info("No smart match found. Please enter YMMT manually.")
        cols = st.columns(4)
        manual_ymmt["Year"] = cols[0].text_input("Year")
        manual_ymmt["Make"] = cols[1].text_input("Make")
        manual_ymmt["Model"] = cols[2].text_input("Model")
        manual_ymmt["Trim"] = cols[3].text_input("Trim")

# ------------------ Display matching CADS lines ------------------
if not filtered_matches.empty or manual_ymmt:
    if filtered_matches.empty:
        # Build single row from manual input
        filtered_matches = pd.DataFrame([manual_ymmt])
    
    # Join with CADS.csv to get applicable lines
    def get_applicable_cads(row):
        df = cads_df.copy()
        # Match Year
        if row.get("Year"):
            df = df[df["MODEL_YEAR"].astype(str) == str(row["Year"])]
        # Match Make
        if row.get("Make"):
            df = df[df["MAKE"].str.lower() == str(row["Make"]).lower()]
        # Match Model
        if row.get("Model"):
            df = df[df["MODEL_NAME"].str.lower() == str(row["Model"]).lower()]
        # Match Trim if present
        if row.get("Trim"):
            df = df[df["TRIM"].str.lower() == str(row["Trim"]).lower()]
        return df

    close_matches_df = pd.concat([get_applicable_cads(r) for _, r in filtered_matches.iterrows()])
    
    if close_matches_df.empty:
        st.warning("No CADS lines found for this vehicle/YMMT.")
    else:
        st.subheader("Close Matches")
        # Add checkbox column
        close_matches_df["Select"] = False
        cols = ["Select","MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"]
        table = close_matches_df[cols]
        # Display table with checkboxes
        for i, row in table.iterrows():
            key = f"select_{i}"
            table.at[i,"Select"] = st.checkbox(
                label=f"{row['MODEL_YEAR']} | {row['MAKE']} {row['MODEL_NAME']} {row['TRIM']}",
                key=key
            )

        # ------------------ Save Selected Mappings ------------------
        selected_indices = table[table["Select"]].index.tolist()
        if selected_indices:
            if st.button("Save Selected Mappings"):
                selected_rows = close_matches_df.loc[selected_indices].copy()
                selected_rows["Vehicle"] = vehicle_input
                # Reorder columns for output
                selected_rows = selected_rows[["Vehicle","MODEL_YEAR","MAKE","MODEL_NAME","TRIM","AD_MFGCODE","STYLE_ID"]]
                # Rename columns to match Mappings.csv
                selected_rows.rename(columns={
                    "MODEL_YEAR":"Year",
                    "MAKE":"Make",
                    "MODEL_NAME":"Model",
                    "TRIM":"Trim"
                }, inplace=True)
                mappings_df = pd.concat([mappings_df, selected_rows], ignore_index=True)
                mappings_df.drop_duplicates(subset=["Vehicle","STYLE_ID"], inplace=True)
                mappings_df.to_csv(MAPPINGS_FILE, index=False)
                st.success(f"Saved {len(selected_rows)} mapping(s) with model codes!")

# ------------------ Show total mappings ------------------
st.subheader("Total Mappings")
st.write(f"Total Mappings: {len(mappings_df)}")
st.dataframe(mappings_df)
