import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import os

# ----------------------
# Config / File Paths
# ----------------------
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"
MAPPINGS_FILE = "Mappings.csv"

# ----------------------
# Load CSV
# ----------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, sep=None, engine="python")

# ----------------------
# Smart Vehicle Match
# ----------------------
def smart_vehicle_match(df, vehicle_input):
    # Create a searchable string combining relevant columns
    df['vehicle_search'] = df[['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM']].astype(str).agg(' '.join, axis=1)
    
    choices = df['vehicle_search'].tolist()
    matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=50)
    
    matched_indices = [i for i, (val, score, i) in enumerate(matches) if score > 60]
    
    if matched_indices:
        matches_df = df.iloc[matched_indices].copy()
        raw_matches = [choices[i] for i in matched_indices]
        return matches_df, raw_matches
    else:
        return pd.DataFrame(), []

# ----------------------
# Load data
# ----------------------
cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# Load existing mappings or create new
if os.path.exists(MAPPINGS_FILE):
    mappings_df = pd.read_csv(MAPPINGS_FILE)
else:
    mappings_df = pd.DataFrame(columns=["Vehicle_Input","MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"])

# ----------------------
# Streamlit UI
# ----------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
st.markdown("### YMMT Filter (optional)")
col1, col2, col3, col4 = st.columns(4)
year_input = col1.text_input("Year")
make_input = col2.text_input("Make")
model_input = col3.text_input("Model")
trim_input = col4.text_input("Trim")

if st.button("Search Vehicle"):
    # Apply optional YMMT filter
    filtered_cads = cads_df.copy()
    if year_input:
        filtered_cads = filtered_cads[filtered_cads["MODEL_YEAR"].astype(str).str.contains(year_input)]
    if make_input:
        filtered_cads = filtered_cads[filtered_cads["AD_MAKE"].str.contains(make_input, case=False, na=False)]
    if model_input:
        filtered_cads = filtered_cads[filtered_cads["AD_MODEL"].str.contains(model_input, case=False, na=False)]
    if trim_input:
        filtered_cads = filtered_cads[filtered_cads["TRIM"].str.contains(trim_input, case=False, na=False)]

    # Smart match based on vehicle input
    matches_df, raw_matches = smart_vehicle_match(filtered_cads, vehicle_input)
    
    if matches_df.empty:
        st.warning("No matching vehicles found.")
    else:
        st.success(f"Found {len(matches_df)} possible matches")

        # Display table with checkboxes
        st.markdown("### Select Applicable Vehicles")
        selected_indices = []
        for idx, row in matches_df.iterrows():
            checked = st.checkbox(
                f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}",
                key=f"chk_{idx}"
            )
            if checked:
                selected_indices.append(idx)

        if st.button("Submit Mapping"):
            if not selected_indices:
                st.warning("No vehicles selected for mapping.")
            else:
                selected_rows = matches_df.loc[selected_indices, ["MODEL_YEAR","AD_MAKE","AD_MODEL","TRIM","AD_MFGCODE"]]
                for _, row in selected_rows.iterrows():
                    new_mapping = {
                        "Vehicle_Input": vehicle_input,
                        "MODEL_YEAR": row["MODEL_YEAR"],
                        "AD_MAKE": row["AD_MAKE"],
                        "AD_MODEL": row["AD_MODEL"],
                        "TRIM": row["TRIM"],
                        "AD_MFGCODE": row["AD_MFGCODE"]
                    }
                    mappings_df = pd.concat([mappings_df, pd.DataFrame([new_mapping])], ignore_index=True)
                
                mappings_df.to_csv(MAPPINGS_FILE, index=False)
                st.success(f"Mapping for {vehicle_input} saved.")
                st.dataframe(selected_rows.rename(columns={"AD_MFGCODE":"Model Code"}))
