import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# -----------------------------
# File paths
# -----------------------------
CADS_FILE = "CADS.csv"          # source CADS file
MAPPINGS_FILE = "Mappings.csv"  # file to store mappings

# -----------------------------
# Load CSVs
# -----------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

cads_df = load_csv(CADS_FILE)

try:
    mappings_df = load_csv(MAPPINGS_FILE)
except FileNotFoundError:
    mappings_df = pd.DataFrame(columns=["vehicle_input","MODEL_YEAR","AD_MAKE","AD_MODEL","STYLE_NAME","AD_MFGCODE"])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AFF Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle (freeform)")

st.write("Optional YMMT filters:")
col1, col2, col3, col4 = st.columns(4)
year_filter = col1.text_input("Year")
make_filter = col2.text_input("Make")
model_filter = col3.text_input("Model")
trim_filter = col4.text_input("Trim")

# -----------------------------
# Search Button
# -----------------------------
if st.button("Search"):
    if not vehicle_input.strip():
        st.warning("Please enter a vehicle name.")
    else:
        # Prepare vehicle_search column
        cads_df['vehicle_search'] = cads_df[['MODEL_YEAR','AD_MAKE','AD_MODEL','STYLE_NAME']].astype(str).agg(' '.join, axis=1)
        
        # Fuzzy match input against CADS
        choices = cads_df['vehicle_search'].tolist()
        matches = process.extract(vehicle_input, choices, scorer=fuzz.token_sort_ratio, limit=50)
        
        # Keep matches above threshold
        threshold = 40
        matched_indices = [idx for val, score, idx in matches if score >= threshold]
        matches_df = cads_df.iloc[matched_indices].copy()
        
        # Apply optional YMMT filters
        if year_filter:
            matches_df = matches_df[matches_df['MODEL_YEAR'].astype(str).str.contains(year_filter)]
        if make_filter:
            matches_df = matches_df[matches_df['AD_MAKE'].str.contains(make_filter, case=False)]
        if model_filter:
            matches_df = matches_df[matches_df['AD_MODEL'].str.contains(model_filter, case=False)]
        if trim_filter:
            matches_df = matches_df[matches_df['STYLE_NAME'].str.contains(trim_filter, case=False)]
        
        if matches_df.empty:
            st.info("No close matches found.")
        else:
            st.write("Matching Vehicles:")
            selected_rows = []
            
            # Display checkboxes for each row
            for idx, row in matches_df.iterrows():
                label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['STYLE_NAME']} | Model Code: {row['AD_MFGCODE']}"
                if st.checkbox(label, key=f"chk_{idx}"):
                    selected_rows.append(row)
            
            # Submit Mapping button
            if st.button("Submit Mapping"):
                if not selected_rows:
                    st.warning("Please select at least one vehicle to map.")
                else:
                    for row in selected_rows:
                        new_entry = {
                            "vehicle_input": vehicle_input,
                            "MODEL_YEAR": row['MODEL_YEAR'],
                            "AD_MAKE": row['AD_MAKE'],
                            "AD_MODEL": row['AD_MODEL'],
                            "STYLE_NAME": row['STYLE_NAME'],
                            "AD_MFGCODE": row['AD_MFGCODE']
                        }
                        mappings_df = pd.concat([mappings_df, pd.DataFrame([new_entry])], ignore_index=True)
                    
                    # Save to CSV
                    mappings_df.to_csv(MAPPINGS_FILE, index=False)
                    st.success(f"Mapping saved for '{vehicle_input}'.")

# -----------------------------
# Show existing mappings
# -----------------------------
if not mappings_df.empty:
    st.write("Existing Mappings:")
    st.dataframe(mappings_df)
