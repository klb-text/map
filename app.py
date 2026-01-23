# app.py
import pandas as pd
import streamlit as st
from thefuzz import process, fuzz

# ---------------------------
# Helper functions
# ---------------------------

@st.cache_data
def load_csv(path):
    """Load CSV safely with basic cleaning."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]  # trim column names
        df.fillna("", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

def normalize_text(s):
    return str(s).strip().lower()

def filter_by_make_model_year(cads_df, year, make, model):
    """Deterministically filter CADS by year, make, model."""
    filtered = cads_df[
        (cads_df['MODEL_YEAR'].astype(str) == str(year)) &
        (cads_df['AD_MAKE'].apply(normalize_text) == normalize_text(make)) &
        (cads_df['AD_MODEL'].apply(normalize_text) == normalize_text(model))
    ].copy()
    return filtered

def fuzzy_trim_match(df, user_trim, limit=10):
    """Apply fuzzy matching on TRIM column only for filtered rows."""
    if df.empty or not user_trim:
        return df
    df['trim_score'] = df['TRIM'].apply(lambda x: fuzz.partial_ratio(user_trim.lower(), str(x).lower()))
    return df.sort_values('trim_score', ascending=False).head(limit)

# ---------------------------
# Load data
# ---------------------------

st.title("AFF Vehicle Mapping")

vehicle_ref_df = load_csv("vehicle_example.txt")
cads_df = load_csv("CADS.csv")

# ---------------------------
# Vehicle input
# ---------------------------

vehicle_input = st.text_input("Enter Vehicle (freeform)")

# Optional YMMT filters
st.subheader("YMMT Filters (optional)")
col1, col2, col3, col4 = st.columns(4)
filter_year = col1.text_input("Year")
filter_make = col2.text_input("Make")
filter_model = col3.text_input("Model")
filter_trim = col4.text_input("Trim")

search_button = st.button("Search Vehicles")

# ---------------------------
# Vehicle search logic
# ---------------------------

if search_button and vehicle_input:
    # Try to reference example file for make/model
    ref_row = vehicle_ref_df[vehicle_ref_df['Vehicle'].apply(normalize_text) == normalize_text(vehicle_input)]
    if not ref_row.empty:
        example_year = ref_row.iloc[0].get('Year', filter_year) or filter_year
        example_make = ref_row.iloc[0].get('Make', filter_make) or filter_make
        example_model = ref_row.iloc[0].get('Model', filter_model) or filter_model
        example_trim = ref_row.iloc[0].get('Trim', filter_trim) or filter_trim
    else:
        example_year = filter_year
        example_make = filter_make
        example_model = filter_model
        example_trim = filter_trim

    # Deterministic filter by year/make/model
    filtered_cads = filter_by_make_model_year(cads_df, example_year, example_make, example_model)

    # Optional fuzzy trim matching
    if example_trim:
        filtered_cads = fuzzy_trim_match(filtered_cads, example_trim)

    if filtered_cads.empty:
        st.warning(f"No matching vehicles found for: {vehicle_input}")
    else:
        st.subheader("Matching Vehicles")
        selected_rows = []

        # Show checkboxes for each row
        for idx, row in filtered_cads.iterrows():
            label = f"{row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']}"
            if st.checkbox(label, key=f"chk_{idx}"):
                selected_rows.append(idx)

        # Commit mapping button
        if st.button("Commit Mapping") and selected_rows:
            mapped_df = filtered_cads.loc[selected_rows, ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']]
            st.success("Mapping committed for selected vehicles:")
            st.dataframe(mapped_df.reset_index(drop=True))
