import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from pathlib import Path

# ===================== Page Config =====================
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ===================== Files =====================
BASE_DIR = Path(__file__).parent
CADS_FILE = BASE_DIR / "CADS.csv"
VEHICLE_REF_FILE = BASE_DIR / "vehicle_example.txt"  # can be .csv or .txt

FUZZY_THRESHOLD = 90

# ===================== Loaders =====================
@st.cache_data
def load_cads(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]

    # Canonical working columns
    df["YEAR"] = df["MODEL_YEAR"]
    df["MAKE"] = df["DIVISION_NAME"]
    df["MODEL"] = df["MODEL_NAME"]
    df["TRIM_CANON"] = df["TRIM"].fillna("")
    df["MODEL_CODE"] = df["AD_MFGCODE"]

    return df


@st.cache_data
def load_vehicle_ref(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


cads_df = load_cads(CADS_FILE)
vehicle_ref_df = load_vehicle_ref(VEHICLE_REF_FILE)

# ===================== Vehicle Search =====================
st.subheader("Enter Vehicle (freeform)")

vehicle_input = st.text_input(
    "Vehicle Name",
    placeholder="e.g. 2025 TLX TECH 10 Speed Automatic"
)

search_clicked = st.button("Search")

matches_df = pd.DataFrame()

if search_clicked and vehicle_input.strip():
    choices = vehicle_ref_df["Vehicle"].dropna().unique().tolist()

    match = process.extractOne(
        vehicle_input,
        choices,
        scorer=fuzz.token_sort_ratio
    )

    if match and match[1] >= FUZZY_THRESHOLD:
        ref_row = vehicle_ref_df[vehicle_ref_df["Vehicle"] == match[0]].iloc[0]

        st.success(
            f"Smart match found: {match[0]} (score {match[1]})"
        )

        year = ref_row.get("Year")
        make = ref_row.get("Make")
        model = ref_row.get("Model")
        trim = str(ref_row.get("Trim", "")).strip()

        matches_df = cads_df.copy()

        if pd.notna(year):
            matches_df = matches_df[matches_df["YEAR"] == year]
        if pd.notna(make):
            matches_df = matches_df[matches_df["MAKE"].str.contains(str(make), case=False, na=False)]
        if pd.notna(model):
            matches_df = matches_df[matches_df["MODEL"].str.contains(str(model), case=False, na=False)]
        if trim:
            matches_df = matches_df[matches_df["TRIM_CANON"].str.contains(trim, case=False, na=False)]

    else:
        st.warning("No strong vehicle match found. Use YMMT filter below.")

# ===================== YMMT Fallback =====================
st.subheader("YMMT Filter (optional)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    year_f = st.text_input("Year")
with col2:
    make_f = st.text_input("Make")
with col3:
    model_f = st.text_input("Model")
with col4:
    trim_f = st.text_input("Trim")

apply_ymmt = st.button("Apply YMMT Filter")

if apply_ymmt:
    matches_df = cads_df.copy()

    if year_f.strip():
        matches_df = matches_df[matches_df["YEAR"].astype(str) == year_f.strip()]
    if make_f.strip():
        matches_df = matches_df[matches_df["MAKE"].str.contains(make_f, case=False, na=False)]
    if model_f.strip():
        matches_df = matches_df[matches_df["MODEL"].str.contains(model_f, case=False, na=False)]
    if trim_f.strip():
        matches_df = matches_df[matches_df["TRIM_CANON"].str.contains(trim_f, case=False, na=False)]

# ===================== Results Table =====================
st.subheader("Applicable Vehicle Lines")

if not matches_df.empty:
    display_df = matches_df[
        ["YEAR", "MAKE", "MODEL", "TRIM_CANON", "MODEL_CODE", "STYLE_ID"]
    ].copy()

    display_df.rename(columns={
        "YEAR": "Year",
        "MAKE": "Make",
        "MODEL": "Model",
        "TRIM_CANON": "Trim",
        "MODEL_CODE": "Model Code",
        "STYLE_ID": "Style ID"
    }, inplace=True)

    display_df.insert(0, "Select", False)

    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        use_container_width=True
    )

    selected = edited_df[edited_df["Select"]]

    st.write(f"Selected rows: {len(selected)}")

    if not selected.empty:
        st.dataframe(
            selected.drop(columns=["Select"]),
            use_container_width=True
        )
else:
    st.info("No vehicle lines to display yet.")
