
import streamlit as st
import pandas as pd
from thefuzz import fuzz  # if you want speed: pip install "thefuzz[speedup]"

# --- File paths ---
CADS_FILE = "CADS.csv"
VEHICLE_REF_FILE = "vehicle_example.txt"

# --- Data loading ---
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV/TXT using python engine to auto-detect delimiter.
    Coerce all fields to string and fill NaNs to stabilize .str ops.
    """
    try:
        df = pd.read_csv(path, sep=None, engine='python', dtype=str)
        return df.fillna('')
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

cads_df = load_csv(CADS_FILE)
vehicle_ref_df = load_csv(VEHICLE_REF_FILE)

# --- Text normalization helpers ---
def normalize(s: str) -> str:
    """
    Lowercase, trim, collapse whitespace, and replace common separators
    so fuzzy match is more forgiving.
    """
    s = str(s or "")
    s = s.strip().lower()
    s = s.replace("-", " ").replace("/", " ")
    s = " ".join(s.split())
    return s

# --- Helper: get example make/model from reference file (exact or fuzzy) ---
def get_example_make_model(vehicle_name: str):
    if vehicle_ref_df.empty or 'Vehicle' not in vehicle_ref_df.columns:
        return None, None

    vn = normalize(vehicle_name)
    # Build a normalized column once (safe even if re-run)
    vehicle_ref_df['_vnorm'] = vehicle_ref_df['Vehicle'].astype(str).map(normalize)

    # Try exact normalized match first
    ref_row = vehicle_ref_df[vehicle_ref_df['_vnorm'] == vn]
    if not ref_row.empty:
        make = ref_row['Make'].values[0] if 'Make' in ref_row.columns else None
        model = ref_row['Model'].values[0] if 'Model' in ref_row.columns else None
        return make, model

    # Fuzzy fallback if no exact reference match
    if not vehicle_ref_df.empty:
        scores = vehicle_ref_df['_vnorm'].map(lambda x: fuzz.token_set_ratio(vn, x))
        top_idx = scores.idxmax()
        if pd.notna(top_idx) and scores.loc[top_idx] >= 80:
            row = vehicle_ref_df.loc[top_idx]
            return row.get('Make', None), row.get('Model', None)

    return None, None

# --- Page ---
st.title("AFF Vehicle Mapping")
st.caption("Type a vehicle (e.g., '2024 Toyota RAV4 XLE') and optionally filter by Year/Make/Model/Trim.")

# --- Inputs in a form to avoid mid-typing reruns ---
with st.form("search_form"):
    vehicle_input = st.text_input("Vehicle (freeform)", placeholder="e.g., 2024 Toyota RAV4 XLE")

    st.write("YMMT Filter (optional)")
    col1, col2, col3, col4 = st.columns(4)
    year_input = col1.text_input("Year")
    make_input = col2.text_input("Make")
    model_input = col3.text_input("Model")
    trim_input = col4.text_input("Trim")

    top_n = st.slider("How many matches to show", min_value=5, max_value=50, value=20, step=5)
    score_cutoff = st.slider("Minimum match score", min_value=0, max_value=100, value=60, step=5)

    submitted = st.form_submit_button("Search Vehicles")

# --- Show reference suggestion if available ---
example_make, example_model = (None, None)
if vehicle_input:
    example_make, example_model = get_example_make_model(vehicle_input)
    if example_make or example_model:
        st.info(f"Example Make/Model from reference: {example_make or '—'} / {example_model or '—'}")

# --- Smart Match Function ---
def smart_vehicle_match(
    df: pd.DataFrame,
    vehicle_q: str,
    year: str = "",
    make: str = "",
    model: str = "",
    trim: str = "",
    example_make: str | None = None,
    example_model: str | None = None,
    top_n: int = 20,
    score_cutoff: int = 60
) -> pd.DataFrame:
    """
    Returns a dataframe of top candidates with a fuzzy 'score' column.
    """
    if df.empty or not vehicle_q:
        return pd.DataFrame()

    # Ensure required columns exist (avoid KeyErrors downstream)
    needed = ['MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID']
    for col in needed:
        if col not in df.columns:
            df[col] = ""

    work = df.copy()
    # Normalize text columns to strings
    for col in ['MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM']:
        work[col] = work[col].astype(str).fillna('')

    # Apply explicit filters first (narrow the candidate space)
    if year:
        work = work[work['MODEL_YEAR'].astype(str) == str(year)]
    if make:
        work = work[work['AD_MAKE'].str.lower() == make.lower()]
    if model:
        work = work[work['AD_MODEL'].str.lower() == model.lower()]
    if trim:
        # contains allows partial matches for trim
        work = work[work['TRIM'].str.lower().str.contains(trim.lower())]

    # If nothing found and we have example make/model, try that as a fallback
    if work.empty and example_make and example_model:
        work = df[
            (df['AD_MAKE'].astype(str).str.lower() == example_make.lower()) &
            (df['AD_MODEL'].astype(str).str.lower() == example_model.lower())
        ].copy()

    if work.empty:
        return pd.DataFrame()

    # Build a combined search string per row
    work = work.copy()
    work['vehicle_search'] = (
        work['MODEL_YEAR'].astype(str).str.strip() + ' ' +
        work['AD_MAKE'].astype(str).str.strip() + ' ' +
        work['AD_MODEL'].astype(str).str.strip() + ' ' +
        work['TRIM'].astype(str).str.strip()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Normalize and score per-row (keeps 1:1 mapping)
    q_norm = normalize(vehicle_q)
    work['vs_norm'] = work['vehicle_search'].map(normalize)
    work['score'] = work['vs_norm'].map(lambda s: fuzz.token_set_ratio(q_norm, s))

    # Keep top results above cutoff
    work = (
        work[work['score'] >= score_cutoff]
        .sort_values(['score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM'],
                     ascending=[False, False, True, True, True])
        .head(top_n)
    )

    cols = ['score', 'MODEL_YEAR', 'AD_MAKE', 'AD_MODEL', 'TRIM', 'AD_MFGCODE', 'STYLE_ID', 'vehicle_search']
    return work[cols]

# --- Search & Results ---
if submitted:
    # Reset selection on every new search
    st.session_state['selected_vehicles'] = []

    matches_df = smart_vehicle_match(
        cads_df,
        vehicle_input,
        year=year_input,
        make=make_input,
        model=model_input,
        trim=trim_input,
        example_make=example_make,
        example_model=example_model,
        top_n=top_n,
        score_cutoff=score_cutoff
    )

    if matches_df.empty:
        st.warning(f"No matching vehicles found for: {vehicle_input}")
    else:
        st.subheader("Matching Vehicles")
        st.dataframe(matches_df.reset_index(drop=True))

        st.write("Select vehicles to map:")
        select_all = st.checkbox("Select all shown")

        selected = []
        for idx, row in matches_df.iterrows():
            label = f"[{row['score']}] {row['MODEL_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['TRIM']} | Model Code: {row['AD_MFGCODE']} | Style ID: {row['STYLE_ID']}"
            checked = st.checkbox(label, key=f"chk_{idx}", value=select_all)
            if checked:
                selected.append(idx)

        st.session_state['selected_vehicles'] = selected

        if selected:
            final_df = matches_df.loc[selected, ['MODEL_YEAR','AD_MAKE','AD_MODEL','TRIM','AD_MFGCODE','STYLE_ID']].copy()
            st.success(f"{len(final_df)} row(s) ready.")
            st.dataframe(final_df.reset_index(drop=True))

            csv = final_df.to_csv(index=False)
            st.download_button(
                "Download mapping as CSV",
                data=csv,
                file_name="vehicle_mapping.csv",
                mime="text/csv"
            )
# --- EOF ---
