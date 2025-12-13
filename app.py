
# app.py
import pandas as pd
import streamlit as st

# -----------------------------
# Data loading stubs (replace with your real loaders)
# -----------------------------
@st.cache_data
def load_cads_df() -> pd.DataFrame:
    """
    Stub: returns a small CADS-like DataFrame so the UI runs without external deps.
    Replace this with your real CADS loader.
    """
    data = [
        {"ad_year": 2024, "ad_make": "Acura", "ad_model": "TLX", "ad_trim": "A-Spec", "ad_mfgcode": "TLXASPEC"},
        {"ad_year": 2024, "ad_make": "Acura", "ad_model": "TLX", "ad_trim": "Type S", "ad_mfgcode": "TLXTYPE-S"},
        {"ad_year": 2025, "ad_make": "Acura", "ad_model": "MDX", "ad_trim": "Tech", "ad_mfgcode": "MDXTECH"},
    ]
    return pd.DataFrame(data)


@st.cache_data
def load_maps_df() -> pd.DataFrame:
    """
    Stub: returns a small maps DataFrame for demonstration.
    Replace this with your real maps file loader.
    """
    data = [
        {"src_year": 2024, "src_make": "Acura", "src_model": "TLX", "src_trim": "A-Spec", "ad_mfgcode": "TLXASPEC"},
    ]
    return pd.DataFrame(data)


# -----------------------------
# Candidate filtering (replace/extend with your logic)
# -----------------------------
def candidates_by_ymmt(
    cads_df: pd.DataFrame,
    src_year: int,
    src_make: str,
    src_model: str,
    src_trim: str,
) -> pd.DataFrame:
    """
    Filters CADS rows by Year/Make/Model. Trim is presented as options for the chosen YMM.
    Adjust normalization or include trim strictness to match your catalog rules.
    """
    df = cads_df.copy()
    mask = (
        (df["ad_year"] == src_year)
        & (df["ad_make"].str.casefold() == src_make.casefold())
        & (df["ad_model"].str.casefold() == src_model.casefold())
    )
    return df.loc[mask].reset_index(drop=True)


# -----------------------------
# Mapping update + write stubs
# -----------------------------
def save_mapping(
    maps_df: pd.DataFrame,
    src_year: int,
    src_make: str,
    src_model: str,
    src_trim: str,
    cad_row: pd.Series,
) -> pd.DataFrame:
    """
    Appends/updates a mapping record based on the selected CADS row.
    Deduplicates on the src key. Replace with your exact business rules if needed.
    """
    new_row = {
        "src_year": src_year,
        "src_make": src_make,
        "src_model": src_model,
        "src_trim": src_trim,
        "ad_year": int(cad_row.get("ad_year")),
        "ad_make": str(cad_row.get("ad_make")),
        "ad_model": str(cad_row.get("ad_model")),
        "ad_trim": str(cad_row.get("ad_trim")),
        "ad_mfgcode": str(cad_row.get("ad_mfgcode")),
    }

    # Remove any existing mapping for the same src key
    dedup_mask = (
        (maps_df["src_year"] == src_year)
        & (maps_df["src_make"].str.casefold() == src_make.casefold())
        & (maps_df["src_model"].str.casefold() == src_model.casefold())
        & (maps_df["src_trim"].str.casefold() == src_trim.casefold())
    )
    maps_df = maps_df.loc[~dedup_mask].copy()

    # Append the new mapping
    maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)
    return maps_df


def write_maps(maps_df: pd.DataFrame) -> None:
    """
    Stub: shows the resulting DataFrame.
    Replace this with your persistence logic (e.g., write to CSV/JSON, push to GitHub, etc.).
    """
    st.subheader("Updated mappings (preview)")
    st.dataframe(maps_df, use_container_width=True)


# -----------------------------
# UI
# -----------------------------
st.title("Vehicle Mapping")

with st.sidebar:
    st.header("Source vehicle (YMMT)")
    src_year = st.number_input("Year", min_value=1990, max_value=2100, value=2024, step=1)
    src_make = st.text_input("Make", value="Acura")
    src_model = st.text_input("Model", value="TLX")
    src_trim = st.text_input("Trim", value="A-Spec")

# Load data
cads_df = load_cads_df()
maps_df = load_maps_df()

# Build candidate list
cands = candidates_by_ymmt(cads_df, src_year, src_make, src_model, src_trim)
if cands.empty:
    st.error("No CADS candidates found.")
    st.stop()

view_cols = ["ad_year", "ad_make", "ad_model", "ad_trim", "ad_mfgcode"]
st.dataframe(cands[view_cols], use_container_width=True, height=260)

labels = [
    f"{r['ad_year']} {r['ad_make']} {r['ad_model']} {r['ad_trim']} | code={r['ad_mfgcode']}"
    for _, r in cands[view_cols].reset_index(drop=True).iterrows()
]

selected_pos = st.radio(
    "Choose a candidate",
    options=list(range(len(labels))),
    format_func=lambda i: labels[i] if i < len(labels) else "—",
    index=0,
)

if st.button("💾 Save Mapping", type="primary"):
    # Defensive checks
    if len(cands) == 0:
        st.error("No CADS candidates available to save.")
        st.stop()
    if selected_pos < 0 or selected_pos >= len(cands):
        st.error(f"Invalid selection index: {selected_pos}")
        st.stop()

    cad_row = cands.iloc[selected_pos]

    # Single, complete call (no duplication)
    new_maps = save_mapping(
        maps_df=maps_df,
        src_year=src_year,
        src_make=src_make,
        src_model=src_model,
        src_trim=src_trim,
        cad_row=cad_row,
    )

    write_maps(new_maps)
    st.success("✅ Mapping saved")
   
