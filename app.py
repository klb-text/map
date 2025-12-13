
# app.py
import os
import sys
from typing import Optional

import pandas as pd
import streamlit as st


# -----------------------------
# Configuration (Option 2)
# -----------------------------
def get_config():
    """Load config from Streamlit secrets or environment variables with sensible defaults."""
    REPO = st.secrets.get("REPO") or os.environ.get("REPO")
    BRANCH = st.secrets.get("BRANCH") or os.environ.get("BRANCH", "main")
    GH_TOKEN = st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")
    GIT_USER_NAME = st.secrets.get("GIT_USER_NAME") or os.environ.get("GIT_USER_NAME")
    GIT_USER_EMAIL = st.secrets.get("GIT_USER_EMAIL") or os.environ.get("GIT_USER_EMAIL")

    missing = []
    # REQUIRED: REPO and GH_TOKEN to perform write operations to GitHub
    if not REPO:
        missing.append("REPO")
    if not GH_TOKEN:
        missing.append("GH_TOKEN")

    if missing:
        st.error(
            "Missing required configuration keys: "
            + ", ".join(missing)
            + "\n\nSet them in `.streamlit/secrets.toml` (local) or Streamlit Cloud → Manage app → Secrets."
        )
        st.stop()

    # Show non-sensitive config for debugging
    st.caption(f"Repo: {REPO} | Branch: {BRANCH}")

    return {
        "REPO": REPO,
        "BRANCH": BRANCH,
        "GH_TOKEN": GH_TOKEN,
        "GIT_USER_NAME": GIT_USER_NAME,
        "GIT_USER_EMAIL": GIT_USER_EMAIL,
    }


CONFIG = get_config()


# -----------------------------
# Utility / Data loading stubs
# -----------------------------
@st.cache_data
def load_cads_df() -> pd.DataFrame:
    """
    TODO: Replace with real CADS loader.
    This stub returns a small DataFrame to allow the UI to run.
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
    TODO: Replace with real map file loader (e.g., from GitHub via GH_TOKEN).
    """
    # Example existing mappings:
    data = [
        {"src_year": 2024, "src_make": "Acura", "src_model": "TLX", "src_trim": "A-Spec", "ad_mfgcode": "TLXASPEC"},
    ]
    return pd.DataFrame(data)


def candidates_by_ymmt(
    cads_df: pd.DataFrame, src_year: int, src_make: str, src_model: str, src_trim: str
) -> pd.DataFrame:
    """
    TODO: Replace with your real filtering logic.
    Filters CADS catalog rows by year/make/model/trim (trim relax logic can be applied if needed).
    """
    df = cads_df.copy()
    # Basic filter; adjust for normalization as needed (strip, upper, etc.)
    mask = (
        (df["ad_year"] == src_year)
        & (df["ad_make"].str.casefold() == src_make.casefold())
        & (df["ad_model"].str.casefold() == src_model.casefold())
    )
    # You may choose to include trim in the strict filter or present all trims for the chosen YMM.
    # Here we show all trims for selected YMM:
    return df.loc[mask].reset_index(drop=True)


# -----------------------------
# Mapping write stubs
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
    TODO: Replace with your real mapping update logic.
    Appends/updates a mapping record based on selected CADS row.
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

    # Deduplicate: remove any existing row for the same src key, then append
    dedup_mask = (
        (maps_df["src_year"] == src_year)
        & (maps_df["src_make"].str.casefold() == src_make.casefold())
        & (maps_df["src_model"].str.casefold() == src_model.casefold())
        & (maps_df["src_trim"].str.casefold() == src_trim.casefold())
    )
    maps_df = maps_df.loc[~dedup_mask].copy()
    maps_df = pd.concat([maps_df, pd.DataFrame([new_row])], ignore_index=True)
    return maps_df


def write_maps(maps_df: pd.DataFrame) -> None:
    """
    TODO: Replace with your real persistence to GitHub (using CONFIG["GH_TOKEN"]).
    This stub just shows the DataFrame in the app.

    For real GitHub writes, you might:
      - Use the GitHub REST API to PUT file contents (Base64) to a path in CONFIG["REPO"] on CONFIG["BRANCH"].
      - Or use a lightweight git library if available (ensure compatibility in Streamlit Cloud).
    """
    st.subheader("Preview: Updated mappings (stub)")
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

# Candidates for selection
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

    # ✅ Single, complete call
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
