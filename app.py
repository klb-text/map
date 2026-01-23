import base64, json, io, re, difflib, time
from typing import Optional, List, Dict, Tuple, Set, Any
import requests, pandas as pd, streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ===================== Page Config =====================
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ===================== Secrets / Config =====================
gh_cfg = st.secrets.get("github", {})
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")
GH_TOKEN  = gh_cfg.get("token")

# ===================== Helper Functions =====================
def _setup_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5)
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = _setup_session()

def load_json_from_github(path: str) -> Dict:
    url = f"https://raw.githubusercontent.com/{GH_REPO}/{GH_BRANCH}/{path}"
    r = session.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        st.warning(f"Could not load {path} from GitHub, status {r.status_code}")
        return {}

def commit_json_to_github(path: str, data: Dict, message: str):
    """
    Commit JSON file to GitHub repo.
    """
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    # Get SHA if exists
    r = session.get(url, headers=headers)
    sha = None
    if r.status_code == 200:
        sha = r.json().get("sha")
    content = base64.b64encode(json.dumps(data, indent=2).encode()).decode()
    payload = {"message": message, "content": content}
    if sha:
        payload["sha"] = sha
    r2 = session.put(url, headers=headers, json=payload)
    if r2.status_code in [200, 201]:
        st.success(f"Committed {path} to GitHub successfully.")
    else:
        st.error(f"Failed to commit {path}: {r2.text}")

# ===================== Load CADS / Catalog =====================
@st.cache_data
def load_cads_df_ui(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

@st.cache_data
def load_vehicle_catalog(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def parse_vehicle_against_catalog(vehicle: str, catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to smart match a vehicle string against the catalog dataframe.
    Return closest matches.
    """
    vehicle_lower = vehicle.lower()
    matches = catalog_df[catalog_df['vehicle'].str.lower().str.contains(vehicle_lower)]
    if len(matches) == 0:
        # fallback to YMMT match logic
        words = vehicle_lower.split()
        pattern = "|".join(words)
        matches = catalog_df[catalog_df['vehicle'].str.lower().str.contains(pattern)]
    return matches

# ===================== Session State =====================
if "vehicle_input" not in st.session_state:
    st.session_state.vehicle_input = ""
if "vehicle_matches" not in st.session_state:
    st.session_state.vehicle_matches = pd.DataFrame()
if "selected_mappings" not in st.session_state:
    st.session_state.selected_mappings = []

# ===================== Sidebar =====================
st.sidebar.header("Vehicle Mapping")
vehicle_input = st.sidebar.text_input("Enter vehicle for mapping", st.session_state.vehicle_input)
st.session_state.vehicle_input = vehicle_input

# ===================== Vehicle Mapping =====================
if vehicle_input:
    catalog_df = load_vehicle_catalog("catalog.csv")
    matches_df = parse_vehicle_against_catalog(vehicle_input, catalog_df)
    st.session_state.vehicle_matches = matches_df
    st.subheader("Closest Catalog Matches")
    st.dataframe(matches_df.head(20))

# ===================== Manual Mapping =====================
st.sidebar.header("Manual Mapping")
if not st.session_state.vehicle_matches.empty:
    st.sidebar.write("Select rows to map to this vehicle:")
    for idx, row in st.session_state.vehicle_matches.iterrows():
        key = f"select_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        st.session_state[key] = st.sidebar.checkbox(f"{row['vehicle']} ({row['ymmt']})", st.session_state[key])
        if st.session_state[key]:
            if row['vehicle'] not in st.session_state.selected_mappings:
                st.session_state.selected_mappings.append(row['vehicle'])

st.write("Selected Mappings:", st.session_state.selected_mappings)

# ===================== Commit Mappings =====================
if st.sidebar.button("Commit Mappings"):
    mappings_data = {"vehicle": vehicle_input, "mappings": st.session_state.selected_mappings}
    commit_json_to_github("mappings.json", mappings_data, f"Mapping for {vehicle_input}")

# ===================== End Block 1 =====================
# ===================== Block 2: Harvest Modes & CADS =====================

st.sidebar.header("Harvest Options")
harvest_mode = st.sidebar.selectbox(
    "Select Harvest Mode",
    ["inputs", "mapped", "quick_vehicle", "quick_ymmt", "unmapped", "catalog"]
)

cads_file_path = st.sidebar.text_input("CADS CSV Path", "cads.csv")

@st.cache_data
def load_cads_generic(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure required columns exist
    for col in ["vehicle", "make", "model", "trim", "year", "code"]:
        if col not in df.columns:
            df[col] = ""
    return df

import os
import pandas as pd
import requests

# Make sure the path points to the local CADS.csv
cads_file_path = os.path.join(os.path.dirname(__file__), "CADS.csv")

# If the file doesn't exist locally, download it from GitHub
if not os.path.exists(cads_file_path):
    url = "https://raw.githubusercontent.com/map/main/CADS.csv"  # <-- replace with your repo path
    r = requests.get(url)
    with open(cads_file_path, "wb") as f:
        f.write(r.content)

# Load the CADS file
cads_df = pd.read_csv(cads_file_path)


# ===================== Harvest Mode Logic =====================
st.subheader("Harvest Mode Output")
filtered_df = pd.DataFrame()

if harvest_mode == "inputs":
    st.write("Harvesting from manual inputs...")
    filtered_df = cads_df.head(50)

elif harvest_mode == "mapped":
    st.write("Harvesting previously mapped vehicles...")
    mapped_vehicles = st.session_state.selected_mappings
    filtered_df = cads_df[cads_df['vehicle'].isin(mapped_vehicles)]

elif harvest_mode == "quick_vehicle":
    st.write("Harvesting using quick vehicle match...")
    if vehicle_input:
        filtered_df = parse_vehicle_against_catalog(vehicle_input, cads_df)
    else:
        filtered_df = pd.DataFrame()

elif harvest_mode == "quick_ymmt":
    st.write("Harvesting using YMMT fallback...")
    ymmt_input = st.sidebar.text_input("Enter YMMT for fallback")
    if ymmt_input:
        pattern = "|".join(ymmt_input.lower().split())
        filtered_df = cads_df[cads_df['vehicle'].str.lower().str.contains(pattern)]

elif harvest_mode == "unmapped":
    st.write("Harvesting unmapped vehicles...")
    mapped_vehicles = st.session_state.selected_mappings
    filtered_df = cads_df[~cads_df['vehicle'].isin(mapped_vehicles)]

elif harvest_mode == "catalog":
    st.write("Harvesting full catalog...")
    filtered_df = cads_df.copy()

# ===================== Filter Utilities =====================
def filter_cads_by_text(df: pd.DataFrame, text: str) -> pd.DataFrame:
    text = text.lower()
    return df[df.apply(lambda x: text in str(x.values).lower(), axis=1)]

text_filter = st.text_input("Filter table by text")
if text_filter:
    filtered_df = filter_cads_by_text(filtered_df, text_filter)

# ===================== Table Display & Selection =====================
st.subheader("CADS Results")
if not filtered_df.empty:
    filtered_df = filtered_df.reset_index(drop=True)
    selection_keys = []
    for idx, row in filtered_df.iterrows():
        key = f"select_cads_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        #st.session_state[key] = st.checkbox(f"{row['vehicle']} ({row['ymmt']})", st.session_state[key])
        st.session_state[key] = st.checkbox(
    f"{row['AD_YEAR']} {row['AD_MAKE']} {row['AD_MODEL']} {row['AD_TRIM']} ({row['STYLE_ID']})",
    st.session_state.get(key, False)
)

        if st.session_state[key]:
            selection_keys.append(idx)
    st.session_state.selected_rows = filtered_df.loc[selection_keys]
    st.write("Selected CADS rows:", st.session_state.selected_rows)
else:
    st.info("No CADS results to display for this mode/filter.")

# ===================== YMM/YMMT Picker =====================
st.sidebar.header("YMMT Quick Search")
year_input = st.sidebar.text_input("Year")
make_input = st.sidebar.text_input("Make")
model_input = st.sidebar.text_input("Model")
trim_input = st.sidebar.text_input("Trim")

if year_input or make_input or model_input or trim_input:
    pattern = "|".join(filter(None, [year_input, make_input, model_input, trim_input]))
    ymmt_matches = cads_df[cads_df.apply(lambda x: pattern.lower() in str(x.values).lower(), axis=1)]
    st.subheader("YMMT Quick Matches")
    st.dataframe(ymmt_matches.head(20))

# ===================== End Block 2 =====================
# ===================== Block 3: Vehicle Matching & GitHub Integration =====================

st.sidebar.header("Vehicle Mapping")

vehicle_input = st.text_input("Enter Vehicle Name for Mapping")

def closest_vehicle_match(vehicle: str, cads: pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Returns top n closest matches for a given vehicle using difflib.
    """
    if not vehicle:
        return pd.DataFrame()
    matches = difflib.get_close_matches(vehicle, cads['vehicle'].tolist(), n=n, cutoff=0.6)
    return cads[cads['vehicle'].isin(matches)]

# Session state initialization
if "selected_mappings" not in st.session_state:
    st.session_state.selected_mappings = []

if vehicle_input:
    st.subheader(f"Closest Matches for '{vehicle_input}'")
    close_matches_df = closest_vehicle_match(vehicle_input, cads_df)
    if close_matches_df.empty:
        st.warning("No close matches found. You can fallback to YMMT.")
    else:
        st.dataframe(close_matches_df)
        for idx, row in close_matches_df.iterrows():
            key = f"map_vehicle_{idx}"
            if key not in st.session_state:
                st.session_state[key] = False
            st.session_state[key] = st.checkbox(f"Select {row['vehicle']} ({row['ymmt']})", st.session_state[key])
            if st.session_state[key] and row['vehicle'] not in st.session_state.selected_mappings:
                st.session_state.selected_mappings.append(row['vehicle'])

st.subheader("Currently Mapped Vehicles")
st.write(st.session_state.selected_mappings)

# ===================== GitHub Integration =====================
st.sidebar.header("GitHub Integration")
gh_token = st.secrets.get("github", {}).get("token", "")
gh_repo = st.secrets.get("github", {}).get("repo", "")
gh_branch = st.secrets.get("github", {}).get("branch", "main")

def save_mappings_to_github(selected_vehicles: list):
    """
    Saves current mappings to GitHub JSON file
    """
    if not (gh_token and gh_repo):
        st.error("GitHub token or repo not set in secrets.")
        return False
    content = json.dumps({"mappings": selected_vehicles}, indent=2)
    url = f"https://api.github.com/repos/{gh_repo}/contents/mappings.json"
    headers = {"Authorization": f"token {gh_token}"}
    # Get SHA for existing file
    r = requests.get(url + f"?ref={gh_branch}", headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")
    else:
        sha = None
    data = {
        "message": f"Update mappings ({len(selected_vehicles)} vehicles)",
        "content": base64.b64encode(content.encode()).decode(),
        "branch": gh_branch,
    }
    if sha:
        data["sha"] = sha
    resp = requests.put(url, headers=headers, data=json.dumps(data))
    if resp.status_code in [200, 201]:
        st.success("Mappings saved to GitHub!")
    else:
        st.error(f"Failed to save mappings. {resp.status_code}: {resp.text}")

if st.button("Commit Mappings to GitHub"):
    save_mappings_to_github(st.session_state.selected_mappings)

# ===================== Helper: Parse Vehicle Against Catalog =====================
def parse_vehicle_against_catalog(vehicle: str, catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses input vehicle and returns matching catalog rows.
    Uses exact match first, then difflib closest match fallback.
    """
    exact_match = catalog_df[catalog_df['vehicle'].str.lower() == vehicle.lower()]
    if not exact_match.empty:
        return exact_match
    return closest_vehicle_match(vehicle, catalog_df)

# ===================== Logging Helper =====================
def log_action(action: str, details: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {action}: {details}")

# ===================== End Block 3 =====================
# ===================== Block 4: Catalog Parsing & Harvest Logic =====================

st.sidebar.header("Catalog / Harvest Options")

catalog_file = st.sidebar.file_uploader("Upload Vehicle Catalog CSV", type="csv")

def load_vehicle_catalog(file) -> pd.DataFrame:
    """
    Load catalog CSV into DataFrame, ensuring consistent column names.
    """
    try:
        df = pd.read_csv(file)
        if 'vehicle' not in df.columns:
            st.error("Catalog CSV missing 'vehicle' column.")
            return pd.DataFrame()
        df['vehicle'] = df['vehicle'].astype(str)
        df['ymmt'] = df.get('ymmt', df['vehicle']).astype(str)
        return df
    except Exception as e:
        st.error(f"Failed to load catalog: {e}")
        return pd.DataFrame()

if catalog_file:
    catalog_df = load_vehicle_catalog(catalog_file)
    st.success(f"Catalog loaded ({len(catalog_df)} vehicles)")
else:
    catalog_df = pd.DataFrame()

# ===================== Filter Helpers =====================
def filter_cads_generic(df: pd.DataFrame, term: Optional[str] = None) -> pd.DataFrame:
    """
    Filters CADS DataFrame for term. Case-insensitive contains search.
    """
    if term:
        return df[df.apply(lambda x: x.astype(str).str.contains(term, case=False).any(), axis=1)]
    return df

def filter_cads_by_ymmt(df: pd.DataFrame, year=None, make=None, model=None, trim=None):
    """
    Filters CADS by optional Y/M/M/T.
    """
    filtered = df.copy()
    if year:
        filtered = filtered[filtered['year'].astype(str) == str(year)]
    if make:
        filtered = filtered[filtered['make'].str.lower() == make.lower()]
    if model:
        filtered = filtered[filtered['model'].str.lower() == model.lower()]
    if trim:
        filtered = filtered[filtered['trim'].str.lower() == trim.lower()]
    return filtered

# ===================== Harvest Mode =====================
st.sidebar.header("Harvest Mode")
harvest_mode = st.sidebar.selectbox("Mode", ["All", "Mapped", "Quick Vehicle", "Quick YMMT", "Unmapped", "Catalog"])

def _run_harvest(cads: pd.DataFrame, catalog: pd.DataFrame, mode: str, vehicle_input: str) -> pd.DataFrame:
    """
    Main harvest logic: filters CADS based on mode and catalog/vehicle inputs.
    """
    if mode == "All":
        return cads
    elif mode == "Mapped":
        if not st.session_state.selected_mappings:
            st.warning("No vehicles mapped yet.")
            return pd.DataFrame()
        return cads[cads['vehicle'].isin(st.session_state.selected_mappings)]
    elif mode == "Quick Vehicle":
        if not vehicle_input:
            st.warning("Enter a vehicle first.")
            return pd.DataFrame()
        return closest_vehicle_match(vehicle_input, cads)
    elif mode == "Quick YMMT":
        # Placeholder: Could provide sidebar inputs for Y/M/M/T
        year = st.sidebar.text_input("Year (Quick YMMT)")
        make = st.sidebar.text_input("Make")
        model = st.sidebar.text_input("Model")
        trim = st.sidebar.text_input("Trim")
        return filter_cads_by_ymmt(cads, year, make, model, trim)
    elif mode == "Unmapped":
        if st.session_state.selected_mappings:
            return cads[~cads['vehicle'].isin(st.session_state.selected_mappings)]
        return cads
    elif mode == "Catalog":
        if catalog.empty:
            st.warning("No catalog loaded.")
            return pd.DataFrame()
        return parse_vehicle_against_catalog(vehicle_input, catalog)
    return pd.DataFrame()

# ===================== Display Harvested CADS =====================
if not cads_df.empty:
    harvested_df = _run_harvest(cads_df, catalog_df, harvest_mode, vehicle_input)
    st.subheader(f"Harvest Results ({len(harvested_df)})")
    st.dataframe(harvested_df)

# ===================== Advanced Logging / Debug =====================
st.sidebar.header("Debug / Logging")
debug_mode = st.sidebar.checkbox("Enable Debug Logs")

def debug_log(msg: str):
    if debug_mode:
        print(f"[DEBUG] {msg}")

if harvested_df is not None:
    debug_log(f"Harvest mode: {harvest_mode}")
    debug_log(f"Number of rows returned: {len(harvested_df)}")

# ===================== Row Selection / Session Helpers =====================
def toggle_selection(idx: int):
    key = f"selected_row_{idx}"
    if key not in st.session_state:
        st.session_state[key] = False
    st.session_state[key] = not st.session_state[key]
    debug_log(f"Toggled selection for row {idx}: {st.session_state[key]}")

# Add a button column for selection
if harvested_df is not None and not harvested_df.empty:
    selection_keys = []
    for idx, _ in harvested_df.iterrows():
        key = f"selected_row_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        selection_keys.append(st.session_state[key])
    st.write("Use checkboxes in next block for individual row selection.")

# ===================== End Block 4 =====================
# ===================== Block 5: Row Selection, Mapping & GitHub Save =====================

st.sidebar.header("Mapping / GitHub")

save_to_github = st.sidebar.checkbox("Enable GitHub Save")
gh_token = st.secrets.get("github_token", "")
gh_file_path = st.sidebar.text_input("GitHub JSON Path", "mappings.json")

# ===================== Row Selection Table =====================
def render_selection_table(df: pd.DataFrame):
    """
    Render table with checkboxes for selecting mapped vehicles.
    """
    if df.empty:
        st.info("No data to display in table.")
        return []

    selected_indices = []
    st.write("Select vehicles to map:")
    for idx, row in df.iterrows():
        key = f"select_row_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        st.session_state[key] = st.checkbox(
            label=f"{row['vehicle']} | {row.get('year','')}-{row.get('make','')}-{row.get('model','')}-{row.get('trim','')}",
            value=st.session_state[key],
            key=key
        )
        if st.session_state[key]:
            selected_indices.append(idx)
    return selected_indices

# ===================== Mapping / Commit =====================
def commit_mappings(df: pd.DataFrame, selected_indices: List[int]):
    """
    Save selected mappings to session state and optionally GitHub.
    """
    if not selected_indices:
        st.warning("No rows selected for mapping.")
        return

    mapped_vehicles = df.loc[selected_indices, 'vehicle'].tolist()
    if 'selected_mappings' not in st.session_state:
        st.session_state.selected_mappings = []
    for v in mapped_vehicles:
        if v not in st.session_state.selected_mappings:
            st.session_state.selected_mappings.append(v)

    st.success(f"{len(mapped_vehicles)} vehicles added to mappings.")

    # Save to GitHub
    if save_to_github and gh_token:
        save_mappings_to_github(st.session_state.selected_mappings)

def save_mappings_to_github(mappings: List[str]):
    """
    Save mappings JSON to GitHub using provided token.
    """
    import base64

    url = f"https://api.github.com/repos/{GH_REPO}/contents/{gh_file_path}"
    headers = {"Authorization": f"token {gh_token}"}

    # Get SHA if file exists
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        sha = resp.json()['sha']
    else:
        sha = None

    content = json.dumps(mappings, indent=2)
    data = {
        "message": f"Update vehicle mappings ({len(mappings)} items)",
        "content": base64.b64encode(content.encode()).decode(),
    }
    if sha:
        data["sha"] = sha

    r = requests.put(url, headers=headers, data=json.dumps(data))
    if r.status_code in [200, 201]:
        st.success("Mappings saved to GitHub successfully.")
    else:
        st.error(f"Failed to save to GitHub: {r.text}")

# ===================== Render Selection & Commit =====================
if harvested_df is not None and not harvested_df.empty:
    selected_indices = render_selection_table(harvested_df)

    if st.button("Commit Selected Mappings"):
        commit_mappings(harvested_df, selected_indices)

# ===================== Selected Mappings Display =====================
if 'selected_mappings' in st.session_state:
    st.sidebar.subheader("Mapped Vehicles")
    st.sidebar.write(st.session_state.selected_mappings)

# ===================== Utility: Closest Vehicle Match =====================
def closest_vehicle_match(vehicle: str, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Return top_n closest matches from CADS DataFrame for manual mapping.
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['score'] = df['vehicle'].apply(lambda x: difflib.SequenceMatcher(None, vehicle.lower(), str(x).lower()).ratio())
    df = df.sort_values(by='score', ascending=False)
    return df.head(top_n)

# ===================== Utility: Catalog Parsing =====================
def parse_vehicle_against_catalog(vehicle: str, catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Match vehicle input against catalog and return candidate rows.
    """
    if catalog.empty or not vehicle:
        return pd.DataFrame()
    # Try exact first
    exact_matches = catalog[catalog['vehicle'].str.lower() == vehicle.lower()]
    if not exact_matches.empty:
        return exact_matches
    # Fallback: partial contains match
    contains_matches = catalog[catalog['vehicle'].str.lower().str.contains(vehicle.lower())]
    return contains_matches.head(10)

# ===================== Session State Initialization =====================
if 'selected_mappings' not in st.session_state:
    st.session_state.selected_mappings = []

if 'cads_df' not in st.session_state:
    st.session_state.cads_df = cads_df if 'cads_df' in locals() else pd.DataFrame()

# ===================== End of App =====================
st.markdown("---")
st.markdown("AFF Vehicle Mapping App — fully rebuilt with vehicle-centered mapping and full harvest/selection logic.")
