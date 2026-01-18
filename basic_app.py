
import base64, json, re
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
import streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ========================== Page / Config ==========================
st.set_page_config(page_title="Mapped Vehicle Lookup (Minimal Harvest)", layout="wide")
st.title("Mapped Vehicle Lookup (Minimal Harvest)")

# ---- GitHub config (from secrets, with sidebar overrides) ----
gh_cfg = st.secrets.get("github", {})
DEF_TOKEN  = gh_cfg.get("token", "")
DEF_OWNER  = gh_cfg.get("owner", "")
DEF_REPO   = gh_cfg.get("repo", "")
DEF_BRANCH = gh_cfg.get("branch", "main")
MAPPINGS_PATH = "data/mappings.json"  # adjust if your file lives elsewhere

st.sidebar.header("GitHub Source")
GH_TOKEN  = st.sidebar.text_input("Token", value=DEF_TOKEN, type="password")
GH_OWNER  = st.sidebar.text_input("Owner", value=DEF_OWNER or "your-org-or-user")
GH_REPO   = st.sidebar.text_input("Repo", value=DEF_REPO or "your-repo")
GH_BRANCH = st.sidebar.text_input("Branch", value=DEF_BRANCH or "main")
MAPPINGS_PATH = st.sidebar.text_input("Path to mappings.json", value=MAPPINGS_PATH)

# ========================== HTTP session ==========================
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
_session.mount("https://", HTTPAdapter(max_retries=_retries))
_session.mount("http://",  HTTPAdapter(max_retries=_retries))

def gh_headers(token: str) -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def gh_contents_url(owner: str, repo: str, path: str) -> str:
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

# ========================== Canon helpers ==========================
def canon_text(val: str, for_trim: bool=False) -> str:
    s = (val or "").replace("\u00A0", " ")
    s = s.strip().lower()
    s = re.sub(r"^[\s\.,;:!]+", "", s)
    s = re.sub(r"[\s\.,;:!]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    if for_trim:
        repl = {
            "all wheel drive":"awd","all-wheel drive":"awd","4wd":"awd","4x4":"awd",
            "front wheel drive":"fwd","front-wheel drive":"fwd",
            "rear wheel drive":"rwd","rear-wheel drive":"rwd",
            "two wheel drive":"2wd","two-wheel drive":"2wd",
            "plug-in hybrid":"phev","electric":"ev","bev":"ev",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
    return s

def tokens(s: str, min_len: int = 1) -> List[str]:
    s = canon_text(s)
    tks = re.split(r"[^\w]+", s)
    return [t for t in tks if t and len(t) >= min_len]

def _extract_years_from_text(s: str) -> set:
    s = (s or "").strip().lower()
    years = set()
    for m in re.finditer(r"\b(19[5-9]\d|20[0-4]\d|2050)\b", s):
        years.add(int(m.group(0)))
    for m in re.finditer(r"\bmy\s*([0-9]{2})\b", s):
        years.add(2000 + int(m.group(1)))
    if not years:
        for m in re.finditer(r"\b([0-9]{2})\b", s):
            cand = 2000 + int(m.group(1))
            if 2000 <= cand <= 2050:
                years.add(cand)
    return years

def extract_primary_year(val: str) -> Optional[int]:
    ys = _extract_years_from_text(str(val))
    return max(ys) if ys else None

def year_token_matches(mapping_year: str, user_year: str) -> bool:
    uy_set = _extract_years_from_text(user_year)
    my_set = _extract_years_from_text(mapping_year)
    if not uy_set: return True
    if not my_set: return False
    return bool(uy_set.intersection(my_set))

def trim_tokens(s: str) -> set:
    return set(tokens(canon_text(s, True), min_len=1))

def trim_matches(row_trim: str, user_trim: str, exact_only: bool=False) -> bool:
    row = canon_text(row_trim, True)
    usr = canon_text(user_trim, True)
    if not usr:
        return True
    if row == usr:
        return True
    if exact_only:
        return False
    return trim_tokens(usr).issubset(trim_tokens(row))

# ========================== GitHub loader ==========================
@st.cache_data(ttl=60)
def fetch_mappings(owner: str, repo: str, path: str, token: str, ref: str) -> Dict[str, Dict[str, str]]:
    """
    Loads mappings.json from GitHub contents API.
    Returns {} if not found or parse error.
    """
    url = gh_contents_url(owner, repo, path)
    r = _session.get(url, headers=gh_headers(token), params={"ref": ref}, timeout=15)
    if r.status_code == 200:
        j = r.json()
        if "content" in j:
            decoded = base64.b64decode(j["content"]).decode("utf-8", errors="replace")
            try:
                data = json.loads(decoded)
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        # fallback to download_url if present
        dl = j.get("download_url")
        if dl:
            r2 = _session.get(dl, timeout=15)
            if r2.status_code == 200:
                try:
                    data = json.loads(r2.text)
                    return data if isinstance(data, dict) else {}
                except Exception:
                    return {}
        return {}
    elif r.status_code == 404:
        return {}
    else:
        raise RuntimeError(f"GitHub fetch failed ({r.status_code}): {r.text}")

# ========================== Lookups ==========================
def pick_mapping_by_vehicle(mappings: Dict[str, Dict[str, str]], vehicle: str) -> List[Tuple[str, Dict[str, str]]]:
    cv = canon_text(vehicle)
    out = []
    for k, v in mappings.items():
        if canon_text(v.get("vehicle", "")) == cv:
            out.append((k, v))
    return out

def find_mappings_by_ymmt(
    mappings: Dict[str, Dict[str, str]],
    year: str, make: str, model: str, trim: Optional[str] = None
) -> List[Tuple[str, Dict[str, str]]]:
    cy  = (year or "")
    cmk = canon_text(make)
    cmd = canon_text(model)
    ctr = canon_text(trim or "", True)
    out = []
    for k, v in mappings.items():
        vy  = v.get("year", "")
        vmk = v.get("make", "")
        vmd = v.get("model", "")
        vtr = v.get("trim", "")
        if canon_text(vmk) != cmk:
            continue
        if not year_token_matches(vy, cy):
            continue
        if canon_text(vmd) != cmd:
            continue
        if ctr and not trim_matches(vtr, ctr, exact_only=False):
            continue
        out.append((k, v))
    return out

def parse_vehicle_freeform(s: str) -> Tuple[Optional[int], str, str, str]:
    """
    Heuristic: Year Make Model [Trim ...]
    """
    year = extract_primary_year(s)
    tks  = tokens(s, min_len=1)
    make = tks[1] if len(tks) >= 2 else ""
    model= tks[2] if len(tks) >= 3 else ""
    trim = " ".join(tks[3:]) if len(tks) > 3 else ""
    return (year, make, model, trim)

# ========================== Harvest Table ==========================
def _esc(s: str) -> str:
    return (str(s)
            .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            .replace('"',"&quot;").replace("'","&#39;"))

def render_harvest_table(rows: List[Dict[str, str]], table_id="mapped_harvest", caption="Mapped Vehicles", plain: bool=False):
    """
    rows: list of dicts with keys: 'Key','Year','Make','Model','Trim','Vehicle','Code','Model Code','YMMT'
    Renders a simple semantic table with per-cell IDs and data-col-key for Mozenda.
    """
    cols = ["Key","Year","Make","Model","Trim","Vehicle","Code","Model Code","YMMT","__source"]
    css = ""
    if not plain:
        css = """
        <style>
          table#mapped_harvest { border-collapse: collapse; width: 100%; font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Arial; }
          table#mapped_harvest th, table#mapped_harvest td { border: 1px solid #999; padding: 6px 8px; }
          table#mapped_harvest thead th { background: #eee; }
          td:hover { outline: 2px solid #00a; }
        </style>
        """
    parts = [css, f"<table id='{_esc(table_id)}' data-source='mapped'>"]
    parts.append(f"<caption>{_esc(caption)}</caption>")
    # header
    parts.append("<thead><tr>")
    for c in cols:
        parts.append(f"<th scope='col' data-col-key='{_esc(c)}'>{_esc(c)}</th>")
    parts.append("</tr></thead>")
    # body
    parts.append("<tbody>")
    for r in rows:
        rk  = r.get("Key","")
        yr  = r.get("Year","")
        mk  = r.get("Make","")
        md  = r.get("Model","")
        trm = r.get("Trim","")
        # Row attributes for short XPath filters:
        row_attrs = (
            f"data-row-key='{_esc(rk)}' "
            f"data-year='{_esc(yr)}' "
            f"data-make='{_esc(canon_text(mk))}' "
            f"data-model='{_esc(canon_text(md))}' "
            f"data-trim='{_esc(canon_text(trm, True))}'"
        )
        parts.append(f"<tr {row_attrs}>")
        for c in cols:
            cell_id = f"cell-{rk}-{c}".replace(" ", "_")
            parts.append(f"<td id='{_esc(cell_id)}' data-col-key='{_esc(c)}'>{_esc(r.get(c,''))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)

# ========================== UI Inputs ==========================
vehicle_input = st.text_input("Enter Vehicle (e.g., 2025 Audi SQ5 Premium Plus)", key="veh_input", placeholder="Year Make Model [Trim]")

st.sidebar.header("Options")
HARVEST_ONLY = st.sidebar.checkbox("Harvest Mode (table-only)", value=("1" == st.experimental_get_query_params().get("harvest", ["0"])[0]))
PLAIN = st.sidebar.checkbox("Plain (no CSS)", value=("1" == st.experimental_get_query_params().get("plain", ["0"])[0]))

# ========================== Fetch mappings ==========================
try:
    mappings = fetch_mappings(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH)
    st.caption(f"Loaded {len(mappings)} mapping entries from GitHub: {GH_OWNER}/{GH_REPO}@{GH_BRANCH}/{MAPPINGS_PATH}")
except Exception as e:
    st.error(f"Failed to load mappings: {e}")
    mappings = {}

# ========================== Lookup & Render ==========================
rows_out: List[Dict[str, str]] = []
by_source = ""

if vehicle_input.strip() and mappings:
    # 1) Try exact vehicle match
    direct_hits = pick_mapping_by_vehicle(mappings, vehicle_input)
    if direct_hits:
        for k, v in direct_hits:
            rows_out.append({
                "Key": k,
                "Year": v.get("year",""),
                "Make": v.get("make",""),
                "Model": v.get("model",""),
                "Trim": v.get("trim",""),
                "Vehicle": v.get("vehicle",""),
                "Code": v.get("code",""),
                "Model Code": v.get("model_code",""),
                "YMMT": v.get("ymmt",""),
                "__source": "by_vehicle"
            })
        by_source = "by_vehicle"

    # 2) If no direct vehicle hit, try Y/M/M/T
    if not rows_out:
        y, mk, md, tr = parse_vehicle_freeform(vehicle_input)
        ymmt_hits = find_mappings_by_ymmt(mappings, str(y or ""), mk, md, tr or None)
        for k, v in ymmt_hits:
            rows_out.append({
                "Key": k,
                "Year": v.get("year",""),
                "Make": v.get("make",""),
                "Model": v.get("model",""),
                "Trim": v.get("trim",""),
                "Vehicle": v.get("vehicle",""),
                "Code": v.get("code",""),
                "Model Code": v.get("model_code",""),
                "YMMT": v.get("ymmt",""),
                "__source": "by_ymmt"
            })
        if ymmt_hits:
            by_source = "by_ymmt"

# Render
if rows_out:
    if not HARVEST_ONLY:
        st.success(f"Found {len(rows_out)} mapped entr{'y' if len(rows_out)==1 else 'ies'} ({by_source}).")
    render_harvest_table(rows_out, table_id="mapped_harvest", caption="Mapped Vehicles", plain=PLAIN)
else:
    st.info("No mapped entries found for the provided vehicle." if vehicle_input.strip() else "Enter a vehicle to look up mapped entries.")

# Quick XPath tips (visible only in non-harvest mode)
if not HARVEST_ONLY:
    st.caption("XPath samples:")
    st.code(
        """//table[@id='mapped_harvest']/tbody/tr
//table[@id='mapped_harvest']/tbody/tr[@data-row-key='<<mapping-key>>']/td[@data-col-key='Model']
//*[@id='cell-<<mapping-key>>-Code']""",
        language="text"
    )

# Harvest-only via URL param (optional and non-blocking)
params = st.experimental_get_query_params()
if params.get("harvest", ["0"])[0] == "1":
    # force table-only (no extra widgets below)
    st.stop()

# --- EOF ---
