
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# + Single-best mapping on Y/M/M/T (Trim exact/subset; Make exact; Year token-aware; Model similarity)
# + Land Rover family enforcement (Range Rover / Discovery / Defender) to prevent cross-family mismatches
# + Canonicalization; Code-first CADS search; detailed matching trace
# Repo: klb-text/map, Branch: main

import base64
import json
import time
import io
import re
import difflib
from typing import Optional, List, Dict, Tuple
import requests
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# ---------------------------------------------------------------------
# Secrets / Config
# ---------------------------------------------------------------------
gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

MAPPINGS_PATH   = "data/mappings.json"
AUDIT_LOG_PATH  = "data/mappings_log.jsonl"
CADS_PATH       = "CADS.csv"
CADS_IS_EXCEL   = False
CADS_SHEET_NAME_DEFAULT = "0"

# Candidate ID columns in CADS
CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ---------------------------------------------------------------------
# Resilient HTTP
# ---------------------------------------------------------------------
from requests.adapters import HTTPAdapter, Retry
_session = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "PUT", "POST"],
)
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def _get(url, headers=None, params=None, timeout=15):  return _session.get(url, headers=headers, params=params, timeout=timeout)
def _put(url, headers=None, json=None, timeout=15):    return _session.put(url, headers=headers, json=json, timeout=timeout)
def _post(url, headers=None, json=None, timeout=15):   return _session.post(url, headers=headers, json=json, timeout=timeout)

def gh_headers(token: str): return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
def gh_contents_url(owner, repo, path): return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
def gh_ref_heads(owner, repo, branch):  return f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"

def get_file(owner, repo, path, token, ref=None):
    params = {"ref": ref} if ref else {}
    return _get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params)

def get_file_sha(owner, repo, path, token, ref=None):
    r = get_file(owner, repo, path, token, ref)
    if r.status_code == 200: return r.json()["sha"]
    if r.status_code == 404: return None
    raise RuntimeError(f"Failed to fetch SHA ({r.status_code}): {r.text}")

def load_json_from_github(owner, repo, path, token, ref=None):
    r = get_file(owner, repo, path, token, ref)
    if r.status_code == 200:
        j = r.json()
        decoded = base64.b64decode(j["content"]).decode("utf-8")
        return json.loads(decoded)
    if r.status_code == 404: return None
    raise RuntimeError(f"Failed to load file ({r.status_code}): {r.text}")

def get_branch_head_sha(owner, repo, branch, token):
    r = _get(gh_ref_heads(owner, repo, branch), headers=gh_headers(token))
    if r.status_code == 200: return r.json()["object"]["sha"]
    if r.status_code == 404: return None
    raise RuntimeError(f"Failed to read branch {branch} head ({r.status_code}): {r.text}")

def ensure_feature_branch(owner, repo, token, source_branch, feature_branch):
    base_sha = get_branch_head_sha(owner, repo, source_branch, token)
    if not base_sha: return False
    r_feat = _get(gh_ref_heads(owner, repo, feature_branch), headers=gh_headers(token))
    if r_feat.status_code == 200: return True
    if r_feat.status_code != 404: raise RuntimeError(f"Failed checking feature branch ({r_feat.status_code}): {r_feat.text}")
    r_create = _post(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs",
        headers=gh_headers(token),
        json={"ref": f"refs/heads/{feature_branch}", "sha": base_sha},
    )
    return r_create.status_code in (201, 422)

def save_json_to_github(owner, repo, path, token, branch, payload_dict, commit_message,
                        author_name=None, author_email=None, use_feature_branch=False, feature_branch_name="aff-mapping-app"):
    content = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    target_branch = branch
    if use_feature_branch and ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
        target_branch = feature_branch_name
    sha = get_file_sha(owner, repo, path, token, ref=target_branch)
    data = {"message": commit_message, "content": content_b64, "branch": target_branch}
    if sha: data["sha"] = sha
    if author_name and author_email: data["committer"] = {"name": author_name, "email": author_email}
    r = _put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
    if r.status_code in (200, 201): return r.json()
    if r.status_code == 409:
        latest_sha = get_file_sha(owner, repo, path, token, ref=target_branch)
        if latest_sha and not data.get("sha"):
            data["sha"] = latest_sha
            r2 = _put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
            if r2.status_code in (200, 201): return r2.json()
    raise RuntimeError(f"Failed to save file ({r.status_code}): {r.text}")

def append_jsonl_to_github(owner, repo, path, token, branch, record, commit_message,
                           use_feature_branch=False, feature_branch_name="aff-mapping-app"):
    target_branch = branch
    if use_feature_branch and ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
        target_branch = feature_branch_name
    r = get_file(owner, repo, path, token, ref=target_branch)
    lines, sha = "", None
    if r.status_code == 200:
        sha = r.json()["sha"]
        existing = base64.b64decode(r.json()["content"]).decode("utf-8")
        lines = existing if existing.endswith("\n") else (existing + "\n")
    elif r.status_code != 404:
        raise RuntimeError(f"Failed to read log file ({r.status_code}): {r.text}")
    lines += json.dumps(record, ensure_ascii=False) + "\n"
    content_b64 = base64.b64encode(lines.encode("utf-8")).decode("utf-8")
    data = {"message": commit_message, "content": content_b64, "branch": target_branch}
    if sha: data["sha"] = sha
    r2 = _put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data)
    if r2.status_code in (200, 201): return r2.json()
    raise RuntimeError(f"Failed to append log ({r2.status_code}): {r.text}")

# ---------------------------------------------------------------------
# CADS loaders
# ---------------------------------------------------------------------
def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0: df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df

@st.cache_data(ttl=600)
def _decode_bytes_to_text(raw: bytes) -> tuple[str, str]:
    if not raw or raw.strip() == b"": return ("", "empty")
    encoding = "utf-8"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"): encoding = "utf-16"
    elif raw.startswith(b"\xef\xbb\xbf"): encoding = "utf-8-sig"
    text = raw.decode(encoding, errors="replace")
    return (text, encoding)

@st.cache_data(ttl=600)
def load_cads_from_github_csv(owner, repo, path, token, ref=None) -> pd.DataFrame:
    import csv
    params = {"ref": ref} if ref else {}
    r = _get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params)
    if r.status_code == 200:
        j = r.json(); raw = None
        if "content" in j and j["content"]:
            try: raw = base64.b64decode(j["content"])
            except Exception: raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = _get(j["download_url"]); raw = r2.content if r2.status_code == 200 else None
        if raw is None or raw.strip() == b"": raise ValueError(f"CADS `{path}` empty or unavailable.")
        text, _ = _decode_bytes_to_text(raw)
        sample = text[:4096]
        delimiter = None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",","\t",";","|"])
            delimiter = dialect.delimiter
        except Exception:
            for cand in [",","\t",";","|"]:
                if cand in sample: delimiter = cand; break
        if delimiter is None:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, on_bad_lines="skip")
        else:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, on_bad_lines="skip", engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")
        if df.empty or len(df.columns) == 0: raise ValueError("CADS CSV parsed but produced no columns/rows.")
        return _strip_object_columns(df)
    if r.status_code == 404: raise FileNotFoundError(f"CADS not found: {path}")
    raise RuntimeError(f"Failed to load CADS CSV ({r.status_code}): {r.text}")

@st.cache_data(ttl=600)
def load_cads_from_github_excel(owner, repo, path, token, ref=None, sheet_name=0) -> pd.DataFrame:
    params = {"ref": ref} if ref else {}
    r = _get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params)
    if r.status_code == 200:
        j = r.json(); raw = None
        if "content" in j and j["content"]:
            try: raw = base64.b64decode(j["content"])
            except Exception: raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = _get(j["download_url"]); raw = r2.content if r2.status_code == 200 else None
        if raw is None or raw.strip() == b"": raise ValueError(f"CADS `{path}` empty or unavailable.")
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, engine="openpyxl")
        return _strip_object_columns(df)
    if r.status_code == 404: raise FileNotFoundError(f"CADS not found: {path}")
    raise RuntimeError(f"Failed to load CADS Excel ({r.status_code}): {r.text}")

# ---------------------------------------------------------------------
# Canonicalization + helpers
# ---------------------------------------------------------------------
def canon_text(val: str, for_trim: bool=False) -> str:
    """
    Lowercases, strips spaces, removes leading/trailing punctuation (keeps hyphens/slashes).
    Normalizes trim synonyms when for_trim=True.
    """
    s = (val or "").strip().lower()
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
        for k, v in repl.items(): s = s.replace(k, v)
    return s

def _extract_years_from_text(s: str) -> set:
    """
    Extract potential year tokens from mapping or input text.
    Supports:
      - 4-digit years: 2024, 2025
      - 'MY25', 'MY 25' -> 2025
      - ranges/composites: '2024/2025', '2025-2026', '2024 | 2025'
      - quarters/phrases: 'Q2-2025', '2025 Q4' -> 2025
      - bare 2-digit '25' -> 2025 (only if no 4-digit found)
    """
    s = (s or "").strip().lower()
    years = set()

    # 4-digit
    for m in re.finditer(r"\b(19[5-9]\d|20[0-4]\d|2050)\b", s):
        years.add(int(m.group(0)))

    # MY25 → 2025
    for m in re.finditer(r"\bmy\s*([0-9]{2})\b", s):
        years.add(2000 + int(m.group(1)))

    # Qx-2025 or 2025-Qx → 2025
    for m in re.finditer(r"\b(?:q[1-4][\-\s]*)?(19[5-9]\d|20[0-4]\d|2050)\b", s):
        years.add(int(m.group(1)))

    # bare 2-digit if nothing else
    if not years:
        for m in re.finditer(r"\b([0-9]{2})\b", s):
            yy = int(m.group(1)); years.add(2000 + yy)

    return years

def year_token_matches(mapping_year: str, user_year: str) -> bool:
    uy_set = _extract_years_from_text(user_year)
    my_set = _extract_years_from_text(mapping_year)
    if not uy_set:  # user did not supply year → accept
        return True
    if not my_set:
        return False
    return bool(uy_set.intersection(my_set))

def _trim_tokens(s: str) -> set:
    s = canon_text(s, for_trim=True)
    toks = re.split(r"[^\w]+", s)
    return {t for t in toks if t}

def trim_matches(row_trim: str, user_trim: str, exact_only: bool=False) -> Tuple[bool, float]:
    """
    Returns (match, score_weight):
      - exact_only=True  -> only exact match counts (True, 1.0)
      - exact_only=False -> exact (1.0) or token-subset (0.8)
    """
    row = canon_text(row_trim, True)
    usr = canon_text(user_trim, True)
    if not usr:
        return (True, 0.5)
    if row == usr:
        return (True, 1.0)
    if exact_only:
        return (False, 0.0)
    if _trim_tokens(usr).issubset(_trim_tokens(row)):
        return (True, 0.8)
    return (False, 0.0)

def model_similarity(a: str, b: str) -> float:
    a = canon_text(a); b = canon_text(b)
    if not a and not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a: return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------------------------------------------------------------------
# Land Rover family taxonomy & detector
# ---------------------------------------------------------------------
LAND_ROVER_FAMILIES = {
    "range rover": {
        "range rover", "range rover sport", "range rover velar", "range rover evoque"
    },
    "discovery": {
        "discovery", "discovery sport"
    },
    "defender": {
        "defender", "defender 90", "defender 110", "defender 130"
    },
}

def detect_lr_family(model_text: str) -> Optional[str]:
    """
    Return 'range rover' | 'discovery' | 'defender' if model_text belongs to that family, else None.
    """
    md = canon_text(model_text)
    if not md:
        return None
    for fam, names in LAND_ROVER_FAMILIES.items():
        for name in names:
            n = canon_text(name)
            if md == n or (n in md) or (md in n):
                return fam
    # Keyword fallback
    if "range rover" in md: return "range rover"
    if "discovery" in md:   return "discovery"
    if "defender" in md:    return "defender"
    return None

# ---------------------------------------------------------------------
# Single-best mapping picker (strict trim+make; token-aware year; tolerant model; LR family constraint)
# ---------------------------------------------------------------------
def pick_best_mapping(
    mappings: Dict[str, Dict[str, str]],
    year: str, make: str, model: str, trim: str,
    trim_exact_only: bool = False,
    enforce_lr_family: bool = True,
    model_exact_when_full: bool = True,
) -> Optional[Tuple[str, Dict[str, str], float]]:
    cmk = canon_text(make)
    ctr = canon_text(trim, True)
    cy  = (year or "")
    cmd = canon_text(model)

    if not cmk:
        return None

    # Land Rover family enforcement (user-side)
    user_lr_family = detect_lr_family(model) if cmk == "land rover" else None
    force_exact_model = model_exact_when_full and len(cmd.split()) >= 2  # treat multi-word as "full name"

    candidates: List[Tuple[str, Dict[str, str], float]] = []
    for k, v in mappings.items():
        vmk = v.get("make","")
        vy  = v.get("year","")
        vtr = v.get("trim","")
        vmd = v.get("model","")

        # Make exact
        if canon_text(vmk) != cmk:
            continue
        # Year token-aware
        if not year_token_matches(vy, cy):
            continue
        # Trim exact or subset (strict if requested)
        tmatch, tscore = trim_matches(vtr, ctr, exact_only=trim_exact_only)
        if not tmatch:
            continue

        # Land Rover family guard
        if enforce_lr_family and cmk == "land rover" and user_lr_family:
            candidate_family = detect_lr_family(vmd)
            if candidate_family != user_lr_family:
                continue

        # Model similarity, optionally penalize non-exact when user typed full model
        ms = model_similarity(vmd, cmd)
        if force_exact_model and canon_text(vmd) != cmd:
            ms = ms * 0.5  # dampen non-exact to keep exact on top

        score = tscore * 0.6 + ms * 0.4
        candidates.append((k, v, score))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[0]  # (key, value, score)

# ---------------------------------------------------------------------
# CADS filter – strict Trim (exact or subset); model contains/exact; token-aware year; LR family constraint
# ---------------------------------------------------------------------
def filter_cads(
    df: pd.DataFrame,
    year: str, make: str, model: str, trim: str, vehicle: str, model_code: str = "",
    exact_mmt: bool = False, case_sensitive: bool = False,
    strict_and: bool = True,
    trim_exact_only: bool = False,
    enforce_lr_family: bool = True,
) -> pd.DataFrame:
    df2 = _strip_object_columns(df.copy())

    YEAR_CANDS  = ["AD_YEAR","Year","MY","ModelYear","Model Year"]
    MAKE_CANDS  = ["AD_MAKE","Make","MakeName","Manufacturer"]
    MODEL_CANDS = ["AD_MODEL","Model","Line","Carline","Series"]
    TRIM_CANDS  = ["AD_TRIM","Trim","Grade","Variant","Submodel"]

    year_col  = next((c for c in YEAR_CANDS  if c in df2.columns), None)
    make_col  = next((c for c in MAKE_CANDS  if c in df2.columns), None)
    model_col = next((c for c in MODEL_CANDS if c in df2.columns), None)
    trim_col  = next((c for c in TRIM_CANDS  if c in df2.columns), None)

    y  = (year or "")
    mk = canon_text(make)
    md = canon_text(model)
    tr = canon_text(trim, True)

    masks = []

    # Make exact
    if make_col and mk:
        s = df2[make_col].astype(str).str.lower()
        masks.append(s == mk)

    # Model: exact or contains; apply LR family constraint if enabled
    if model_col and md:
        s = df2[model_col].astype(str).str.lower()
        if enforce_lr_family and mk == "land rover":
            user_lr_family = detect_lr_family(model)
            if user_lr_family:
                family_names = LAND_ROVER_FAMILIES[user_lr_family]
                fam_mask = False
                for name in family_names:
                    n = canon_text(name)
                    fam_mask = fam_mask | s.str.contains(n, na=False)
                masks.append(fam_mask)
            else:
                masks.append((s == md) if exact_mmt else s.str.contains(md, na=False))
        else:
            masks.append((s == md) if exact_mmt else s.str.contains(md, na=False))

    # Trim: exact or subset
    if trim_col and tr:
        s = df2[trim_col].astype(str)
        m_exact  = s.str.lower() == tr
        if trim_exact_only:
            masks.append(m_exact)
        else:
            m_subset = s.apply(lambda x: _trim_tokens(tr).issubset(_trim_tokens(x)))
            masks.append(m_exact | m_subset)

    # Year: token-aware
    if year_col and y:
        s = df2[year_col].astype(str)
        masks.append(s.apply(lambda vy: year_token_matches(vy, y)))

    if not masks:
        return df2.iloc[0:0]

    m = masks[0]
    for mm in masks[1:]:
        m = m & mm if strict_and else m | mm

    return df2[m]

# ---------------------------------------------------------------------
# CADS matching for a single mapping: Code → Model Code → fallback filter
# ---------------------------------------------------------------------
def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_CODE_PREFS if c in df.columns] or list(df.columns)

def get_model_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_MODEL_CODE_PREFS if c in df.columns] or list(df.columns)

def match_cads_rows_for_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    exact_mmt: bool = False,
    strict_and: bool = True,
    trim_exact_only: bool = False,
    enforce_lr_family: bool = True,
) -> pd.DataFrame:
    df2 = _strip_object_columns(df.copy())

    # Code union
    code_val = (mapping.get("code","") or "").strip()
    if code_val:
        hits = []
        for col in get_cads_code_candidates(df2):
            if col in df2.columns:
                series = df2[col].astype(str).str.strip().str.lower()
                mask = series == code_val.lower()
                if mask.any(): hits.append(df2[mask])
        if hits:
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True)

    # Model Code union
    model_code_val = (mapping.get("model_code","") or "").strip()
    if model_code_val:
        hits = []
        for col in get_model_code_candidates(df2):
            if col in df2.columns:
                series = df2[col].astype(str).str.strip().str.lower()
                mask = series == model_code_val.lower()
                if mask.any(): hits.append(df2[mask])
        if hits:
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True)

    # Fallback strict YMMT
    return filter_cads(
        df2,
        mapping.get("year",""), mapping.get("make",""), mapping.get("model",""), mapping.get("trim",""),
        "", "", exact_mmt=exact_mmt, case_sensitive=False, strict_and=strict_and,
        trim_exact_only=trim_exact_only, enforce_lr_family=enforce_lr_family
    ).reset_index(drop=True)

# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
def get_query_param(name: str, default: str = "") -> str:
    try:
        params = st.experimental_get_query_params()
        val = params.get(name, [default])
        return str(val[0]).strip()
    except Exception:
        return default

def _extract_year_token(s: str) -> str:
    s = (s or "").strip()
    if not s: return ""
    m = re.search(r"\b(19[5-9]\d|20[0-4]\d|2050)\b", s)
    return m.group(0) if m else ""

def secrets_status():
    missing = []
    if not GH_TOKEN:  missing.append("github.token")
    if not GH_OWNER:  missing.append("github.owner")
    if not GH_REPO:   missing.append("github.repo")
    if not GH_BRANCH: missing.append("github.branch")
    return missing

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("AFF Vehicle Mapping")

# --------------------- Agent Mode ------------------------------------
AGENT_MODE = (get_query_param("agent").lower() == "mozenda")
if AGENT_MODE:
    q_year    = get_query_param("year")
    q_make    = get_query_param("make")
    q_model   = get_query_param("model")
    q_trim    = get_query_param("trim")
    q_vehicle = get_query_param("vehicle")
    q_model_code = get_query_param("model_code")
    q_code    = get_query_param("code")

    if not q_year and q_vehicle:
        y_guess = _extract_year_token(q_vehicle)
        if y_guess: q_year = y_guess

    if "mappings" not in st.session_state:
        try:
            existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
            st.session_state.mappings = existing or {}
        except Exception:
            st.session_state.mappings = {}

    # Defaults for agent mode
    TRIM_EXACT_ONLY_AGENT      = (get_query_param("trim_exact_only", "false").lower() == "true")
    ENFORCE_LR_FAMILY_AGENT    = (get_query_param("enforce_lr_family", "true").lower() == "true")
    MODEL_EXACT_WHEN_FULL_AGENT= (get_query_param("model_exact_full", "true").lower() == "true")

    best = pick_best_mapping(
        st.session_state.mappings, q_year, q_make, q_model, q_trim,
        trim_exact_only=TRIM_EXACT_ONLY_AGENT,
        enforce_lr_family=ENFORCE_LR_FAMILY_AGENT,
        model_exact_when_full=MODEL_EXACT_WHEN_FULL_AGENT,
    )

    st.subheader("Mozenda Agent Mode")
    st.caption("Output minimized for scraper consumption.")

    if st.button("🧹 Clear (Agent)", key="agent_clear_btn"):
        for k in ["year_input","make_input","model_input","trim_input","vehicle_input",
                  "code_input","model_code_input","prev_inputs","results_df",
                  "code_candidates","model_code_candidates","code_column","model_code_column","last_matches"]:
            st.session_state.pop(k, None)
        st.success("Agent state cleared. Ready for next vehicle.")
        st.text("STATUS=CLEARED")
        st.write({"status":"CLEARED"})
        st.stop()

    if best:
        k, v, score = best
        row = {
            "key": k, "score": round(score,3),
            "year": v.get("year",""), "make": v.get("make",""), "model": v.get("model",""),
            "trim": v.get("trim",""), "vehicle": v.get("vehicle",""),
            "code": v.get("code",""), "model_code": v.get("model_code",""),
            "reason": "exact_trim_or_subset_best_model_token_year_lr_family",
        }
        st.success("Mapped: 1")
        st.dataframe(pd.DataFrame([row]), use_container_width=True)
        st.text("STATUS=MAPPED")
        st.write({"status":"MAPPED","count":1,"data":[row]})
        st.stop()
    else:
        st.error("NEEDS_MAPPING")
        st.text("STATUS=NEEDS_MAPPING")
        st.write({"status":"NEEDS_MAPPING","inputs":{
            "year":q_year,"make":q_make,"model":q_model,"trim":q_trim,
            "vehicle":q_vehicle,"model_code":q_model_code,"code":q_code
        }})
        st.stop()

# --------------------- Diagnostics -----------------------------------
with st.expander("📦 Data source / diagnostics"):
    try:
        st.write({"owner":GH_OWNER,"repo":GH_REPO,"branch":GH_BRANCH,"mappings_path":MAPPINGS_PATH})
        st.write({"loaded_mappings_count": len(st.session_state.get("mappings", {}))})
        r_meta = get_file(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        if r_meta.status_code == 200:
            meta = r_meta.json()
            st.write({"file_sha": meta.get("sha",""), "path": meta.get("path",""), "size_bytes": meta.get("size","")})
        elif r_meta.status_code == 404:
            st.info("Mappings file does not exist yet (will be created on first commit).")
        else:
            st.warning(f"Could not read file metadata ({r_meta.status_code}).")
        if st.button("🔄 Reload mappings (diagnostics)", key="diag_reload_btn"):
            existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
            st.session_state.mappings = existing or {}
            st.success(f"Reloaded. Current count: {len(st.session_state.mappings)}")
    except Exception as diag_err:
        st.error(f"Diagnostics error: {diag_err}")

# Load mappings on first interactive run
if "mappings" not in st.session_state:
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
    except Exception as e:
        st.session_state.mappings = {}
        st.warning(f"Starting with empty mappings (load error): {e}")

# --------------------- Sidebar ---------------------------------------
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Reload from GitHub"):
    try:
        existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
        st.session_state.mappings = existing or {}
        st.sidebar.success("Reloaded from GitHub.")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

commit_msg = st.sidebar.text_input("Commit message", value="chore(app): update AFF vehicle mappings via Streamlit")
use_feature_branch = st.sidebar.checkbox("Use feature branch (aff-mapping-app)", value=False)

_miss = secrets_status()
if _miss: st.sidebar.warning("Missing secrets: " + ", ".join(_miss))

if st.sidebar.button("💾 Commit mappings to GitHub"):
    if _miss:
        st.sidebar.error("Cannot commit: fix missing secrets first.")
    else:
        try:
            resp = save_json_to_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH,
                                       st.session_state.mappings, commit_msg,
                                       author_name="AFF Mapping App", author_email="aff-app@coxautoinc.com",
                                       use_feature_branch=use_feature_branch)
            st.sidebar.success("Mappings committed ✅")
            st.sidebar.caption(f"Commit: {resp['commit']['sha'][:7]}")
            try:
                append_jsonl_to_github(GH_OWNER, GH_REPO, AUDIT_LOG_PATH, GH_TOKEN, GH_BRANCH,
                    {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "user":"streamlit-app", "action":"commit",
                     "count": len(st.session_state.mappings), "path": MAPPINGS_PATH,
                     "branch": GH_BRANCH if not use_feature_branch else "aff-mapping-app"},
                    commit_message="chore(app): append audit commit entry",
                    use_feature_branch=use_feature_branch)
            except Exception as log_err:
                st.sidebar.warning(f"Audit log append failed (non-blocking): {log_err}")
        except Exception as e:
            st.sidebar.error(f"Commit failed: {e}")
            st.sidebar.info("If main is protected, enable feature branch and merge via PR.")

st.sidebar.subheader("Backup / Restore")
backup = json.dumps(st.session_state.mappings, indent=2, ensure_ascii=False)
st.sidebar.download_button("⬇️ Download mappings.json", data=backup, file_name="mappings.json", mime="application/json")
uploaded = st.sidebar.file_uploader("⬆️ Upload mappings.json (local restore)", type=["json"])
if uploaded:
    try:
        st.session_state.mappings = json.load(uploaded)
        st.sidebar.success("Local restore complete. Remember to Commit to GitHub.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded JSON: {e}")

# --------------------- CADS settings / matching controls --------------
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

st.sidebar.subheader("Matching Controls")
TRIM_EXACT_ONLY = st.sidebar.checkbox("Trim must be exact (no token-subset)", value=True)
MODEL_EXACT_WHEN_FULL = st.sidebar.checkbox("Model exact when input is multi-word", value=True)
ENFORCE_LR_FAMILY = st.sidebar.checkbox(
    "Require same Land Rover family (Range Rover / Discovery / Defender)",
    value=True,
    help="When Make=Land Rover, only mappings from the same family as the input Model are allowed."
)
EXACT_MMT  = st.sidebar.checkbox("CADS fallback: Model exact (otherwise contains)", value=False)
STRICT_AND = st.sidebar.checkbox("Require strict AND across provided filters", value=True)
TABLE_HEIGHT = st.sidebar.slider("Results table height (px)", min_value=400, max_value=1200, value=700, step=50)

if st.sidebar.button("🧹 Clear (Interactive)", key="sidebar_clear_btn"):
    for k in ["year_input","make_input","model_input","trim_input","vehicle_input",
              "code_input","model_code_input","prev_inputs","results_df",
              "code_candidates","model_code_candidates","code_column","model_code_column","last_matches"]:
        st.session_state.pop(k, None)
    st.sidebar.success("Interactive inputs/state cleared.")

# --------------------- Mapping editor inputs --------------------------
st.subheader("Edit / Add Mapping")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: year = st.text_input("Year", key="year_input", placeholder="e.g., 2025")
with c2: make = st.text_input("Make", key="make_input", placeholder="e.g., Land Rover")
with c3: model = st.text_input("Model", key="model_input", placeholder="e.g., Range Rover Evoque / Discovery Sport")
with c4: trim = st.text_input("Trim", key="trim_input", placeholder="e.g., S / SE / R-Dynamic")
with c5: vehicle = st.text_input("Vehicle (alt)", key="vehicle_input", placeholder="Optional")
with c6: mapped_code = st.text_input("Mapped Code", key="code_input", placeholder="Optional (STYLE_ID/AD_VEH_ID/etc.)")
model_code_input = st.text_input("Model Code (optional)", key="model_code_input", placeholder="AD_MFGCODE/MODEL_CODE/etc.")

# Clear stale state when inputs change
current_inputs = (canon_text(year), canon_text(make), canon_text(model), canon_text(trim, True), canon_text(vehicle), (model_code_input or "").strip())
prev_inputs = st.session_state.get("prev_inputs")
if prev_inputs != current_inputs:
    for k in ["results_df","code_candidates","model_code_candidates","code_column","model_code_column","last_matches"]:
        st.session_state.pop(k, None)
    st.session_state["prev_inputs"] = current_inputs

# Debug banner for canonical inputs & LR family
detected_family = detect_lr_family(model) if canon_text(make) == "land rover" else None
st.caption(
    f"🔎 Canonical → Year='{canon_text(year)}' Make='{canon_text(make)}' "
    f"Model='{canon_text(model)}' (LR family={detected_family or 'n/a'}) "
    f"Trim='{canon_text(trim, True)}' | TRIM_EXACT_ONLY={TRIM_EXACT_ONLY}, "
    f"MODEL_EXACT_WHEN_FULL={MODEL_EXACT_WHEN_FULL}, ENFORCE_LR_FAMILY={ENFORCE_LR_FAMILY}"
)

# --------------------- Existing mapping detection (interactive) -------
best = pick_best_mapping(
    st.session_state.mappings, year, make, model, trim,
    trim_exact_only=TRIM_EXACT_ONLY,
    enforce_lr_family=ENFORCE_LR_FAMILY,
    model_exact_when_full=MODEL_EXACT_WHEN_FULL,
)

st.subheader("Existing Mapping (for current inputs)")
if best:
    k, v, score = best
    rows = [{
        "Match Level": "exact_trim_or_subset_best_model_token_year_lr_family",
        "Score": round(score,3),
        "Key": k, "Year": v.get("year",""), "Make": v.get("make",""),
        "Model": v.get("model",""), "Trim": v.get("trim",""),
        "Vehicle": v.get("vehicle",""), "Code": v.get("code",""),
        "Model Code": v.get("model_code",""),
    }]
    st.success("Already mapped: 1 match.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No existing mapping detected for current inputs.")
    # Trace: why rows were excluded
    cmk = canon_text(make); ctr = canon_text(trim, True)
    trace = {"trim_mismatch":0, "make_mismatch":0, "year_mismatch":0, "lr_family_mismatch":0, "candidates":0}
    sample_rows = []
    for k2, v2 in st.session_state.mappings.items():
        vmk = v2.get("make",""); vy = v2.get("year",""); vtr = v2.get("trim",""); vmd = v2.get("model","")
        # Trim gates
        tmatch, _ = trim_matches(vtr, ctr, exact_only=TRIM_EXACT_ONLY)
        if not tmatch:
            trace["trim_mismatch"] += 1; continue
        # Make gate
        if canon_text(vmk) != cmk:
            trace["make_mismatch"] += 1; continue
        # Year gate
        if not year_token_matches(vy, year):
            trace["year_mismatch"] += 1; continue
        # LR family gate
        if ENFORCE_LR_FAMILY and cmk == "land rover":
            uf = detect_lr_family(model)
            cf = detect_lr_family(vmd)
            if uf and cf and uf != cf:
                trace["lr_family_mismatch"] += 1; continue
        trace["candidates"] += 1
        sample_rows.append({
            "Key": k2, "Year": vy, "Make": vmk, "Model": vmd, "Trim": vtr,
            "ModelSimilarity": round(model_similarity(vmd, canon_text(model)), 3),
            "LRFamily": detect_lr_family(vmd)
        })
        if len(sample_rows) >= 12: break
    with st.expander("🧪 Matching trace"):
        st.write(trace)
        if sample_rows:
            st.dataframe(pd.DataFrame(sample_rows), use_container_width=True)

cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    if st.button("📋 Copy mapped Code to input", key="copy_code_btn"):
        if best:
            st.session_state["code_input"] = (best[1].get("code","") or "").strip()
            st.success("Copied mapped Code to input.")
        else:
            st.info("No mapped vehicle to copy from.")

# --------------------- Search CADS — single mapped record -------------
with cc3:
    if st.button("🔎 Search CADS", key="search_cads"):
        try:
            # Load CADS
            if cads_upload is not None:
                if cads_upload.name.lower().endswith(".xlsx"):
                    df_cads = pd.read_excel(cads_upload, engine="openpyxl")
                else:
                    df_cads = pd.read_csv(cads_upload)
            else:
                if CADS_IS_EXCEL:
                    sheet_arg = CADS_SHEET_NAME
                    try: sheet_arg = int(sheet_arg)
                    except Exception: pass
                    df_cads = load_cads_from_github_excel(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH, sheet_name=sheet_arg)
                else:
                    df_cads = load_cads_from_github_csv(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH)

            df_cads = _strip_object_columns(df_cads)

            if best:
                mapping = best[1]
                df_match = match_cads_rows_for_mapping(
                    df_cads, mapping,
                    exact_mmt=EXACT_MMT,
                    strict_and=STRICT_AND,
                    trim_exact_only=TRIM_EXACT_ONLY,
                    enforce_lr_family=ENFORCE_LR_FAMILY
                )
                if len(df_match) > 0:
                    st.success(f"Found {len(df_match)} CADS row(s) for mapped vehicle (Code→ModelCode→YMMT).")
                    selectable = df_match.copy()
                    if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                    st.session_state["results_df"] = selectable
                    st.session_state["code_candidates"] = get_cads_code_candidates(selectable)
                    st.session_state["model_code_candidates"] = get_model_code_candidates(selectable)
                    st.session_state["code_column"] = st.session_state["code_candidates"][0] if st.session_state["code_candidates"] else None
                    st.session_state["model_code_column"] = st.session_state["model_code_candidates"][0] if st.session_state["model_code_candidates"] else None
                else:
                    st.warning("No CADS rows found by Code/ModelCode/YMMT for the mapped vehicle.")
            else:
                st.info("No mapped vehicle found yet; running filter on current inputs (Trim exact/subset; Model contains; Year token-aware).")
                results = filter_cads(
                    df_cads, year, make, model, trim, "", "",
                    exact_mmt=EXACT_MMT, case_sensitive=False, strict_and=STRICT_AND,
                    trim_exact_only=TRIM_EXACT_ONLY, enforce_lr_family=ENFORCE_LR_FAMILY
                )
                if len(results) == 0:
                    st.warning("No CADS rows matched inputs. Try toggling Model exact/contains or relaxing Trim exact-only.")
                else:
                    st.success(f"Found {len(results)} CADS row(s).")
                    selectable = results.copy()
                    if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                    st.session_state["results_df"] = selectable
                    st.session_state["code_candidates"] = get_cads_code_candidates(selectable)
                    st.session_state["model_code_candidates"] = get_model_code_candidates(selectable)
                    st.session_state["code_column"] = st.session_state["code_candidates"][0] if st.session_state["code_candidates"] else None
                    st.session_state["model_code_column"] = st.session_state["model_code_candidates"][0] if st.session_state["model_code_candidates"] else None
        except FileNotFoundError as fnf:
            st.error(str(fnf))
            st.info(f"Ensure CADS exists at `{CADS_PATH}` in `{GH_OWNER}/{GH_REPO}` @ `{GH_BRANCH}`.")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

with cc4:
    if st.button("📋 Copy first CADS Model Code to input", key="copy_model_code_btn"):
        if "results_df" in st.session_state:
            df_r = st.session_state["results_df"]
            mc_cols = [c for c in CADS_MODEL_CODE_PREFS if c in df_r.columns] or list(df_r.columns)
            mc_col = mc_cols[0] if mc_cols else None
            st.session_state["model_code_column"] = mc_col
            if mc_col:
                vals = df_r[mc_col].dropna().tolist()
                if vals:
                    st.session_state["model_code_input"] = str(vals[0]).strip()
                    st.success(f"Copied model code '{vals[0]}' to Model Code input.")
                else:
                    st.info("No model code values in current CADS results.")
            else:
                st.info("No model code column detected in current CADS results.")

# Clear button (interactive area)
if st.button("🧹 Clear Inputs (Interactive)", key="clear_inputs_btn"):
    for k in ["year_input","make_input","model_input","trim_input","vehicle_input",
              "code_input","model_code_input","prev_inputs","results_df",
              "code_candidates","model_code_candidates","code_column","model_code_column","last_matches"]:
        st.session_state.pop(k, None)
    st.success("Inputs cleared.")

st.caption("Local changes persist while you navigate pages. Use **Commit mappings to GitHub** (sidebar) to save permanently.")

# --------------------- Results selection ------------------------------
if "results_df" in st.session_state:
    st.subheader("Select vehicles from CADS results")
    code_candidates = st.session_state.get("code_candidates", [])
    model_code_candidates = st.session_state.get("model_code_candidates", [])
    st.session_state["code_column"] = st.selectbox(
        "Mapped Code column (from CADS results)",
        options=code_candidates if code_candidates else list(st.session_state["results_df"].columns),
        index=0 if code_candidates else 0,
        key="code_column_select",
    )
    st.session_state["model_code_column"] = st.selectbox(
        "Model Code column (from CADS results)",
        options=model_code_candidates if model_code_candidates else list(st.session_state["results_df"].columns),
        index=0 if model_code_candidates else 0,
        key="model_code_column_select",
    )

    df_show = st.session_state["results_df"]
    front_cols = [c for c in ["Select","Similarity"] if c in df_show.columns]
    col_order = front_cols + [c for c in df_show.columns if c not in front_cols]

    csel1, csel2 = st.columns(2)
    with csel1:
        if st.button("✅ Select All", key="select_all_btn"):
            df_tmp = df_show.copy(); df_tmp["Select"] = True
            st.session_state["results_df"] = df_tmp; df_show = df_tmp
    with csel2:
        if st.button("🧹 Clear Selection", key="clear_selection_btn"):
            df_tmp = df_show.copy(); df_tmp["Select"] = False
            st.session_state["results_df"] = df_tmp; df_show = df_tmp

    edited = st.data_editor(
        df_show, key="results_editor", use_container_width=True,
        num_rows="dynamic", column_order=col_order, height=TABLE_HEIGHT
    )
    st.session_state["results_df"] = edited

    selected_rows = edited[edited["Select"] == True]
    st.caption(f"Selected {len(selected_rows)} vehicle(s).")

    if st.button("➕ Add selected vehicle(s) to mappings", key="add_selected_to_mappings"):
        if selected_rows.empty:
            st.warning("No rows selected.")
        else:
            df2 = selected_rows.copy()
            year_col    = next((c for c in ["AD_YEAR","Year","MY","ModelYear","Model Year"] if c in df2.columns), None)
            make_col    = next((c for c in ["AD_MAKE","Make","MakeName","Manufacturer"] if c in df2.columns), None)
            model_col   = next((c for c in ["AD_MODEL","Model","Line","Carline","Series"] if c in df2.columns), None)
            trim_col    = next((c for c in ["AD_TRIM","Trim","Grade","Variant","Submodel"] if c in df2.columns), None)
            vehicle_col = next((c for c in ["Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series"] if c in df2.columns), None)
            code_col        = st.session_state.get("code_column")
            model_code_col  = st.session_state.get("model_code_column")

            added = 0
            for _, row in df2.iterrows():
                yv  = (row.get(year_col, "") if year_col else "").strip()
                mkv = (row.get(make_col, "") if make_col else "").strip()
                mdv = (row.get(model_col,"") if model_col else "").strip()
                trv = (row.get(trim_col, "") if trim_col else "").strip()
                vhv = (row.get(vehicle_col,"") if vehicle_col else "").strip()
                key = f"{yv}-{mkv}-{mdv}-{trv}".strip("-")

                code_val       = (str(row.get(code_col, "")) if code_col else "").strip()
                model_code_val = (str(row.get(model_code_col, "")) if model_code_col else "").strip()

                existing = st.session_state.mappings.get(key)
                if existing and (existing.get("code") != code_val or existing.get("model_code") != model_code_val):
                    st.warning(f"Key '{key}' exists with different Code/Model Code. Overwriting.")
                st.session_state.mappings[key] = {
                    "year": yv, "make": mkv, "model": mdv, "trim": trv,
                    "vehicle": vhv, "code": code_val, "model_code": model_code_val,
                }
                added += 1
            st.success(f"Added/updated {added} mapping(s). Commit them in the sidebar.")

# --------------------- Current mappings table -------------------------
st.subheader("Current Mappings (session)")
if st.session_state.mappings:
    rows = []
    for k, v in st.session_state.mappings.items():
        rows.append({
            "Key": k,
            "Year": v.get("year",""),
            "Make": v.get("make",""),
            "Model": v.get("model",""),
            "Trim": v.get("trim",""),
            "Vehicle": v.get("vehicle",""),
            "Code": v.get("code",""),
            "Model Code": v.get("model_code",""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No mappings yet. Add one above or select CADS rows to add mappings.")

# --- EOF ---
