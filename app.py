
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# Generic model matching for 38+ OEMs using:
#  - "Effective model" (concat of model-like CADS columns)
#  - Per-make token frequencies to identify stopwords (high-frequency tokens)
#  - Tiered matching: Token-AND → Token-OR → Contains (non-stopword) → Contains (full text) → Fuzzy suggestions
#  - Trim exact or token-subset; Year token-aware; Make exact
# No hardcoded model lists per OEM.
# Repo: klb-text/map, Branch: main

import base64, json, time, io, re, difflib
from typing import Optional, List, Dict, Tuple, Set
import requests, pandas as pd, streamlit as st

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

# ID columns in CADS (union scans)
CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ---------------------------------------------------------------------
# Resilient HTTP
# ---------------------------------------------------------------------
from requests.adapters import HTTPAdapter, Retry
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","PUT","POST"])
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
    r_create = _post(f"https://api.github.com/repos/{owner}/{repo}/git/refs", headers=gh_headers(token),
                     json={"ref": f"refs/heads/{feature_branch}", "sha": base_sha})
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
# Canonicalization / tokens / year / trim
# ---------------------------------------------------------------------
def canon_text(val: str, for_trim: bool=False) -> str:
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

def tokens(s: str, min_len: int = 2) -> List[str]:
    s = canon_text(s)
    tks = re.split(r"[^\w]+", s)
    return [t for t in tks if t and len(t) >= min_len]

def _trim_tokens(s: str) -> Set[str]:
    return set(tokens(canon_text(s, True)))

def _extract_years_from_text(s: str) -> set:
    s = (s or "").strip().lower()
    years = set()
    for m in re.finditer(r"\b(19[5-9]\d|20[0-4]\d|2050)\b", s):
        years.add(int(m.group(0)))
    for m in re.finditer(r"\bmy\s*([0-9]{2})\b", s):
        years.add(2000 + int(m.group(1)))
    for m in re.finditer(r"\b(?:q[1-4][\-\s]*)?(19[5-9]\d|20[0-4]\d|2050)\b", s):
        years.add(int(m.group(1)))
    if not years:
        for m in re.finditer(r"\b([0-9]{2})\b", s):
            years.add(2000 + int(m.group(1)))
    return years

def year_token_matches(mapping_year: str, user_year: str) -> bool:
    uy_set = _extract_years_from_text(user_year)
    my_set = _extract_years_from_text(mapping_year)
    if not uy_set: return True
    if not my_set: return False
    return bool(uy_set.intersection(my_set))

def trim_matches(row_trim: str, user_trim: str, exact_only: bool=False) -> Tuple[bool, float]:
    row = canon_text(row_trim, True)
    usr = canon_text(user_trim, True)
    if not usr: return (True, 0.5)
    if row == usr: return (True, 1.0)
    if exact_only: return (False, 0.0)
    if _trim_tokens(usr).issubset(_trim_tokens(row)): return (True, 0.8)
    return (False, 0.0)

def model_similarity(a: str, b: str) -> float:
    a = canon_text(a); b = canon_text(b)
    if not a and not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a: return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------------------------------------------------------------------
# Detect model-like columns & build "effective model"
# ---------------------------------------------------------------------
# Expanded detection: covers common variants across 38 OEMs
MODEL_LIKE_REGEX  = re.compile(r"(?:^|_|\s)(model(name)?|car\s*line|carline|line|series)(?:$|_|\s)", re.I)
SERIES_LIKE_REGEX = re.compile(r"(?:^|_|\s)(series(name)?|sub(?:_|-)?model|body(?:_|-)?style|body|trim|grade|variant|description|modeltrim|name)(?:$|_|\s)", re.I)

def detect_model_like_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    model_cols  = [c for c in cols if MODEL_LIKE_REGEX.search(c)]
    series_cols = [c for c in cols if SERIES_LIKE_REGEX.search(c) and c not in model_cols]
    # Dedup; keep order stable
    return (list(dict.fromkeys(model_cols)), list(dict.fromkeys(series_cols)))

def effective_model_row(row: pd.Series, model_cols: List[str], series_cols: List[str]) -> str:
    parts = []
    for c in model_cols + series_cols:
        if c in row.index:
            v = str(row.get(c, "") or "").strip()
            if v:
                parts.append(v)
    # collapse into canonical effective model
    return canon_text(" ".join(parts))

def add_effective_model_column(df: pd.DataFrame, override_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    If override_cols provided, use only these columns to build effective model.
    Otherwise auto-detect model-like & series-like columns.
    Returns (df_with_effective_model, used_model_cols, used_series_cols).
    """
    if override_cols:
        used_model_cols  = [c for c in override_cols if c in df.columns]
        used_series_cols = []
    else:
        used_model_cols, used_series_cols = detect_model_like_columns(df)
    if not used_model_cols and not used_series_cols:
        df["__effective_model__"] = ""
        return df, used_model_cols, used_series_cols
    df["__effective_model__"] = df.apply(lambda r: effective_model_row(r, used_model_cols, used_series_cols), axis=1)
    return df, used_model_cols, used_series_cols

# ---------------------------------------------------------------------
# Per-make stopwords (data-driven)
# ---------------------------------------------------------------------
def compute_per_make_stopwords(
    df: pd.DataFrame, stopword_threshold: float = 0.40, token_min_len: int = 2
) -> Set[str]:
    """
    Data-driven: tokens that appear in >= stopword_threshold of rows (already make-sliced)
    are treated as non-discriminant.
    """
    if "__effective_model__" not in df.columns:
        df, _, _ = add_effective_model_column(df)
    total = len(df)
    if total == 0: return set()
    freq: Dict[str,int] = {}
    for _, row in df.iterrows():
        toks = set(tokens(row["__effective_model__"], min_len=token_min_len))
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
    return {t for t, c in freq.items() if (c / total) >= float(stopword_threshold)}

# ---------------------------------------------------------------------
# Single-best mapping picker (generic)
# ---------------------------------------------------------------------
def pick_best_mapping(
    mappings: Dict[str, Dict[str, str]],
    year: str, make: str, model: str, trim: str,
    trim_exact_only: bool = False,
    model_exact_when_full: bool = True,
) -> Optional[Tuple[str, Dict[str, str], float]]:
    cmk = canon_text(make)
    ctr = canon_text(trim, True)
    cy  = (year or "")
    cmd = canon_text(model)

    if not cmk:
        return None

    force_exact_model = model_exact_when_full and len(cmd.split()) >= 2

    candidates: List[Tuple[str, Dict[str, str], float]] = []
    for k, v in mappings.items():
        vmk = v.get("make","")
        vy  = v.get("year","")
        vtr = v.get("trim","")
        vmd = v.get("model","")

        if canon_text(vmk) != cmk:                   # Make exact
            continue
        if not year_token_matches(vy, cy):           # Year token-aware
            continue
        tmatch, tscore = trim_matches(vtr, ctr, exact_only=trim_exact_only)  # Trim gate
        if not tmatch:
            continue

        ms = model_similarity(vmd, cmd)
        if force_exact_model and canon_text(vmd) != cmd:
            ms = ms * 0.5  # damp non-exact for multi-word input

        score = tscore * 0.6 + ms * 0.4  # prioritize trim, then model
        candidates.append((k, v, score))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[0]  # (key, value, score)

# ---------------------------------------------------------------------
# Generic CADS filter using effective model + stopwords (tiered)
# ---------------------------------------------------------------------
def _tiered_model_mask(eff: pd.Series, md: str, discriminant: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Return masks for the 4 tiers:
      and_mask: token-AND on discriminants
      or_mask:  token-OR  on discriminants
      ns_mask:  contains on non-stopword text (join discriminants)
      full_mask: contains on full model text
    """
    if not md:
        true_mask = pd.Series([True]*len(eff), index=eff.index)
        return true_mask, true_mask, true_mask, true_mask
    and_mask  = eff.apply(lambda s: all(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    or_mask   = eff.apply(lambda s: any(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    ns_text   = " ".join(discriminant).strip()
    ns_mask   = eff.str.contains(ns_text, na=False) if ns_text else eff.str.contains(md, na=False)
    full_mask = eff.str.contains(md, na=False)
    return and_mask, or_mask, ns_mask, full_mask

def filter_cads_generic(
    df: pd.DataFrame,
    year: str, make: str, model: str, trim: str,
    exact_model_when_full: bool,
    trim_exact_only: bool,
    strict_and: bool,
    stopword_threshold: float,
    token_min_len: int,
    effective_model_cols_override: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    df2 = _strip_object_columns(df.copy())
    df2, used_model_cols, used_series_cols = add_effective_model_column(df2, override_cols=effective_model_cols_override)

    YEAR_CANDS  = ["AD_YEAR","Year","MY","ModelYear","Model Year"]
    MAKE_CANDS  = ["AD_MAKE","Make","MakeName","Manufacturer"]
    TRIM_CANDS  = ["AD_TRIM","Trim","Grade","Variant","Submodel"]

    year_col = next((c for c in YEAR_CANDS if c in df2.columns), None)
    make_col = next((c for c in MAKE_CANDS if c in df2.columns), None)
    trim_col = next((c for c in TRIM_CANDS if c in df2.columns), None)

    y  = (year or "")
    mk = canon_text(make)
    md = canon_text(model)
    tr = canon_text(trim, True)

    masks = []

    # Make slice and stopwords
    if make_col and mk:
        s_make = df2[make_col].astype(str).str.lower()
        masks.append(s_make == mk)
        df_make_slice = df2[s_make == mk]
    else:
        df_make_slice = df2

    eff = df2["__effective_model__"]
    user_tokens = tokens(md, min_len=token_min_len)
    make_stopwords = compute_per_make_stopwords(df_make_slice, stopword_threshold, token_min_len)
    discriminant = [t for t in user_tokens if t not in make_stopwords]

    # MODEL tiered masks
    and_mask, or_mask, ns_mask, full_mask = _tiered_model_mask(eff, md, discriminant)
    if md:
        # Start with AND (primary)
        masks.append(and_mask)

    # Trim gate
    if trim_col and tr:
        s_trim = df2[trim_col].astype(str)
        m_exact  = s_trim.str.lower() == tr
        if trim_exact_only:
            masks.append(m_exact)
        else:
            m_subset = s_trim.apply(lambda x: _trim_tokens(tr).issubset(_trim_tokens(x)))
            masks.append(m_exact | m_subset)

    # Year token-aware
    if year_col and y:
        s_year = df2[year_col].astype(str)
        masks.append(s_year.apply(lambda vy: year_token_matches(vy, y)))

    # Combine masks
    if not masks:
        result = df2.iloc[0:0]
    else:
        m = masks[0]
        for mm in masks[1:]:
            m = (m & mm) if strict_and else (m | mm)
        result = df2[m]

    # Fallback tiers if empty
    tier_used = "AND"
    if md and len(result) == 0:
        # Try OR
        tier_used = "OR"
        m2 = or_mask
        for mm in masks[1:]:  # reapply non-model masks
            m2 = (m2 & mm) if strict_and else (m2 | mm)
        result = df2[m2]
        if len(result) == 0:
            # Try non-stopword contains
            tier_used = "NS_CONTAINS"
            m3 = ns_mask
            for mm in masks[1:]:
                m3 = (m3 & mm) if strict_and else (m3 | mm)
            result = df2[m3]
            if len(result) == 0:
                # Try full model contains
                tier_used = "FULL_CONTAINS"
                m4 = full_mask
                for mm in masks[1:]:
                    m4 = (m4 & mm) if strict_and else (m4 | mm)
                result = df2[m4]

    diag = {
        "used_model_cols": used_model_cols,
        "used_series_cols": used_series_cols,
        "make_stopwords": sorted(list(make_stopwords))[:50],
        "user_tokens": user_tokens,
        "discriminant_tokens": discriminant,
        "tier_used": tier_used,
        "rows_after_tier": len(result),
    }
    return result, diag

# ---------------------------------------------------------------------
# CADS matching for a single mapping: Code → Model Code → generic fallback
# ---------------------------------------------------------------------
def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_CODE_PREFS if c in df.columns] or list(df.columns)

def get_model_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_MODEL_CODE_PREFS if c in df.columns] or list(df.columns)

def match_cads_rows_for_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    exact_model_when_full: bool,
    trim_exact_only: bool,
    strict_and: bool,
    stopword_threshold: float,
    token_min_len: int,
    effective_model_cols_override: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    df2 = _strip_object_columns(df.copy())
    df2, used_model_cols, used_series_cols = add_effective_model_column(df2, override_cols=effective_model_cols_override)

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
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True), {
                "used_model_cols": used_model_cols, "used_series_cols": used_series_cols, "tier_used": "CODE"
            }

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
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True), {
                "used_model_cols": used_model_cols, "used_series_cols": used_series_cols, "tier_used": "MODEL_CODE"
            }

    # Generic fallback using effective model + stopwords
    res, diag = filter_cads_generic(
        df2,
        mapping.get("year",""), mapping.get("make",""), mapping.get("model",""), mapping.get("trim",""),
        exact_model_when_full=exact_model_when_full,
        trim_exact_only=trim_exact_only,
        strict_and=strict_and,
        stopword_threshold=stopword_threshold,
        token_min_len=token_min_len,
        effective_model_cols_override=effective_model_cols_override,
    )
    diag.update({"used_model_cols": used_model_cols, "used_series_cols": used_series_cols})
    return res, diag

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
        st.sidebar.success("Reloaded.")
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
            st.sidebar.success("Committed ✅")
            st.sidebar.caption(f"Commit: {resp['commit']['sha'][:7]}")
            try:
                append_jsonl_to_github(GH_OWNER, GH_REPO, AUDIT_LOG_PATH, GH_TOKEN, GH_BRANCH,
                    {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "user":"streamlit-app","action":"commit",
                     "count": len(st.session_state.mappings),"path": MAPPINGS_PATH,
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
uploaded = st.sidebar.file_uploader("⬆️ Upload mappings.json (restore)", type=["json"])
if uploaded:
    try:
        st.session_state.mappings = json.load(uploaded)
        st.sidebar.success("Local restore complete. Remember to Commit.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded JSON: {e}")

# --------------------- CADS settings & matching controls --------------
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

st.sidebar.subheader("Matching Controls")
TRIM_EXACT_ONLY = st.sidebar.checkbox("Trim must be exact (no token-subset)", value=False)
MODEL_EXACT_WHEN_FULL = st.sidebar.checkbox("Model exact when input is multi-word", value=False)
STRICT_AND = st.sidebar.checkbox("Require strict AND across provided filters", value=True)
STOPWORD_THRESHOLD = st.sidebar.slider("Per-make stopword threshold", 0.1, 0.9, 0.50, 0.05,
                                       help="Tokens appearing in ≥ threshold of rows for the Make are treated as non-discriminant.")
TOKEN_MIN_LEN = st.sidebar.slider("Token minimum length", 1, 5, 2, 1,
                                  help="Ignore tokens shorter than this for effective-model matching.")

st.sidebar.subheader("Effective Model (override)")
EFFECTIVE_MODEL_COLS_OVERRIDE = st.sidebar.text_input(
    "Comma-separated CADS column names to use for effective model (optional)",
    value="",
    help="Example: AD_MODEL,AD_SERIES,ModelTrim,Description. If blank, auto-detect."
)
OVERRIDE_COLS = [c.strip() for c in EFFECTIVE_MODEL_COLS_OVERRIDE.split(",") if c.strip()] or None

TABLE_HEIGHT = st.sidebar.slider("Results table height (px)", min_value=400, max_value=1200, value=700, step=50)

if st.sidebar.button("🧹 Clear (Interactive)", key="sidebar_clear_btn"):
    for k in ["year_input","make_input","model_input","trim_input","vehicle_input",
              "code_input","model_code_input","prev_inputs",
              "results_df_mapped","results_df_inputs",
              "code_candidates_mapped","code_candidates_inputs",
              "model_code_candidates_mapped","model_code_candidates_inputs",
              "code_column_mapped","code_column_inputs","model_code_column_mapped","model_code_column_inputs",
              "last_diag_inputs","last_diag_mapped"]:
        st.session_state.pop(k, None)
    st.sidebar.success("Interactive state cleared.")

# --------------------- Mapping editor inputs --------------------------
st.subheader("Edit / Add Mapping")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: year = st.text_input("Year", key="year_input", placeholder="e.g., 2025")
with c2: make = st.text_input("Make", key="make_input", placeholder="e.g., Land Rover")
with c3: model = st.text_input("Model", key="model_input", placeholder="e.g., Range Rover Evoque")
with c4: trim = st.text_input("Trim", key="trim_input", placeholder="e.g., S / SE / R-Dynamic")
with c5: vehicle = st.text_input("Vehicle (alt)", key="vehicle_input", placeholder="Optional")
with c6: mapped_code = st.text_input("Mapped Code", key="code_input", placeholder="Optional (STYLE_ID/AD_VEH_ID/etc.)")
model_code_input = st.text_input("Model Code (optional)", key="model_code_input", placeholder="AD_MFGCODE/MODEL_CODE/etc.")

# Clear stale state when inputs change
current_inputs = (canon_text(year), canon_text(make), canon_text(model), canon_text(trim, True), canon_text(vehicle), (model_code_input or "").strip())
prev_inputs = st.session_state.get("prev_inputs")
if prev_inputs != current_inputs:
    for k in ["results_df_mapped","results_df_inputs",
              "code_candidates_mapped","code_candidates_inputs",
              "model_code_candidates_mapped","model_code_candidates_inputs",
              "code_column_mapped","code_column_inputs","model_code_column_mapped","model_code_column_inputs",
              "last_diag_inputs","last_diag_mapped"]:
        st.session_state.pop(k, None)
    st.session_state["prev_inputs"] = current_inputs

# Debug banner
st.caption(
    f"🔎 Inputs → Year='{canon_text(year)}' Make='{canon_text(make)}' "
    f"Model='{canon_text(model)}' Trim='{canon_text(trim, True)}' | "
    f"TRIM_EXACT_ONLY={TRIM_EXACT_ONLY}, MODEL_EXACT_WHEN_FULL={MODEL_EXACT_WHEN_FULL}, "
    f"STOPWORD_THRESHOLD={STOPWORD_THRESHOLD}, TOKEN_MIN_LEN={TOKEN_MIN_LEN}, "
    f"OverrideCols={OVERRIDE_COLS or '(auto)'}"
)

# --------------------- Existing mapping detection (interactive) -------
best = pick_best_mapping(
    st.session_state.mappings, year, make, model, trim,
    trim_exact_only=TRIM_EXACT_ONLY,
    model_exact_when_full=MODEL_EXACT_WHEN_FULL,
)

st.subheader("Existing Mapping (for current inputs)")
if best:
    k, v, score = best
    rows = [{
        "Match Level": "generic_best_trim_model_year",
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

# --------------------- CADS search buttons ----------------------------
b1, b2, b3, b4 = st.columns(4)
with b2:
    if st.button("🔎 Search CADS (mapped vehicle)", key="search_cads_mapped"):
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
                df_match, diag = match_cads_rows_for_mapping(
                    df_cads, mapping,
                    exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                    trim_exact_only=TRIM_EXACT_ONLY,
                    strict_and=STRICT_AND,
                    stopword_threshold=STOPWORD_THRESHOLD,
                    token_min_len=TOKEN_MIN_LEN,
                    effective_model_cols_override=OVERRIDE_COLS,
                )
                st.session_state["last_diag_mapped"] = diag
                if len(df_match) > 0:
                    st.success(f"Found {len(df_match)} CADS row(s) for mapped vehicle (tier={diag.get('tier_used')}).")
                    selectable = df_match.copy()
                    if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                    st.session_state["results_df_mapped"] = selectable
                    st.session_state["code_candidates_mapped"] = get_cads_code_candidates(selectable)
                    st.session_state["model_code_candidates_mapped"] = get_model_code_candidates(selectable)
                    st.session_state["code_column_mapped"] = st.session_state["code_candidates_mapped"][0] if st.session_state["code_candidates_mapped"] else None
                    st.session_state["model_code_column_mapped"] = st.session_state["model_code_candidates_mapped"][0] if st.session_state["model_code_candidates_mapped"] else None
                else:
                    st.warning("No CADS rows found via Code/ModelCode/tiered fallback for the mapped vehicle.")
            else:
                st.info("No mapped vehicle; use 'Search CADS (use current inputs)'.")
        except FileNotFoundError as fnf:
            st.error(str(fnf))
        except Exception as e:
            st.error(f"CADS search failed: {e}")

with b3:
    if st.button("🔎 Search CADS (use current inputs)", key="search_cads_inputs"):
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

            results, diag = filter_cads_generic(
                df_cads,
                year, make, model, trim,
                exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                trim_exact_only=TRIM_EXACT_ONLY,
                strict_and=STRICT_AND,
                stopword_threshold=STOPWORD_THRESHOLD,
                token_min_len=TOKEN_MIN_LEN,
                effective_model_cols_override=OVERRIDE_COLS,
            )
            st.session_state["last_diag_inputs"] = diag
            if len(results) == 0:
                st.warning("No CADS rows matched inputs. Try columns override, raise stopword threshold (e.g., 0.6), turn OFF 'Model exact when full', or relax Trim.")
            else:
                st.success(f"Found {len(results)} CADS row(s) for current inputs (tier={diag.get('tier_used')}).")
                selectable = results.copy()
                if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                st.session_state["results_df_inputs"] = selectable
                st.session_state["code_candidates_inputs"] = get_cads_code_candidates(selectable)
                st.session_state["model_code_candidates_inputs"] = get_model_code_candidates(selectable)
                st.session_state["code_column_inputs"] = st.session_state["code_candidates_inputs"][0] if st.session_state["code_candidates_inputs"] else None
                st.session_state["model_code_column_inputs"] = st.session_state["model_code_candidates_inputs"][0] if st.session_state["model_code_candidates_inputs"] else None
        except FileNotFoundError as fnf:
            st.error(str(fnf))
        except Exception as e:
            st.error(f"CADS search failed: {e}")

# --------------------- Diagnostics: effective model & tokens ----------
with st.expander("🔬 Effective model diagnostics (current CADS / Make slice)"):
    try:
        # Load CADS once for preview
        if cads_upload is not None:
            if cads_upload.name.lower().endswith(".xlsx"):
                df_prev = pd.read_excel(cads_upload, engine="openpyxl")
            else:
                df_prev = pd.read_csv(cads_upload)
        else:
            if CADS_IS_EXCEL:
                sheet_arg = CADS_SHEET_NAME
                try: sheet_arg = int(sheet_arg)
                except Exception: pass
                df_prev = load_cads_from_github_excel(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH, sheet_name=sheet_arg)
            else:
                df_prev = load_cads_from_github_csv(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH)
        df_prev = _strip_object_columns(df_prev)
        df_prev, used_m_cols, used_s_cols = add_effective_model_column(df_prev, override_cols=OVERRIDE_COLS)
        st.write({"used_model_cols": used_m_cols, "used_series_cols": used_s_cols})

        # Show top 50 effective models for current Make slice
        mk = canon_text(make)
        make_col = next((c for c in ["AD_MAKE","Make","MakeName","Manufacturer"] if c in df_prev.columns), None)
        if make_col and mk:
            df_slice = df_prev[df_prev[make_col].astype(str).str.lower() == mk]
        else:
            df_slice = df_prev
        st.write({"rows_in_make_slice": len(df_slice)})

        preview = (df_slice["__effective_model__"].value_counts().reset_index().rename(columns={"index":"effective_model","__effective_model__":"count"})).head(50)
        st.dataframe(preview, use_container_width=True)

        # Show tokens/stopwords/discriminants for current inputs
        u_tokens = tokens(canon_text(model), min_len=TOKEN_MIN_LEN)
        stopwords = compute_per_make_stopwords(df_slice, STOPWORD_THRESHOLD, TOKEN_MIN_LEN)
        disc = [t for t in u_tokens if t not in stopwords]
        st.write({"user_tokens": u_tokens, "stopwords": sorted(list(stopwords))[:50], "discriminant_tokens": disc})
    except Exception as e:
        st.warning(f"Diagnostics preview failed: {e}")

# --------------------- Results tables: Mapped Vehicle -----------------
if "results_df_mapped" in st.session_state:
    st.subheader("Select vehicles from CADS results — Mapped Vehicle")
    df_show = st.session_state["results_df_mapped"]
    code_candidates = st.session_state.get("code_candidates_mapped", [])
    model_code_candidates = st.session_state.get("model_code_candidates_mapped", [])

    st.session_state["code_column_mapped"] = st.selectbox(
        "Mapped Code column (mapped results)",
        options=code_candidates if code_candidates else list(df_show.columns),
        index=0 if code_candidates else 0,
        key="code_column_select_mapped",
    )
    st.session_state["model_code_column_mapped"] = st.selectbox(
        "Model Code column (mapped results)",
        options=model_code_candidates if model_code_candidates else list(df_show.columns),
        index=0 if model_code_candidates else 0,
        key="model_code_column_select_mapped",
    )

    st.caption(f"Tier used: {st.session_state.get('last_diag_mapped', {}).get('tier_used', '(n/a)')}")
    st.caption(f"Effective model columns: {st.session_state.get('last_diag_mapped', {}).get('used_model_cols', [])} + {st.session_state.get('last_diag_mapped', {}).get('used_series_cols', [])}")

    front_cols = [c for c in ["Select","__effective_model__"] if c in df_show.columns]
    col_order = front_cols + [c for c in df_show.columns if c not in front_cols]

    csel1, csel2 = st.columns(2)
    with csel1:
        if st.button("✅ Select All (mapped)", key="select_all_mapped_btn"):
            df_tmp = df_show.copy(); df_tmp["Select"] = True
            st.session_state["results_df_mapped"] = df_tmp; df_show = df_tmp
    with csel2:
        if st.button("🧹 Clear Selection (mapped)", key="clear_selection_mapped_btn"):
            df_tmp = df_show.copy(); df_tmp["Select"] = False
            st.session_state["results_df_mapped"] = df_tmp; df_show = df_tmp

    edited = st.data_editor(df_show, key="results_editor_mapped", use_container_width=True,
                            num_rows="dynamic", column_order=col_order, height=TABLE_HEIGHT)
    st.session_state["results_df_mapped"] = edited
    selected_rows = edited[edited["Select"] == True]
    st.caption(f"(Mapped) Selected {len(selected_rows)} vehicle(s).")

    if st.button("➕ Add selected (mapped) to mappings", key="add_selected_to_mappings_mapped"):
        if selected_rows.empty:
            st.warning("No rows selected (mapped).")
        else:
            df2 = selected_rows.copy()
            year_col    = next((c for c in ["AD_YEAR","Year","MY","ModelYear","Model Year"] if c in df2.columns), None)
            make_col    = next((c for c in ["AD_MAKE","Make","MakeName","Manufacturer"] if c in df2.columns), None)
            model_col   = next((c for c in ["AD_MODEL","Model","Line","Carline","Series"] if c in df2.columns), None)
            trim_col    = next((c for c in ["AD_TRIM","Trim","Grade","Variant","Submodel"] if c in df2.columns), None)
            vehicle_col = next((c for c in ["Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series"] if c in df2.columns), None)
            code_col        = st.session_state.get("code_column_mapped")
            model_code_col  = st.session_state.get("model_code_column_mapped")

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

                st.session_state.mappings[key] = {
                    "year": yv, "make": mkv, "model": mdv, "trim": trv,
                    "vehicle": vhv, "code": code_val, "model_code": model_code_val,
                }
                added += 1
            st.success(f"[Mapped] Added/updated {added} mapping(s).")

# --------------------- Results tables: Direct Input -------------------
if "results_df_inputs" in st.session_state:
    st.subheader("Select vehicles from CADS results — Direct Input Search")
    df_show = st.session_state["results_df_inputs"]
    code_candidates = st.session_state.get("code_candidates_inputs", [])
    model_code_candidates = st.session_state.get("model_code_candidates_inputs", [])

    st.session_state["code_column_inputs"] = st.selectbox(
        "Mapped Code column (input results)",
        options=code_candidates if code_candidates else list(df_show.columns),
        index=0 if code_candidates else 0,
        key="code_column_select_inputs",
    )
    st.session_state["model_code_column_inputs"] = st.selectbox(
        "Model Code column (input results)",
        options=model_code_candidates if model_code_candidates else list(df_show.columns),
        index=0 if model_code_candidates else 0,
        key="model_code_column_select_inputs",
    )

    st.caption(f"Tier used: {st.session_state.get('last_diag_inputs', {}).get('tier_used', '(n/a)')}")
    st.caption(f"Effective model columns: {st.session_state.get('last_diag_inputs', {}).get('used_model_cols', [])} + {st.session_state.get('last_diag_inputs', {}).get('used_series_cols', [])}")

    front_cols = [c for c in ["Select","__effective_model__"] if c in df_show.columns]
    col_order = front_cols + [c for c in df_show.columns if c not in front_cols]

    csel1, csel2 = st.columns(2)
    with csel1:
        if st.button("✅ Select All (inputs)", key="select_all_inputs_btn"):
            df_tmp = df_show.copy(); df_tmp["Select"] = True
            st.session_state["results_df_inputs"] = df_tmp; df_show = df_tmp
    with csel2:
        if st.button("🧹 Clear Selection (inputs)", key="clear_selection_inputs_btn"):
            df_tmp = df_show.copy(); df_tmp["Select"] = False
            st.session_state["results_df_inputs"] = df_tmp; df_show = df_tmp

    edited = st.data_editor(df_show, key="results_editor_inputs", use_container_width=True,
                            num_rows="dynamic", column_order=col_order, height=TABLE_HEIGHT)
    st.session_state["results_df_inputs"] = edited
    selected_rows = edited[edited["Select"] == True]
    st.caption(f"(Inputs) Selected {len(selected_rows)} vehicle(s).")

    if st.button("➕ Add selected (inputs) to mappings", key="add_selected_to_mappings_inputs"):
        if selected_rows.empty:
            st.warning("No rows selected (inputs).")
        else:
            df2 = selected_rows.copy()
            year_col    = next((c for c in ["AD_YEAR","Year","MY","ModelYear","Model Year"] if c in df2.columns), None)
            make_col    = next((c for c in ["AD_MAKE","Make","MakeName","Manufacturer"] if c in df2.columns), None)
            model_col   = next((c for c in ["AD_MODEL","Model","Line","Carline","Series"] if c in df2.columns), None)
            trim_col    = next((c for c in ["AD_TRIM","Trim","Grade","Variant","Submodel"] if c in df2.columns), None)
            vehicle_col = next((c for c in ["Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series"] if c in df2.columns), None)
            code_col        = st.session_state.get("code_column_inputs")
            model_code_col  = st.session_state.get("model_code_column_inputs")

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

                st.session_state.mappings[key] = {
                    "year": yv, "make": mkv, "model": mdv, "trim": trv,
                    "vehicle": vhv, "code": code_val, "model_code": model_code_val,
                }
                added += 1
            st.success(f"[Inputs] Added/updated {added} mapping(s).")

# --------------------- Current Mappings table -------------------------
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
