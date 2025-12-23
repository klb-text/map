
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# + Exact YMMT lookup first (canonicalized), strict Trim gate, Code-first CADS search
# + Canonicalization: lower, trim, remove trailing punctuation for robust equality
# + Snapshot cleared on input changes; minimal debug banner for inputs/flags
# Repo: klb-text/map, Branch: main

import base64, json, time, io, re, difflib
from typing import Optional, List, Dict, Tuple
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

CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]   # expanded union; AD_VEH_ID is common
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ---------------------------------------------------------------------
# Resilient HTTP
# ---------------------------------------------------------------------
from requests.adapters import HTTPAdapter, Retry
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5,
                                 status_forcelist=[429,500,502,503,504], allowed_methods=["GET","PUT","POST"])))
_session.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5,
                                 status_forcelist=[429,500,502,503,504], allowed_methods=["GET","PUT","POST"])))

def _get(url, headers=None, params=None, timeout=15):  return _session.get(url, headers=headers, params=params, timeout=timeout)
def _put(url, headers=None, json=None, timeout=15):    return _session.put(url, headers=headers, json=json, timeout=timeout)
def _post(url, headers=None, json=None, timeout=15):   return _session.post(url, headers=headers, json=json, timeout=timeout)

def gh_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
def gh_contents_url(owner, repo, path):
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
def gh_ref_heads(owner, repo, branch):
    return f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"

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
    if r_feat.status_code != 404:
        raise RuntimeError(f"Failed checking feature branch ({r_feat.status_code}): {r_feat.text}")
    r_create = _post(f"https://api.github.com/repos/{owner}/{repo}/git/refs",
                     headers=gh_headers(token),
                     json={"ref": f"refs/heads/{feature_branch}", "sha": base_sha})
    return r_create.status_code in (201, 422)

def save_json_to_github(owner, repo, path, token, branch, payload_dict,
                        commit_message, author_name=None, author_email=None,
                        use_feature_branch=False, feature_branch_name="aff-mapping-app"):
    content = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    target_branch = branch
    if use_feature_branch and ensure_feature_branch(owner, repo, token, branch, feature_branch_name):
        target_branch = feature_branch_name
    sha = get_file_sha(owner, repo, path, token, ref=target_branch)
    data = {"message": commit_message, "content": content_b64, "branch": target_branch}
    if sha: data["sha"] = sha
    if author_name and author_email:
        data["committer"] = {"name": author_name, "email": author_email}
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
    raise RuntimeError(f"Failed to append log ({r2.status_code}): {r2.text}")

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
        j = r.json()
        raw = None
        if "content" in j and j["content"]:
            try: raw = base64.b64decode(j["content"])
            except Exception: raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = _get(j["download_url"])
            if r2.status_code == 200: raw = r2.content
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
        j = r.json()
        raw = None
        if "content" in j and j["content"]:
            try: raw = base64.b64decode(j["content"])
            except Exception: raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = _get(j["download_url"])
            if r2.status_code == 200: raw = r2.content
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
    Lowercases, strips spaces, and removes leading/trailing punctuation.
    Keeps hyphens, slashes, and alphanumerics.
    Examples: 'SE.' -> 'se', ' S, ' -> 's', 'e-tron 45' -> 'e-tron 45'
    """
    s = (val or "").strip().lower()
    # remove leading/trailing punctuation except hyphen and slash
    s = re.sub(r"^[\s\.,;:!]+", "", s)
    s = re.sub(r"[\s\.,;:!]+$", "", s)
    # collapse inner whitespace
    s = re.sub(r"\s+", " ", s)
    if for_trim:
        # normalize common synonyms
        repl = {
            "all wheel drive": "awd", "all-wheel drive": "awd", "4wd":"awd","4x4":"awd",
            "front wheel drive":"fwd","front-wheel drive":"fwd",
            "rear wheel drive":"rwd","rear-wheel drive":"rwd",
            "two wheel drive":"2wd","two-wheel drive":"2wd",
            "plug-in hybrid":"phev","electric":"ev","bev":"ev",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
    return s

def canon_key(year: str, make: str, model: str, trim: str) -> str:
    return f"{canon_text(year)}|{canon_text(make)}|{canon_text(model)}|{canon_text(trim, for_trim=True)}"

def secrets_status():
    missing = []
    if not GH_TOKEN:  missing.append("github.token")
    if not GH_OWNER:  missing.append("github.owner")
    if not GH_REPO:   missing.append("github.repo")
    if not GH_BRANCH: missing.append("github.branch")
    return missing

def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_CODE_PREFS if c in df.columns] or list(df.columns)

def get_model_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_MODEL_CODE_PREFS if c in df.columns] or list(df.columns)

# ---------------------------------------------------------------------
# Filtering (unchanged core but uses canonical inputs at call sites)
# ---------------------------------------------------------------------
def filter_cads(
    df: pd.DataFrame,
    year: str, make: str, model: str, trim: str, vehicle: str, model_code: str = "",
    exact_year: bool = True, exact_mmt: bool = False, case_sensitive: bool = False,
    strict_and: bool = True, lock_modelcode_make_model: bool = True, tokenize_year: bool = True,
    trim_match_mode: str = "Exact", trim_synonyms: bool = True, trim_fuzzy_threshold: float = 0.65,
) -> pd.DataFrame:
    y, mk, md, tr, vh, mc = (year or ""), (make or ""), (model or ""), (trim or ""), (vehicle or ""), (model_code or "")
    df2 = _strip_object_columns(df.copy())

    YEAR_CANDS  = ["AD_YEAR","Year","MY","ModelYear","Model Year"]
    MAKE_CANDS  = ["AD_MAKE","Make","MakeName","Manufacturer"]
    MODEL_CANDS = ["AD_MODEL","Model","Line","Carline","Series"]
    TRIM_CANDS  = ["AD_TRIM","Trim","Grade","Variant","Submodel"]
    VEH_CANDS   = ["Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series"]

    year_col    = next((c for c in YEAR_CANDS  if c in df2.columns), None)
    make_col    = next((c for c in MAKE_CANDS  if c in df2.columns), None)
    model_col   = next((c for c in MODEL_CANDS if c in df2.columns), None)
    trim_col    = next((c for c in TRIM_CANDS  if c in df2.columns), None)
    vehicle_col = next((c for c in VEH_CANDS   if c in df2.columns), None)

    def col_contains(col, val):
        if not col or val == "": return None
        s = df2[col].astype(str)
        return s.str.contains(val, case=case_sensitive, na=False)

    def col_equals(col, val):
        if not col or val == "": return None
        s = df2[col].astype(str)
        return (s == val) if case_sensitive else (s.str.lower() == val.lower())

    masks = []

    # Model Code exact union
    if mc:
        mc_union = None
        for mc_col in get_model_code_candidates(df2):
            if mc_col in df2.columns:
                m = col_equals(mc_col, mc)
                if m is not None:
                    mc_union = m if mc_union is None else (mc_union | m)
        if mc_union is not None:
            masks.append(mc_union)
            if lock_modelcode_make_model:
                if make_col and mk:
                    mm = col_equals(make_col, mk);  masks.append(mm) if mm is not None else None
                if model_col and md:
                    mo = col_equals(model_col, md); masks.append(mo) if mo is not None else None

    # Make / Model
    if make_col and mk:
        masks.append(col_equals(make_col, mk) if exact_mmt else col_contains(make_col, mk))
    if model_col and md:
        masks.append(col_equals(model_col, md) if exact_mmt else col_contains(model_col, md))

    # Trim
    if trim_col and tr:
        mode = (trim_match_mode or "Exact").strip().lower()
        if mode == "exact":      masks.append(col_equals(trim_col, tr))
        elif mode == "contains": masks.append(col_contains(trim_col, tr))
        elif mode == "token or":
            # simple token OR
            toks = [t for t in re.split(r"[^\w]+", canon_text(tr, True)) if t]
            series_toks = df2[trim_col].astype(str).apply(lambda s: set(re.split(r"[^\w]+", canon_text(s, True))))
            masks.append(series_toks.apply(lambda s: any(t in s for t in toks)))
        elif mode == "fuzzy":
            series = df2[trim_col].astype(str).apply(lambda s: canon_text(s, True))
            target = canon_text(tr, True)
            masks.append(series.apply(lambda s: difflib.SequenceMatcher(None, s, target).ratio() >= float(trim_fuzzy_threshold)))
        else:
            # Token AND default
            toks = [t for t in re.split(r"[^\w]+", canon_text(tr, True)) if t]
            series_toks = df2[trim_col].astype(str).apply(lambda s: set(re.split(r"[^\w]+", canon_text(s, True))))
            masks.append(series_toks.apply(lambda s: all(t in s for t in toks)))

    # Year
    if year_col and y:
        try:
            df_year_int = df2[year_col].astype(int)
            masks.append(df_year_int == int(y))
        except Exception:
            series = df2[year_col].astype(str)
            if exact_year:
                if tokenize_year:
                    masks.append(series.apply(lambda s: y in re.split(r"[\/\-\|\s,;]+", s.strip())))
                else:
                    masks.append(col_equals(year_col, y))
            else:
                masks.append(col_contains(year_col, y))

    if not masks: return df2.iloc[0:0]

    final = masks[0]
    for m in masks[1:]:
        final = (final & m) if strict_and else (final | m)
    res = df2[final]

    # MC-only fallback
    if model_code and strict_and and len(res) == 0:
        mc_only = None
        for mc_col in get_model_code_candidates(df2):
            if mc_col in df2.columns:
                m = col_equals(mc_col, mc)
                mc_only = m if mc_only is None else (mc_only | m)
        if mc_only is not None: res = df2[mc_only]

    return res

# ---------------------------------------------------------------------
# Mapping search utilities
# ---------------------------------------------------------------------
def build_key(year: str, make: str, model: str, trim: str, vehicle: str) -> str:
    y, mk, md, tr, vh = map(_normalize, (year, make, model, trim, vehicle))
    if mk and (y or md or tr): return f"{y}-{mk}-{md}-{tr}".strip("-")
    elif mk and vh:            return f"{mk}:{vh}"
    elif mk and md:            return f"{mk}:{md}"
    else:                      return mk or vh or "UNSPECIFIED"

def _normalize(val: str) -> str:
    return (val or "").strip()

def find_existing_mappings(
    mappings: Dict[str, Dict[str, str]],
    year: str, make: str, model: str, trim: str, vehicle: str, code_input: str,
    exact_year: bool = True, case_sensitive: bool = False,
    ignore_year: bool = False, ignore_trim: bool = False,
    require_trim_exact_if_provided: bool = True, disallow_lenient_when_trim: bool = True,
) -> List[Tuple[str, Dict[str, str], str]]:
    # Canonicalize inputs
    y_in  = canon_text(year)
    mk_in = canon_text(make)
    md_in = canon_text(model)
    tr_in = canon_text(trim, for_trim=True)
    code_in = (code_input or "").strip()

    results = []
    trim_provided = (tr_in != "")

    if trim_provided and disallow_lenient_when_trim:
        ignore_trim = False  # hard override

    for k, v in mappings.items():
        vy  = canon_text(v.get("year",""))
        vmk = canon_text(v.get("make",""))
        vmd = canon_text(v.get("model",""))
        vtr = canon_text(v.get("trim",""), for_trim=True)
        vcode = (v.get("code","") or "").strip()

        # Code path (require exact trim if provided)
        if code_in and vcode and (vcode.lower() == code_in.lower()):
            if (not trim_provided) or (vtr == tr_in):
                results.append((k, v, "by_code"))
            continue

        # Strict YMMT path
        strict_ok = True
        if y_in and not ignore_year:
            strict_ok = (vy == y_in)
        if mk_in and strict_ok:
            strict_ok = (vmk == mk_in)
        if md_in and strict_ok:
            strict_ok = (vmd == md_in)
        if trim_provided and require_trim_exact_if_provided and not ignore_trim and strict_ok:
            strict_ok = (vtr == tr_in)
        elif tr_in and strict_ok:
            strict_ok = (vtr == tr_in)

        if strict_ok and (mk_in or md_in or tr_in or y_in):
            results.append((k, v, "strict_ymmt"))
            continue

        # Lenient suppressed when trim provided
        if trim_provided and disallow_lenient_when_trim and not ignore_trim:
            continue

        # Lenient: no year
        if mk_in and md_in and (vmk == mk_in) and (vmd == md_in) and (not tr_in or vtr == tr_in):
            results.append((k, v, "lenient_no_year"))
            continue

        # Lenient: no trim
        if mk_in and md_in and (vmk == mk_in) and (vmd == md_in) and (not y_in or vy == y_in):
            results.append((k, v, "lenient_no_trim"))
            continue

        # Make+Model only
        if mk_in and md_in and (vmk == mk_in) and (vmd == md_in):
            results.append((k, v, "make_model_only"))
            continue

    return results

def match_cads_rows_for_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Code-first, fallback strict YMMT; uses canonical text comparisons."""
    df2 = _strip_object_columns(df.copy())

    # 1) by CODE union
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

    # 2) strict YMMT fallback (Exact everything; canonicalized)
    y  = canon_text(mapping.get("year",""))
    mk = canon_text(mapping.get("make",""))
    md = canon_text(mapping.get("model",""))
    tr = canon_text(mapping.get("trim",""), for_trim=True)

    year_col    = next((c for c in ["AD_YEAR","Year","MY","ModelYear","Model Year"] if c in df2.columns), None)
    make_col    = next((c for c in ["AD_MAKE","Make","MakeName","Manufacturer"] if c in df2.columns), None)
    model_col   = next((c for c in ["AD_MODEL","Model","Line","Carline","Series"] if c in df2.columns), None)
    trim_col    = next((c for c in ["AD_TRIM","Trim","Grade","Variant","Submodel"] if c in df2.columns), None)

    mask = pd.Series([True] * len(df2), index=df2.index)
    if year_col and y:
        try:
            mask = mask & (df2[year_col].astype(int) == int(y))
        except Exception:
            mask = mask & (df2[year_col].astype(str).str.lower() == y.lower())
    if make_col and mk:
        mask = mask & (df2[make_col].astype(str).str.lower() == mk.lower())
    if model_col and md:
        mask = mask & (df2[model_col].astype(str).str.lower() == md.lower())
    if trim_col and tr:
        # canonical compare for trim
        mask = mask & (df2[trim_col].astype(str).apply(lambda s: canon_text(s, True)) == tr)

    res = df2[mask]
    return res.reset_index(drop=True)

# ---------------------------------------------------------------------
# Query params helpers
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

    q_ignore_year = (get_query_param("ignore_year", "true").lower() == "true")
    q_ignore_trim_param = get_query_param("ignore_trim", "")
    if q_ignore_trim_param == "":
        q_ignore_trim = (False if canon_text(q_trim, True) else True)
    else:
        q_ignore_trim = (q_ignore_trim_param.lower() == "true")
    q_lenient_trim = (get_query_param("lenient_trim", "false").lower() == "true")

    # Exact-key (canonical) lookup first
    strict_key_agent = canon_key(q_year, q_make, q_model, q_trim)
    exact_hit_agent = None
    for k, v in st.session_state.mappings.items():
        if canon_key(v.get("year",""), v.get("make",""), v.get("model",""), v.get("trim","")) == strict_key_agent:
            exact_hit_agent = (k, v, "exact_key")
            break

    if exact_hit_agent:
        matches = [exact_hit_agent]
    else:
        matches = find_existing_mappings(
            st.session_state.mappings,
            q_year, q_make, q_model, q_trim, q_vehicle, q_code,
            exact_year=True, case_sensitive=False,
            ignore_year=q_ignore_year,
            ignore_trim=(q_ignore_trim or q_lenient_trim),
            require_trim_exact_if_provided=(not q_lenient_trim),
            disallow_lenient_when_trim=True
        )
        # HARD gate by canonical trim
        tr_in = canon_text(q_trim, True)
        if tr_in:
            matches = [(k, v, r) for (k, v, r) in matches if canon_text(v.get("trim",""), True) == tr_in]

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

    if matches:
        rows = []
        for k, v, reason in matches:
            rows.append({
                "key": k,
                "year": v.get("year",""),
                "make": v.get("make",""),
                "model": v.get("model",""),
                "trim": v.get("trim",""),
                "vehicle": v.get("vehicle",""),
                "code": v.get("code",""),
                "model_code": v.get("model_code",""),
                "reason": reason,
            })
        st.success(f"Mapped: {len(rows)}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.text("STATUS=MAPPED")
        st.write({"status":"MAPPED","count":len(rows),"data":rows})
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
        if st.button("🔄 Reload mappings (diagnostics)"):
            existing = load_json_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref=GH_BRANCH)
            st.session_state.mappings = existing or {}
            st.success(f"Reloaded. Count: {len(st.session_state.mappings)}")
    except Exception as diag_err:
        st.error(f"Diagnostics error: {diag_err}")

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

miss = secrets_status()
if miss: st.sidebar.warning("Missing secrets: " + ", ".join(miss))

if st.sidebar.button("💾 Commit mappings to GitHub"):
    if miss:
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

st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

st.sidebar.subheader("Matching Controls")
EXACT_YEAR = st.sidebar.checkbox("Exact Year match", value=True)
EXACT_MMT  = st.sidebar.checkbox("Exact Make/Model/Trim match", value=False)
CASE_SENSITIVE = st.sidebar.checkbox("Case sensitive matching", value=False)
BLOCK_SEARCH_IF_MAPPED = st.sidebar.checkbox("Block CADS search if mapping exists", value=True)
IGNORE_YEAR = st.sidebar.checkbox("Ignore Year when detecting existing mapping", value=False)
IGNORE_TRIM = st.sidebar.checkbox("Ignore Trim when detecting existing mapping", value=False)
LENIENT_TRIM_THIS_RUN = st.sidebar.checkbox("Temporarily ignore Trim (this run only)", value=False)

LOAD_CADS_DETAILS_ON_MATCH = st.sidebar.checkbox("Load CADS details when mapping exists", value=True)
MAX_CADS_ROWS_PER_MATCH = st.sidebar.number_input("Max CADS rows to show per match", min_value=1, max_value=10000, value=1000, step=50)

STRICT_AND = st.sidebar.checkbox("Require strict AND across provided filters", value=True)
LOCK_MODEL_CODE_MAKE_MODEL = st.sidebar.checkbox("Lock Model Code to Make+Model (exact)", value=True)
TOKENIZE_YEAR = st.sidebar.checkbox("Tokenize Year (handle '2024/2025' style values)", value=True)

TRIM_MATCH_MODE = st.sidebar.selectbox("Trim match mode", options=["Exact","Contains","Token AND","Token OR","Fuzzy"], index=0)
TRIM_SYNONYMS = st.sidebar.checkbox("Trim: normalize AWD/FWD/RWD/2WD + PHEV/EV synonyms", value=True)
TRIM_FUZZY_THRESHOLD = st.sidebar.slider("Trim: fuzzy similarity threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01)

SUGGESTION_COUNT = st.sidebar.number_input("Top N trim suggestions when no results", min_value=1, max_value=200, value=50, step=5)
TABLE_HEIGHT = st.sidebar.slider("Results table height (px)", min_value=400, max_value=1200, value=700, step=50)

if st.sidebar.button("🧹 Clear (Interactive)"):
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
with c3: model = st.text_input("Model", key="model_input", placeholder="e.g., Discovery Sport")
with c4: trim = st.text_input("Trim", key="trim_input", placeholder="e.g., S / SE / Technology Package")
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

# Debug banner for current canonical inputs
st.caption(f"🔎 Canonical inputs → Year='{canon_text(year)}' Make='{canon_text(make)}' Model='{canon_text(model)}' Trim='{canon_text(trim, True)}'")

# --------------------- Existing mapping detection (interactive) -------
strict_key = canon_key(year, make, model, trim)
exact_hit = None
for k, v in st.session_state.mappings.items():
    if canon_key(v.get("year",""), v.get("make",""), v.get("model",""), v.get("trim","")) == strict_key:
        exact_hit = (k, v, "exact_key")
        break

if exact_hit:
    matches = [exact_hit]
else:
    matches = find_existing_mappings(
        st.session_state.mappings, year, make, model, trim, vehicle, mapped_code,
        exact_year=EXACT_YEAR, case_sensitive=False, ignore_year=IGNORE_YEAR,
        ignore_trim=(IGNORE_TRIM or LENIENT_TRIM_THIS_RUN),
        require_trim_exact_if_provided=(not LENIENT_TRIM_THIS_RUN),
        disallow_lenient_when_trim=True
    )
    # Hard gate by canonical trim
    tr_in = canon_text(trim, True)
    if tr_in:
        matches = [(k, v, r) for (k, v, r) in matches if canon_text(v.get("trim",""), True) == tr_in]

st.subheader("Existing Mapping (for current inputs)")
if matches:
    st.success(f"Already mapped: {len(matches)} match(es).")
    rows = []
    for k, v, reason in matches:
        rows.append({
            "Match Level": reason,
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
    st.info("No existing mapping detected for current inputs.")

cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    if st.button("📋 Copy first match's Code to input"):
        if matches:
            st.session_state["code_input"] = (matches[0][1].get("code","") or "").strip()
            st.success(f"Copied code to input.")
        else:
            st.info("No matches to copy from.")

# --------------------- Search CADS (Code-first, then strict YMMT) -----
with cc3:
    if st.button("🔎 Search CADS"):
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

            mapped_record = exact_hit[1] if exact_hit else (matches[0][1] if matches else None)
            if mapped_record:
                # 1) CODE-first (union across candidate code columns)
                code_val = (mapped_record.get("code","") or "").strip()
                model_code_val = (mapped_record.get("model_code","") or "").strip()

                df_hits = []
                if code_val:
                    for col in get_cads_code_candidates(df_cads):
                        if col in df_cads.columns:
                            series = df_cads[col].astype(str).str.strip().str.lower()
                            mask = series == code_val.lower()
                            if mask.any(): df_hits.append(df_cads[mask])

                # If no 'code' hits, try model_code union (many OEMs)
                if not df_hits and model_code_val:
                    for col in get_model_code_candidates(df_cads):
                        if col in df_cads.columns:
                            series = df_cads[col].astype(str).str.strip().str.lower()
                            mask = series == model_code_val.lower()
                            if mask.any(): df_hits.append(df_cads[mask])

                if df_hits:
                    df_res = pd.concat(df_hits, axis=0).drop_duplicates().reset_index(drop=True)
                    st.success(f"Found {len(df_res)} CADS row(s) by Code/Model Code.")
                    selectable = df_res.copy()
                    if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                    st.session_state["results_df"] = selectable
                    st.session_state["code_candidates"] = get_cads_code_candidates(selectable)
                    st.session_state["model_code_candidates"] = get_model_code_candidates(selectable)
                    st.session_state["code_column"] = st.session_state["code_candidates"][0] if st.session_state["code_candidates"] else None
                    st.session_state["model_code_column"] = st.session_state["model_code_candidates"][0] if st.session_state["model_code_candidates"] else None
                else:
                    # 2) Strict YMMT fallback (canonical, exact everything)
                    yv = canon_text(mapped_record.get("year",""))
                    mkv = canon_text(mapped_record.get("make",""))
                    mdv = canon_text(mapped_record.get("model",""))
                    trv = canon_text(mapped_record.get("trim",""), True)

                    results = filter_cads(
                        df_cads, yv, mkv, mdv, trv, "", "",
                        exact_year=True, exact_mmt=True, case_sensitive=False,
                        strict_and=True, lock_modelcode_make_model=False, tokenize_year=True,
                        trim_match_mode="Exact", trim_synonyms=True, trim_fuzzy_threshold=0.65
                    )

                    if len(results) > 0:
                        st.success(f"Found {len(results)} CADS row(s) by strict YMMT.")
                        selectable = results.copy()
                        if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                        st.session_state["results_df"] = selectable
                        st.session_state["code_candidates"] = get_cads_code_candidates(selectable)
                        st.session_state["model_code_candidates"] = get_model_code_candidates(selectable)
                        st.session_state["code_column"] = st.session_state["code_candidates"][0] if st.session_state["code_candidates"] else None
                        st.session_state["model_code_column"] = st.session_state["model_code_candidates"][0] if st.session_state["model_code_candidates"] else None
                    else:
                        st.warning("No CADS rows found via Code or strict YMMT for the mapped vehicle. "
                                   "If CADS Model differs (e.g., 'Discovery Sport' vs 'Discovery'), toggle Exact MMT OFF and rerun.")
            else:
                # No mapped record – run filter with current inputs (canonical)
                effective_trim_mode = "Exact" if canon_text(trim, True) and not (IGNORE_TRIM or LENIENT_TRIM_THIS_RUN) else TRIM_MATCH_MODE
                results = filter_cads(
                    df_cads, canon_text(year), canon_text(make), canon_text(model), canon_text(trim, True), canon_text(vehicle), model_code_input,
                    exact_year=EXACT_YEAR, exact_mmt=EXACT_MMT, case_sensitive=False,
                    strict_and=STRICT_AND, lock_modelcode_make_model=LOCK_MODEL_CODE_MAKE_MODEL, tokenize_year=TOKENIZE_YEAR,
                    trim_match_mode=effective_trim_mode, trim_synonyms=TRIM_SYNONYMS, trim_fuzzy_threshold=TRIM_FUZZY_THRESHOLD
                )
                if len(results) == 0:
                    st.warning("No CADS rows matched your input. Try toggling Exact MMT OFF, or omit Trim to broaden.")
                    sugg_df, _ = suggest_top_trims(
                        df_cads, canon_text(year), canon_text(make), canon_text(model), canon_text(trim, True),
                        top_n=SUGGESTION_COUNT, trim_synonyms=TRIM_SYNONYMS, restrict_to_make_model=True, case_sensitive=False
                    )
                    if sugg_df.empty:
                        st.info("No suggestions found (relax filters or check spelling).")
                    else:
                        selectable = sugg_df.copy()
                        if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                        st.session_state["results_df"] = selectable
                        st.session_state["code_candidates"] = get_cads_code_candidates(selectable)
                        st.session_state["model_code_candidates"] = get_model_code_candidates(selectable)
                        st.session_state["code_column"] = st.session_state["code_candidates"][0] if st.session_state["code_candidates"] else None
                        st.session_state["model_code_column"] = st.session_state["model_code_candidates"][0] if st.session_state["model_code_candidates"] else None
                        st.success(f"Loaded {len(selectable)} trim suggestions.")
        except FileNotFoundError as fnf:
            st.error(str(fnf))
            st.info(f"Ensure CADS exists at `{CADS_PATH}` in `{GH_OWNER}/{GH_REPO}` @ `{GH_BRANCH}`.")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

with cc4:
    if st.button("📋 Copy first CADS Model Code to input"):
        if "results_df" in st.session_state and "model_code_column" in st.session_state:
            df_r = st.session_state["results_df"]
            mc_col = st.session_state["model_code_column"]
            if mc_col and mc_col in df_r.columns:
                vals = df_r[mc_col].dropna().tolist()
                if vals:
                    st.session_state["model_code_input"] = str(vals[0]).strip()
                    st.success(f"Copied model code '{vals[0]}' to Model Code input.")
                else:
                    st.info("No model code values in current results.")
            else:
                st.info("No model code column detected in current results.")

if st.button("🧹 Clear Inputs (Interactive)"):
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
        if st.button("✅ Select All"):
            df_tmp = df_show.copy(); df_tmp["Select"] = True
            st.session_state["results_df"] = df_tmp; df_show = df_tmp
    with csel2:
        if st.button("🧹 Clear Selection"):
            df_tmp = df_show.copy(); df_tmp["Select"] = False
            st.session_state["results_df"] = df_tmp; df_show = df_tmp

    edited = st.data_editor(df_show, key="results_editor", use_container_width=True,
                            num_rows="dynamic", column_order=col_order, height=TABLE_HEIGHT)
    st.session_state["results_df"] = edited
    selected_rows = edited[edited["Select"] == True]
    st.caption(f"Selected {len(selected_rows)} vehicle(s).")

    if st.button("➕ Add selected vehicle(s) to mappings"):
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
                yv  = _normalize(row.get(year_col, ""))    if year_col else ""
                mkv = _normalize(row.get(make_col, ""))    if make_col else ""
                mdv = _normalize(row.get(model_col, ""))   if model_col else ""
                trv = _normalize(row.get(trim_col, ""))    if trim_col else ""
                vhv = _normalize(row.get(vehicle_col, "")) if vehicle_col else ""
                key = build_key(yv, mkv, mdv, trv, vhv)

                code_val       = _normalize(str(row.get(code_col, ""))) if code_col else ""
                model_code_val = _normalize(str(row.get(model_code_col, ""))) if model_code_col else ""

                existing = st.session_state.mappings.get(key)
                if existing and (existing.get("code") != code_val or existing.get("model_code") != model_code_val):
                    st.warning(f"Key '{key}' already exists with a different Code/Model Code. Overwriting.")
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
