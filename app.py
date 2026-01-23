
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# Catalog‑accelerated Vehicle parsing (NO alias memory) + YMM/YMMT fallback + Harvest mode
# Build: 2026-01-22

import base64, json, io, re, difflib
from typing import Optional, List, Dict, Tuple, Set, Any
import requests, pandas as pd, streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ===================== Page Config =====================
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# ===================== Secrets / Config =====================
gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

MAPPINGS_PATH   = "data/mappings.json"
AUDIT_LOG_PATH  = "data/mappings_log.jsonl"
CADS_PATH       = "data/CADS.csv"        # default; can override in sidebar
CADS_IS_EXCEL   = False
CADS_SHEET_NAME_DEFAULT = "0"

# Preferred columns in CADS
CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ===================== Canonicalization / Helpers =====================

def canon_text(val: str, for_trim: bool=False) -> str:
    s = (val or "").replace(" ", " ")
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

def tokens(s: str, min_len: int = 2) -> List[str]:
    s = canon_text(s)
    tks = re.split(r"[^\w]+", s)
    return [t for t in tks if t and len(t) >= min_len]

# Year parsing

def _extract_years_from_text(s: str) -> set:
    s = (s or "").strip().lower()
    years = set()
    for m in re.finditer(r"(19[5-9]\d|20[0-4]\d|2050)", s):
        years.add(int(m.group(0)))
    for m in re.finditer(r"my\s*(\d{2})", s):
        years.add(2000 + int(m.group(1)))
    if not years:
        for m in re.finditer(r"(\d{2})", s):
            years.add(2000 + int(m.group(1)))
    return years

def extract_primary_year(val: str) -> Optional[int]:
    ys = _extract_years_from_text(str(val))
    if not ys: return None
    return max(ys)

def year_token_matches(mapping_year: str, user_year: str) -> bool:
    uy_set = _extract_years_from_text(user_year)
    my_set = _extract_years_from_text(mapping_year)
    if not uy_set: return True
    if not my_set: return False
    return bool(uy_set.intersection(my_set))

# Trim matching

def _trim_tokens(s: str) -> Set[str]:
    return set(tokens(canon_text(s, True)))

def trim_matches(row_trim: str, user_trim: str, exact_only: bool=False) -> Tuple[bool, float]:
    row = canon_text(row_trim, True)
    usr = canon_text(user_trim, True)
    if not usr: return (True, 0.5)
    if row == usr: return (True, 1.0)
    if exact_only: return (False, 0.0)
    if _trim_tokens(usr).issubset(_trim_tokens(row)): return (True, 0.8)
    return (False, 0.0)

def trim_match_type_and_score(row_trim: str, user_trim: str) -> Tuple[str, float]:
    ok, score = trim_matches(row_trim, user_trim, exact_only=False)
    if not (user_trim or "").strip():
        return ("none", 0.0)
    if not ok:
        return ("none", 0.0)
    row = canon_text(row_trim, True)
    usr = canon_text(user_trim, True)
    if row == usr: return ("exact", 1.0)
    return ("subset", 0.8)

# Model similarity

def model_similarity(a: str, b: str) -> float:
    a = canon_text(a); b = canon_text(b)
    if not a and not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a: return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()

# ===================== Resilient HTTP Session =====================
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","PUT","POST"])
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def gh_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

def gh_contents_url(owner, repo, path):
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

def gh_ref_heads(owner, repo, branch):
    return f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"

# ===================== CADS Loaders =====================

def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df

@st.cache_data(ttl=600)
def _decode_bytes_to_text(raw: bytes) -> tuple[str, str]:
    if not raw or raw.strip() == b"": return ("", "empty")
    encoding = "utf-8"
    if raw.startswith(b"ÿþ") or raw.startswith(b"þÿ"): encoding = "utf-16"
    elif raw.startswith(b"ï»¿"): encoding = "utf-8-sig"
    text = raw.decode(encoding, errors="replace")
    return (text, encoding)


@st.cache_data(ttl=600)
def load_cads_from_github_csv(owner, repo, path, token, ref=None) -> pd.DataFrame:
    import csv
    params = {"ref": ref} if ref else {}
    r = _session.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params, timeout=15)
    if r.status_code == 200:
        j = r.json(); raw = None
        if "content" in j and j["content"]:
            try:
                raw = base64.b64decode(j["content"])
            except Exception:
                raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = _session.get(j["download_url"], timeout=15)
            if r2.status_code == 200:
                raw = r2.content
        if raw is None or raw.strip() == b"":
            raise ValueError(f"CADS `{path}` empty or unavailable.")

        # --- Robust delimiter detection ---
        text, _ = _decode_bytes_to_text(raw)
        sample = text[:4096]
        delimiter = None
        try:
            # Use explicit candidates (NO newline in this list)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            delimiter = dialect.delimiter
        except Exception:
            # Fallback: pick first present candidate
            for cand in [",", "\t", ";", "|"]:
                if cand in sample:
                    delimiter = cand
                    break

        # Read with the decided delimiter or let pandas infer
        if delimiter is None:
            # Let pandas detect (slower, but robust)
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, on_bad_lines="skip")
            # Optional hardening: if pandas returned one column, retry with common seps
            if df.shape[1] == 1:
                for cand in [",", "\t", ";", "|"]:
                    try:
                        df_try = pd.read_csv(io.StringIO(text), sep=cand, dtype=str, on_bad_lines="skip", engine="python")
                        if df_try.shape[1] > 1:
                            df = df_try
                            break
                    except Exception:
                        pass
        else:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, on_bad_lines="skip", engine="python")

        df.columns = [str(c).strip() for c in df.columns]
        return _strip_object_columns(df.dropna(how="all"))

    if r.status_code == 404:
        raise FileNotFoundError(f"CADS not found: {path}")
    raise RuntimeError(f"Failed to load CADS CSV ({r.status_code}): {r.text}")

@st.cache_data(ttl=600)
def load_cads_from_github_excel(owner, repo, path, token, ref=None, sheet_name=0) -> pd.DataFrame:
    params = {"ref": ref} if ref else {}
    r = _session.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params, timeout=15)
    if r.status_code == 200:
        j = r.json(); raw = None
        if "content" in j and j["content"]:
            try: raw = base64.b64decode(j["content"]) 
            except Exception: raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = _session.get(j["download_url"], timeout=15)
            if r2.status_code == 200: raw = r2.content
        if raw is None or raw.strip() == b"": raise ValueError(f"CADS `{path}` empty or unavailable.")
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, engine="openpyxl")
        return _strip_object_columns(df)
    if r.status_code == 404: raise FileNotFoundError(f"CADS not found: {path}")
    raise RuntimeError(f"Failed to load CADS Excel ({r.status_code}): {r.text}")

# ===================== Effective Model & Stopwords =====================
MODEL_LIKE_REGEX  = re.compile(r"(?:^|_|\s)(model(name)?|car\s*line|carline|line|series)(?:$|_|\s)", re.I)
SERIES_LIKE_REGEX = re.compile(r"(?:^|_|\s)(series(name)?|sub(?:_|-)?model|body(?:_|-)?style|body|trim|grade|variant|description|modeltrim|name)(?:$|_|\s)", re.I)

def detect_model_like_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    model_cols  = [c for c in cols if MODEL_LIKE_REGEX.search(c)]
    series_cols = [c for c in cols if SERIES_LIKE_REGEX.search(c) and c not in model_cols]
    return (list(dict.fromkeys(model_cols)), list(dict.fromkeys(series_cols)))

def effective_model_row(row: pd.Series, model_cols: List[str], series_cols: List[str]) -> str:
    parts = []
    for c in model_cols + series_cols:
        if c in row.index:
            v = str(row.get(c, "") or "").strip()
            if v: parts.append(v)
    return canon_text(" ".join(parts))

def add_effective_model_column(df: pd.DataFrame, override_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    auto_model_cols, auto_series_cols = detect_model_like_columns(df)
    if override_cols:
        model_cols = [c for c in override_cols if c in df.columns]
        series_cols = []
    else:
        model_cols, series_cols = auto_model_cols, auto_series_cols
    for c in ["AD_MODEL", "MODEL_NAME", "STYLE_NAME", "AD_SERIES"]:
        if c in df.columns and c not in model_cols and c not in series_cols:
            model_cols.append(c)
    if not model_cols and not series_cols:
        df["__effective_model__"] = ""
        return df, model_cols, series_cols
    df["__effective_model__"] = df.apply(lambda r: effective_model_row(r, model_cols, series_cols), axis=1)
    return df, model_cols, series_cols

def compute_per_make_stopwords(df_make_slice: pd.DataFrame, stopword_threshold: float = 0.40, token_min_len: int = 2) -> Set[str]:
    if "__effective_model__" not in df_make_slice.columns:
        df_make_slice, _, _ = add_effective_model_column(df_make_slice)
    total = len(df_make_slice)
    if total == 0: return set()
    freq: Dict[str,int] = {}
    for _, row in df_make_slice.iterrows():
        toks = set(tokens(row["__effective_model__"], min_len=token_min_len))
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
    return {t for t, c in freq.items() if (c / total) >= float(stopword_threshold)}

# ===================== GitHub Persistence =====================

def save_json_to_github(owner, repo, path, token, branch, payload_dict, commit_message,
                        author_name=None, author_email=None, use_feature_branch=False, feature_branch_name="aff-mapping-app"):
    content = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    target_branch = branch
    if use_feature_branch:
        r_feat = _session.get(gh_ref_heads(owner, repo, feature_branch_name), headers=gh_headers(token), timeout=15)
        if r_feat.status_code != 200:
            r_base = _session.get(gh_ref_heads(owner, repo, branch), headers=gh_headers(token), timeout=15)
            if r_base.status_code == 200:
                base_sha = r_base.json()["object"]["sha"]
                _session.post(f"https://api.github.com/repos/{owner}/{repo}/git/refs",
                              headers=gh_headers(token),
                              json={"ref": f"refs/heads/{feature_branch_name}", "sha": base_sha}, timeout=15)
        target_branch = feature_branch_name
    r = _session.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params={"ref": target_branch}, timeout=15)
    sha = r.json().get("sha") if r.status_code == 200 else None
    data = {"message": commit_message, "content": content_b64, "branch": target_branch}
    if sha: data["sha"] = sha
    if author_name and author_email:
        data["committer"] = {"name": author_name, "email": author_email}
    r2 = _session.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data, timeout=15)
    if r2.status_code in (200, 201): return r2.json()
    if r2.status_code == 409:
        r3 = _session.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params={"ref": target_branch}, timeout=15)
        latest_sha = r3.json().get("sha") if r3.status_code == 200 else None
        if latest_sha:
            data["sha"] = latest_sha
            r4 = _session.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data, timeout=15)
            if r4.status_code in (200, 201): return r4.json()
    raise RuntimeError(f"Failed to save file ({r2.status_code}): {r2.text}")

@st.cache_data(ttl=60)
def fetch_mappings_from_github(owner, repo, path, token, ref) -> Dict[str, Dict[str, str]]:
    r = _session.get(
        gh_contents_url(owner, repo, path),
        headers=gh_headers(token),
        params={"ref": ref},
        timeout=15,
    )
    if r.status_code == 200:
        decoded = base64.b64decode(r.json()["content"]).decode("utf-8")
        try:
            data = json.loads(decoded)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    elif r.status_code == 404:
        return {}
    else:
        raise RuntimeError(f"Failed to load mappings ({r.status_code}): {r.text}")

# ===================== Mapping utilities =====================

def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_CODE_PREFS if c in df.columns] or list(df.columns)

def get_model_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_MODEL_CODE_PREFS if c in df.columns] or list(df.columns)

# ===================== CADS filtering pipeline =====================

def _tiered_model_mask(eff: pd.Series, md: str, discriminant: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if not md:
        true_mask = pd.Series([True]*len(eff), index=eff.index)
        return true_mask, true_mask, true_mask, true_mask
    and_mask = eff.apply(lambda s: all(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    or_mask  = eff.apply(lambda s: any(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    ns_text  = " ".join(discriminant).strip()
    ns_mask  = eff.str.contains(ns_text, na=False) if ns_text else eff.str.contains(md, na=False)
    full_mask= eff.str.contains(md, na=False)
    return and_mask, or_mask, ns_mask, full_mask

VEHICLE_LIKE_CANDS = ["Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series","STYLE_NAME","AD_MODEL","MODEL_NAME"]

def find_rows_by_vehicle_text(df: pd.DataFrame, vehicle: str) -> Optional[pd.DataFrame]:
    cv = canon_text(vehicle)
    if not cv: return None
    hits = []
    for col in VEHICLE_LIKE_CANDS:
        if col in df.columns:
            ser = df[col].astype(str).str.lower()
            mask = ser.str.contains(cv, na=False)
            if mask.any():
                hits.append(df[mask])
    if hits:
        return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True)
    return None

def filter_cads_generic(
    df: pd.DataFrame,
    year: str, make: str, model: str, trim: str,
    exact_model_when_full: bool, trim_exact_only: bool, strict_and: bool,
    stopword_threshold: float, token_min_len: int,
    effective_model_cols_override: Optional[List[str]] = None,
    trim_as_hint: bool = False, year_require_exact: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df2 = _strip_object_columns(df.copy())
    df2, used_model_cols, used_series_cols = add_effective_model_column(df2, override_cols=effective_model_cols_override)
    YEAR_CANDS = ["AD_YEAR","Year","MY","ModelYear","Model Year"]
    MAKE_CANDS = ["AD_MAKE","Make","MakeName","Manufacturer"]
    TRIM_CANDS = ["AD_TRIM","Trim","Grade","Variant","Submodel"]
    year_col = next((c for c in YEAR_CANDS if c in df2.columns), None)
    make_col = next((c for c in MAKE_CANDS if c in df2.columns), None)
    trim_col = next((c for c in TRIM_CANDS if c in df2.columns), None)

    y  = (year or "")
    mk = canon_text(make)
    md = canon_text(model)
    tr = canon_text(trim, True)

    masks = []
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
    and_mask, or_mask, ns_mask, full_mask = _tiered_model_mask(eff, md, discriminant)
    if md:
        masks.append(and_mask)

    # Trim gate
    if trim_col and tr:
        s_trim  = df2[trim_col].astype(str)
        m_exact = s_trim.str.lower() == tr
        m_subset= s_trim.apply(lambda x: _trim_tokens(tr).issubset(_trim_tokens(x)))
        if trim_as_hint:
            df2["__trim_match_type__"]  = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[0])
            df2["__trim_match_score__"] = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[1])
        else:
            masks.append(m_exact if trim_exact_only else (m_exact | m_subset))
    else:
        df2["__trim_match_type__"]  = "none"
        df2["__trim_match_score__"] = 0.0

    # Year gate
    if year_col and y:
        s_year = df2[year_col].astype(str)
        if year_require_exact:
            uy = extract_primary_year(y)
            if uy is not None:
                masks.append(s_year.apply(lambda vy: (uy in _extract_years_from_text(vy)) if _extract_years_from_text(vy) else True))
            else:
                masks.append(s_year.apply(lambda vy: year_token_matches(vy, y)))
        else:
            masks.append(s_year.apply(lambda vy: year_token_matches(vy, y)))

    # Apply masks
    if not masks:
        result = df2.iloc[0:0]
    else:
        m = masks[0]
        for mm in masks[1:]:
            m = (m & mm) if strict_and else (m | mm)
        result = df2[m]

    # Fallback tiers
    tier_used = "AND"
    if md and len(result) == 0:
        tier_used = "OR"; m2 = or_mask
        for mm in masks[1:]: m2 = (m2 & mm) if strict_and else (m2 | mm)
        result = df2[m2]
    if len(result) == 0:
        tier_used = "NS_CONTAINS"; m3 = ns_mask
        for mm in masks[1:]: m3 = (m3 & mm) if strict_and else (m3 | mm)
        result = df2[m3]
    if len(result) == 0:
        tier_used = "FULL_CONTAINS"; m4 = full_mask
        for mm in masks[1:]: m4 = (m4 & mm) if strict_and else (m4 | mm)
        result = df2[m4]

    if md and trim_as_hint and len(result) > 0:
        sort_cols = []
        if "__trim_match_score__" in result.columns: sort_cols.append("__trim_match_score__")
        if "__effective_model__" in result.columns: sort_cols.append("__effective_model__")
        if sort_cols:
            if len(sort_cols) == 2: result = result.sort_values(by=sort_cols, ascending=[False, True])
            else: result = result.sort_values(by=sort_cols, ascending=False)

    diag = {
        "used_model_cols": used_model_cols,
        "used_series_cols": used_series_cols,
        "make_stopwords": sorted(list(make_stopwords))[:50],
        "user_tokens": user_tokens,
        "discriminant_tokens": discriminant,
        "tier_used": tier_used,
        "rows_after_tier": len(result),
        "trim_as_hint": trim_as_hint,
    }
    return result, diag

# ===================== URL Params & Harvest Mode =====================
st.title("AFF Vehicle Mapping")
params = st.experimental_get_query_params()
HARVEST_MODE   = (params.get("harvest", ["0"]) [0] == "1")
HARVEST_SOURCE = (params.get("source",  ["mapped"]) [0])  # mapped | inputs | quick_ymmt | quick_vehicle | unmapped | catalog

def _get_bool(name: str, default: bool) -> bool:
    v = params.get(name, [None])[0]
    if v is None: return default
    return str(v).strip() in ("1","true","True","yes","on")

def _get_float(name: str, default: float) -> float:
    v = params.get(name, [None])[0]
    try:
        return float(v) if v is not None else default
    except:
        return default

def _get_int(name: str, default: int) -> int:
    v = params.get(name, [None])[0]
    try:
        return int(v) if v is not None else default
    except:
        return default

def _get_str(name: str, default: str = "") -> str:
    v = params.get(name, [None])[0]
    return v if v is not None else default

@st.cache_data(ttl=600)
def _load_cads_df(cads_path: Optional[str] = None, cads_is_excel: Optional[bool] = None, sheet_name: Optional[str] = None, ref: Optional[str] = None):
    path = cads_path if cads_path is not None else CADS_PATH
    is_xlsx = cads_is_excel if cads_is_excel is not None else CADS_IS_EXCEL
    ref = ref or GH_BRANCH
    if is_xlsx:
        sn = sheet_name if sheet_name is not None else CADS_SHEET_NAME_DEFAULT
        try: sn = int(sn)
        except Exception: pass
        return load_cads_from_github_excel(GH_OWNER, GH_REPO, path, GH_TOKEN, ref=ref, sheet_name=sn)
    return load_cads_from_github_csv(GH_OWNER, GH_REPO, path, GH_TOKEN, ref=ref)

HARVEST_PREF_ORDER = ["AD_YEAR","AD_MAKE","AD_MODEL","MODEL_NAME","STYLE_NAME","AD_SERIES","Trim","AD_TRIM","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"]

def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def _first_nonempty(*vals) -> str:
    for v in vals:
        if v is not None:
            sv = str(v).strip()
            if sv: return sv
    return ""

def render_harvest_table(
    df: pd.DataFrame,
    table_id: str = "cads_harvest_table",
    preferred_order: Optional[List[str]] = None,
    visible_only_cols: Optional[List[str]] = None,
    include_attr_cols: Optional[List[str]] = None,
    caption: Optional[str] = None,
    plain: bool = False,
):
    if df is None or len(df) == 0:
        st.markdown("<p id='harvest-empty'>No rows</p>", unsafe_allow_html=True)
        return
    cols = list(df.columns)
    if visible_only_cols:
        cols = [c for c in cols if c in visible_only_cols]
    if preferred_order:
        front = [c for c in preferred_order if c in cols]
        back  = [c for c in cols if c not in front]
        cols = front + back
    style_key = "STYLE_ID" if "STYLE_ID" in df.columns else None
    veh_key   = "AD_VEH_ID" if "AD_VEH_ID" in df.columns else None
    attr_cols = include_attr_cols or []

    parts = []
    parts.append(f"<table id='{_html_escape(table_id)}' class='cads-harvest' data-source='harvest'>")
    if caption:
        parts.append(f"<caption>{_html_escape(caption)}</caption>")
    parts.append("<thead><tr>")
    for c in cols:
        parts.append(f"<th scope='col' data-col-key='{_html_escape(c)}'>{_html_escape(c)}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for idx, row in df.iterrows():
        row_key = _first_nonempty(row.get(style_key), row.get(veh_key), idx)
        tr_attrs = []
        if row_key != "":
            tr_attrs.append(f"data-row-key='{_html_escape(row_key)}'")
        for c in attr_cols:
            if c in df.columns:
                val = _first_nonempty(row.get(c))
                if val:
                    tr_attrs.append(f"data-{_html_escape(c).lower().replace(' ','_').replace('/','-') }='{_html_escape(val)}'")
        parts.append(f"<tr {' '.join(tr_attrs)}>")
        for c in cols:
            val = row.get(c, "")
            parts.append(f"<td data-col-key='{_html_escape(c)}'>{_html_escape(val)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    if not plain:
        css = """
        <style>
        table.cads-harvest { border-collapse: collapse; width: 100%; font: 13px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Arial; }
        table.cads-harvest th, table.cads-harvest td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
        table.cads-harvest thead th { position: sticky; top: 0; background: #f8f8f8; z-index: 2; }
        table.cads-harvest caption { text-align:left; font-weight:600; margin: 6px 0; }
        .harvest-note { margin: 6px 0 14px; color: #444; font-size: 12px; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    st.markdown("
".join(parts), unsafe_allow_html=True)

# ===================== SERVER-EXECUTED HARVEST MODE =====================

def _run_harvest():
    trim_as_hint         = _get_bool("trim_as_hint", True)
    trim_exact_only      = _get_bool("trim_exact_only", False)
    strict_and           = _get_bool("strict_and", True)
    model_exact_when_full= _get_bool("model_exact_when_full", False)
    year_require_exact   = _get_bool("year_require_exact", True)
    stopword_threshold   = _get_float("stopword_threshold", 0.60)
    token_min_len        = _get_int("token_min_len", 2)
    plain                = _get_bool("plain", False)

    cads_path  = _get_str("cads_path", CADS_PATH)
    cads_is_xl = _get_bool("cads_is_excel", CADS_IS_EXCEL)
    cads_sheet = _get_str("cads_sheet", CADS_SHEET_NAME_DEFAULT)
    ref_branch = _get_str("ref", GH_BRANCH)

    oc = _get_str("override_cols", "AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES")
    override_cols = [c.strip() for c in oc.split(",") if c.strip()] or None

    mappings = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref_branch)
    df_cads  = _load_cads_df(cads_path, cads_is_xl, cads_sheet, ref=ref_branch)
    df_cads  = _strip_object_columns(df_cads)

    source = HARVEST_SOURCE
    if source in ("inputs", "quick_ymmt"):
        year  = _get_str("year", "")
        make  = _get_str("make", "")
        model = _get_str("model", "")
        trim  = _get_str("trim",  "")
        results, _ = filter_cads_generic(
            df_cads, year, make, model, trim,
            exact_model_when_full=model_exact_when_full,
            trim_exact_only=trim_exact_only, strict_and=strict_and,
            stopword_threshold=stopword_threshold, token_min_len=token_min_len,
            effective_model_cols_override=override_cols,
            trim_as_hint=trim_as_hint, year_require_exact=year_require_exact,
        )
        render_harvest_table(
            results,
            table_id="cads_inputs_results" if source=="inputs" else "cads_mapped_quick_ymmt",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Input-driven results" if source=="inputs" else "CADS – Quick YMM(/T) mapped results",
            plain=plain,
        ); st.stop()

    elif source == "unmapped":
        veh_txt = canon_text(_get_str("vehicle", ""))
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_unmapped_results", caption="No vehicle provided", plain=plain); st.stop()
        hits = []
        for col in VEHICLE_LIKE_CANDS:
            if col in df_cads.columns:
                ser = df_cads[col].astype(str).str.lower(); mask = ser.str.contains(veh_txt, na=False)
                if mask.any(): hits.append(df_cads[mask])
        df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True) if hits else df_cads.iloc[0:0]
        render_harvest_table(
            df_union,
            table_id="cads_unmapped_results",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Unmapped search results",
            plain=plain,
        ); st.stop()


    elif source == "catalog":
        veh_txt = _get_str("vehicle", "")
        cat_path = _get_str("catalog_path", "data/AFF Vehicles YMMT.csv")
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_catalog_results", caption="No vehicle provided", plain=plain)
            st.stop()
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, cat_path, GH_TOKEN, ref=ref_branch)
            cat_idx = build_catalog_index(df_cat)
            parsed = parse_vehicle_against_catalog(veh_txt, cat_idx)
            if not parsed:
                render_harvest_table(df_cads.iloc[0:0], table_id="cads_catalog_results", caption="Catalog did not find a close match", plain=plain)
                st.stop()
            y_s, mk_s, md_s, tr_s = parsed["year"], parsed["make"], parsed["model"], parsed["trim"]
            results, _ = filter_cads_generic(
                df_cads, y_s, mk_s, md_s, tr_s,
                exact_model_when_full=model_exact_when_full,
                trim_exact_only=False, strict_and=strict_and,
                stopword_threshold=stopword_threshold, token_min_len=token_min_len,
                effective_model_cols_override=override_cols, trim_as_hint=True, year_require_exact=year_require_exact,
            )
            render_harvest_table(
                results,
                table_id="cads_catalog_results",
                preferred_order=HARVEST_PREF_ORDER,
                include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
                caption="CADS – Catalog-accelerated results",
                plain=plain,
            )
            st.stop()
        except Exception as e:
            st.markdown(f"<p id='harvest-error'>Catalog harvest failed: {e}</p>", unsafe_allow_html=True)
            st.stop()

    # Default if no source matched
    st.markdown("<p id='harvest-empty'>No harvest source matched or insufficient parameters.</p>", unsafe_allow_html=True)
    st.stop()


# If HARVEST_MODE is on, run and stop.
if HARVEST_MODE:
    _run_harvest()

# ===================== Interactive UI =====================
# Sidebar: CADS Settings
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

# Sidebar: Vehicle Catalog (no alias)
st.sidebar.subheader("Vehicle Catalog")
VEH_CATALOG_PATH = st.sidebar.text_input("Vehicle Catalog path in repo", value="data/AFF Vehicles YMMT.csv")
st.session_state["veh_catalog_path"] = VEH_CATALOG_PATH

# Sidebar: Actions & Matching
st.sidebar.header("Actions")
st.sidebar.subheader("Mappings Source")
load_branch = st.sidebar.text_input("Branch to load mappings from", value=st.session_state.get("load_branch", GH_BRANCH), help="Branch we read mappings.json from.")
st.session_state["load_branch"] = load_branch
if st.sidebar.button("🔄 Reload mappings from GitHub"):
    try:
        fetched = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, st.session_state["load_branch"])
        st.session_state["mappings"] = fetched
        st.session_state["local_mappings_modified"] = False
        st.sidebar.success(f"Reloaded {len(fetched)} mapping(s) from {st.session_state['load_branch']}.")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

commit_msg = st.sidebar.text_input("Commit message", value="chore(app): update AFF vehicle mappings via Streamlit")
use_feature_branch = st.sidebar.checkbox("Use feature branch (aff-mapping-app)", value=False)

st.sidebar.subheader("Matching Controls")
TRIM_AS_HINT = st.sidebar.checkbox("Use Trim as hint (do not filter)", value=True)
TRIM_EXACT_ONLY = st.sidebar.checkbox("Trim must be exact (no token-subset)", value=False)
MODEL_EXACT_WHEN_FULL = st.sidebar.checkbox("Model exact when input is multi-word", value=False)
STRICT_AND = st.sidebar.checkbox("Require strict AND across provided filters", value=True)
YEAR_REQUIRE_EXACT = st.sidebar.checkbox("Require exact year match", value=True, help="Only include rows where the entered year appears in the CADS row year tokens.")
STOPWORD_THRESHOLD = st.sidebar.slider("Per-make stopword threshold", 0.1, 0.9, 0.60, 0.05)
TOKEN_MIN_LEN = st.sidebar.slider("Token minimum length", 1, 5, 2, 1)

st.sidebar.subheader("Effective Model (override)")
EFFECTIVE_MODEL_COLS_OVERRIDE = st.sidebar.text_input("Comma-separated CADS columns (optional)", value="AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES", help="If blank, auto-detect model/series columns.")
OVERRIDE_COLS = [c.strip() for c in EFFECTIVE_MODEL_COLS_OVERRIDE.split(",") if c.strip()] or None
TABLE_HEIGHT = st.sidebar.slider("Results table height (px)", 400, 1200, 700, 50)

# Helper to load CADS for UI

def _load_cads_df_ui():
    if cads_upload is not None:
        if cads_upload.name.lower().endswith(".xlsx"):
            return pd.read_excel(cads_upload, engine="openpyxl")
        return pd.read_csv(cads_upload)
    if CADS_IS_EXCEL:
        sheet_arg = CADS_SHEET_NAME
        try: sheet_arg = int(sheet_arg)
        except Exception: pass
        return load_cads_from_github_excel(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=st.session_state["load_branch"], sheet_name=sheet_arg)
    return load_cads_from_github_csv(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=st.session_state["load_branch"])

# Auto refresh mappings unless locally modified
local_mod_flag = st.session_state.get("local_mappings_modified", False)
try:
    fetched = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, st.session_state["load_branch"])
    if not local_mod_flag:
        st.session_state["mappings"] = fetched
except Exception as e:
    st.sidebar.warning(f"Could not load mappings.json: {e}")

mappings: Dict[str, Dict[str, Any]] = st.session_state.get("mappings", {}) or {}

# ===================== Vehicle → Catalog → CADS (primary) =====================
st.header("Vehicle-Only Quick Lookup (catalog → CADS, alias-free)")
veh_catalog_txt = st.text_input("Vehicle (Year Make Model [Trim])", key="quick_vehicle_catalog_input", placeholder="e.g., 2026 Lexus RX 350 Premium AWD")

if st.button("⚡ Find by Vehicle (catalog → CADS)"):
    if not (veh_catalog_txt or "").strip():
        st.warning("Enter a vehicle string first.")
    else:
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"), GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
            cat_idx = build_catalog_index(df_cat)
            parsed = parse_vehicle_against_catalog(veh_catalog_txt, cat_idx)
            if not parsed:
                st.info("Could not parse Vehicle deterministically. Use the YMM/YMMT fallback below.")
            else:
                y_s, mk_s, md_s, tr_s = parsed["year"], parsed["make"], parsed["model"], parsed["trim"]
                st.caption(f"📌 Parsed → Y='{y_s}'  Make='{mk_s}'  Model='{md_s}'  Trim='{tr_s}'  (confidence={parsed['confidence']}, reason={parsed['reason']})")
                df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
                results, diag = filter_cads_generic(
                    df_cads, y_s, mk_s, md_s, tr_s,
                    exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                    trim_exact_only=False,
                    strict_and=STRICT_AND,
                    stopword_threshold=STOPWORD_THRESHOLD,
                    token_min_len=TOKEN_MIN_LEN,
                    effective_model_cols_override=OVERRIDE_COLS,
                    trim_as_hint=True,
                    year_require_exact=YEAR_REQUIRE_EXACT,
                )
                st.session_state["last_diag_inputs"] = {"catalog_parse": parsed, "cads_filter": diag}
                st.caption(f"Tier used (catalog→CADS): {diag.get('tier_used')}")
                if len(results) == 0:
                    st.warning("No CADS rows matched. Use the YMM or YMMT fallback below.")
                else:
                    st.dataframe(results, use_container_width=True, height=TABLE_HEIGHT)
                    st.session_state["results_df_inputs"] = results
        except Exception as e:
            st.error(f"Catalog-based vehicle search failed: {e}")

# ===================== Fallback: YMM / YMMT from catalog =====================
st.subheader("Fallback: Pick by YMM/YMMT (from catalog)")
col1, col2 = st.columns(2)
with col1:
    if st.button("Open YMM pickers"):
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"), GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
            years = sorted(df_cat["Year"].unique().tolist())
            makes = sorted(df_cat["Make"].unique().tolist())
            st.session_state["ymm_years"] = years
            st.session_state["ymm_makes"] = makes
        except Exception as e:
            st.error(f"Could not load catalog for YMM: {e}")

years = st.session_state.get("ymm_years", [])
makes = st.session_state.get("ymm_makes", [])
if years or makes:
    y_pick = st.selectbox("Year", options=[""] + years, index=0)
    mk_pick = st.selectbox("Make", options=[""] + makes, index=0)
    md_opts = sorted(st.session_state.get("ymm_md_opts", []))
    if mk_pick:
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"), GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
            md_opts = sorted(df_cat[df_cat["Make"]==mk_pick]["Model"].unique().tolist())
            st.session_state["ymm_md_opts"] = md_opts
        except Exception:
            pass
    md_pick = st.selectbox("Model", options=[""] + md_opts, index=0)
    tr_opts = []
    if mk_pick and md_pick:
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"), GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
            filt = (df_cat["Make"]==mk_pick) & (df_cat["Model"]==md_pick)
            if y_pick: filt &= (df_cat["Year"]==y_pick)
            tr_opts = sorted([t for t in df_cat[filt]["Trim"].unique().tolist() if t])
        except Exception:
            pass
    tr_pick = st.selectbox("Trim (optional)", options=[""] + tr_opts, index=0)
    if st.button("🔎 Search CADS with YMM(T)"):
        df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
        results, diag = filter_cads_generic(
            df_cads, str(y_pick or ""), mk_pick or "", md_pick or "", tr_pick or "",
            exact_model_when_full=MODEL_EXACT_WHEN_FULL,
            trim_exact_only=False,
            strict_and=STRICT_AND,
            stopword_threshold=STOPWORD_THRESHOLD,
            token_min_len=TOKEN_MIN_LEN,
            effective_model_cols_override=OVERRIDE_COLS,
            trim_as_hint=True,
            year_require_exact=YEAR_REQUIRE_EXACT,
        )
        st.session_state["last_diag_inputs"] = {"ymmt_pick": {"y":y_pick,"mk":mk_pick,"md":md_pick,"tr":tr_pick}, "cads_filter": diag}
        if len(results) == 0:
            st.warning("No CADS rows matched YMM(T).")
        else:
            st.dataframe(results, use_container_width=True, height=TABLE_HEIGHT)
            st.session_state["results_df_inputs"] = results

# ===================== Add selected mapping (manual confirm) =====================
st.header("Confirm & Save Mapping")
with st.expander("Prepare mapping payload", expanded=False):
    vehicle_text = st.text_input("Vehicle (as typed or canonical)")
    year_val = st.text_input("Year")
    make_val = st.text_input("Make")
    model_val= st.text_input("Model")
    trim_val = st.text_input("Trim (optional)")
    code_col  = st.selectbox("Code column (if you want to store a primary code)", options=st.session_state.get("code_candidates_inputs", []) or [], index=0 if st.session_state.get("code_candidates_inputs") else 0)
    model_code_col = st.selectbox("Model Code column (optional)", options=st.session_state.get("model_code_candidates_inputs", []) or [], index=0 if st.session_state.get("model_code_candidates_inputs") else 0)
    code_val  = st.text_input("Code value (optional)")
    model_code_val = st.text_input("Model Code value (optional)")

    # Helper buttons to pull first row values from current results
    if st.button("Use first row's codes from results"):
        df_res = st.session_state.get("results_df_inputs")
        if df_res is not None and len(df_res) > 0:
            if code_col and code_col in df_res.columns:
                st.session_state["code_val_auto"] = str(df_res.iloc[0][code_col])
            if model_code_col and model_code_col in df_res.columns:
                st.session_state["model_code_val_auto"] = str(df_res.iloc[0][model_code_col])
    code_val = st.text_input("Code value (auto/override)", value=st.session_state.get("code_val_auto",""))
    model_code_val = st.text_input("Model Code value (auto/override)", value=st.session_state.get("model_code_val_auto",""))

save_cols = st.columns(3)
with save_cols[0]:
    if st.button("💾 Save mapping to GitHub"):
        try:
            new_map = {
                "vehicle": vehicle_text,
                "year": year_val,
                "make": make_val,
                "model": model_val,
                "trim": trim_val,
                "code": code_val,
                "model_code": model_code_val,
            }
            # Load current mappings, update with a deterministic key
            mappings = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, st.session_state.get("load_branch", GH_BRANCH))
            key = f"{canon_text(make_val)}::{canon_text(model_val)}::{canon_text(trim_val, True)}::{year_val}"
            mappings[key] = new_map
            save_json_to_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH, mappings, commit_msg, use_feature_branch=use_feature_branch)
            st.success(f"Saved mapping for {make_val} {model_val} {trim_val} {year_val}.")
        except Exception as e:
            st.error(f"Failed to save mapping: {e}")

with save_cols[1]:
    if st.button("📋 Copy payload JSON"):
        st.code(json.dumps({
            "vehicle": vehicle_text,
            "year": year_val,
            "make": make_val,
            "model": model_val,
            "trim": trim_val,
            "code": code_val,
            "model_code": model_code_val,
        }, indent=2, ensure_ascii=False))

with save_cols[2]:
    if st.button("🔧 Show last diagnostics"):
        st.json(st.session_state.get("last_diag_inputs", {}))

# --- EOF ---
