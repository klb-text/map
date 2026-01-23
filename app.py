
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# Additive: Catalog-accelerated Vehicle parsing (NO alias memory) + YMM/YMMT fallback + Harvest mode
# Keeps existing functionality (Vehicle/YMM/YMMT flows, selection, save/commit, harvest routes)
# Build: 2026-01-23

import base64, json, io, re, difflib, time
from typing import Optional, List, Dict, Tuple, Set, Any
import requests, pandas as pd, streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ===================== Page Config =====================
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")
st.title("AFF Vehicle Mapping")

# ===================== Secrets / Config =====================
gh_cfg = st.secrets.get("github", {})
GH_TOKEN  = gh_cfg.get("token")
GH_OWNER  = gh_cfg.get("owner")
GH_REPO   = gh_cfg.get("repo")
GH_BRANCH = gh_cfg.get("branch", "main")

# Data paths in repo
MAPPINGS_PATH   = "data/mappings.json"
AUDIT_LOG_PATH  = "data/mappings_log.jsonl"   # (not written in this UI, but kept for compatibility)
CADS_PATH       = "data/CADS.csv"             # default; override in sidebar
CADS_IS_EXCEL   = False
CADS_SHEET_NAME_DEFAULT = "0"

# Preferred columns in CADS
CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ===================== Canonicalization / Helpers =====================
def canon_text(val: str, for_trim: bool=False) -> str:
    # NOTE: operate on Python strings; not HTML entities
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

def tokens(s: str, min_len: int = 2) -> List[str]:
    s = canon_text(s)
    tks = re.split(r"[^\w]+", s)
    return [t for t in tks if t and len(t) >= min_len]

def _extract_years_from_text(s: str) -> set:
    s = (s or "").strip().lower()
    years = set()
    # e.g., 1950..2050 bounded
    for m in re.finditer(r"\b(19[5-9]\d|20[0-4]\d|2050)\b", s):
        years.add(int(m.group(0)))
    # "my25" → 2025; "my26" → 2026
    for m in re.finditer(r"\bmy\s*(\d{2})\b", s):
        years.add(2000 + int(m.group(1)))
    # Bare 2-digit if nothing else found
    if not years:
        for m in re.finditer(r"\b(\d{2})\b", s):
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

# ===================== CADS Loaders (CSV/Excel) =====================
def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df

@st.cache_data(ttl=600)
def _decode_bytes_to_text(raw: bytes) -> tuple[str, str]:
    if not raw or raw.strip() == b"":
        return ("", "empty")
    # ASCII-safe BOM detection
    encoding = "utf-8"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        encoding = "utf-16"
    elif raw.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
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

        # Robust delimiter detection
        text, _ = _decode_bytes_to_text(raw)
        sample = text[:4096]
        delimiter = None
        try:
            # Explicit candidates (NO newline)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            delimiter = dialect.delimiter
        except Exception:
            for cand in [",", "\t", ";", "|"]:
                if cand in sample:
                    delimiter = cand
                    break

        if delimiter is None:
            # Let pandas detect (fallback)
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, on_bad_lines="skip")
            # Harden: if pandas inferred 1 column, retry common separators
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
        if "content" in j and j["content"]]:
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
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, engine="openpyxl")
        return _strip_object_columns(df)

    if r.status_code == 404:
        raise FileNotFoundError(f"CADS not found: {path}")
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
    # add common columns if available
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
        # ensure feature branch exists
        r_feat = _session.get(gh_ref_heads(owner, repo, feature_branch_name), headers=gh_headers(token), timeout=15)
        if r_feat.status_code != 200:
            r_base = _session.get(gh_ref_heads(owner, repo, branch), headers=gh_headers(token), timeout=15)
            if r_base.status_code == 200:
                base_sha = r_base.json()["object"]["sha"]
                _session.post(f"https://api.github.com/repos/{owner}/{repo}/git/refs",
                              headers=gh_headers(token),
                              json={"ref": f"refs/heads/{feature_branch_name}", "sha": base_sha}, timeout=15)
        target_branch = feature_branch_name
    # read old sha if any
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

# ===================== Mapping pickers / helpers =====================
def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_CODE_PREFS if c in df.columns] or list(df.columns)

def get_model_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_MODEL_CODE_PREFS if c in df.columns] or list(df.columns)

def pick_mapping_by_vehicle(mappings: Dict[str, Dict[str, str]], vehicle: str) -> Optional[Tuple[str, Dict[str, str]]]:
    cv = canon_text(vehicle)
    if not cv: return None
    for k, v in mappings.items():
        if canon_text(v.get("vehicle", "")) == cv:
            return (k, v)
    return None

def find_mappings_by_ymmt_all(mappings: Dict[str, Dict[str, str]], year: str, make: str, model: str, trim: Optional[str] = None) -> List[Tuple[str, Dict[str, str]]]:
    cy = (year or ""); cmk = canon_text(make); cmd = canon_text(model); ctr = canon_text(trim or "", True)
    out: List[Tuple[str, Dict[str, str]]] = []
    for k, v in mappings.items():
        vy, vmk, vmd, vtr = v.get("year",""), v.get("make",""), v.get("model",""), v.get("trim","")
        if canon_text(vmk) != cmk: continue
        if not year_token_matches(vy, cy): continue
        if canon_text(vmd) != cmd: continue
        if ctr:
            ok, _ = trim_matches(vtr, ctr, exact_only=False)
            if not ok: continue
        out.append((k, v))
    return out

# ===================== CADS filtering =====================
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

# ===================== Match rows for a single mapping (used in harvest) =====================
def match_cads_rows_for_mapping(
    df: pd.DataFrame, mapping: Dict[str, str],
    exact_model_when_full: bool, trim_exact_only: bool, strict_and: bool,
    stopword_threshold: float, token_min_len: int,
    effective_model_cols_override: Optional[List[str]] = None,
    trim_as_hint: bool = False, year_require_exact: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df2 = _strip_object_columns(df.copy())
    df2, used_model_cols, used_series_cols = add_effective_model_column(df2, override_cols=effective_model_cols_override)

    # Code tier
    code_val = (mapping.get("code","") or "").strip()
    if code_val:
        hits = []
        for col in get_cads_code_candidates(df2):
            if col in df2.columns:
                series = df2[col].astype(str).str.strip().str.lower()
                mask = series == code_val.lower()
                if mask.any(): hits.append(df2[mask])
        if hits:
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True), {"tier_used": "CODE"}

    # Model code tier
    model_code_val = (mapping.get("model_code","") or "").strip()
    if model_code_val:
        hits = []
        for col in get_model_code_candidates(df2):
            if col in df2.columns:
                series = df2[col].astype(str).str.strip().str.lower()
                mask = series == model_code_val.lower()
                if mask.any(): hits.append(df2[mask])
        if hits:
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True), {"tier_used": "MODEL_CODE"}

    # Vehicle text tier
    veh_val = (mapping.get("vehicle","") or "").strip()
    if veh_val:
        veh_hits = find_rows_by_vehicle_text(df2, veh_val)
        if veh_hits is not None and len(veh_hits) > 0:
            return veh_hits, {"tier_used": "VEHICLE_TEXT"}

    # Generic fallback
    res, diag = filter_cads_generic(
        df2,
        mapping.get("year",""), mapping.get("make",""), mapping.get("model",""), mapping.get("trim",""),
        exact_model_when_full=exact_model_when_full,
        trim_exact_only=trim_exact_only, strict_and=strict_and,
        stopword_threshold=stopword_threshold, token_min_len=token_min_len,
        effective_model_cols_override=effective_model_cols_override,
        trim_as_hint=trim_as_hint, year_require_exact=year_require_exact,
    )
    diag.update({"used_model_cols": used_model_cols, "used_series_cols": used_series_cols})
    return res, diag

# ===================== Vehicle Catalog (no alias memory) =====================
@st.cache_data(ttl=600)
def load_vehicle_catalog(owner, repo, path, token, ref=None):
    """Return normalized DataFrame with Year, Make, Model, Trim, Vehicle (if present)."""
    params = {"ref": ref} if ref else {}
    r = _session.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Vehicle catalog load failed ({r.status_code}): {r.text}")
    j = r.json(); raw = None
    if "content" in j and j["content"]:
        try: raw = base64.b64decode(j["content"])
        except Exception: raw = None
    if (raw is None or raw.strip() == b"") and j.get("download_url"):
        r2 = _session.get(j["download_url"], timeout=15)
        if r2.status_code == 200: raw = r2.content
    if raw is None or raw.strip() == b"":
        raise ValueError(f"Vehicle catalog `{path}` empty or unavailable.")
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    else:
        df = pd.read_csv(io.BytesIO(raw))
    for col in ["Year","Make","Model","Trim","Vehicle","VehicleAttributes"]:
        if col in df.columns: df[col] = df[col].fillna("").astype(str).str.strip()
    keep = [c for c in ["Year","Make","Model","Trim","Vehicle"] if c in df.columns]
    return _strip_object_columns(df[keep].copy())

@st.cache_data(ttl=600)
def build_catalog_index(df_cat: pd.DataFrame) -> Dict[str, Any]:
    df = df_cat.copy()
    makes = sorted(df["Make"].unique().tolist())
    def _lc(s): return (s or "").lower().strip()
    def _trc(s): return canon_text(s or "", for_trim=True)

    per_make_models: Dict[str, List[str]] = {}
    mm_years_trims: Dict[Tuple[str,str], Dict[str, Set[str]]] = {}
    for mk, g_mk in df.groupby("Make"):
        mk_l = _lc(mk)
        models = sorted(g_mk["Model"].unique().tolist(), key=lambda x: len(x or ""), reverse=True)
        per_make_models[mk_l] = models
        for md, g_md in g_mk.groupby("Model"):
            md_l = _lc(md)
            for yr, g_yr in g_md.groupby("Year"):
                trims = set(_trc(t) for t in g_yr["Trim"].fillna("").astype(str))
                mm_years_trims.setdefault((mk_l, md_l), {}).setdefault(str(yr).strip(), set()).update(trims)

    makes_by_len_desc = sorted(makes, key=lambda x: len(x or ""), reverse=True)
    return {
        "makes": makes,
        "makes_len_desc": makes_by_len_desc,
        "per_make_models": per_make_models,
        "mm_years_trims": mm_years_trims,
    }

def parse_vehicle_against_catalog(vehicle_text: str, cat_idx: Dict[str, Any]):
    txt = (vehicle_text or "").strip()
    if not txt: return None
    yr = extract_primary_year(txt)
    yr_s = str(yr) if yr else ""
    low = txt.lower()

    # Make: longest appearance
    chosen_make = None
    for mk in cat_idx["makes_len_desc"]:
        mk_l = mk.lower()
        if mk_l and mk_l in low:
            chosen_make = mk
            break
    if not chosen_make:
        return None

    # Model: longest within make
    per_models = cat_idx["per_make_models"].get(chosen_make.lower(), [])
    chosen_model = None
    for md in per_models:
        if (md or "").strip() and md.lower() in low:
            chosen_model = md; break
    if not chosen_model:
        return {"year": yr_s, "make": chosen_make, "model": "", "trim": "", "confidence": 0.4, "reason": "make_only"}

    # Trim tokens
    tr_free = low
    if yr_s: tr_free = tr_free.replace(yr_s, "")
    tr_free = tr_free.replace(chosen_make.lower(), "")
    tr_free = tr_free.replace(chosen_model.lower(), "")
    tr_user = canon_text(tr_free, for_trim=True)
    user_tokens = set(re.split(r"\W+", tr_user)) - {""}

    mm_key = (chosen_make.lower(), chosen_model.lower())
    trims_by_year = cat_idx["mm_years_trims"].get(mm_key, {})
    known_trims = set()
    if yr_s and yr_s in trims_by_year and trims_by_year[yr_s]:
        known_trims = trims_by_year[yr_s]
    else:
        for s in trims_by_year.values(): known_trims |= s

    chosen_trim = ""
    if user_tokens and known_trims:
        for kt in known_trims:
            kt_tokens = set(re.split(r"\W+", kt)) - {""}
            if user_tokens.issubset(kt_tokens):
                chosen_trim = kt; break

    conf = 0.0
    conf += 0.4 if chosen_make else 0.0
    conf += 0.4 if chosen_model else 0.0
    if yr_s and ((yr_s in trims_by_year) or (len(trims_by_year) > 0)): conf += 0.1
    if chosen_trim: conf += 0.1

    return {"year": yr_s, "make": chosen_make, "model": chosen_model, "trim": chosen_trim, "confidence": round(conf,3), "reason": "ok"}

# ===================== Harvest helpers (HTML table) =====================
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

HARVEST_PREF_ORDER = ["AD_YEAR","AD_MAKE","AD_MODEL","MODEL_NAME","STYLE_NAME","AD_SERIES","Trim","AD_TRIM","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"]

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

    st.markdown("\n".join(parts), unsafe_allow_html=True)

# ===================== URL Params & Shared Loaders =====================
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

    elif source == "mapped":
        year = _get_str("year", ""); make = _get_str("make", ""); model = _get_str("model", ""); trim = _get_str("trim", "")
        ymmt_list = find_mappings_by_ymmt_all(mappings, year, make, model, trim if canon_text(trim, True) else None)
        hits = []
        for _, mp in ymmt_list:
            df_hit, diag = match_cads_rows_for_mapping(
                df_cads, mp,
                exact_model_when_full=model_exact_when_full, trim_exact_only=trim_exact_only, strict_and=strict_and,
                stopword_threshold=stopword_threshold, token_min_len=token_min_len,
                effective_model_cols_override=override_cols, trim_as_hint=True, year_require_exact=year_require_exact,
            )
            if len(df_hit) > 0:
                df_hit = df_hit.copy()
                df_hit["__mapped_key__"] = f"{mp.get('make','')},{mp.get('model','')},{mp.get('trim','')},{mp.get('year','')}"
                df_hit["__tier__"] = diag.get("tier_used")
                hits.append(df_hit)
        df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True) if hits else df_cads.iloc[0:0]
        render_harvest_table(
            df_union,
            table_id="cads_mapped_results",
            preferred_order=HARVEST_PREF_ORDER + ["__mapped_key__","__tier__"],
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE","__mapped_key__","__tier__"],
            caption="CADS – Mapped search results",
            plain=plain,
        ); st.stop()

    elif source == "quick_vehicle":
        veh_txt = _get_str("vehicle", "")
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_mapped_quick_vehicle", caption="No vehicle provided", plain=plain); st.stop()
        veh_hits = find_rows_by_vehicle_text(df_cads, veh_txt)
        if veh_hits is None: veh_hits = df_cads.iloc[0:0]
        render_harvest_table(
            veh_hits,
            table_id="cads_mapped_quick_vehicle",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Quick Vehicle text results",
            plain=plain,
        ); st.stop()

    elif source == "unmapped":
        veh_txt = canon_text(_get_str("vehicle", ""))
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_unmapped_results", caption="No vehicle provided", plain=plain); st.stop()
        hits = []
        for col in ["Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series","STYLE_NAME","AD_MODEL","MODEL_NAME"]:
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
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_catalog_results", caption="No vehicle provided", plain=plain); st.stop()
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, cat_path, GH_TOKEN, ref=ref_branch)
            cat_idx = build_catalog_index(df_cat)
            parsed = parse_vehicle_against_catalog(veh_txt, cat_idx)
            if not parsed:
                render_harvest_table(df_cads.iloc[0:0], table_id="cads_catalog_results", caption="Catalog did not find a close match", plain=plain); st.stop()
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
            ); st.stop()
        except Exception as e:
            st.markdown(f"<p id='harvest-error'>Catalog harvest failed: {e}</p>", unsafe_allow_html=True); st.stop()

    st.markdown("<p id='harvest-empty'>No harvest source matched or insufficient parameters.</p>", unsafe_allow_html=True)
    st.stop()

# If HARVEST_MODE is on, run and stop.
if HARVEST_MODE:
    _run_harvest()

# ===================== Interactive UI (Additive, keeps your existing flows) =====================
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


# ===================== NEW Lane (Additive): Vehicle → Catalog → CADS =====================
st.header("Vehicle-Only Quick Lookup (catalog → CADS, alias-free)")
veh_catalog_txt = st.text_input("Vehicle (Year Make Model [Trim])", key="quick_vehicle_catalog_input", placeholder="e.g., 2026 Lexus RX 350 Premium AWD")

if st.button("⚡ Find by Vehicle (catalog → CADS)", key="btn_quick_vehicle_catalog"):
    if not (veh_catalog_txt or "").strip():
        st.warning("Enter a vehicle string first.")
    else:
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"),
                                          GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
            cat_idx = build_catalog_index(df_cat)
            parsed = parse_vehicle_against_catalog(veh_catalog_txt, cat_idx)
            if not parsed:
                st.info("Could not parse Vehicle deterministically. Use your existing panels or the YMM/YMMT fallback below.")
            else:
                y_s, mk_s, md_s, tr_s = parsed["year"], parsed["make"], parsed["model"], parsed["trim"]
                st.caption(f"📌 Catalog parsed → Y='{y_s}'  Make='{mk_s}'  Model='{md_s}'  Trim='{tr_s}'  (confidence={parsed['confidence']}, reason={parsed['reason']})")
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
                    st.warning("No CADS rows matched. Use your existing Vehicle/YMM/YMMT panels.")
                else:
                    if "Select" not in results.columns: results.insert(0, "Select", False)
                    st.session_state["results_df_inputs"] = results
                    st.session_state["code_candidates_inputs"] = get_cads_code_candidates(results)
                    st.session_state["model_code_candidates_inputs"] = get_model_code_candidates(results)
                    st.session_state["code_column_inputs"] = st.session_state["code_candidates_inputs"][0] if st.session_state.get("code_candidates_inputs") else None
                    st.session_state["model_code_column_inputs"] = st.session_state["model_code_candidates_inputs"][0] if st.session_state.get("model_code_candidates_inputs") else None
                    st.success(f"Found {len(results)} CADS row(s). Use your existing selection + save workflow below.")
        except Exception as e:
            st.error(f"Catalog-based vehicle search failed: {e}")

# ===================== Fallback (Additive): Catalog-powered YMM/YMMT pickers =====================
st.subheader("Fallback: Pick by YMM/YMMT (from catalog)")
col1, col2 = st.columns(2)
with col1:
    if st.button("Open YMM pickers", key="btn_open_ymm"):
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"),
                                          GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
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
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"),
                                          GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
            md_opts = sorted(df_cat[df_cat["Make"]==mk_pick]["Model"].unique().tolist())
            st.session_state["ymm_md_opts"] = md_opts
        except Exception:
            pass
    md_pick = st.selectbox("Model", options=[""] + md_opts, index=0)
    tr_opts = []
    if mk_pick and md_pick:
        try:
            df_cat = load_vehicle_catalog(GH_OWNER, GH_REPO, st.session_state.get("veh_catalog_path","data/AFF Vehicles YMMT.csv"),
                                          GH_TOKEN, ref=st.session_state.get("load_branch", GH_BRANCH))
            filt = (df_cat["Make"]==mk_pick) & (df_cat["Model"]==md_pick)
            if y_pick: filt &= (df_cat["Year"]==y_pick)
            tr_opts = sorted([t for t in df_cat[filt]["Trim"].unique().tolist() if t])
        except Exception:
            pass
    tr_pick = st.selectbox("Trim (optional)", options=[""] + tr_opts, index=0)
    if st.button("🔎 Search CADS with YMM(T)", key="btn_search_ymmt"):
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
            if "Select" not in results.columns: results.insert(0, "Select", False)
            st.session_state["results_df_inputs"] = results
            st.session_state["code_candidates_inputs"] = get_cads_code_candidates(results)
            st.session_state["model_code_candidates_inputs"] = get_model_code_candidates(results)
            st.session_state["code_column_inputs"] = st.session_state["code_candidates_inputs"][0] if st.session_state.get("code_candidates_inputs") else None
            st.session_state["model_code_column_inputs"] = st.session_state["model_code_candidates_inputs"][0] if st.session_state.get("model_code_candidates_inputs") else None
            st.success(f"Found {len(results)} CADS row(s). Continue with selection + save below.")

# ===================== Legacy quick vehicle contains (unchanged behavior) =====================
st.header("Legacy: Quick Vehicle Text Search (contains across common columns)")
veh_legacy = st.text_input("Vehicle text (legacy contains search)", key="legacy_vehicle_contains", placeholder="e.g., 2026 Pacifica Select AWD")
if st.button("🔎 Find by Vehicle (legacy contains)"):
    df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
    res = find_rows_by_vehicle_text(df_cads, veh_legacy)
    if res is None or len(res)==0:
        st.info("No rows matched vehicle text.")
    elseYou're right—let’s finish this cleanly.  
Below is the **rest of `app.py` (Part 2/2)** you can **append directly** after your last line:

---

## **app.py — Part 2/2 (append below your last line)**

```python
# ===================== Results review & selection (keeps your existing flow) =====================
st.header("Results – Review & Select Rows")
if "results_df_inputs" in st.session_state and st.session_state["results_df_inputs"] is not None:
    try:
        results_df = st.session_state["results_df_inputs"].copy()
        # Ensure 'Select' column exists and is boolean
        if "Select" not in results_df.columns:
            results_df.insert(0, "Select", False)
        # Streamlit data_editor lets you flip 'Select' checkboxes inline
        st.caption("Tip: Toggle the **Select** checkboxes to mark row(s) you want to use.")
        edited_df = st.data_editor(
            results_df,
            use_container_width=True,
            height=TABLE_HEIGHT,
            num_rows="dynamic",  # allow filtering/resize; we only read 'Select'
            key="results_editor_inputs",
        )
        st.session_state["results_df_inputs"] = edited_df

        # Selected subset summary
        selected_mask = edited_df["Select"] == True
        selected_count = int(selected_mask.sum()) if "Select" in edited_df.columns else 0
        st.caption(f"✅ Selected rows: **{selected_count}**")

        sel_cols = st.columns(3)
        with sel_cols[0]:
            if st.button("Use first SELECTED row's codes"):
                try:
                    if selected_count > 0:
                        first_row = edited_df[selected_mask].iloc[0]
                    else:
                        first_row = edited_df.iloc[0]
                    # Prefer current chosen columns if available; else, fall back to candidates
                    code_col = st.session_state.get("code_column_inputs")
                    model_code_col = st.session_state.get("model_code_column_inputs")
                    # If none chosen or missing, choose first available candidate from the edited frame
                    if not code_col or code_col not in edited_df.columns:
                        cands = [c for c in ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"] if c in edited_df.columns]
                        code_col = cands[0] if cands else None
                    if not model_code_col or model_code_col not in edited_df.columns:
                        mcands = [c for c in ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"] if c in edited_df.columns]
                        model_code_col = mcands[0] if mcands else None

                    if code_col:
                        st.session_state["code_val_auto"] = str(first_row.get(code_col, "") or "")
                    if model_code_col:
                        st.session_state["model_code_val_auto"] = str(first_row.get(model_code_col, "") or "")
                    st.success("Filled auto code fields from the first selected row.")
                except Exception as e:
                    st.error(f"Could not pull codes from selected row: {e}")

        with sel_cols[1]:
            if st.button("Preview mapping JSON from first SELECTED row"):
                try:
                    if selected_count > 0:
                        row0 = edited_df[selected_mask].iloc[0]
                    else:
                        row0 = edited_df.iloc[0]
                    # Heuristics to build a suggested mapping preview
                    year_guess = str(row0.get("AD_YEAR", "")) or str(row0.get("Year", "")) or ""
                    make_guess = str(row0.get("AD_MAKE", "")) or str(row0.get("Make", "")) or ""
                    model_guess = str(row0.get("AD_MODEL", "")) or str(row0.get("MODEL_NAME", "")) or str(row0.get("__effective_model__", "")) or ""
                    trim_guess  = str(row0.get("AD_TRIM", "")) or str(row0.get("Trim", "")) or ""

                    code_col = st.session_state.get("code_column_inputs")
                    model_code_col = st.session_state.get("model_code_column_inputs")
                    if not code_col or code_col not in edited_df.columns:
                        cands = [c for c in ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"] if c in edited_df.columns]
                        code_col = cands[0] if cands else ""
                    if not model_code_col or model_code_col not in edited_df.columns:
                        mcands = [c for c in ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"] if c in edited_df.columns]
                        model_code_col = mcands[0] if mcands else ""

                    code_guess = str(row0.get(code_col, "")) if code_col else ""
                    model_code_guess = str(row0.get(model_code_col, "")) if model_code_col else ""

                    mapping_preview = {
                        "vehicle": "",  # Freeform; leave blank or paste your canonical text
                        "year": year_guess,
                        "make": make_guess,
                        "model": model_guess,
                        "trim": trim_guess,
                        "code": code_guess,
                        "model_code": model_code_guess,
                    }
                    st.code(json.dumps(mapping_preview, indent=2, ensure_ascii=False))
                    st.info("Paste these into the **Prepare mapping payload** fields above, or use the 'Use first row's codes' button to auto-fill code fields.")
                except Exception as e:
                    st.error(f"Could not build mapping preview: {e}")

        with sel_cols[2]:
            if st.button("Clear selection flags"):
                try:
                    edited_df["Select"] = False
                    st.session_state["results_df_inputs"] = edited_df
                    st.success("Cleared all Select flags.")
                except Exception as e:
                    st.error(f"Could not clear selection: {e}")

    except Exception as e:
        st.error(f"Error displaying results editor: {e}")
else:
    st.caption("No results to display yet. Run one of the searches above.")

# --- EOF ---
