
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# Full-Length Patched Build (2026-01-13)
# Includes: Trim-as-hint, Vehicle-only lookup, YMMT persistence, robust year gate, lenient trim matching, full UI

import base64, json, time, io, re, difflib
from typing import Optional, List, Dict, Tuple, Set
import requests, pandas as pd, streamlit as st
from requests.adapters import HTTPAdapter, Retry

# ---- Page Config ----
st.set_page_config(page_title="AFF Vehicle Mapping", layout="wide")

# ---- Secrets / Config ----
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

CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ---- Canonicalization / Helpers ----
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
        for k, v in repl.items():
            s = s.replace(k, v)
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
    if not years:
        for m in re.finditer(r"\b([0-9]{2})\b", s):
            years.add(2000 + int(m.group(1)))
    return years

def extract_primary_year(val: str) -> Optional[int]:
    ys = _extract_years_from_text(str(val))
    if not ys:
        return None
    return max(ys)

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

def trim_match_type_and_score(row_trim: str, user_trim: str) -> Tuple[str, float]:
    ok, score = trim_matches(row_trim, user_trim, exact_only=False)
    if not (user_trim or "").strip():
        return ("none", 0.0)
    if not ok:
        return ("none", 0.0)
    row = canon_text(row_trim, True)
    usr = canon_text(user_trim, True)
    if row == usr:
        return ("exact", 1.0)
    return ("subset", 0.8)

def model_similarity(a: str, b: str) -> float:
    a = canon_text(a); b = canon_text(b)
    if not a and not b: return 0.0
    if a == b: return 1.0
    if a in b or b in a: return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---- Resilient HTTP Session ----
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","PUT","POST"])
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def gh_headers(token: str): return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
def gh_contents_url(owner, repo, path): return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
def gh_ref_heads(owner, repo, branch): return f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"

# ---- CADS Loaders ----
def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
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
    r = _session.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params, timeout=15)
    if r.status_code == 200:
        j = r.json()
        raw = None
        if "content" in j and j["content"]:
            try: raw = base64.b64decode(j["content"])
            except Exception: raw = None
        if (raw is None or raw.strip() == b"") and j.get("download_url"):
            r2 = _session.get(j["download_url"], timeout=15)
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
        return _strip_object_columns(df.dropna(how="all"))
    if r.status_code == 404: raise FileNotFoundError(f"CADS not found: {path}")
    raise RuntimeError(f"Failed to load CADS CSV ({r.status_code}): {r.text}")

@st.cache_data(ttl=600)
def load_cads_from_github_excel(owner, repo, path, token, ref=None, sheet_name=0) -> pd.DataFrame:
    params = {"ref": ref} if ref else {}
    r = _session.get(gh_contents_url(owner, repo, path), headers=gh_headers(token), params=params, timeout=15)
    if r.status_code == 200:
        j = r.json()
        raw = None
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

# ---- Effective Model & Stopwords ----
MODEL_LIKE_REGEX  = re.compile(r"(?:^|_|\s)(model(name)?|car\s*line|carline|line|series)(?:$|_|\s)", re.I)
SERIES_LIKE_REGEX = re.compile(r"(?:^|_|\s)(series(name)?|sub(?:_|-)?model|body(?:_|-)?style|body|trim|grade|variant|description|modeltrim|name)(?:$|_|\s)", re.I)

def detect_model_like_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    model_cols  = [c for c in cols if MODEL_LIKE_REGEX.search(c)]
    series_cols = [c for c in cols if SERIES_LIKE_REGEX.search(c) and c not in model_cols]
    return (list(dict.fromkeys(model_cols)), list(dict.fromkeys(series_cols]))

def effective_model_row(row: pd.Series, model_cols: List[str], series_cols: List[str]) -> str:
    parts = []
    for c in model_cols + series_cols:
        if c in row.index:
            v = str(row.get(c, "") or "").strip()
            if v:
                parts.append(v)
    return canon_text(" ".join(parts))

def add_effective_model_column(df: pd.DataFrame, override_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    auto_model_cols, auto_series_cols = detect_model_like_columns(df)
    if override_cols:
        model_cols  = [c for c in override_cols if c in df.columns]
        series_cols = []
    else:
        model_cols, series_cols = auto_model_cols, auto_series_cols
    always_add = ["AD_MODEL", "MODEL_NAME", "STYLE_NAME", "AD_SERIES"]
    for c in always_add:
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

# ---- GitHub Persistence Helpers ----
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

def append_jsonl_to_github(owner, repo, path, token, branch, record, commit_message,
                           use_feature_branch=False, feature_branch_name="aff-mapping-app"):
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
    r2 = _session.put(gh_contents_url(owner, repo, path), headers=gh_headers(token), json=data, timeout=15)
    if r2.status_code in (200, 201): return r2.json()
    raise RuntimeError(f"Failed to append log ({r2.status_code}): {r2.text}")

# ---- Matching pickers (lenient Trim) ----
def pick_best_mapping(
    mappings: Dict[str, Dict[str, str]],
    year: str,
    make: str,
    model: str,
    trim: str,
    trim_exact_only: bool = False,
    model_exact_when_full: bool = True
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
        vmk = v.get("make", "")
        vy  = v.get("year", "")
        vtr = v.get("trim", "")
        vmd = v.get("model", "")

        # Must match make and year (token-aware)
        if canon_text(vmk) != cmk:
            continue
        if not year_token_matches(vy, cy):
            continue

        # Trim matching (lenient): if user provided Trim but it doesn't match, keep candidate with lower score
        tmatch, tscore = trim_matches(vtr, ctr, exact_only=trim_exact_only)
        if not tmatch:
            if ctr:
                tscore = 0.3  # penalize but keep
            else:
                continue

        # Model similarity with optional exact bias when multi-word input
        ms = model_similarity(vmd, cmd)
        if force_exact_model and canon_text(vmd) != cmd:
            ms *= 0.5

        score = tscore * 0.6 + ms * 0.4
        candidates.append((k, v, score))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[0]


def pick_mapping_by_vehicle(mappings: Dict[str, Dict[str, str]], vehicle: str) -> Optional[Tuple[str, Dict[str, str]]]:
    cv = canon_text(vehicle)
    if not cv:
        return None
    for k, v in mappings.items():
        if canon_text(v.get("vehicle", "")) == cv:
            return (k, v)
    return None


def find_mappings_by_vehicle_all(mappings: Dict[str, Dict[str, str]], vehicle: str) -> List[Tuple[str, Dict[str, str]]]:
    """Return all mappings whose stored 'vehicle' equals the provided vehicle text (canonicalized)."""
    cv = canon_text(vehicle)
    if not cv:
        return []
    out: List[Tuple[str, Dict[str, str]]] = []
    for k, v in mappings.items():
        if canon_text(v.get("vehicle", "")) == cv:
            out.append((k, v))
    return out


# ---- CADS filtering (Trim-as-hint, tiers) ----
def _tiered_model_mask(eff: pd.Series, md: str, discriminant: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if not md:
        true_mask = pd.Series([True] * len(eff), index=eff.index)
        return true_mask, true_mask, true_mask, true_mask

    and_mask  = eff.apply(lambda s: all(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    or_mask   = eff.apply(lambda s: any(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    ns_text   = " ".join(discriminant).strip()
    ns_mask   = eff.str.contains(ns_text, na=False) if ns_text else eff.str.contains(md, na=False)
    full_mask = eff.str.contains(md, na=False)
    return and_mask, or_mask, ns_mask, full_mask


# Vehicle-like columns & helper for vehicle-text tier
VEHICLE_LIKE_CANDS = [
    "Vehicle", "Description", "ModelTrim", "ModelName",
    "AD_SERIES", "Series", "STYLE_NAME", "AD_MODEL", "MODEL_NAME"
]


def find_rows_by_vehicle_text(df: pd.DataFrame, vehicle: str) -> Optional[pd.DataFrame]:
    cv = canon_text(vehicle)
    if not cv:
        return None

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
    year: str,
    make: str,
    model: str,
    trim: str,
    exact_model_when_full: bool,
    trim_exact_only: bool,
    strict_and: bool,
    stopword_threshold: float,
    token_min_len: int,
    effective_model_cols_override: Optional[List[str]] = None,
    trim_as_hint: bool = False,
    year_require_exact: bool = False
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    df2 = _strip_object_columns(df.copy())
    df2, used_model_cols, used_series_cols = add_effective_model_column(df2, override_cols=effective_model_cols_override)

    YEAR_CANDS  = ["AD_YEAR", "Year", "MY", "ModelYear", "Model Year"]
    MAKE_CANDS  = ["AD_MAKE", "Make", "MakeName", "Manufacturer"]
    TRIM_CANDS  = ["AD_TRIM", "Trim", "Grade", "Variant", "Submodel"]

    year_col = next((c for c in YEAR_CANDS if c in df2.columns), None)
    make_col = next((c for c in MAKE_CANDS if c in df2.columns), None)
    trim_col = next((c for c in TRIM_CANDS if c in df2.columns), None)

    y  = (year or "")
    mk = canon_text(make)
    md = canon_text(model)
    tr = canon_text(trim, True)

    masks = []

    # Make gate
    if make_col and mk:
        s_make = df2[make_col].astype(str).str.lower()
        masks.append(s_make == mk)
        df_make_slice = df2[s_make == mk]
    else:
        df_make_slice = df2

    # Model gate (tiered discriminant tokens)
    eff = df2["__effective_model__"]
    user_tokens = tokens(md, min_len=token_min_len)
    make_stopwords = compute_per_make_stopwords(df_make_slice, stopword_threshold, token_min_len)
    discriminant = [t for t in user_tokens if t not in make_stopwords]
    and_mask, or_mask, ns_mask, full_mask = _tiered_model_mask(eff, md, discriminant)
    if md:
        masks.append(and_mask)

    # Trim gate
    if trim_col and tr:
        s_trim = df2[trim_col].astype(str)
        m_exact  = s_trim.str.lower() == tr
        m_subset = s_trim.apply(lambda x: _trim_tokens(tr).issubset(_trim_tokens(x)))
        if trim_as_hint:
            df2["__trim_match_type__"]  = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[0])
            df2["__trim_match_score__"] = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[1])
        else:
            if trim_exact_only:
                masks.append(m_exact)
            else:
                masks.append(m_exact | m_subset)
    else:
        df2["__trim_match_type__"]  = "none"
        df2["__trim_match_score__"] = 0.0

    # Year gate (robust exact; don't eliminate rows with unparseable year text)
    if year_col and y:
        s_year = df2[year_col].astype(str)
        if year_require_exact:
            uy = extract_primary_year(y)
            if uy is not None:
                masks.append(
                    s_year.apply(
                        lambda vy: (uy in _extract_years_from_text(vy)) if _extract_years_from_text(vy) else True
                    )
                )
            else:
                masks.append(s_year.apply(lambda vy: year_token_matches(vy, y)))
        else:
            masks.append(s_year.apply(lambda vy: year_token_matches(vy, y)))

    # Combine masks
    if not masks:
        result = df2.iloc[0:0]
    else:
        m = masks[0]
        for mm in masks[1:]:
            m = (m & mm) if strict_and else (m | mm)
        result = df2[m]

    # Fallback tiers (OR → ns_contains → full_contains)
    tier_used = "AND"
    if md and len(result) == 0:
        tier_used = "OR"
        m2 = or_mask
        for mm in masks[1:]:
            m2 = (m2 & mm) if strict_and else (m2 | mm)
        result = df2[m2]

        if len(result) == 0:
            tier_used = "NS_CONTAINS"
            m3 = ns_mask
            for mm in masks[1:]:
                m3 = (m3 & mm) if strict_and else (m3 | mm)
            result = df2[m3]

            if len(result) == 0:
                tier_used = "FULL_CONTAINS"
                m4 = full_mask
                for mm in masks[1:]:
                    m4 = (m4 & mm) if strict_and else (m4 | mm)
                result = df2[m4]

    # Ranking when Trim is a hint
    if md and trim_as_hint and len(result) > 0:
        sort_cols = []
        if "__trim_match_score__" in result.columns:
            sort_cols.append("__trim_match_score__")
        if "__effective_model__" in result.columns:
            sort_cols.append("__effective_model__")
        if sort_cols:
            if len(sort_cols) == 2:
                result = result.sort_values(by=sort_cols, ascending=[False, True])
            else:
                result = result.sort_values(by=sort_cols, ascending=False)

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


# ---- CADS matching for a single mapping ----
def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    # Prefer typical ID columns; fall back to all columns for flexibility
    return [c for c in CADS_CODE_PREFS if c in df.columns] or list(df.columns)


def get_model_code_candidates(df: pd.DataFrame) -> List[str]:
    # Prefer typical model-code columns; fall back to all columns
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
    trim_as_hint: bool = False,
    year_require_exact: bool = False
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    df2 = _strip_object_columns(df.copy())
    df2, used_model_cols, used_series_cols = add_effective_model_column(df2, override_cols=effective_model_cols_override)

    # 1) Code union
    code_val = (mapping.get("code", "") or "").strip()
    if code_val:
        hits = []
        for col in get_cads_code_candidates(df2):
            if col in df2.columns:
                series = df2[col].astype(str).str.strip().str.lower()
                mask = series == code_val.lower()
                if mask.any():
                    hits.append(df2[mask])
        if hits:
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True), {"tier_used": "CODE"}

    # 2) Model Code union
    model_code_val = (mapping.get("model_code", "") or "").strip()
    if model_code_val:
        hits = []
        for col in get_model_code_candidates(df2):
            if col in df2.columns:
                series = df2[col].astype(str).str.strip().str.lower()
                mask = series == model_code_val.lower()
                if mask.any():
                    hits.append(df2[mask])
        if hits:
            return pd.concat(hits, axis=0).drop_duplicates().reset_index(drop=True), {"tier_used": "MODEL_CODE"}

    # 2.5) Vehicle-text tier
    veh_val = (mapping.get("vehicle", "") or "").strip()
    if veh_val:
        veh_hits = find_rows_by_vehicle_text(df2, veh_val)
        if veh_hits is not None and len(veh_hits) > 0:
            return veh_hits, {"tier_used": "VEHICLE_TEXT"}

    # 3) Generic fallback (tiered model, trim-as-hint optional, robust year gate)
    res, diag = filter_cads_generic(
        df2,
        mapping.get("year", ""),
        mapping.get("make", ""),
        mapping.get("model", ""),
        mapping.get("trim", ""),
        exact_model_when_full=exact_model_when_full,
        trim_exact_only=trim_exact_only,
        strict_and=strict_and,
        stopword_threshold=stopword_threshold,
        token_min_len=token_min_len,
        effective_model_cols_override=effective_model_cols_override,
        trim_as_hint=trim_as_hint,
        year_require_exact=year_require_exact,
    )

    diag.update({"used_model_cols": used_model_cols, "used_series_cols": used_series_cols})
    return res, diag

# ===================== UI =====================
st.title("AFF Vehicle Mapping")

# ---- Vehicle-Only Quick Lookup (NEW) ----
st.header("Vehicle-Only Quick Lookup (mapped)")
quick_vehicle = st.text_input("Vehicle (exact text as mapped)", key="quick_vehicle_input", placeholder="e.g., Q3 S line 45 TFSI quattro Premium")
do_quick = st.button("🔎 Quick Search by Vehicle (mapped only)", key="btn_quick_vehicle")

def _quick_load_cads_df():
    if cads_upload is not None:
        if cads_upload.name.lower().endswith(".xlsx"):
            return pd.read_excel(cads_upload, engine="openpyxl")
        return pd.read_csv(cads_upload)
    if CADS_IS_EXCEL:
        sheet_arg = CADS_SHEET_NAME
        try: sheet_arg = int(sheet_arg)
        except Exception: pass
        return load_cads_from_github_excel(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH, sheet_name=sheet_arg)
    return load_cads_from_github_csv(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH)

if do_quick:
    veh_txt = (quick_vehicle or "").strip()
    if not veh_txt:
        st.warning("Enter a vehicle string first.")
    else:
        mappings = st.session_state.get("mappings", {})
        vm_list = find_mappings_by_vehicle_all(mappings, veh_txt)
        if not vm_list:
            st.error("❌ Not mapped. Map this vehicle below (enter YMMT + Vehicle, add to mappings), commit, then retry.")
        else:
            try:
                df_cads = _quick_load_cads_df()
                df_cads = _strip_object_columns(df_cads)
                hits = []
                for _, mp in vm_list:
                    df_hit, diag = match_cads_rows_for_mapping(
                        df_cads, mp,
                        exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                        trim_exact_only=False,
                        strict_and=STRICT_AND,
                        stopword_threshold=STOPWORD_THRESHOLD,
                        token_min_len=TOKEN_MIN_LEN,
                        effective_model_cols_override=OVERRIDE_COLS,
                        trim_as_hint=True,
                        year_require_exact=YEAR_REQUIRE_EXACT,
                    )
                    if len(df_hit) > 0:
                        df_hit = df_hit.copy()
                        df_hit["__mapped_vehicle__"] = mp.get("vehicle", "")
                        df_hit["__tier__"] = diag.get("tier_used")
                        hits.append(df_hit)
                if hits:
                    df_all = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True)
                    st.success(f"Found {len(df_all)} CADS row(s) for mapped vehicle '{veh_txt}'.")
                    st.dataframe(df_all, use_container_width=True, height=TABLE_HEIGHT)
                else:
                    st.warning("No CADS rows matched for the mapped vehicle(s). Check Code/Model Code/Vehicle text.")
            except Exception as e:
                st.error(f"Quick vehicle search failed: {e}")

# ---- Sidebar: CADS Settings ----
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

# ---- Sidebar: Actions & Matching ----
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Reload mappings from GitHub"):
    try:
        r_load = _session.get(gh_contents_url(GH_OWNER, GH_REPO, MAPPINGS_PATH), headers=gh_headers(GH_TOKEN), params={"ref": GH_BRANCH}, timeout=15)
        if r_load.status_code == 200:
            decoded = base64.b64decode(r_load.json()["content"]).decode("utf-8")
            st.session_state.mappings = json.loads(decoded)
        else:
            st.session_state.mappings = {}
        st.sidebar.success("Reloaded.")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

commit_msg = st.sidebar.text_input("Commit message", value="chore(app): update AFF vehicle mappings via Streamlit")
use_feature_branch = st.sidebar.checkbox("Use feature branch (aff-mapping-app)", value=False)

st.sidebar.subheader("Matching Controls")
TRIM_AS_HINT = st.sidebar.checkbox("Use Trim as hint (do not filter)", value=True)
TRIM_EXACT_ONLY = st.sidebar.checkbox("Trim must be exact (no token-subset)", value=False)
MODEL_EXACT_WHEN_FULL = st.sidebar.checkbox("Model exact when input is multi-word", value=False)
STRICT_AND = st.sidebar.checkbox("Require strict AND across provided filters", value=True)
YEAR_REQUIRE_EXACT = st.sidebar.checkbox("Require exact year match", value=True)
STOPWORD_THRESHOLD = st.sidebar.slider("Per-make stopword threshold", 0.1, 0.9, 0.60, 0.05)
TOKEN_MIN_LEN = st.sidebar.slider("Token minimum length", 1, 5, 2, 1)

st.sidebar.subheader("Effective Model (override)")
EFFECTIVE_MODEL_COLS_OVERRIDE = st.sidebar.text_input("Comma-separated CADS columns to use (optional)", value="AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES")
OVERRIDE_COLS = [c.strip() for c in EFFECTIVE_MODEL_COLS_OVERRIDE.split(",") if c.strip()] or None

TABLE_HEIGHT = st.sidebar.slider("Results table height (px)", 400, 1200, 700, 50)

# ---- Inputs for Mapping ----
st.subheader("Edit / Add Mapping")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: year = st.text_input("Year", key="year_input", placeholder="e.g., 2025")
with c2: make = st.text_input("Make", key="make_input", placeholder="e.g., Audi")
with c3: model = st.text_input("Model", key="model_input", placeholder="e.g., Q3")
with c4: trim = st.text_input("Trim", key="trim_input", placeholder="e.g., S line 45 TFSI quattro Premium")
with c5: vehicle = st.text_input("Vehicle (alt)", key="vehicle_input", placeholder="Optional")
with c6: mapped_code = st.text_input("Mapped Code", key="code_input", placeholder="Optional")
model_code_input = st.text_input("Model Code (optional)", key="model_code_input", placeholder="AD_MFGCODE/MODEL_CODE/etc.")

# ---- Existing Mapping ----
st.subheader("Existing Mapping (for current inputs)")
existing_rows = []
vehicle_first = False
if canon_text(vehicle) and not (canon_text(year) or canon_text(make) or canon_text(model) or canon_text(trim, True)):
    vm = pick_mapping_by_vehicle(st.session_state.get("mappings", {}), vehicle)
    vehicle_first = True
    if vm:
        k, v = vm
        existing_rows.append({"Match Level": "by_vehicle", "Score": 1.0, "Key": k, "Year": v.get("year",""), "Make": v.get("make",""), "Model": v.get("model",""), "Trim": v.get("trim",""), "Vehicle": v.get("vehicle",""), "Code": v.get("code",""), "Model Code": v.get("model_code","")})

if not existing_rows:
    best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, trim, trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
    if not best and (year or make or model):
        best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, "", trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
    if best:
        k, v, score = best
        existing_rows.append({"Match Level": "generic_best_trim_model_year" if canon_text(trim, True) else "generic_best_by_ymm", "Score": round(score,3), "Key": k, "Year": v.get("year",""), "Make": v.get("make",""), "Model": v.get("model",""), "Trim": v.get("trim",""), "Vehicle": v.get("vehicle",""), "Code": v.get("code",""), "Model Code": v.get("model_code","")})

if existing_rows:
    st.dataframe(pd.DataFrame(existing_rows), use_container_width=True)
else:
    st.info("No existing mapping detected for current inputs.")

# ---- CADS Search Buttons ----
b1, b2, b3, b4 = st.columns(4)
def _load_cads_df():
    if cads_upload is not None:
        if cads_upload.name.lower().endswith(".xlsx"):
            return pd.read_excel(cads_upload, engine="openpyxl")
        return pd.read_csv(cads_upload)
    if CADS_IS_EXCEL:
        sheet_arg = CADS_SHEET_NAME
        try: sheet_arg = int(sheet_arg)
        except Exception: pass
        return load_cads_from_github_excel(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH, sheet_name=sheet_arg)
    return load_cads_from_github_csv(GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH)

# (1) Search CADS using mapped vehicle
with b2:
    if st.button("🔎 Search CADS (mapped vehicle)"):
        try:
            df_cads = _load_cads_df()
            df_cads = _strip_object_columns(df_cads)
            mapping_to_use = None
            vm = pick_mapping_by_vehicle(st.session_state.get("mappings", {}), vehicle)
            if canon_text(vehicle) and vm:
                mapping_to_use = vm[1]
            else:
                best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, trim, trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
                if not best and (year or make or model):
                    best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, "", trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
                if best:
                    mapping_to_use = best[1]
            if not mapping_to_use:
                st.warning("No mapped vehicle detected.")
            else:
                df_match, diag = match_cads_rows_for_mapping(df_cads, mapping_to_use, exact_model_when_full=MODEL_EXACT_WHEN_FULL, trim_exact_only=TRIM_EXACT_ONLY, strict_and=STRICT_AND, stopword_threshold=STOPWORD_THRESHOLD, token_min_len=TOKEN_MIN_LEN, effective_model_cols_override=OVERRIDE_COLS, trim_as_hint=True, year_require_exact=YEAR_REQUIRE_EXACT)
                if len(df_match) > 0:
                    st.success(f"Found {len(df_match)} CADS row(s).")
                    selectable = df_match.copy()
                    if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                    st.session_state["results_df_mapped"] = selectable
                    st.session_state["code_candidates_mapped"] = get_cads_code_candidates(selectable)
                    st.session_state["model_code_candidates_mapped"] = get_model_code_candidates(selectable)
                else:
                    st.warning("No CADS rows found.")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

# ---- Results Table for Mapped ----
if "results_df_mapped" in st.session_state:
    st.subheader("Select vehicles from CADS results — Mapped Vehicle")
    st.data_editor(st.session_state["results_df_mapped"], use_container_width=True, height=TABLE_HEIGHT, key="editor_mapped")
    if st.button("➕ Add selected (mapped) to mappings"):
        selected = st.session_state["results_df_mapped"][st.session_state["results_df_mapped"]["Select"] == True]
        for _, row in selected.iterrows():
            key = f"{canon_text(make)}|{canon_text(model)}|{canon_text(trim, True)}|{(year or '').strip()}"
            st.session_state.mappings[key] = {
                "year": year, "make": make, "model": model, "trim": trim,
                "vehicle": vehicle or row.get("Vehicle",""),
                "code": row.get(st.session_state.get("code_candidates_mapped",[None])[0],""),
                "model_code": row.get(st.session_state.get("model_code_candidates_mapped",[None])[0],""),
                "ymmt": f"{year}|{make}|{model}|{trim}"
            }
        st.success(f"Added {len(selected)} mapping(s).")

# ---- Current Mappings ----
st.subheader("Current Mappings (session)")
if st.session_state.get("mappings"):
    st.dataframe(pd.DataFrame.from_dict(st.session_state.mappings, orient="index"), use_container_width=True)
else:
    st.info("No mappings yet.")

# ---- Commit to GitHub ----
if st.sidebar.button("💾 Commit mappings to GitHub"):
    try:
        resp = save_json_to_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH, st.session_state.get("mappings", {}), commit_msg, author_name="AFF Mapping App", author_email="aff-mapping@app.local", use_feature_branch=use_feature_branch)
        st.sidebar.success("Committed ✅")
    except Exception as e:
        st.sidebar.error(f"Commit failed: {e}")
