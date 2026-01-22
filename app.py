
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# Harvester Mode: server-executed, stateless; Mozenda-friendly semantic table
# Full-Length Build (2026-01-18)

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
UNBUILDABLE_PATH = "data/unbuildable_vehicles.json"
CADS_IS_EXCEL   = False
CADS_SHEET_NAME_DEFAULT = "0"

CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ---- Canonicalization / Helpers ----
def canon_text(val: str, for_trim: bool=False) -> str:
    s = (val or "").replace("\u00A0", " ")  # normalize NBSP
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
    if row == usr: return ("exact", 1.0)
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

# ---- GitHub loader (always fetch mappings on startup / reload) ----

@st.cache_data(ttl=60)
def fetch_unbuildable_from_github(owner, repo, path, token, ref):
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
        raise RuntimeError(f"Failed to load unbuildable list ({r.status_code}): {r.text}")
        
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

def save_unbuildable_to_github(owner, repo, path, token, branch, payload_dict):
    content = json.dumps(payload_dict, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    r = _session.get(
        gh_contents_url(owner, repo, path),
        headers=gh_headers(token),
        params={"ref": branch},
        timeout=15,
    )
    sha = r.json().get("sha") if r.status_code == 200 else None

    data = {
        "message": "chore(app): mark vehicle as missing CADS data",
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha

    r2 = _session.put(
        gh_contents_url(owner, repo, path),
        headers=gh_headers(token),
        json=data,
        timeout=15,
    )
    if r2.status_code not in (200, 201):
        raise RuntimeError(f"Failed to save unbuildable list ({r2.status_code}): {r2.text}")


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

# ---- Mapping pickers ----
def pick_best_mapping(
    mappings: Dict[str, Dict[str, str]],
    year: str, make: str, model: str, trim: str,
    trim_exact_only: bool = False, model_exact_when_full: bool = True
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
        vmk = v.get("make", ""); vy = v.get("year", ""); vtr = v.get("trim", ""); vmd = v.get("model", "")
        if canon_text(vmk) != cmk: continue
        if not year_token_matches(vy, cy): continue

        tmatch, tscore = trim_matches(vtr, ctr, exact_only=trim_exact_only)
        if not tmatch:
            if ctr: tscore = 0.3
            else:   continue

        ms = model_similarity(vmd, cmd)
        if force_exact_model and canon_text(vmd) != cmd: ms *= 0.5
        score = tscore * 0.6 + ms * 0.4
        candidates.append((k, v, score))

    if not candidates: return None
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[0]

def pick_mapping_by_vehicle(mappings: Dict[str, Dict[str, str]], vehicle: str) -> Optional[Tuple[str, Dict[str, str]]]:
    cv = canon_text(vehicle)
    if not cv: return None
    for k, v in mappings.items():
        if canon_text(v.get("vehicle", "")) == cv:
            return (k, v)
    return None

def find_mappings_by_ymm_all(mappings: Dict[str, Dict[str, str]], year: str, make: str, model: str, trim: Optional[str] = None) -> List[Tuple[str, Dict[str, str]]]:
    cy  = (year or "")
    cmk = canon_text(make)
    cmd = canon_text(model)
    ctr = canon_text(trim or "", True)
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

# ---- CADS filtering ----
def _tiered_model_mask(eff: pd.Series, md: str, discriminant: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if not md:
        true_mask = pd.Series([True]*len(eff), index=eff.index)
        return true_mask, true_mask, true_mask, true_mask
    and_mask  = eff.apply(lambda s: all(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    or_mask   = eff.apply(lambda s: any(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    ns_text   = " ".join(discriminant).strip()
    ns_mask   = eff.str.contains(ns_text, na=False) if ns_text else eff.str.contains(md, na=False)  # FIX: str.contains
    full_mask = eff.str.contains(md, na=False)
    return and_mask, or_mask, ns_mask, full_mask

VEHICLE_LIKE_CANDS = [
    "Vehicle","Description","ModelTrim","ModelName","AD_SERIES","Series","STYLE_NAME","AD_MODEL","MODEL_NAME"
]

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
        s_trim = df2[trim_col].astype(str)
        m_exact  = s_trim.str.lower() == tr
        m_subset = s_trim.apply(lambda x: _trim_tokens(tr).issubset(_trim_tokens(x)))
        if trim_as_hint:
            df2["__trim_match_type__"]  = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[0])
            df2["__trim_match_score__"] = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[1])
        else:
            if trim_exact_only: masks.append(m_exact)
            else: masks.append(m_exact | m_subset)
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

# ---- CADS matching for a single mapping ----
def get_cads_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_CODE_PREFS if c in df.columns] or list(df.columns)

def get_model_code_candidates(df: pd.DataFrame) -> List[str]:
    return [c for c in CADS_MODEL_CODE_PREFS if c in df.columns] or list(df.columns)

def match_cads_rows_for_mapping(
    df: pd.DataFrame, mapping: Dict[str, str],
    exact_model_when_full: bool, trim_exact_only: bool, strict_and: bool,
    stopword_threshold: float, token_min_len: int,
    effective_model_cols_override: Optional[List[str]] = None,
    trim_as_hint: bool = False, year_require_exact: bool = False
) -> Tuple[pd.DataFrame, Dict[str, any]]:
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

    # Model Code tier
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

    # Vehicle-text tier
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

# ===================== Mozenda Harvest Helpers =====================
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
            if sv:
                return sv
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
        back = [c for c in cols if c not in front]
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

# ===================== UI (normal mode) =====================
st.title("AFF Vehicle Mapping")

# Parse URL params early (used by both modes)
params = st.experimental_get_query_params()
HARVEST_MODE   = (params.get("harvest", ["0"])[0] == "1")
HARVEST_SOURCE = (params.get("source", ["mapped"])[0])  # mapped | inputs | quick_ymmt | quick_vehicle | unmapped

# Helpers to parse bool/float/int params
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

# Loaders usable in both modes
def _load_cads_df(cads_path: Optional[str] = None, cads_is_excel: Optional[bool] = None, sheet_name: Optional[str] = None, ref: Optional[str] = None):
    path = cads_path if cads_path is not None else CADS_PATH
    is_xlsx = cads_is_excel if cads_is_excel is not None else CADS_IS_EXCEL
    ref = ref or GH_BRANCH
    if is_xlsx:
        sn = sheet_name if sheet_name is not None else CADS_SHEET_NAME_DEFAULT
        try:
            sn = int(sn)  # allow numeric index
        except Exception:
            pass
        return load_cads_from_github_excel(GH_OWNER, GH_REPO, path, GH_TOKEN, ref=ref, sheet_name=sn)
    return load_cads_from_github_csv(GH_OWNER, GH_REPO, path, GH_TOKEN, ref=ref)

# ===================== SERVER-EXECUTED HARVEST MODE =====================
def _run_harvest():
    """
    Stateless, server-executed harvest. No buttons, no prior session.
    Use URL params to compute results and immediately emit a semantic table.
    """
    # Read common matching controls from params (with safe defaults)
    trim_as_hint         = _get_bool("trim_as_hint", True)
    trim_exact_only      = _get_bool("trim_exact_only", False)
    strict_and           = _get_bool("strict_and", True)
    model_exact_when_full= _get_bool("model_exact_when_full", False)
    year_require_exact   = _get_bool("year_require_exact", True)
    stopword_threshold   = _get_float("stopword_threshold", 0.60)
    token_min_len        = _get_int("token_min_len", 2)
    plain                = _get_bool("plain", False)

    # CADS source override via params (optional)
    cads_path     = _get_str("cads_path", CADS_PATH)
    cads_is_excel = _get_bool("cads_is_excel", CADS_IS_EXCEL)
    cads_sheet    = _get_str("cads_sheet", CADS_SHEET_NAME_DEFAULT)
    ref_branch    = _get_str("ref", GH_BRANCH)

    # Effective model override columns (comma-separated)
    oc = _get_str("override_cols", "AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES")
    override_cols = [c.strip() for c in oc.split(",") if c.strip()] or None

    # Load artifacts
    mappings = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, ref_branch)
    df_cads  = _load_cads_df(cads_path, cads_is_excel, cads_sheet, ref=ref_branch)
    df_cads  = _strip_object_columns(df_cads)

    source = HARVEST_SOURCE

    # Route by source
    if source in ("inputs", "quick_ymmt"):
        # inputs: year/make/model/trim from params
        year = _get_str("year", "")
        make = _get_str("make", "")
        model = _get_str("model", "")
        trim = _get_str("trim", "")
        results, diag = filter_cads_generic(
            df_cads, year, make, model, trim,
            exact_model_when_full=model_exact_when_full,
            trim_exact_only=trim_exact_only,
            strict_and=strict_and,
            stopword_threshold=stopword_threshold,
            token_min_len=token_min_len,
            effective_model_cols_override=override_cols,
            trim_as_hint=trim_as_hint,
            year_require_exact=year_require_exact,
        )
        render_harvest_table(
            results,
            table_id="cads_inputs_results" if source == "inputs" else "cads_mapped_quick_ymmt",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Input-driven results" if source == "inputs" else "CADS – Quick YMM(/T) mapped results",
            plain=plain,
        )
        st.stop()

    elif source == "mapped":
        # mapped: union all mappings for given YMM(/T)
        year = _get_str("year", ""); make = _get_str("make", ""); model = _get_str("model", ""); trim = _get_str("trim", "")
        ymmt_list = find_mappings_by_ymm_all(mappings, year, make, model, trim if canon_text(trim, True) else None)
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
                df_hit["__mapped_key__"] = f"{mp.get('make','')}|{mp.get('model','')}|{mp.get('trim','')}|{mp.get('year','')}"
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
        )
        st.stop()

    elif source == "quick_vehicle":
        # quick_vehicle: `vehicle` param, map via tolerant trim logic
        veh_txt = _get_str("vehicle", "")
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_mapped_quick_vehicle", caption="No vehicle provided", plain=plain)
            st.stop()

        year_val = extract_primary_year(veh_txt)
        toks = tokens(veh_txt, min_len=1)  # preserve single-letter like 'S' in 'S line'
        make_val = toks[1] if len(toks) >= 2 else ""
        model_val = toks[2] if len(toks) >= 3 else ""
        trim_val  = " ".join(toks[3:]) if len(toks) > 3 else ""

        all_mps = []
        for _, v in mappings.items():
            if canon_text(v.get("make","")) != canon_text(make_val): continue
            if not year_token_matches(v.get("year",""), str(year_val)): continue
            if canon_text(v.get("model","")) != canon_text(model_val): continue
            if trim_val:
                ok, _ = trim_matches(v.get("trim",""), trim_val, exact_only=False)
                if not ok: continue
            else:
                if v.get("trim","").strip(): continue
            all_mps.append(v)

        hits = []
        for mp in all_mps:
            df_hit, diag = match_cads_rows_for_mapping(
                df_cads, mp,
                exact_model_when_full=model_exact_when_full, trim_exact_only=False, strict_and=strict_and,
                stopword_threshold=stopword_threshold, token_min_len=token_min_len,
                effective_model_cols_override=override_cols, trim_as_hint=True, year_require_exact=year_require_exact,
            )
            if len(df_hit) > 0:
                df_hit = df_hit.copy()
                df_hit.insert(0, "Mapped Vehicle", mp.get("vehicle",""))
                df_hit["__mapped_key__"] = f"{mp.get('make','')}|{mp.get('model','')}|{mp.get('trim','')}|{mp.get('year','')}"
                df_hit["__tier__"] = diag.get("tier_used")
                hits.append(df_hit)
        df_all = pd.concat(hits, ignore_index=True) if hits else df_cads.iloc[0:0]
        if len(df_all) > 0:
            df_all = df_all.drop_duplicates(subset=["STYLE_ID","AD_VEH_ID"], keep="first").reset_index(drop=True)
        render_harvest_table(
            df_all,
            table_id="cads_mapped_quick_vehicle",
            preferred_order=["Mapped Vehicle"] + HARVEST_PREF_ORDER + ["__mapped_key__","__tier__"],
            include_attr_cols=["Mapped Vehicle","AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE","__mapped_key__","__tier__"],
            caption="CADS – Quick Vehicle mapped results",
            plain=plain,
        )
        st.stop()

    elif source == "unmapped":
        veh_txt = canon_text(_get_str("vehicle", ""))
        if not veh_txt:
            render_harvest_table(df_cads.iloc[0:0], table_id="cads_unmapped_results", caption="No vehicle provided", plain=plain)
            st.stop()
        hits = []
        for col in VEHICLE_LIKE_CANDS:
            if col in df_cads.columns:
                ser = df_cads[col].astype(str).str.lower()
                mask = ser.str.contains(veh_txt, na=False)
                if mask.any():
                    hits.append(df_cads[mask])
        df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True) if hits else df_cads.iloc[0:0]
        render_harvest_table(
            df_union,
            table_id="cads_unmapped_results",
            preferred_order=HARVEST_PREF_ORDER,
            include_attr_cols=["AD_YEAR","AD_MAKE","AD_MODEL","Trim","STYLE_ID","AD_VEH_ID","AD_MFGCODE","MODEL_CODE"],
            caption="CADS – Unmapped search results",
            plain=plain,
        )
        st.stop()

    # Default: nothing matched
    st.markdown("<p id='harvest-empty'>No harvest source matched or insufficient parameters.</p>", unsafe_allow_html=True)
    st.stop()

# If HARVEST_MODE is on, run the stateless path and stop.
if HARVEST_MODE:
    _run_harvest()

# ===================== Normal Interactive UI (unchanged core UX) =====================
# Sidebar: CADS Settings
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

# Sidebar: Actions & Matching
st.sidebar.header("Actions")
st.sidebar.subheader("Mappings Source")
load_branch = st.sidebar.text_input(
    "Branch to load mappings from",
    value=st.session_state.get("load_branch", GH_BRANCH),
    help="This is the branch we read mappings.json from on startup and on Reload."
)
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
YEAR_REQUIRE_EXACT = st.sidebar.checkbox(
    "Require exact year match", value=True,
    help="Only include rows where the user-entered year appears among the CADS row's year tokens. Rows with unparseable year won't be eliminated."
)
STOPWORD_THRESHOLD = st.sidebar.slider("Per-make stopword threshold", 0.1, 0.9, 0.60, 0.05)
TOKEN_MIN_LEN = st.sidebar.slider("Token minimum length", 1, 5, 2, 1)

st.sidebar.subheader("Effective Model (override)")
EFFECTIVE_MODEL_COLS_OVERRIDE = st.sidebar.text_input(
    "Comma-separated CADS columns to use (optional)",
    value="AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES",
    help="If blank, auto-detect model-like and series-like columns."
)
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

# Always refresh mappings unless locally modified
local_mod_flag = st.session_state.get("local_mappings_modified", False)
try:
    fetched = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, st.session_state["load_branch"])
    if not local_mod_flag:
        st.session_state["mappings"] = fetched
        st.caption(f"🔁 Loaded {len(fetched)} mapping(s) from GitHub @ {st.session_state['load_branch']}/{MAPPINGS_PATH}")
    else:
        st.caption("⚠️ Using locally modified mappings (not reloaded from GitHub).")
except Exception as e:
    st.warning(f"Could not load mappings from GitHub: {e}")

# ---- Quick Lookups & full UI (unchanged logic) ----
st.header("YMM Quick Lookup (mapped)")
ymmt_col1, ymmt_col2, ymmt_col3, ymmt_col4 = st.columns(4)
with ymmt_col1: q_year  = st.text_input("Quick Year",  key="quick_year_input",  placeholder="e.g., 2025")
with ymmt_col2: q_make  = st.text_input("Quick Make",  key="quick_make_input",  placeholder="e.g., Audi")
with ymmt_col3: q_model = st.text_input("Quick Model", key="quick_model_input", placeholder="e.g., Q7")
with ymmt_col4: q_trim  = st.text_input("Quick Trim (optional)", key="quick_trim_input", placeholder="e.g., 45 TFSI quattro Premium")

if st.button("🔎 Quick Search by YMM(/T) (mapped)", key="btn_quick_ymmt_v1"):
    if not (q_year or q_make or q_model):
        st.warning("Enter at least Year, Make, and Model.")
    else:
        try:
            all_mps = find_mappings_by_ymm_all(st.session_state.get("mappings", {}), q_year, q_make, q_model, q_trim or None)
            if not all_mps:
                st.error("❌ No mapped entries for this YMM(/T). If expected, add mappings below and commit, then retry.")
            else:
                df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
                hits = []
                for _, mp in all_mps:
                    df_hit, diag = match_cads_rows_for_mapping(
                        df_cads, mp,
                        exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                        trim_exact_only=False, strict_and=STRICT_AND,
                        stopword_threshold=STOPWORD_THRESHOLD, token_min_len=TOKEN_MIN_LEN,
                        effective_model_cols_override=OVERRIDE_COLS, trim_as_hint=True, year_require_exact=YEAR_REQUIRE_EXACT,
                    )
                    if len(df_hit) > 0:
                        df_hit = df_hit.copy()
                        df_hit["__mapped_key__"] = f"{mp.get('make','')}|{mp.get('model','')}|{mp.get('trim','')}|{mp.get('year','')}"
                        df_hit["__tier__"] = diag.get("tier_used")
                        hits.append(df_hit)
                if hits:
                    df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True)
                    st.success(f"Found {len(df_union)} CADS row(s) for YMM(/T) '{q_year} {q_make} {q_model} {q_trim or ''}'.")
                    st.dataframe(df_union, use_container_width=True, height=TABLE_HEIGHT)
                else:
                    st.warning("No CADS rows matched for the mapped YMM(/T). Check Code/Model Code/Vehicle text or adjust Matching Controls.")
        except Exception as e:
            st.error(f"Quick YMM(/T) search failed: {e}")

st.header("Vehicle-Only Quick Lookup (mapped)")
quick_vehicle = st.text_input("Vehicle (Year Make Model [Trim])", key="quick_vehicle_input", placeholder="e.g., 2025 Audi SQ5 or 2025 Audi SQ5 Premium Plus")

if st.button("🔎 Quick Search by Vehicle (mapped)", key="btn_quick_vehicle_v2"):
    veh_txt = (quick_vehicle or "").strip()
    if not veh_txt:
        st.warning("Enter a vehicle string first.")
    else:
        try:
            year_val = extract_primary_year(veh_txt)
            tokens_list = tokens(veh_txt, min_len=1)
            make_val = tokens_list[1] if len(tokens_list) >= 2 else ""
            model_val = tokens_list[2] if len(tokens_list) >= 3 else ""
            trim_val  = " ".join(tokens_list[3:]) if len(tokens_list) > 3 else ""
            mappings = st.session_state.get("mappings", {})

            all_mps = []
            for _, v in mappings.items():
                if canon_text(v.get("make","")) != canon_text(make_val): continue
                if not year_token_matches(v.get("year",""), str(year_val)): continue
                if canon_text(v.get("model","")) != canon_text(model_val): continue
                if trim_val:
                    ok, _ = trim_matches(v.get("trim",""), trim_val, exact_only=False)
                    if not ok: continue
                else:
                    if v.get("trim","").strip(): continue
                all_mps.append(v)

            if not all_mps:
                st.error("❌ No mapped entries for this vehicle. If expected, add mappings below and commit, then retry.")
            else:
                df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
                hits = []
                for mp in all_mps:
                    df_hit, diag = match_cads_rows_for_mapping(
                        df_cads, mp,
                        exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                        trim_exact_only=False, strict_and=STRICT_AND,
                        stopword_threshold=STOPWORD_THRESHOLD, token_min_len=TOKEN_MIN_LEN,
                        effective_model_cols_override=OVERRIDE_COLS, trim_as_hint=True, year_require_exact=YEAR_REQUIRE_EXACT,
                    )
                    if len(df_hit) > 0:
                        df_hit = df_hit.copy()
                        df_hit.insert(0, "Mapped Vehicle", mp.get("vehicle",""))
                        df_hit["__mapped_key__"] = f"{mp.get('make','')}|{mp.get('model','')}|{mp.get('trim','')}|{mp.get('year','')}"
                        df_hit["__tier__"] = diag.get("tier_used")
                        hits.append(df_hit)

                if hits:
                    df_all = pd.concat(hits, ignore_index=True)
                    df_all = df_all.drop_duplicates(subset=["STYLE_ID", "AD_VEH_ID"]).reset_index(drop=True)
                    st.success(f"Found {len(df_all)} CADS row(s) for '{veh_txt}'.")
                    st.dataframe(df_all, use_container_width=True, height=TABLE_HEIGHT)
                else:
                    st.warning("No CADS rows matched for the mapped vehicle(s). Check Code/Model Code/Vehicle text.")
        except Exception as e:
            st.error(f"Quick vehicle search failed: {e}")

st.header("Search CADS for Unmapped Vehicle")
unmapped_vehicle = st.text_input("Vehicle (search in CADS)", key="unmapped_vehicle_input_v1", placeholder="e.g., Q3 S line 45 TFSI quattro Premium")
if st.button("🔍 Search CADS for Unmapped Vehicle", key="btn_unmapped_v1"):
    veh_txt = canon_text(unmapped_vehicle)
    if not veh_txt:
        st.warning("Enter a vehicle string first.")
    else:
        try:
            df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
            hits = []
            for col in VEHICLE_LIKE_CANDS:
                if col in df_cads.columns:
                    ser = df_cads[col].astype(str).str.lower()
                    mask = ser.str.contains(veh_txt, na=False)
                    if mask.any(): hits.append(df_cads[mask])
            if hits:
                df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True)
                st.success(f"Found {len(df_union)} CADS row(s) matching '{unmapped_vehicle}'.")
                if "Select" not in df_union.columns:
                    df_union.insert(0, "Select", False)
                st.session_state["results_df_unmapped"] = df_union
                st.session_state["code_candidates_unmapped"] = get_cads_code_candidates(df_union)
                st.session_state["model_code_candidates_unmapped"] = get_model_code_candidates(df_union)
                st.session_state["code_column_unmapped"] = st.session_state["code_candidates_unmapped"][0] if st.session_state.get("code_candidates_unmapped") else None
                st.session_state["model_code_column_unmapped"] = st.session_state["model_code_candidates_unmapped"][0] if st.session_state.get("model_code_candidates_unmapped") else None
            else:
                st.warning("No CADS rows matched that vehicle text.")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

# ---- Selection helpers & add-to-mappings
def _select_all(df_key: str):
    if df_key in st.session_state:
        df = st.session_state[df_key].copy()
        if "Select" in df.columns:
            df["Select"] = True
        st.session_state[df_key] = df

def _clear_selection(df_key: str):
    if df_key in st.session_state:
        df = st.session_state[df_key].copy()
        if "Select" in df.columns:
            df["Select"] = False
        st.session_state[df_key] = df

def _effective_vehicle_text(row: pd.Series) -> str:
    for col in ["Vehicle", "__effective_model__", "AD_MODEL", "MODEL_NAME", "STYLE_NAME", "AD_SERIES"]:
        if col in row.index:
            val = str(row.get(col, "") or "").strip()
            if val:
                return val
    return ""

def _add_selected_rows_to_mappings(df_key, code_col_key: str, model_code_col_key: str, year_val: str, make_val: str, model_val: str, trim_val: str, vehicle_val: str, rows_override=None):

    # === ENFORCE VEHICLE REQUIRED (STATIC, VERBATIM) ===
    vehicle_text = (vehicle_val or "").strip()

    if not vehicle_text:
        st.warning(
            "Vehicle is required.\n\n"
            "Paste the exact vehicle string from the source website "
            "(e.g., '2026 Acura TLX FWD 2.4L Automatic').\n\n"
            "Vehicle is treated as a static, authoritative identifier."
        )
        return
    # ==================================================

    import pandas as pd
    if rows_override is not None:
        selected = rows_override.copy()
    else:
        df = st.session_state.get(df_key)
        if df is None or len(df) == 0:
            st.warning("No rows to add."); return
        if "Select" not in df.columns:
            st.warning("Selection column is missing."); return
        selected = df[df["Select"].astype(bool) == True]

    if len(selected) == 0:
        st.warning("Select at least one row."); return

    code_col = st.session_state.get(code_col_key)
    model_code_col = st.session_state.get(model_code_col_key)
    added = 0
    st.session_state.setdefault("mappings", {})

    for _, row in selected.iterrows():
        ymmt = f"{(year_val or '').strip()}|{canon_text(make_val)}|{canon_text(model_val)}|{canon_text(trim_val, True)}"
        vehicle_text = (vehicle_val or "").strip() or _effective_vehicle_text(row)
        code_val = str(row.get(code_col, "") or "").strip() if code_col else ""
        model_code_val = str(row.get(model_code_col, "") or "").strip() if model_code_col else ""
        unique_id = str(row.get("STYLE_ID") or row.get("AD_VEH_ID") or "")
        key = f"{canon_text(make_val)}|{canon_text(model_val)}|{canon_text(trim_val, True)}|{(year_val or '').strip()}|{unique_id}"
        st.session_state.mappings[key] = {
            "year": (year_val or "").strip(),
            "make": (make_val or "").strip(),
            "model": (model_val or "").strip(),
            "trim": (trim_val or "").strip(),
            "vehicle": vehicle_text,
            "code": code_val,
            "model_code": model_code_val,
            "ymmt": ymmt,
        }
        added += 1
    st.session_state["local_mappings_modified"] = True
    st.success(f"Added {added} mapping(s).")

# --- Mapped results table (edited DataFrame)
if "results_df_unmapped" in st.session_state:
    st.subheader("Select CADS rows to map this vehicle")
    cUA, cUB = st.columns(2)
    with cUA:
        st.session_state["code_column_unmapped"] = st.selectbox("Code column (unmapped results)", options=st.session_state.get("code_candidates_unmapped", []), index=0 if st.session_state.get("code_candidates_unmapped") else None)
    with cUB:
        st.session_state["model_code_column_unmapped"] = st.selectbox("Model Code column (unmapped results)", options=st.session_state.get("model_code_candidates_unmapped", []), index=0 if st.session_state.get("model_code_candidates_unmapped") else None)
    edited_unmapped_df = st.data_editor(st.session_state["results_df_unmapped"], use_container_width=True, height=TABLE_HEIGHT, key="editor_unmapped", column_config={"Select": st.column_config.CheckboxColumn(required=False)})
    if st.button("➕ Add selected (unmapped) to mappings", key="btn_unmapped_add_v1"):
        df_to_use = edited_unmapped_df if edited_unmapped_df is not None else st.session_state["results_df_unmapped"]
        selected = df_to_use[df_to_use["Select"].astype(bool) == True] if "Select" in df_to_use.columns else df_to_use.iloc[0:0]
        if len(selected) == 0:
            st.warning("Select at least one row.")
        else:
            y_val = (st.session_state.get("year_input", "") or "").strip()
            mk_val = (st.session_state.get("make_input", "") or "").strip()
            md_val = (st.session_state.get("model_input", "") or "").strip()
            tr_val = (st.session_state.get("trim_input", "") or "").strip()
            _add_selected_rows_to_mappings(None, "code_column_unmapped", "model_code_column_unmapped", y_val, mk_val, md_val, tr_val, st.session_state.get("unmapped_vehicle_input_v1",""), selected)

# ---- Inputs (YMMT + Vehicle) for standard mapping flow
st.subheader("Edit / Add Mapping")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: year = st.text_input("Year", key="year_input", placeholder="e.g., 2025")
with c2: make = st.text_input("Make", key="make_input", placeholder="e.g., Audi")
with c3: model = st.text_input("Model", key="model_input", placeholder="e.g., Q7")
with c4: trim = st.text_input("Trim", key="trim_input", placeholder="e.g., 45 TFSI quattro Premium")

with c5: vehicle = st.text_input(
    "Vehicle (REQUIRED – paste exact value from source website)",
    key="vehicle_input",
    placeholder="e.g., 2026 Acura TLX FWD 2.4L Automatic"
)

with c6: mapped_code = st.text_input("Mapped Code", key="code_input", placeholder="Optional (STYLE_ID/AD_VEH_ID/etc.)")
model_code_input = st.text_input("Model Code (optional)", key="model_code_input", placeholder="AD_MFGCODE/MODEL_CODE/etc.")

st.caption(
    f"🔎 Inputs → Year='{canon_text(year)}' Make='{canon_text(make)}' Model='{canon_text(model)}' Trim='{canon_text(trim, True)}' | "
    f"TRIM_AS_HINT={TRIM_AS_HINT}, TRIM_EXACT_ONLY={TRIM_EXACT_ONLY}, MODEL_EXACT_WHEN_FULL={MODEL_EXACT_WHEN_FULL}, STRICT_AND={STRICT_AND}, "
    f"YEAR_REQUIRE_EXACT={YEAR_REQUIRE_EXACT}, STOPWORD_THRESHOLD={STOPWORD_THRESHOLD}, TOKEN_MIN_LEN={TOKEN_MIN_LEN}, OverrideCols={OVERRIDE_COLS or '(auto)'}"
)

# ---- Existing Mapping (for current inputs)
st.subheader("Existing Mapping (for current inputs)")
existing_rows = []
vehicle_first = False
if canon_text(vehicle) and not (canon_text(year) or canon_text(make) or canon_text(model) or canon_text(trim, True)):
    vm = pick_mapping_by_vehicle(st.session_state.get("mappings", {}), vehicle)
    vehicle_first = True
    if vm:
        k, v = vm
        existing_rows.append({"Match Level":"by_vehicle","Score":1.0,"Key":k,"Year":v.get("year",""),"Make":v.get("make",""),"Model":v.get("model",""),"Trim":v.get("trim",""),"Vehicle":v.get("vehicle",""),"Code":v.get("code",""),"Model Code":v.get("model_code","")})

if not existing_rows:
    best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, trim, trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
    if not best and (year or make or model):
        best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, "", trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
    if best:
        k, v, score = best
        existing_rows.append({"Match Level":"generic_best_trim_model_year" if canon_text(trim, True) else "generic_best_by_ymm","Score":round(score,3),"Key":k,"Year":v.get("year",""),"Make":v.get("make",""),"Model":v.get("model",""),"Trim":v.get("trim",""),"Vehicle":v.get("vehicle",""),"Code":v.get("code",""),"Model Code":v.get("model_code","")})


if existing_rows:
    st.dataframe(pd.DataFrame(existing_rows), use_container_width=True)
else:
    st.info("No existing mapping detected for current inputs.")

    vehicle_text = (vehicle or "").strip()
    if vehicle_text:
        # Check if already marked unbuildable
        unbuildable = fetch_unbuildable_from_github(
            GH_OWNER, GH_REPO, UNBUILDABLE_PATH, GH_TOKEN, st.session_state["load_branch"]
        )

        if canon_text(vehicle_text) in {canon_text(v) for v in unbuildable.keys()}:
            st.warning("🚧 Missing Vehicle Data")
            st.caption("This vehicle is not yet available in the CADS file.")
        else:
            if st.button("🚧 No Vehicle Data Yet"):
                try:
                    now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    unbuildable[vehicle_text] = {
                        "vehicle": vehicle_text,
                        "year": year.strip(),
                        "make": make.strip(),
                        "model": model.strip(),
                        "reason": "CADS does not contain this model year yet",
                        "added_at": now_ts,
                        "added_by": "dashboard-app",
                    }

                    save_unbuildable_to_github(
                        GH_OWNER,
                        GH_REPO,
                        UNBUILDABLE_PATH,
                        GH_TOKEN,
                        st.session_state["load_branch"],
                        unbuildable,
                    )

                    st.success("Vehicle marked as missing CADS data.")
                    st.cache_data.clear()

                except Exception as e:
                    st.error(f"Failed to mark vehicle as missing: {e}")


# ---- CADS Search Buttons
b1, b2, b3, b4 = st.columns(4)

with b2:
    if st.button("🔎 Search CADS (mapped vehicle)"):
        try:
            df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
            mapping_to_use = None
            vm = pick_mapping_by_vehicle(st.session_state.get("mappings", {}), vehicle)

            if canon_text(vehicle) and vm:
                mapping_to_use = vm[1]
            else:
                ymmt_list = find_mappings_by_ymm_all(st.session_state.get("mappings", {}), year, make, model, trim if canon_text(trim, True) else None)
                if ymmt_list:
                    hits = []
                    for _, mp in ymmt_list:
                        df_hit, diag = match_cads_rows_for_mapping(df_cads, mp, MODEL_EXACT_WHEN_FULL, TRIM_EXACT_ONLY, STRICT_AND, STOPWORD_THRESHOLD, TOKEN_MIN_LEN, OVERRIDE_COLS, True, YEAR_REQUIRE_EXACT)
                        if len(df_hit) > 0:
                            df_hit = df_hit.copy()
                            df_hit["__mapped_key__"] = f"{mp.get('make','')}|{mp.get('model','')}|{mp.get('trim','')}|{mp.get('year','')}"
                            df_hit["__tier__"] = diag.get("tier_used")
                            hits.append(df_hit)
                    if hits:
                        df_union = pd.concat(hits, ignore_index=True).drop_duplicates().reset_index(drop=True)
                        st.success(f"Found {len(df_union)} CADS row(s) for YMM(/T) (multiple mappings).")
                        if "Select" not in df_union.columns: df_union.insert(0, "Select", False)
                        st.session_state["results_df_mapped"] = df_union
                        st.session_state["code_candidates_mapped"] = get_cads_code_candidates(df_union)
                        st.session_state["model_code_candidates_mapped"] = get_model_code_candidates(df_union)
                        st.session_state["code_column_mapped"] = st.session_state["code_candidates_mapped"][0] if st.session_state.get("code_candidates_mapped") else None
                        st.session_state["model_code_column_mapped"] = st.session_state["model_code_candidates_mapped"][0] if st.session_state.get("model_code_candidates_mapped") else None
                        st.stop()
                best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, trim, trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
                if not best and (year or make or model):
                    best = pick_best_mapping(st.session_state.get("mappings", {}), year, make, model, "", trim_exact_only=TRIM_EXACT_ONLY, model_exact_when_full=MODEL_EXACT_WHEN_FULL)
                if best: mapping_to_use = best[1]

            if not mapping_to_use:
                st.warning("No mapped vehicle detected.")
            else:
                df_match, diag = match_cads_rows_for_mapping(df_cads, mapping_to_use, MODEL_EXACT_WHEN_FULL, TRIM_EXACT_ONLY, STRICT_AND, STOPWORD_THRESHOLD, TOKEN_MIN_LEN, OVERRIDE_COLS, True, YEAR_REQUIRE_EXACT)
                st.session_state["last_diag_mapped"] = diag
                st.caption(f"Tier used (mapped): {diag.get('tier_used')}")
                if len(df_match) > 0:
                    selectable = df_match.copy()
                    if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                    st.session_state["results_df_mapped"] = selectable
                    st.session_state["code_candidates_mapped"] = get_cads_code_candidates(selectable)
                    st.session_state["model_code_candidates_mapped"] = get_model_code_candidates(selectable)
                    st.session_state["code_column_mapped"] = st.session_state["code_candidates_mapped"][0] if st.session_state.get("code_candidates_mapped") else None
                    st.session_state["model_code_column_mapped"] = st.session_state["model_code_candidates_mapped"][0] if st.session_state.get("model_code_candidates_mapped") else None
                    st.success(f"Found {len(selectable)} CADS row(s).")
                else:
                    st.warning("No CADS rows found.")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

with b3:
    if st.button("🔎 Search CADS (use current inputs)"):
        try:
            df_cads = _load_cads_df_ui(); df_cads = _strip_object_columns(df_cads)
            results, diag = filter_cads_generic(
                df_cads, year, make, model, trim,
                exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                trim_exact_only=TRIM_EXACT_ONLY,
                strict_and=STRICT_AND,
                stopword_threshold=STOPWORD_THRESHOLD,
                token_min_len=TOKEN_MIN_LEN,
                effective_model_cols_override=OVERRIDE_COLS,
                trim_as_hint=TRIM_AS_HINT,
                year_require_exact=YEAR_REQUIRE_EXACT,
            )
            st.session_state["last_diag_inputs"] = diag
            st.caption(f"Tier used (inputs): {diag.get('tier_used')}")
            if len(results) == 0:
                st.warning("No CADS rows matched inputs. Try adjusting controls.")
            else:
                selectable = results.copy()
                if "Select" not in selectable.columns: selectable.insert(0, "Select", False)
                st.session_state["results_df_inputs"] = selectable
                st.session_state["code_candidates_inputs"] = get_cads_code_candidates(selectable)
                st.session_state["model_code_candidates_inputs"] = get_model_code_candidates(selectable)
                st.session_state["code_column_inputs"] = st.session_state["code_candidates_inputs"][0] if st.session_state.get("code_candidates_inputs") else None
                st.session_state["model_code_column_inputs"] = st.session_state["model_code_candidates_inputs"][0] if st.session_state.get("model_code_candidates_inputs") else None
                st.success(f"Found {len(selectable)} CADS row(s).")
        except Exception as e:
            st.error(f"CADS search failed: {e}")

# --- Mapped results editor
if "results_df_mapped" in st.session_state:
    st.subheader("Select vehicles from CADS results — Mapped Vehicle")
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("Select All (mapped)"): _select_all("results_df_mapped")
    with cB:
        if st.button("Clear Selection (mapped)"): _clear_selection("results_df_mapped")
    edited_mapped_df = st.data_editor(st.session_state["results_df_mapped"], use_container_width=True, height=TABLE_HEIGHT, key="editor_mapped", column_config={"Select": st.column_config.CheckboxColumn(required=False)})
    cX, cY, cZ = st.columns(3)
    with cX:
        st.session_state["code_column_mapped"] = st.selectbox("Code column (mapped results)", options=st.session_state.get("code_candidates_mapped", []), index=0 if st.session_state.get("code_candidates_mapped") else None)
    with cY:
        st.session_state["model_code_column_mapped"] = st.selectbox("Model Code column (mapped results)", options=st.session_state.get("model_code_candidates_mapped", []), index=0 if st.session_state.get("model_code_candidates_mapped") else None)
    with cZ:
        if st.button("➕ Add selected (mapped) to mappings", key="btn_mapped_add_v1"):
            df_to_use = edited_mapped_df if edited_mapped_df is not None else st.session_state["results_df_mapped"]
            selected = df_to_use[df_to_use["Select"].astype(bool) == True] if "Select" in df_to_use.columns else df_to_use.iloc[0:0]
            if len(selected) == 0:
                st.warning("Select at least one row.")
            else:
                _add_selected_rows_to_mappings(None, "code_column_mapped", "model_code_column_mapped", year, make, model, trim, vehicle, selected)

# --- Direct input editor
if "results_df_inputs" in st.session_state:
    st.subheader("Select vehicles from CADS results — Direct Input Search")
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("Select All (inputs)"): _select_all("results_df_inputs")
    with cB:
        if st.button("Clear Selection (inputs)"): _clear_selection("results_df_inputs")
    edited_inputs_df = st.data_editor(st.session_state["results_df_inputs"], use_container_width=True, height=TABLE_HEIGHT, key="editor_inputs", column_config={"Select": st.column_config.CheckboxColumn(required=False)})
    cX, cY, cZ = st.columns(3)
    with cX:
        st.session_state["code_column_inputs"] = st.selectbox("Code column (input results)", options=st.session_state.get("code_candidates_inputs", []), index=0 if st.session_state.get("code_candidates_inputs") else None)
    with cY:
        st.session_state["model_code_column_inputs"] = st.selectbox("Model Code column (input results)", options=st.session_state.get("model_code_candidates_inputs", []), index=0 if st.session_state.get("model_code_candidates_inputs") else None)
    with cZ:
        if st.button("➕ Add selected (inputs) to mappings", key="btn_inputs_add_v1"):
            df_to_use = edited_inputs_df if edited_inputs_df is not None else st.session_state["results_df_inputs"]
            selected = df_to_use[df_to_use["Select"].astype(bool) == True] if "Select" in df_to_use.columns else df_to_use.iloc[0:0]
            if len(selected) == 0:
                st.warning("Select at least one row.")
            else:
                _add_selected_rows_to_mappings(None, "code_column_inputs", "model_code_column_inputs", year, make, model, trim, vehicle, selected)

# ---- Diagnostics
st.subheader("Diagnostics")
exp = st.expander("Mapped Search Diagnostics", expanded=False);   exp.write("Last mapped search diagnostics:");   exp.json(st.session_state.get("last_diag_mapped", {}))
exp2 = st.expander("Inputs Search Diagnostics", expanded=False);  exp2.write("Last inputs search diagnostics:");  exp2.json(st.session_state.get("last_diag_inputs", {}))

# ---- Current Mappings
st.subheader("Current Mappings (session)")
if st.session_state.get("mappings"):
    rows = []
    for k, v in st.session_state.mappings.items():
        mapped_vehicle_display = f"{v.get('year','')} {v.get('make','')} {v.get('model','')}"
        if v.get("trim"):
            mapped_vehicle_display += f" {v.get('trim','')}"
        rows.append({
            "Mapped Vehicle": mapped_vehicle_display, "Key": k, "Year": v.get("year",""),
            "Make": v.get("make",""), "Model": v.get("model",""), "Trim": v.get("trim",""),
            "Vehicle": v.get("vehicle",""), "Code": v.get("code",""), "Model Code": v.get("model_code",""),
            "YMMT": v.get("ymmt",""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No mappings yet. Add one above or select CADS rows to add mappings.")

# ---- Commit to GitHub
missing_secrets = []
if not GH_TOKEN:  missing_secrets.append("github.token")
if not GH_OWNER:  missing_secrets.append("github.owner")
if not GH_REPO:   missing_secrets.append("github.repo")
if not GH_BRANCH: missing_secrets.append("github.branch")
if missing_secrets:
    st.sidebar.warning("Missing secrets: " + ", ".join(missing_secrets))

if st.sidebar.button("💾 Commit mappings to GitHub"):
    if missing_secrets:
        st.sidebar.error("Cannot commit: fix missing secrets first.")
    else:
        try:
            resp = save_json_to_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH, st.session_state.get("mappings", {}), commit_msg, author_name="AFF Mapping App", author_email="aff-mapping@app.local", use_feature_branch=use_feature_branch)
            st.sidebar.success("Committed ✅")
            st.sidebar.caption(f"Commit: {resp['commit']['sha'][:7]}")
            try:
                wrote_branch = st.session_state["load_branch"] if not use_feature_branch else "aff-mapping-app"
                fetched_after = fetch_mappings_from_github(GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, wrote_branch)
                st.session_state["mappings"] = fetched_after
                st.session_state["local_mappings_modified"] = False
                st.sidebar.info(f"Reloaded {len(fetched_after)} mapping(s) from GitHub @{wrote_branch} after commit.")
            except Exception as e2:
                st.sidebar.warning(f"Post-commit reload failed: {e2}")
            try:
                append_jsonl_to_github(GH_OWNER, GH_REPO, AUDIT_LOG_PATH, GH_TOKEN, GH_BRANCH, {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "user":"streamlit-app","action":"commit", "count": len(st.session_state.get("mappings", {})), "path": MAPPINGS_PATH, "branch": wrote_branch}, commit_message="chore(app): append audit commit entry", use_feature_branch=use_feature_branch)
            except Exception as log_err:
                st.sidebar.warning(f"Audit log append failed (non-blocking): {log_err}")
        except Exception as e:
            st.sidebar.error(f"Commit failed: {e}")
            st.sidebar.info("If main is protected, enable feature branch and merge via PR.")

# --- EOF ---
