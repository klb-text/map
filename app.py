
# app.py
# AFF Vehicle Mapping – Streamlit + GitHub persistence + CADS search + row selection
# Updated: Trim-as-hint, Vehicle-only lookup, YMMT persistence

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

CADS_CODE_PREFS       = ["STYLE_ID", "AD_VEH_ID", "AD_MFGCODE"]
CADS_MODEL_CODE_PREFS = ["AD_MFGCODE", "MODEL_CODE", "ModelCode", "MFG_CODE", "MFGCODE"]

# ---------------------------------------------------------------------
# Canonicalization / tokens / year / trim
# ---------------------------------------------------------------------
def canon_text(val: str, for_trim: bool=False) -> str:
    s = (val or "").strip().lower()
    s = re.sub(r"^[\\s\\.,;:!]+", "", s)
    s = re.sub(r"[\\s\\.,;:!]+$", "", s)
    s = re.sub(r"\\s+", " ", s)
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
    tks = re.split(r"[^\\w]+", s)
    return [t for t in tks if t and len(t) >= min_len]

def _trim_tokens(s: str) -> Set[str]:
    return set(tokens(canon_text(s, True)))

def _extract_years_from_text(s: str) -> set:
    s = (s or "").strip().lower()
    years = set()
    for m in re.finditer(r"\\b(19[5-9]\\d|20[0-4]\\d|2050)\\b", s):
        years.add(int(m.group(0)))
    for m in re.finditer(r"\\bmy\\s*([0-9]{2})\\b", s):
        years.add(2000 + int(m.group(1)))
    if not years:
        for m in re.finditer(r"\\b([0-9]{2})\\b", s):
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

# ---------------------------------------------------------------------
# CADS loaders (CSV / Excel)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Detect model-like columns & build "effective model"
# ---------------------------------------------------------------------
MODEL_LIKE_REGEX  = re.compile(r"(?:^|_|\\s)(model(name)?|car\\s*line|carline|line|series)(?:$|_|\\s)", re.I)
SERIES_LIKE_REGEX = re.compile(r"(?:^|_|\\s)(series(name)?|sub(?:_|-)?model|body(?:_|-)?style|body|trim|grade|variant|description|modeltrim|name)(?:$|_|\\s)", re.I)

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

# ---------------------------------------------------------------------
# Per-make stopwords
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Resilient HTTP session (GitHub API helpers)
# ---------------------------------------------------------------------
from requests.adapters import HTTPAdapter, Retry
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","PUT","POST"])
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def gh_headers(token: str): return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
def gh_contents_url(owner, repo, path): return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
def gh_ref_heads(owner, repo, branch):  return f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"

# ---------------------------------------------------------------------
# Matching pickers
# ---------------------------------------------------------------------
def pick_best_mapping(
    mappings: Dict[str, Dict[str, str]],
    year: str, make: str, model: str, trim: str,
    trim_exact_only: bool = False,
    model_exact_when_full: bool = True,
) -> Optional[Tuple[str, Dict[str, str], float]]:
    """
    Generic single-best mapping picker using Make exact, Year token-aware, Trim gate (exact / subset),
    and Model similarity (with damping when user input model is multi-word and not exactly equal).
    """
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
    return candidates[0]

def pick_mapping_by_vehicle(
    mappings: Dict[str, Dict[str, str]],
    vehicle: str
) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Direct lookup by 'vehicle' string (canonicalized).
    Returns (key, mapping) if found; otherwise None.
    """
    cv = canon_text(vehicle)
    if not cv:
        return None
    for k, v in mappings.items():
        if canon_text(v.get("vehicle", "")) == cv:
            return (k, v)
    return None

# ---------------------------------------------------------------------
# CADS filtering — Tiered model masks + Trim-as-hint enhancement
# ---------------------------------------------------------------------
def _tiered_model_mask(
    eff: pd.Series,
    md: str,
    discriminant: List[str]
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Return masks for the 4 tiers:
      and_mask: token-AND on discriminants
      or_mask:  token-OR  on discriminants
      ns_mask:  contains on non-stopword text (joined discriminants)
      full_mask: contains on full model text
    """
    if not md:
        true_mask = pd.Series([True]*len(eff), index=eff.index)
        return true_mask, true_mask, true_mask, true_mask

    # Token AND / OR
    and_mask  = eff.apply(lambda s: all(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)
    or_mask   = eff.apply(lambda s: any(t in s for t in discriminant)) if discriminant else eff.str.contains(md, na=False)

    # Non-stopword joined text contains
    ns_text   = " ".join(discriminant).strip()
    ns_mask   = eff.str.contains(ns_text, na=False) if ns_text else eff.str.contains(md, na=False)

    # Full text contains
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
    trim_as_hint: bool = False,  # NEW: hint mode for Trim
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Generic CADS filter using effective model + per-make stopwords, with tiered matching.
    If trim_as_hint=True, Trim does not gate results; instead rows are annotated with
    __trim_match_type__ and __trim_match_score__ and then ranked by score.
    """
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

    # Make slice & stopwords
    if make_col and mk:
        s_make = df2[make_col].astype(str).str.lower()
        masks.append(s_make == mk)
        df_make_slice = df2[s_make == mk]
    else:
        df_make_slice = df2

    eff = df2["__effective_model__"]

    # Build discriminant tokens (user tokens minus per-make stopwords)
    user_tokens = tokens(md, min_len=token_min_len)
    make_stopwords = compute_per_make_stopwords(df_make_slice, stopword_threshold, token_min_len)
    discriminant = [t for t in user_tokens if t not in make_stopwords]

    # Model tiered masks (start with AND)
    and_mask, or_mask, ns_mask, full_mask = _tiered_model_mask(eff, md, discriminant)
    if md:
        masks.append(and_mask)

    # Trim gate (either filter or hint-only)
    if trim_col and tr:
        s_trim = df2[trim_col].astype(str)
        m_exact  = s_trim.str.lower() == tr
        m_subset = s_trim.apply(lambda x: _trim_tokens(tr).issubset(_trim_tokens(x)))

        if trim_as_hint:
            # Annotate (no gating)
            df2["__trim_match_type__"] = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[0])
            df2["__trim_match_score__"] = s_trim.apply(lambda x: trim_match_type_and_score(x, tr)[1])
        else:
            if trim_exact_only:
                masks.append(m_exact)
            else:
                masks.append(m_exact | m_subset)
    else:
        # Ensure columns exist even if no trim or column missing
        df2["__trim_match_type__"] = "none"
        df2["__trim_match_score__"] = 0.0

    # Year gate
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

    # Fallback tiers if empty (only when model is provided)
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
            # Score desc, effective_model asc
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
    trim_as_hint: bool = False,   # NEW
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Try matching CADS rows for a given mapping using:
      1) Code union (STYLE_ID/AD_VEH_ID/etc.)
      2) Model Code union (AD_MFGCODE/MODEL_CODE/etc.)
      3) Generic effective-model matching with tiered fallback (supports trim_as_hint)
    """
    df2 = _strip_object_columns(df.copy())
    df2, used_model_cols, used_series_cols = add_effective_model_column(df2, override_cols=effective_model_cols_override)

    # Code union (first)
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

    # Model Code union (second)
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

    # Generic fallback using effective model + stopwords (with trim_as_hint support)
    res, diag = filter_cads_generic(
        df2,
        mapping.get("year",""), mapping.get("make",""), mapping.get("model",""), mapping.get("trim",""),
        exact_model_when_full=exact_model_when_full,
        trim_exact_only=trim_exact_only,
        strict_and=strict_and,
        stopword_threshold=stopword_threshold,
        token_min_len=token_min_len,
        effective_model_cols_override=effective_model_cols_override,
        trim_as_hint=trim_as_hint,  # NEW
    )
    diag.update({"used_model_cols": used_model_cols, "used_series_cols": used_series_cols})
    return res, diag

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("AFF Vehicle Mapping")

# --------------------- Sidebar: CADS Settings ------------------------
st.sidebar.subheader("CADS Settings")
CADS_PATH = st.sidebar.text_input("CADS path in repo", value=CADS_PATH)
CADS_IS_EXCEL = st.sidebar.checkbox("CADS is Excel (.xlsx)", value=CADS_IS_EXCEL)
CADS_SHEET_NAME = st.sidebar.text_input("Excel sheet name/index", value=CADS_SHEET_NAME_DEFAULT)
cads_upload = st.sidebar.file_uploader("Upload CADS CSV/XLSX (local test)", type=["csv","xlsx"])

# --------------------- Sidebar: Actions & Matching -------------------
st.sidebar.header("Actions")
if st.sidebar.button("🔄 Reload from GitHub"):
    try:
        r_load = _session.get(
            gh_contents_url(GH_OWNER, GH_REPO, MAPPINGS_PATH),
            headers=gh_headers(GH_TOKEN), params={"ref": GH_BRANCH}, timeout=15
        )
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

# NEW: Trim-as-hint toggle
TRIM_AS_HINT = st.sidebar.checkbox(
    "Use Trim as hint (do not filter)",
    value=True,
    help="If enabled, Trim will not restrict results; it will be used to rank/sort matched vehicles."
)

st.sidebar.subheader("Matching Controls")
TRIM_EXACT_ONLY = st.sidebar.checkbox("Trim must be exact (no token-subset)", value=False)
MODEL_EXACT_WHEN_FULL = st.sidebar.checkbox("Model exact when input is multi-word", value=False)
STRICT_AND = st.sidebar.checkbox("Require strict AND across provided filters", value=True)
STOPWORD_THRESHOLD = st.sidebar.slider("Per-make stopword threshold", 0.1, 0.9, 0.60, 0.05)
TOKEN_MIN_LEN = st.sidebar.slider("Token minimum length", 1, 5, 2, 1)

st.sidebar.subheader("Effective Model (override)")
EFFECTIVE_MODEL_COLS_OVERRIDE = st.sidebar.text_input(
    "Comma-separated CADS columns to use (optional)",
    value="AD_MODEL, MODEL_NAME, STYLE_NAME, AD_SERIES"
)
OVERRIDE_COLS = [c.strip() for c in EFFECTIVE_MODEL_COLS_OVERRIDE.split(",") if c.strip()] or None

TABLE_HEIGHT = st.sidebar.slider("Results table height (px)", 400, 1200, 700, 50)

# --------------------- Mapping Inputs --------------------------------
st.subheader("Edit / Add Mapping")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: year = st.text_input("Year", key="year_input", placeholder="e.g., 2025")
with c2: make = st.text_input("Make", key="make_input", placeholder="e.g., Audi")
with c3: model = st.text_input("Model", key="model_input", placeholder="e.g., Q3")
with c4: trim = st.text_input("Trim", key="trim_input", placeholder="e.g., S line 45 TFSI quattro Premium")
with c5: vehicle = st.text_input("Vehicle (alt)", key="vehicle_input", placeholder="Optional")
with c6: mapped_code = st.text_input("Mapped Code", key="code_input", placeholder="Optional (STYLE_ID/AD_VEH_ID/etc.)")
model_code_input = st.text_input("Model Code (optional)", key="model_code_input", placeholder="AD_MFGCODE/MODEL_CODE/etc.")

# --------------------- Existing Mapping (vehicle-first) --------------
st.subheader("Existing Mapping (for current inputs)")
existing_rows = []
vehicle_first = False

if "mappings" not in st.session_state:
    # Load once if not present
    try:
        r_load = _session.get(
            gh_contents_url(GH_OWNER, GH_REPO, MAPPINGS_PATH),
            headers=gh_headers(GH_TOKEN), params={"ref": GH_BRANCH}, timeout=15
        )
        if r_load.status_code == 200:
            decoded = base64.b64decode(r_load.json()["content"]).decode("utf-8")
            st.session_state.mappings = json.loads(decoded)
        else:
            st.session_state.mappings = {}
    except Exception:
        st.session_state.mappings = {}

# Vehicle-only lookup first
if canon_text(vehicle) and not (canon_text(year) or canon_text(make) or canon_text(model) or canon_text(trim, True)):
    vm = pick_mapping_by_vehicle(st.session_state.get("mappings", {}), vehicle)
    vehicle_first = True
    if vm:
        k, v = vm
        existing_rows.append({
            "Match Level": "by_vehicle",
            "Score": 1.0,
            "Key": k, "Year": v.get("year",""), "Make": v.get("make",""),
            "Model": v.get("model",""), "Trim": v.get("trim",""),
            "Vehicle": v.get("vehicle",""), "Code": v.get("code",""),
            "Model Code": v.get("model_code",""),
        })

# Otherwise YMMT-based best mapping
if not existing_rows:
    best = pick_best_mapping(
        st.session_state.get("mappings", {}), year, make, model, trim,
        trim_exact_only=TRIM_EXACT_ONLY,
        model_exact_when_full=MODEL_EXACT_WHEN_FULL,
    )
    if best:
        k, v, score = best
        existing_rows.append({
            "Match Level": "generic_best_trim_model_year",
            "Score": round(score,3),
            "Key": k, "Year": v.get("year",""), "Make": v.get("make",""),
            "Model": v.get("model",""), "Trim": v.get("trim",""),
            "Vehicle": v.get("vehicle",""), "Code": v.get("code",""),
            "Model Code": v.get("model_code",""),
        })

if existing_rows:
    st.success("Existing mapping found." if vehicle_first else "Already mapped: 1 match.")
    st.dataframe(pd.DataFrame(existing_rows), use_container_width=True)
else:
    st.info("No existing mapping detected for current inputs.")

# --------------------- CADS Search Buttons --------------------------
b1, b2, b3, b4 = st.columns(4)

# (1) Search CADS using the mapped vehicle (updated handler)
with b2:
    if st.button("🔎 Search CADS (mapped vehicle)"):
        try:
            # --- Load CADS ---
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
                    df_cads = load_cads_from_github_excel(
                        GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN,
                        ref=GH_BRANCH, sheet_name=sheet_arg
                    )
                else:
                    df_cads = load_cads_from_github_csv(
                        GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH
                    )
            df_cads = _strip_object_columns(df_cads)

            # --- Determine mapping to use (vehicle-first, else YMMT best) ---
            mapping_to_use = None
            vm = pick_mapping_by_vehicle(st.session_state.get("mappings", {}), vehicle)
            if canon_text(vehicle) and vm:
                mapping_to_use = vm[1]
            else:
                best = pick_best_mapping(
                    st.session_state.get("mappings", {}),
                    year, make, model, trim,
                    trim_exact_only=TRIM_EXACT_ONLY,
                    model_exact_when_full=MODEL_EXACT_WHEN_FULL,
                )
                if best:
                    mapping_to_use = best[1]

            if not mapping_to_use:
                st.info("No mapped vehicle detected for current inputs. Try 'Search CADS (use current inputs)'.")
            else:
                # --- Match using mapping (Code → Model Code → generic, with Trim-as-hint ranking) ---
                df_match, diag = match_cads_rows_for_mapping(
                    df_cads, mapping_to_use,
                    exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                    trim_exact_only=TRIM_EXACT_ONLY,
                    strict_and=STRICT_AND,
                    stopword_threshold=STOPWORD_THRESHOLD,
                    token_min_len=TOKEN_MIN_LEN,
                    effective_model_cols_override=OVERRIDE_COLS,
                    trim_as_hint=TRIM_AS_HINT,
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
        except FileNotFoundError as fnf:
            st.error(str(fnf))
        except Exception as e:
            st.error(f"CADS search failed: {e}")

# (2) Search CADS using current inputs (updated handler)
with b3:
    if st.button("🔎 Search CADS (use current inputs)"):
        try:
            # --- Load CADS ---
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
                    df_cads = load_cads_from_github_excel(
                        GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN,
                        ref=GH_BRANCH, sheet_name=sheet_arg
                    )
                else:
                    df_cads = load_cads_from_github_csv(
                        GH_OWNER, GH_REPO, CADS_PATH, GH_TOKEN, ref=GH_BRANCH
                    )
            df_cads = _strip_object_columns(df_cads)

            # --- Generic filter (Trim-as-hint support) ---
            results, diag = filter_cads_generic(
                df_cads,
                year, make, model, trim,
                exact_model_when_full=MODEL_EXACT_WHEN_FULL,
                trim_exact_only=TRIM_EXACT_ONLY,
                strict_and=STRICT_AND,
                stopword_threshold=STOPWORD_THRESHOLD,
                token_min_len=TOKEN_MIN_LEN,
                effective_model_cols_override=OVERRIDE_COLS,
                trim_as_hint=TRIM_AS_HINT,
            )
            st.session_state["last_diag_inputs"] = diag

            if len(results) == 0:
                st.warning("No CADS rows matched inputs. Try adjusting stopword threshold (e.g., 0.65), turning OFF 'Model exact when full', or using Trim-as-hint.")
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

# --------------------- Results Table: Mapped Vehicle -----------------
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

    front_cols = [c for c in ["Select","__effective_model__","__trim_match_type__","__trim_match_score__"] if c in df_show.columns]
    col_order = front_cols + [c for c in df_show.columns if c not in front_cols]

    csel1, csel2 = st.columns(2)
    with csel1:
        if st.button("✅ Select All (mapped)"):
            df_tmp = df_show.copy(); df_tmp["Select"] = True
            st.session_state["results_df_mapped"] = df_tmp; df_show = df_tmp
    with csel2:
        if st.button("🧹 Clear Selection (mapped)"):
            df_tmp = df_show.copy(); df_tmp["Select"] = False
            st.session_state["results_df_mapped"] = df_tmp; df_show = df_tmp

    edited = st.data_editor(
        df_show, key="results_editor_mapped", use_container_width=True,
        num_rows="dynamic", column_order=col_order, height=TABLE_HEIGHT
    )
    st.session_state["results_df_mapped"] = edited
    selected_rows = edited[edited["Select"] == True]
    st.caption(f"(Mapped) Selected {len(selected_rows)} vehicle(s).")

    if st.button("➕ Add selected (mapped) to mappings"):
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

                ymmt = " ".join([v for v in [yv, mkv, mdv, trv] if (v or "").strip()])

                st.session_state.mappings[key] = {
                    "year": yv, "make": mkv, "model": mdv, "trim": trv,
                    "vehicle": vhv, "code": code_val, "model_code": model_code_val,
                    "ymmt": ymmt,
                }
                added += 1
            st.success(f"[Mapped] Added/updated {added} mapping(s).")

# --------------------- Results Table: Direct Input -------------------
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

    front_cols = [c for c in ["Select","__effective_model__","__trim_match_type__","__trim_match_score__"] if c in df_show.columns]
    col_order = front_cols + [c for c in df_show.columns if c not in front_cols]

    csel1, csel2 = st.columns(2)
    with csel1:
        if st.button("✅ Select All (inputs)"):
            df_tmp = df_show.copy(); df_tmp["Select"] = True
            st.session_state["results_df_inputs"] = df_tmp; df_show = df_tmp
    with csel2:
        if st.button("🧹 Clear Selection (inputs)"):
            df_tmp = df_show.copy(); df_tmp["Select"] = False
            st.session_state["results_df_inputs"] = df_tmp; df_show = df_tmp

    edited = st.data_editor(
        df_show, key="results_editor_inputs", use_container_width=True,
        num_rows="dynamic", column_order=col_order, height=TABLE_HEIGHT
    )
    st.session_state["results_df_inputs"] = edited
    selected_rows = edited[edited["Select"] == True]
    st.caption(f"(Inputs) Selected {len(selected_rows)} vehicle(s).")

    if st.button("➕ Add selected (inputs) to mappings"):
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

                ymmt = " ".join([v for v in [yv, mkv, mdv, trv] if (v or "").strip()])

                st.session_state.mappings[key] = {
                    "year": yv, "make": mkv, "model": mdv, "trim": trv,
                    "vehicle": vhv, "code": code_val, "model_code": model_code_val,
                    "ymmt": ymmt,
                }
                added += 1
            st.success(f"[Inputs] Added/updated {added} mapping(s).")

# --------------------- Current Mappings ------------------------------
st.subheader("Current Mappings (session)")
if st.session_state.get("mappings"):
    rows = []
    for k, v in st.session_state.mappings.items():
        rows.append({
            "Key": k, "Year": v.get("year",""), "Make": v.get("make",""),
            "Model": v.get("model",""), "Trim": v.get("trim",""),
            "Vehicle": v.get("vehicle",""), "Code": v.get("code",""),
            "Model Code": v.get("model_code",""), "YMMT": v.get("ymmt",""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No mappings yet. Add one above or select CADS rows to add mappings.")

# --------------------- Commit to GitHub ------------------------------
_miss = []
if not GH_TOKEN:  _miss.append("github.token")
if not GH_OWNER:  _miss.append("github.owner")
if not GH_REPO:   _miss.append("github.repo")
if not GH_BRANCH: _miss.append("github.branch")
if _miss: st.sidebar.warning("Missing secrets: " + ", ".join(_miss))

if st.sidebar.button("💾 Commit mappings to GitHub"):
    if _miss:
        st.sidebar.error("Cannot commit: fix missing secrets first.")
    else:
        try:
            resp = save_json_to_github(
                GH_OWNER, GH_REPO, MAPPINGS_PATH, GH_TOKEN, GH_BRANCH,
                st.session_state.mappings, commit_msg,
                author_name="AFF Mapping App", author_email="aff-mapping@app.local",
                use_feature_branch=use_feature_branch
            )
            st.sidebar.success("Committed ✅")
            st.sidebar.caption(f"Commit: {resp['commit']['sha'][:7]}")
            try:
                append_jsonl_to_github(
                    GH_OWNER, GH_REPO, AUDIT_LOG_PATH, GH_TOKEN, GH_BRANCH,
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "user":"streamlit-app","action":"commit",
                        "count": len(st.session_state.mappings),"path": MAPPINGS_PATH,
                        "branch": GH_BRANCH if not use_feature_branch else "aff-mapping-app"
                    },
                    commit_message="chore(app): append audit commit entry",
                    use_feature_branch=use_feature_branch
                )
            except Exception as log_err:
                st.sidebar.warning(f"Audit log append failed (non-blocking): {log_err}")
        except Exception as e:
            st.sidebar.error(f"Commit failed: {e}")
            st.sidebar.info("If main is protected, enable feature branch and merge via PR.")

# --- EOF ---
