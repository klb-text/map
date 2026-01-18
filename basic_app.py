
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mozenda Basic Harvest (Minimal)", layout="wide")
st.title("Mozenda Basic Harvest (Minimal)")

# -----------------------------
# 1) Minimal demo data (fallback)
# -----------------------------
DEMO_ROWS = [
    {"row_key": "10001", "Year": "2025", "Make": "Audi",   "Model": "SQ5",  "Trim": "Premium Plus", "STYLE_ID": "778899"},
    {"row_key": "10002", "Year": "2025", "Make": "Audi",   "Model": "Q3",   "Trim": "Premium",      "STYLE_ID": "445566"},
    {"row_key": "10003", "Year": "2024", "Make": "Toyota", "Model": "RAV4", "Trim": "XLE",          "STYLE_ID": "112233"},
]
DEFAULT_COLS = ["Year", "Make", "Model", "Trim", "STYLE_ID"]  # Render order

# -----------------------------
# 2) Read URL params (optional)
# -----------------------------
params = st.experimental_get_query_params()
p_year  = (params.get("year",  [""])[0] or "").strip()
p_make  = (params.get("make",  [""])[0] or "").strip()
p_model = (params.get("model", [""])[0] or "").strip()
p_trim  = (params.get("trim",  [""])[0] or "").strip()
p_plain = (params.get("plain", ["0"])[0] == "1")  # if true, no CSS

# -----------------------------
# 3) Optional CSV upload (sidebar)
#    Expected columns (case-sensitive for simplicity):
#    row_key, Year, Make, Model, Trim, STYLE_ID
# -----------------------------
st.sidebar.header("Optional CSV Upload")
uploaded = st.sidebar.file_uploader(
    "Upload CSV with columns: row_key, Year, Make, Model, Trim, STYLE_ID",
    type=["csv"]
)
if uploaded:
    try:
        df_src = pd.read_csv(uploaded, dtype=str).fillna("")
        st.sidebar.success(f"Loaded {len(df_src)} rows from CSV.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        df_src = pd.DataFrame(DEMO_ROWS)
else:
    df_src = pd.DataFrame(DEMO_ROWS)

# Ensure required columns exist; if not, coerce/add
for col in ["row_key"] + DEFAULT_COLS:
    if col not in df_src.columns:
        df_src[col] = ""

# -----------------------------
# 4) Simple filters (URL params only)
# -----------------------------
def _canon(s: str) -> str:
    return (s or "").strip().lower()

mask = pd.Series([True] * len(df_src))
if p_year:
    mask &= (df_src["Year"].astype(str).str.strip() == p_year)
if p_make:
    mask &= (df_src["Make"].astype(str).str.strip().str.lower() == _canon(p_make))
if p_model:
    mask &= (df_src["Model"].astype(str).str.strip().str.lower() == _canon(p_model))
if p_trim:
    mask &= (df_src["Trim"].astype(str).str.strip().str.lower() == _canon(p_trim))

df = df_src[mask].reset_index(drop=True)

# -----------------------------
# 5) Render a plain semantic table
#    - table id: basic_harvest
#    - per-row attributes: data-row-key, data-year, data-make, data-model, data-trim
#    - per-cell: id="cell-<row_key>-<Col>" and data-col-key="<Col>"
# -----------------------------
def render_basic_table(
    df_in: pd.DataFrame,
    table_id: str = "basic_harvest",
    cols: list[str] = None,
    plain: bool = False
):
    cols = cols or DEFAULT_COLS

    css = ""
    if not plain:
        css = """
        <style>
          table#basic_harvest { border-collapse: collapse; width: 100%; font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Arial; }
          table#basic_harvest th, table#basic_harvest td { border: 1px solid #999; padding: 6px 8px; }
          table#basic_harvest thead th { background: #eee; }
          td:hover { outline: 2px solid #00a; }
        </style>
        """

    parts = []
    parts.append(f"{css}<table id='{table_id}' data-source='basic'>")

    # thead
    parts.append("<thead><tr>")
    for c in cols:
        parts.append(f"<th scope='col' data-col-key='{c}'>{c}</th>")
    parts.append("</tr></thead>")

    # tbody
    parts.append("<tbody>")
    for _, row in df_in.iterrows():
        rk    = str(row.get("row_key", "")).strip()
        year  = str(row.get("Year",  "")).strip()
        make  = str(row.get("Make",  "")).strip()
        model = str(row.get("Model", "")).strip()
        trim  = str(row.get("Trim",  "")).strip()

        tr_attrs = (
            f"data-row-key='{rk}' "
            f"data-year='{year}' "
            f"data-make='{make.lower()}' "
            f"data-model='{model.lower()}' "
            f"data-trim='{trim.lower()}'"
        )
        parts.append(f"<tr {tr_attrs}>")

        for c in cols:
            cell_id = f"cell-{rk}-{c}"
            val = str(row.get(c, "") or "")
            parts.append(f"<td id='{cell_id}' data-col-key='{c}'>{val}</td>")

        parts.append("</tr>")
    parts.append("</tbody></table>")

    st.markdown("\n".join(parts), unsafe_allow_html=True)

render_basic_table(df, cols=DEFAULT_COLS, plain=p_plain)

# Helper tips
st.caption("XPath: //table[@id='basic_harvest']/tbody/tr  |  //*[@id='cell-10001-STYLE_ID']  |  //table[@id='basic_harvest']/tbody/tr[@data-row-key='10002']/td[@data-col-key='Model']")

# --- EOF ---
