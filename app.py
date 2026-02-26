import os
import re
import json
import time
from io import BytesIO

import certifi
import httpx
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

APP_TITLE = "Doc Checker PoC — Shipment & Cargo Compare (4 docs)"
MODEL = "gpt-4.1-mini"
MAX_CHARS_TO_SEND = 12000

DOCS = [
    ("BL_SWB", "BL / SWB"),
    ("PACKING_LIST", "Packing List"),
    ("PROFORMA_INVOICE", "Proforma Invoice"),
    ("CUSTOMS_BC16", "Customs BC 1.6"),
]

SHIPMENT_ITEMS = [
    ("document_number", "Document number"),
    ("shipper", "Shipper"),
    ("seller", "Seller"),
    ("plb_operator", "PLB operator"),
    ("cargo_owner", "Cargo owner"),
    ("consignee", "Consignee"),
    ("notify", "Notify party"),
    ("pol", "POL (Port of loading)"),
    ("pod", "POD (Port of discharge)"),
    ("vessel_voy", "Vessel / Voy No."),
    ("container_20", "Container type / amount (20')"),
    ("container_40", "Container type / amount (40')"),
]

CARGO_SUBCOLS = ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)", "Amount (USD)", "Code"]

DOC_COLORS = {
    "BL / SWB": "#e8f2ff",
    "Packing List": "#fff1df",
    "Proforma Invoice": "#f3e8ff",
    "Customs BC 1.6": "#e8ffe8",
}
MISMATCH_RED = "#ffcccc"


def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()


def get_api_key() -> str:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY", "")


def call_openai_once(model: str, prompt: str) -> str:
    with httpx.Client(
        verify=certifi.where(),
        timeout=httpx.Timeout(connect=20.0, read=90.0, write=90.0, pool=90.0),
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
    ) as http_client:
        client = OpenAI(api_key=get_api_key(), http_client=http_client)
        r = client.responses.create(model=model, input=prompt)
        return (r.output_text or "").strip()


def ping_openai() -> str:
    return call_openai_once(MODEL, "Reply with exactly: PONG")


def norm_spaces(v: str) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v)
    return v


def city_only(port: str) -> str:
    s = norm_spaces(port)
    if not s:
        return ""
    s = re.split(r"[,(/]| EX | KRPUS| IDTPP| IDJKT", s, maxsplit=1)[0].strip()
    return s


def norm_pod_equivalence(pod: str) -> str:
    s = norm_spaces(pod).lower()
    if not s:
        return ""
    if "jakarta" in s:
        return "tanjung priok"
    if "tanjung priok" in s or "priok" in s:
        return "tanjung priok"
    return city_only(pod).lower()


def hs6(hs: str) -> str:
    s = re.sub(r"\D", "", (hs or ""))
    return s[:6] if s else ""


def to_mt(value: str) -> str:
    s = norm_spaces(value)
    if not s:
        return ""
    s_low = s.lower()
    num = re.findall(r"[-+]?\d[\d,]*\.?\d*", s_low)
    if not num:
        return s
    n = num[0].replace(",", "")
    try:
        x = float(n)
    except Exception:
        return s

    if "kg" in s_low:
        x = x / 1000.0
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if "mt" in s_low or "ton" in s_low:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{x:.3f}".rstrip("0").rstrip(".")


def safe_parse_json(raw: str):
    if not raw or not raw.strip():
        return None, "Empty response"
    s = raw.strip()

    if s.startswith("```"):
        s = s.strip().strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]

    try:
        return json.loads(s), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def build_prompt(doc_label: str, text: str) -> str:
    ship_keys = [k for k, _ in SHIPMENT_ITEMS]

    # 핵심: Indonesian source(BC 1.6 포함) -> output values must be in English
    rules_translation = (
        "Language rule:\n"
        "- If the source text is Indonesian (e.g., Customs BC 1.6), translate/normalize extracted values into English.\n"
        "- Examples: 'Pelabuhan Muat' -> port of loading value; 'Berat Kotor/Bersih' -> Gross/Net weight; "
        "'Nilai CIF' -> CIF value; 'Pemberitahuan Pabean' -> customs declaration.\n"
        "- Keep company names, invoice/BL numbers as-is.\n"
    )

    prompt = f"""
You extract structured data from shipping/customs documents.

Return ONLY valid JSON (no markdown, no explanation).

Schema:
{{
  "shipment": {{
    {", ".join([f'"{k}": ""' for k in ship_keys])}
  }},
  "cargo": {{
    "hs_code": "",
    "cargo_name": "",
    "totals": {{
      "qty": "",
      "gross_wt": "",
      "net_wt": "",
      "amount_usd": ""
    }},
    "line_items": [
      {{
        "brand": "",
        "packing": "",
        "qty": "",
        "gross_wt": "",
        "net_wt": "",
        "amount_usd": "",
        "code": ""
      }}
    ]
  }}
}}

General rules:
- If not found, use "".
- For weights: include unit if present (KG or MT). Example: "12255 KG" or "12.255 MT".
- HS code: keep as written (we normalize later).
- Keep values short.
{rules_translation}
DOCUMENT TYPE: {doc_label}
TEXT:
-----
{text[:MAX_CHARS_TO_SEND]}
-----
"""
    return prompt


def extract_doc(doc_label: str, text: str) -> dict:
    prompt = build_prompt(doc_label, text)
    last_err = None
    for attempt in range(1, 4):
        try:
            raw = call_openai_once(MODEL, prompt)
            obj, err = safe_parse_json(raw)
            if err:
                raise ValueError(err)
            if not isinstance(obj, dict):
                raise ValueError("Parsed JSON is not a dict")
            return obj
        except Exception as e:
            last_err = e
            time.sleep(2 ** (attempt - 1))
    return {"_error": f"{type(last_err).__name__}: {str(last_err)[:250]}"}


def shipment_table(extracted_by_doc: dict) -> pd.DataFrame:
    colnames = [label for _, label in DOCS]
    rows = [label for _, label in SHIPMENT_ITEMS]
    df = pd.DataFrame("", index=rows, columns=colnames)

    for doc_key, doc_label in DOCS:
        payload = extracted_by_doc.get(doc_key, {})
        ship = payload.get("shipment", {}) if isinstance(payload, dict) else {}
        for k, row_label in SHIPMENT_ITEMS:
            df.loc[row_label, doc_label] = norm_spaces(str(ship.get(k, "")))

    df.loc["POL (Port of loading)"] = df.loc["POL (Port of loading)"].apply(city_only)
    df.loc["POD (Port of discharge)"] = df.loc["POD (Port of discharge)"].apply(city_only)
    return df


def cargo_table(extracted_by_doc: dict) -> pd.DataFrame:
    max_lines = 6
    rows = ["Cargo name", "HS code / pos tarif"] + [f"Line {i}" for i in range(1, max_lines + 1)] + ["Total", "Total check"]

    cols = pd.MultiIndex.from_product([[label for _, label in DOCS], CARGO_SUBCOLS])
    df = pd.DataFrame("", index=rows, columns=cols)

    for doc_key, doc_label in DOCS:
        payload = extracted_by_doc.get(doc_key, {})
        cargo = payload.get("cargo", {}) if isinstance(payload, dict) else {}

        df.loc["Cargo name", (doc_label, "Brand")] = norm_spaces(str(cargo.get("cargo_name", "")))
        df.loc["HS code / pos tarif", (doc_label, "Brand")] = norm_spaces(str(cargo.get("hs_code", "")))

        line_items = cargo.get("line_items", []) if isinstance(cargo, dict) else []
        if not isinstance(line_items, list):
            line_items = []

        for i in range(1, max_lines + 1):
            row = f"Line {i}"
            if i <= len(line_items):
                li = line_items[i - 1] if isinstance(line_items[i - 1], dict) else {}
                df.loc[row, (doc_label, "Brand")] = norm_spaces(str(li.get("brand", "")))
                df.loc[row, (doc_label, "Packing")] = norm_spaces(str(li.get("packing", "")))
                df.loc[row, (doc_label, "QTY")] = norm_spaces(str(li.get("qty", "")))
                df.loc[row, (doc_label, "Gross WT (MT)")] = to_mt(str(li.get("gross_wt", "")))
                df.loc[row, (doc_label, "Net WT (MT)")] = to_mt(str(li.get("net_wt", "")))
                df.loc[row, (doc_label, "Amount (USD)")] = norm_spaces(str(li.get("amount_usd", "")))
                df.loc[row, (doc_label, "Code")] = norm_spaces(str(li.get("code", "")))

        totals = cargo.get("totals", {}) if isinstance(cargo, dict) else {}
        df.loc["Total", (doc_label, "Packing")] = norm_spaces(str(totals.get("qty", "")))
        df.loc["Total", (doc_label, "Gross WT (MT)")] = to_mt(str(totals.get("gross_wt", "")))
        df.loc["Total", (doc_label, "Net WT (MT)")] = to_mt(str(totals.get("net_wt", "")))
        df.loc["Total", (doc_label, "Amount (USD)")] = norm_spaces(str(totals.get("amount_usd", "")))

    return df


def style_shipment(df: pd.DataFrame) :

    for col in df.columns:
        sty = sty.applymap(lambda _: f"background-color: {DOC_COLORS.get(col, '')};", subset=pd.IndexSlice[:, [col]])

    def row_mismatch(row_label: str, values: list[str]) -> bool:
        vals = [norm_spaces(v) for v in values if norm_spaces(v)]
        if not vals:
            return False
        if row_label == "Document number":
            return False
        if row_label == "POL (Port of loading)":
            vals2 = [city_only(v).lower() for v in vals if city_only(v)]
            return len(set(vals2)) >= 2
        if row_label == "POD (Port of discharge)":
            vals2 = [norm_pod_equivalence(v) for v in vals if v]
            return len(set(vals2)) >= 2
        vals2 = [norm_spaces(v).lower() for v in vals]
        return len(set(vals2)) >= 2

    def apply_row(s: pd.Series):
        mismatch = row_mismatch(s.name, list(s.values))
        if not mismatch:
            return [""] * len(s)
        out = []
        for v in s.values:
            out.append(f"background-color: {MISMATCH_RED};" if norm_spaces(str(v)) else "")
        return out

    return sty.apply(apply_row, axis=1)


def style_cargo(df: pd.DataFrame) :

    for doc_label in [label for _, label in DOCS]:
        subset = pd.IndexSlice[:, pd.IndexSlice[doc_label, :]]
        sty = sty.applymap(lambda _: f"background-color: {DOC_COLORS.get(doc_label, '')};", subset=subset)

    # HS 6-digit mismatch highlight
    hs_vals = []
    for doc_label in [label for _, label in DOCS]:
        v = df.loc["HS code / pos tarif", (doc_label, "Brand")]
        h = hs6(str(v))
        if h:
            hs_vals.append(h)

    hs_mismatch = len(set(hs_vals)) >= 2

    if hs_mismatch:
        for doc_label in [label for _, label in DOCS]:
            subset = pd.IndexSlice[["HS code / pos tarif"], pd.IndexSlice[doc_label, ["Brand"]]]
            sty = sty.applymap(lambda _: f"background-color: {MISMATCH_RED};", subset=subset)

    # Total check (line sum vs total)
    def parse_float(s: str):
        s = norm_spaces(s).replace(",", "")
        if not s:
            return None
        m = re.findall(r"[-+]?\d*\.?\d+", s)
        if not m:
            return None
        try:
            return float(m[0])
        except Exception:
            return None

    def close(a, b, tol=0.001):
        if a is None or b is None:
            return None
        return abs(a - b) <= tol

    for doc_label in [label for _, label in DOCS]:
        qty_sum = gross_sum = net_sum = amt_sum = 0.0
        has_any = False
        for i in range(1, 7):
            row = f"Line {i}"
            q = parse_float(str(df.loc[row, (doc_label, "QTY")]))
            g = parse_float(str(df.loc[row, (doc_label, "Gross WT (MT)")]))
            n = parse_float(str(df.loc[row, (doc_label, "Net WT (MT)")]))
            a = parse_float(str(df.loc[row, (doc_label, "Amount (USD)")]))
            if any(x is not None for x in [q, g, n, a]):
                has_any = True
            if q is not None: qty_sum += q
            if g is not None: gross_sum += g
            if n is not None: net_sum += n
            if a is not None: amt_sum += a

        tq = parse_float(str(df.loc["Total", (doc_label, "Packing")]))
        tg = parse_float(str(df.loc["Total", (doc_label, "Gross WT (MT)")]))
        tn = parse_float(str(df.loc["Total", (doc_label, "Net WT (MT)")]))
        ta = parse_float(str(df.loc["Total", (doc_label, "Amount (USD)")]))

        if not has_any:
            continue

        checks = []
        for a, b in [(qty_sum, tq), (gross_sum, tg), (net_sum, tn), (amt_sum, ta)]:
            c = close(a, b)
            if c is not None:
                checks.append(c)

        ok = all(checks) if checks else True
        df.loc["Total check", (doc_label, "Brand")] = "ok" if ok else "no"

        subset = pd.IndexSlice[["Total check"], pd.IndexSlice[doc_label, ["Brand"]]]
        if ok:
            sty = sty.applymap(lambda _: "font-weight: 700;", subset=subset)
        else:
            sty = sty.applymap(lambda _: f"background-color: {MISMATCH_RED}; font-weight: 700;", subset=subset)

    return sty


def export_excel(sh_df: pd.DataFrame, cg_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sh_df.to_excel(writer, sheet_name="Shipment Information")
        cg_df.to_excel(writer, sheet_name="Cargo Information")
    return output.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

top_left, top_right = st.columns([3.5, 1.5], gap="small")
with top_left:
    st.caption("Upload is minimized at top-right. Tables are centered and full-width below.")

with top_right:
    with st.expander("📤 Upload (4 docs)", expanded=False):
        uploaded = {}
        for doc_key, doc_label in DOCS:
            uploaded[doc_key] = st.file_uploader(doc_label, type=["pdf"], key=f"u_{doc_key}")
        colA, colB = st.columns([1, 1])
        with colA:
            run = st.button("Extract & Compare", use_container_width=True)
        with colB:
            mismatch_only = st.checkbox("Mismatch only", value=False)

with st.expander("Debug / Status", expanded=False):
    key = get_api_key()
    st.write("API key loaded:", "✅ Yes" if key.startswith("sk-") else "❌ No (OPENAI_API_KEY missing)")
    if st.button("Ping OpenAI (should return PONG)"):
        try:
            st.success(f"Ping result: {ping_openai()}")
        except Exception as e:
            st.error(f"Ping failed: {type(e).__name__}: {str(e)[:300]}")

if not run:
    st.info("Upload PDFs in the top-right panel, then click **Extract & Compare**.")
    st.stop()

if not get_api_key().startswith("sk-"):
    st.error("OPENAI_API_KEY is missing. Set it in Streamlit Cloud Secrets.")
    st.stop()

extracted_by_doc = {}
errors = []

with st.spinner("Extracting PDFs and calling OpenAI..."):
    for doc_key, doc_label in DOCS:
        f = uploaded.get(doc_key)
        if not f:
            extracted_by_doc[doc_key] = {"_error": "Missing file"}
            continue

        try:
            text = pdf_to_text(f)
            if not text.strip():
                extracted_by_doc[doc_key] = {"_error": "No text extracted (scanned PDF?)"}
                continue
        except Exception as e:
            extracted_by_doc[doc_key] = {"_error": f"PDF read error: {type(e).__name__}: {str(e)[:150]}"}
            continue

        obj = extract_doc(doc_label, text)
        extracted_by_doc[doc_key] = obj
        if "_error" in obj:
            errors.append(f"[{doc_label}] {obj['_error']}")

ship_df = shipment_table(extracted_by_doc)
cargo_df = cargo_table(extracted_by_doc)

# KPI row
hs_vals = []
for doc_label in [label for _, label in DOCS]:
    hs_vals.append(hs6(str(cargo_df.loc["HS code / pos tarif", (doc_label, "Brand")])))
hs_mismatch = len(set([h for h in hs_vals if h])) >= 2

k1, k2, k3, k4 = st.columns(4)
# shipment mismatch count
ship_mis = 0
for idx in ship_df.index:
    vals = [norm_spaces(v) for v in ship_df.loc[idx].values if norm_spaces(v)]
    if not vals:
        continue
    if idx == "Document number":
        continue
    if idx == "POL (Port of loading)":
        if len(set([city_only(v).lower() for v in vals if city_only(v)])) >= 2:
            ship_mis += 1
    elif idx == "POD (Port of discharge)":
        if len(set([norm_pod_equivalence(v) for v in vals if v])) >= 2:
            ship_mis += 1
    else:
        if len(set([v.lower() for v in vals])) >= 2:
            ship_mis += 1

k1.metric("Shipment mismatches", ship_mis)
k2.metric("HS mismatch (6-digit)", "YES" if hs_mismatch else "NO")
k3.metric("Docs uploaded", sum(1 for k, _ in DOCS if uploaded.get(k)))
k4.metric("Extraction errors", len(errors))

if errors:
    with st.expander("⚠️ Extraction errors (details)", expanded=False):
        for e in errors:
            st.error(e)

st.subheader("Shipment Information")
ship_view = ship_df.copy()
if mismatch_only:
    keep = []
    for idx in ship_view.index:
        if idx == "Document number":
            continue
        vals = [norm_spaces(v) for v in ship_view.loc[idx].values if norm_spaces(v)]
        if not vals:
            continue
        if idx == "POL (Port of loading)":
            if len(set([city_only(v).lower() for v in vals if city_only(v)])) >= 2:
                keep.append(idx)
        elif idx == "POD (Port of discharge)":
            if len(set([norm_pod_equivalence(v) for v in vals if v])) >= 2:
                keep.append(idx)
        else:
            if len(set([v.lower() for v in vals])) >= 2:
                keep.append(idx)
    ship_view = ship_view.loc[keep] if keep else ship_view.iloc[0:0]

st.dataframe(style_shipment(ship_view), use_container_width=True)

st.subheader("Cargo Information")
cargo_view = cargo_df.copy()
if mismatch_only:
    rows_keep = ["Cargo name", "HS code / pos tarif", "Total", "Total check"]
    for i in range(1, 7):
        row = f"Line {i}"
        has_any = False
        for doc_label in [label for _, label in DOCS]:
            if any(norm_spaces(str(cargo_view.loc[row, (doc_label, c)])) for c in CARGO_SUBCOLS):
                has_any = True
                break
        if has_any:
            rows_keep.append(row)
    cargo_view = cargo_view.loc[rows_keep]

st.dataframe(style_cargo(cargo_view), use_container_width=True)

st.divider()
excel_bytes = export_excel(ship_df, cargo_df)
st.download_button(
    label="⬇️ Download report (Excel)",
    data=excel_bytes,
    file_name="doc_checker_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "Rules: Doc number mismatch is ignored (no red). POL compares city only. POD compares city only and treats Jakarta as Tanjung Priok. "
    "HS comparison ignores punctuation and matches first 6 digits. KG is converted to MT. Total check compares line sums vs totals (ok/no). "
    "For Indonesian source text (e.g., BC 1.6), extracted values are translated/normalized into English."
)