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

# =========================
# Config
# =========================
APP_TITLE = "Doc Checker PoC — Shipment & Cargo Compare (4 docs)"
MODEL = "gpt-4.1-mini"

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

MAX_CHARS_TO_SEND = 12000
MAX_LINE_ITEMS = 6


# =========================
# Helpers
# =========================
def get_api_key() -> str:
    # Streamlit Cloud Secrets 우선
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"] or "")
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "") or ""


def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()


def norm_spaces(v: str) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v)
    return v


def city_only(port: str) -> str:
    s = norm_spaces(port)
    if not s:
        return ""
    # "Busan, South Korea" -> "Busan" 같은 단순화 목적
    s = re.split(r"[,(/]| EX | KRPUS| IDTPP| IDJKT", s, maxsplit=1)[0].strip()
    return s


def pod_equiv_key(pod: str) -> str:
    s = norm_spaces(pod).lower()
    if not s:
        return ""
    # Jakarta == Tanjung Priok
    if "jakarta" in s:
        return "tanjung priok"
    if "tanjung priok" in s or "priok" in s:
        return "tanjung priok"
    return city_only(pod).lower()


def hs6(hs: str) -> str:
    # 소수점/공백 제거 후 숫자만 추출, 앞 6자리
    s = re.sub(r"\D", "", (hs or ""))
    return s[:6] if s else ""


def to_mt(value: str) -> str:
    """
    Gross/Net weight는 MT 표준.
    "12255 KG" -> 12.255
    "12.255 MT" -> 12.255
    숫자만 있으면 그대로 MT로 간주
    """
    s = norm_spaces(value)
    if not s:
        return ""

    s_low = s.lower()
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", s_low)
    if not nums:
        return s

    n = nums[0].replace(",", "")
    try:
        x = float(n)
    except Exception:
        return s

    if "kg" in s_low:
        x = x / 1000.0
    # mt/ton이면 그대로
    return f"{x:.3f}".rstrip("0").rstrip(".")


def safe_parse_json(raw: str):
    if not raw or not raw.strip():
        return None, "Empty response"
    s = raw.strip()

    # ```json ... ``` 제거
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()

    # 첫 { ~ 마지막 }로 잘라내기
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    try:
        return json.loads(s), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def call_openai_once(prompt: str) -> str:
    """
    Streamlit rerun 안정성 위해 요청마다 새 httpx.Client 생성
    """
    with httpx.Client(
        verify=certifi.where(),
        timeout=httpx.Timeout(connect=20.0, read=90.0, write=90.0, pool=90.0),
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
    ) as http_client:
        client = OpenAI(api_key=get_api_key(), http_client=http_client)
        r = client.responses.create(model=MODEL, input=prompt)
        return (r.output_text or "").strip()


def build_prompt(doc_label: str, text: str) -> str:
    ship_keys = [k for k, _ in SHIPMENT_ITEMS]

    # 핵심: BC 1.6 등 인니어 문서는 영어로 해석/정규화해서 결과를 영어로 반환
    rules_translation = (
        "Language rule:\n"
        "- If the source text is Indonesian (e.g., Customs BC 1.6), translate/normalize extracted values into English.\n"
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
- HS code: keep as written (we normalize later to 6 digits).
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
    for attempt in range(1, 4):  # 3회 시도
        try:
            raw = call_openai_once(prompt)
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


# =========================
# Tables (Always 4 docs)
# =========================
def build_shipment_df(extracted_by_doc: dict) -> pd.DataFrame:
    cols = [label for _, label in DOCS]
    rows = [label for _, label in SHIPMENT_ITEMS]
    df = pd.DataFrame("", index=rows, columns=cols)

    for doc_key, doc_label in DOCS:
        payload = extracted_by_doc.get(doc_key, {}) if isinstance(extracted_by_doc, dict) else {}
        ship = payload.get("shipment", {}) if isinstance(payload, dict) else {}
        for k, row_label in SHIPMENT_ITEMS:
            df.loc[row_label, doc_label] = norm_spaces(str(ship.get(k, "")))

    # 표시용: 도시만
    df.loc["POL (Port of loading)"] = df.loc["POL (Port of loading)"].apply(city_only)
    df.loc["POD (Port of discharge)"] = df.loc["POD (Port of discharge)"].apply(city_only)
    return df


def build_cargo_df(extracted_by_doc: dict) -> pd.DataFrame:
    rows = (
        ["Cargo name", "HS code / pos tarif"]
        + [f"Line {i}" for i in range(1, MAX_LINE_ITEMS + 1)]
        + ["Total", "Total check"]
    )
    cols = pd.MultiIndex.from_product([[label for _, label in DOCS], CARGO_SUBCOLS])
    df = pd.DataFrame("", index=rows, columns=cols)

    for doc_key, doc_label in DOCS:
        payload = extracted_by_doc.get(doc_key, {}) if isinstance(extracted_by_doc, dict) else {}
        cargo = payload.get("cargo", {}) if isinstance(payload, dict) else {}

        df.loc["Cargo name", (doc_label, "Brand")] = norm_spaces(str(cargo.get("cargo_name", "")))
        df.loc["HS code / pos tarif", (doc_label, "Brand")] = norm_spaces(str(cargo.get("hs_code", "")))

        line_items = cargo.get("line_items", []) if isinstance(cargo, dict) else []
        if not isinstance(line_items, list):
            line_items = []

        for i in range(1, MAX_LINE_ITEMS + 1):
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


def compute_kpis(sh_df: pd.DataFrame, cg_df: pd.DataFrame):
    # Shipment mismatches (doc number mismatch ignored)
    ship_mis = 0
    for row in sh_df.index:
        if row == "Document number":
            continue
        vals = [norm_spaces(v) for v in sh_df.loc[row].values if norm_spaces(v)]
        if len(vals) <= 1:
            continue

        if row == "POL (Port of loading)":
            keys = [city_only(v).lower() for v in vals if city_only(v)]
            if len(set(keys)) >= 2:
                ship_mis += 1
        elif row == "POD (Port of discharge)":
            keys = [pod_equiv_key(v) for v in vals if v]
            if len(set(keys)) >= 2:
                ship_mis += 1
        else:
            keys = [v.lower() for v in vals]
            if len(set(keys)) >= 2:
                ship_mis += 1

    # HS mismatch (6-digit)
    hs_vals = []
    for _, doc_label in DOCS:
        hs_vals.append(hs6(str(cg_df.loc["HS code / pos tarif", (doc_label, "Brand")])))
    hs_vals = [h for h in hs_vals if h]
    hs_mismatch = (len(set(hs_vals)) >= 2) if hs_vals else False

    # Docs uploaded count (we infer from whether any non-empty cell exists in shipment doc_number)
    # But more reliable: handled in UI where we know uploaded dict.
    return ship_mis, hs_mismatch


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


def compute_total_check(cg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Total check: line item 합계 vs total 비교하여 ok/no를 "Total check" row에 넣는다.
    (색칠 없음)
    """
    out = cg_df.copy()

    def close(a, b, tol=0.001):
        if a is None or b is None:
            return None
        return abs(a - b) <= tol

    for _, doc_label in DOCS:
        qty_sum = gross_sum = net_sum = amt_sum = 0.0
        has_any = False

        for i in range(1, MAX_LINE_ITEMS + 1):
            row = f"Line {i}"
            q = parse_float(str(out.loc[row, (doc_label, "QTY")]))
            g = parse_float(str(out.loc[row, (doc_label, "Gross WT (MT)")]))
            n = parse_float(str(out.loc[row, (doc_label, "Net WT (MT)")]))
            a = parse_float(str(out.loc[row, (doc_label, "Amount (USD)")]))
            if any(x is not None for x in [q, g, n, a]):
                has_any = True
            if q is not None:
                qty_sum += q
            if g is not None:
                gross_sum += g
            if n is not None:
                net_sum += n
            if a is not None:
                amt_sum += a

        if not has_any:
            out.loc["Total check", (doc_label, "Brand")] = ""
            continue

        tq = parse_float(str(out.loc["Total", (doc_label, "Packing")]))
        tg = parse_float(str(out.loc["Total", (doc_label, "Gross WT (MT)")]))
        tn = parse_float(str(out.loc["Total", (doc_label, "Net WT (MT)")]))
        ta = parse_float(str(out.loc["Total", (doc_label, "Amount (USD)")]))

        checks = []
        for a, b in [(qty_sum, tq), (gross_sum, tg), (net_sum, tn), (amt_sum, ta)]:
            c = close(a, b)
            if c is not None:
                checks.append(c)

        ok = all(checks) if checks else True
        out.loc["Total check", (doc_label, "Brand")] = "ok" if ok else "no"

    return out


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
st.caption("Upload is minimized at top-right. Tables are centered and full-width below. (No text preview)")

# Upload panel (top-right)
top_left, top_right = st.columns([3.5, 1.5], gap="small")

with top_left:
    # Debug kept minimal
    with st.expander("Debug / Status", expanded=False):
        key = get_api_key()
        st.write("API key loaded:", "✅ Yes" if key.startswith("sk-") else "❌ No (OPENAI_API_KEY missing)")
        st.write("Model:", MODEL)

with top_right:
    with st.expander("📤 Upload (4 docs)", expanded=True):
        uploaded = {}
        for doc_key, doc_label in DOCS:
            uploaded[doc_key] = st.file_uploader(doc_label, type=["pdf"], key=f"u_{doc_key}")

        run = st.button("Extract & Compare", use_container_width=True)

# Always create tables (even before extraction)
# Before extraction: empty extracted_by_doc -> all blank tables
extracted_by_doc = {}

# If Extract clicked, only call OpenAI for uploaded docs (missing docs stay blank)
errors = []
if run:
    key = get_api_key()
    if not key.startswith("sk-"):
        st.error("OPENAI_API_KEY is missing. Set it in Streamlit Cloud Secrets.")
    else:
        with st.spinner("Extracting PDFs and calling OpenAI (only for uploaded docs)..."):
            for doc_key, doc_label in DOCS:
                f = uploaded.get(doc_key)
                if not f:
                    # missing -> keep blank payload
                    extracted_by_doc[doc_key] = {"shipment": {}, "cargo": {}}
                    continue

                try:
                    text = pdf_to_text(f)
                    if not text.strip():
                        extracted_by_doc[doc_key] = {"shipment": {}, "cargo": {}}
                        errors.append(f"[{doc_label}] No text extracted (scanned PDF?)")
                        continue
                except Exception as e:
                    extracted_by_doc[doc_key] = {"shipment": {}, "cargo": {}}
                    errors.append(f"[{doc_label}] PDF read error: {type(e).__name__}: {str(e)[:150]}")
                    continue

                obj = extract_doc(doc_label, text)
                if isinstance(obj, dict) and "_error" in obj:
                    errors.append(f"[{doc_label}] {obj['_error']}")
                    # keep blank on error
                    extracted_by_doc[doc_key] = {"shipment": {}, "cargo": {}}
                else:
                    extracted_by_doc[doc_key] = obj

# Build tables
ship_df = build_shipment_df(extracted_by_doc)
cargo_df = build_cargo_df(extracted_by_doc)
cargo_df = compute_total_check(cargo_df)

# KPI row (always shown; meaningful after extraction)
ship_mis, hs_mismatch = compute_kpis(ship_df, cargo_df)
docs_uploaded_count = sum(1 for k, _ in DOCS if uploaded.get(k))  # actual uploaded in UI

k1, k2, k3, k4 = st.columns(4)
k1.metric("Shipment mismatches", ship_mis)
k2.metric("HS mismatch (6-digit)", "YES" if hs_mismatch else "NO")
k3.metric("Docs uploaded", docs_uploaded_count)
k4.metric("Extraction errors", len(errors))

if errors:
    with st.expander("⚠️ Extraction errors (details)", expanded=False):
        for e in errors:
            st.error(e)

# Display tables (NO styling / NO coloring)
st.subheader("Shipment Information")
st.dataframe(ship_df, use_container_width=True)

st.subheader("Cargo Information")
st.dataframe(cargo_df, use_container_width=True)

# Export
st.divider()
excel_bytes = export_excel(ship_df, cargo_df)
st.download_button(
    label="⬇️ Download report (Excel)",
    data=excel_bytes,
    file_name="doc_checker_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "Rules: Doc number mismatch is ignored. POL compares city only. POD compares city only and treats Jakarta as Tanjung Priok. "
    "HS comparison ignores punctuation and matches first 6 digits. KG is converted to MT. "
    "Total check compares line sums vs totals (ok/no). "
    "For Indonesian source text (e.g., BC 1.6), extracted values are translated/normalized into English."