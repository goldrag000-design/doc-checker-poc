import os
import re
import json
import time
import hashlib
from typing import Dict, Any, Tuple, List, Optional

import certifi
import httpx
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =========================
# App config
# =========================
APP_TITLE = "Doc Checker PoC — Shipment & Cargo Compare (4 docs)"
MODEL = "gpt-4.1-mini"

DOCS = [
    ("BL_SWB", "BL / SWB"),
    ("PACKING_LIST", "Packing List"),
    ("PROFORMA_INVOICE", "Proforma Invoice"),
    ("CUSTOMS_BC16", "Customs BC 1.6"),
]

SHIPMENT_ROWS = [
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

CARGO_LINE_COUNT = 5  # Line 1~5

DOC_COLS = {
    # BL/SWB & Packing List: no Amount, no Code
    "BL_SWB": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)"],
    "PACKING_LIST": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)"],
    # Proforma: no Code
    "PROFORMA_INVOICE": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)", "Amount (USD)"],
    # BC 1.6: keep Amount + Code
    "CUSTOMS_BC16": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)", "Amount (USD)", "Code"],
}

MAX_TEXT_CHARS = 12000


# =========================
# Secrets / OpenAI client
# =========================
def get_api_key() -> str:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"] or "")
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "") or ""


def call_openai_once(prompt: str) -> str:
    with httpx.Client(
        verify=certifi.where(),
        timeout=httpx.Timeout(connect=20.0, read=90.0, write=90.0, pool=90.0),
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
    ) as http_client:
        client = OpenAI(api_key=get_api_key(), http_client=http_client)
        r = client.responses.create(model=MODEL, input=prompt)
        return (r.output_text or "").strip()


# =========================
# PDF extraction
# =========================
def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()


def file_fingerprint(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    h = hashlib.sha1(data).hexdigest()
    return f"{uploaded_file.name}:{len(data)}:{h}"


def safe_parse_json(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    if not raw or not raw.strip():
        return None, "Empty response"
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None, "JSON is not an object"
        return obj, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# =========================
# Normalization helpers
# =========================
def norm_spaces(v: str) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v)
    return v


def city_only(port: str) -> str:
    s = norm_spaces(port)
    if not s:
        return ""
    s = re.split(r"[,(/]", s, maxsplit=1)[0].strip()
    return s


def pod_equiv_key(pod: str) -> str:
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


def to_float(s: str) -> Optional[float]:
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


def to_mt(value: str) -> str:
    s = norm_spaces(value)
    if not s:
        return ""
    low = s.lower()
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", low)
    if not nums:
        return s
    n = nums[0].replace(",", "")
    try:
        x = float(n)
    except Exception:
        return s
    if "kg" in low:
        x = x / 1000.0
    return f"{x:.3f}".rstrip("0").rstrip(".")


def vessel_key(v: str) -> str:
    """
    Rule:
    - treat as same if it matches "kmtc hochiminh" and "2510s"
    - ignore punctuation (/ . spaces etc.)
    """
    s = re.sub(r"[^a-z0-9]", "", (v or "").lower())
    if not s:
        return ""
    if ("kmtc" in s) and ("hochiminh" in s) and ("2510s" in s):
        return "kmtchochiminh2510s"
    return s


CONTAINER_NO_RE = re.compile(r"\b[A-Z]{4}\d{7}\b")


def container_qty_only(v: str) -> str:
    """
    Display only quantity number.
    If it looks like a container number (e.g., TEMU0373972), treat as qty=1.
    Else try to extract qty from patterns like x1, 1x, qty 1, 1 container, etc.
    """
    s = norm_spaces(v)
    if not s:
        return ""

    # If container number exists => assume 1
    if CONTAINER_NO_RE.search(s.upper()):
        return "1"

    low = s.lower()

    # x1 / x 1
    m = re.search(r"x\s*(\d+)", low)
    if m:
        return m.group(1)

    # 1x
    m = re.search(r"(\d+)\s*x", low)
    if m:
        return m.group(1)

    # qty: 1
    m = re.search(r"(?:qty|quantity)\s*[:\-]?\s*(\d+)", low)
    if m:
        return m.group(1)

    # 1 container / 1 cont / 1 unit
    m = re.search(r"\b(\d+)\b\s*(?:containers?|conts?|cont|units?|unit|ctn)\b", low)
    if m:
        return m.group(1)

    # if it's just digits already
    if re.fullmatch(r"\d+", s):
        return s

    return ""


# =========================
# Prompt (doc-specific rules included)
# =========================
def build_prompt(doc_key: str, doc_label: str, text: str) -> str:
    schema = """
Return ONLY valid JSON (no markdown).

Schema:
{
  "shipment": {
    "document_number": "",
    "shipper": "",
    "seller": "",
    "plb_operator": "",
    "cargo_owner": "",
    "consignee": "",
    "notify": "",
    "pol": "",
    "pod": "",
    "vessel_voy": "",
    "container_20": "",
    "container_40": ""
  },
  "cargo": {
    "cargo_name": "",
    "hs_code": "",
    "measurement_cbm": "",
    "totals": {
      "qty": "",
      "gross_wt": "",
      "net_wt": "",
      "amount_usd": ""
    },
    "line_items": [
      {
        "brand": "",
        "packing": "",
        "qty": "",
        "gross_wt": "",
        "net_wt": "",
        "amount_usd": "",
        "code": ""
      }
    ]
  }
}

General rules:
- If not found, use "".
- Keep company names and document numbers as-is.
- Weights: include unit if present (KG or MT).
- HS code: keep as written (we normalize later).
- Keep values short.
"""

    doc_rules = """
Shipment rules:
- container_20 and container_40 MUST be quantity ONLY (a number). If only a container number is shown, use 1.
"""

    if doc_key == "CUSTOMS_BC16":
        doc_rules += """
Document-specific rules (Customs BC 1.6):
- Source text is Indonesian; translate extracted values into English (company names stay as-is).
- IMPORTANT: cargo.totals.gross_wt must be the value from 'Berat Kotor' (usually item 31).
- For BC 1.6 line_items: gross_wt can be blank; net_wt and amount/code per line are OK.
"""
    elif doc_key == "PROFORMA_INVOICE":
        doc_rules += """
Document-specific rules (Proforma Invoice):
- If the document shows a single weight per brand/line (e.g., "WT"), treat it as NET WT.
  Put it into line_items[].net_wt (NOT gross_wt).
"""
    elif doc_key == "BL_SWB":
        doc_rules += """
Document-specific rules (BL/SWB):
- Extract measurement_cbm if shown (Measurement/CBM).
"""
    else:
        doc_rules += """
Document-specific rules (Packing List):
- Extract per line qty/gross/net if present.
"""

    return f"""
You extract structured data from shipping/customs documents.

{schema}
{doc_rules}

DOCUMENT TYPE: {doc_label}
TEXT:
-----
{text[:MAX_TEXT_CHARS]}
-----
"""


@st.cache_data(show_spinner=False)
def extract_cached(doc_key: str, doc_label: str, fp: str, text: str) -> Dict[str, Any]:
    last_err = None
    prompt = build_prompt(doc_key, doc_label, text)
    for attempt in range(1, 4):
        try:
            raw = call_openai_once(prompt)
            obj, err = safe_parse_json(raw)
            if err:
                raise ValueError(err)
            return obj
        except Exception as e:
            last_err = e
            time.sleep(2 ** (attempt - 1))
    return {"_error": f"{type(last_err).__name__}: {str(last_err)[:250]}"}


# =========================
# Shipment compare (red text on mismatch)
# =========================
def shipment_value_for_compare(row_key: str, v: str) -> str:
    v = norm_spaces(v)
    if not v:
        return ""
    if row_key == "pol":
        return city_only(v).lower()
    if row_key == "pod":
        return pod_equiv_key(v)
    if row_key == "vessel_voy":
        return vessel_key(v)
    if row_key in ("container_20", "container_40"):
        return container_qty_only(v)
    return v.lower()


def build_shipment_matrix(extracted: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str, Dict[str, str]]]]:
    headers = ["ITEM"] + [label for _, label in DOCS]
    rows = []
    for row_key, row_label in SHIPMENT_ROWS:
        row_map = {}
        for doc_key, doc_label in DOCS:
            payload = extracted.get(doc_key, {})
            ship = payload.get("shipment", {}) if isinstance(payload, dict) else {}
            val = norm_spaces(str(ship.get(row_key, "")))

            # BL/SWB PLB operator always blank
            if doc_key == "BL_SWB" and row_key == "plb_operator":
                val = ""

            # Display rules
            if row_key == "pol":
                val = city_only(val).upper() if val else ""
            if row_key == "pod":
                val = city_only(val).upper() if val else ""

            # Containers: show qty only
            if row_key in ("container_20", "container_40"):
                val = container_qty_only(val)

            row_map[doc_label] = val
        rows.append((row_key, row_label, row_map))
    return headers, rows


def compute_shipment_mismatch_flags(rows) -> Dict[Tuple[str, str], bool]:
    flags = {}
    for row_key, _, row_map in rows:
        # Doc number mismatch ignored
        if row_key == "document_number":
            for _, doc_label in DOCS:
                flags[(row_key, doc_label)] = False
            continue

        comps = []
        for _, doc_label in DOCS:
            comps.append(shipment_value_for_compare(row_key, row_map.get(doc_label, "")))
        non_empty = [c for c in comps if c]
        if len(non_empty) <= 1:
            for _, doc_label in DOCS:
                flags[(row_key, doc_label)] = False
            continue

        mismatch = (len(set(non_empty)) >= 2)
        if mismatch:
            baseline = non_empty[0]
            for idx, (_, doc_label) in enumerate(DOCS):
                c = comps[idx]
                flags[(row_key, doc_label)] = (c != "" and c != baseline)
        else:
            for _, doc_label in DOCS:
                flags[(row_key, doc_label)] = False
    return flags


# =========================
# Cargo building
# =========================
def get_line_item(cargo: dict, idx: int) -> dict:
    li = cargo.get("line_items", []) if isinstance(cargo, dict) else []
    if not isinstance(li, list):
        return {}
    if idx < 0 or idx >= len(li):
        return {}
    it = li[idx]
    return it if isinstance(it, dict) else {}


def line_cell_value(doc_key: str, cargo: dict, line_idx: int, col: str) -> str:
    it = get_line_item(cargo, line_idx)

    brand = norm_spaces(str(it.get("brand", "")))
    packing = norm_spaces(str(it.get("packing", "")))
    qty = norm_spaces(str(it.get("qty", "")))
    gross = to_mt(str(it.get("gross_wt", "")))
    net = to_mt(str(it.get("net_wt", "")))
    amt = norm_spaces(str(it.get("amount_usd", "")))
    code = norm_spaces(str(it.get("code", "")))

    # Proforma: treat WT as NET if gross filled but net empty
    if doc_key == "PROFORMA_INVOICE":
        if gross and not net:
            net = gross
            gross = ""

    # BC 1.6 Gross WT should not appear per line (total only)
    if doc_key == "CUSTOMS_BC16" and col == "Gross WT (MT)":
        return ""

    if col == "Brand":
        return brand
    if col == "Packing":
        return packing
    if col == "QTY":
        return qty
    if col == "Gross WT (MT)":
        return gross
    if col == "Net WT (MT)":
        return net
    if col == "Amount (USD)":
        return amt
    if col == "Code":
        return code
    return ""


def compute_totals_from_lines(doc_key: str, cargo: dict) -> Dict[str, str]:
    li = cargo.get("line_items", []) if isinstance(cargo, dict) else []
    if not isinstance(li, list):
        li = []

    qty_sum = 0.0
    gross_sum = 0.0
    net_sum = 0.0
    amt_sum = 0.0
    has_any = False

    for it in li:
        if not isinstance(it, dict):
            continue
        q = to_float(str(it.get("qty", "")))
        g = to_float(to_mt(str(it.get("gross_wt", ""))))
        n = to_float(to_mt(str(it.get("net_wt", ""))))
        a = to_float(str(it.get("amount_usd", "")))

        # Proforma: WT as net
        if doc_key == "PROFORMA_INVOICE":
            if g is not None and n is None:
                n = g
                g = None

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
        return {"qty": "", "gross": "", "net": "", "amt": ""}

    return {
        "qty": f"{qty_sum:.0f}" if qty_sum.is_integer() else str(qty_sum),
        "gross": f"{gross_sum:.3f}".rstrip("0").rstrip(".") if gross_sum else "",
        "net": f"{net_sum:.3f}".rstrip("0").rstrip(".") if net_sum else "",
        "amt": f"{amt_sum:.2f}".rstrip("0").rstrip(".") if amt_sum else "",
    }


def build_total_check_row(doc_key: str, cols: List[str], totals_doc: Dict[str, str], totals_calc: Dict[str, str]) -> Dict[str, str]:
    """
    Total row = doc totals (unchanged).
    Total check compares (sum of line items) vs (doc totals).
    - ok in black
    - no in red
    If doc total is blank => check cell blank (no judgement).
    """
    def close(a: Optional[float], b: Optional[float], tol=0.001) -> bool:
        if a is None or b is None:
            return False
        return abs(a - b) <= tol

    def decide(doc_val: str, calc_val: str, is_qty=False) -> str:
        if not norm_spaces(doc_val):
            return ""  # can't compare
        da = to_float(doc_val) if is_qty else to_float(doc_val)
        cb = to_float(calc_val) if is_qty else to_float(calc_val)
        if da is None or cb is None:
            # fallback string compare
            return "ok" if norm_spaces(doc_val) == norm_spaces(calc_val) else "no"
        return "ok" if close(da, cb) else "no"

    show = {c: "" for c in cols}

    if doc_key in ["BL_SWB", "PACKING_LIST"]:
        if "QTY" in cols:
            show["QTY"] = decide(totals_doc.get("qty", ""), totals_calc.get("qty", ""), is_qty=True)
        if "Gross WT (MT)" in cols:
            show["Gross WT (MT)"] = decide(totals_doc.get("gross_wt", ""), totals_calc.get("gross", ""))
        if "Net WT (MT)" in cols:
            show["Net WT (MT)"] = decide(totals_doc.get("net_wt", ""), totals_calc.get("net", ""))
    elif doc_key == "PROFORMA_INVOICE":
        if "QTY" in cols:
            show["QTY"] = decide(totals_doc.get("qty", ""), totals_calc.get("qty", ""), is_qty=True)
        if "Net WT (MT)" in cols:
            show["Net WT (MT)"] = decide(totals_doc.get("net_wt", ""), totals_calc.get("net", ""))
        if "Amount (USD)" in cols:
            show["Amount (USD)"] = decide(totals_doc.get("amount_usd", ""), totals_calc.get("amt", ""))
    elif doc_key == "CUSTOMS_BC16":
        if "QTY" in cols:
            show["QTY"] = decide(totals_doc.get("qty", ""), totals_calc.get("qty", ""), is_qty=True)
        if "Gross WT (MT)" in cols:
            show["Gross WT (MT)"] = decide(totals_doc.get("gross_wt", ""), totals_calc.get("gross", ""))
        if "Net WT (MT)" in cols:
            show["Net WT (MT)"] = decide(totals_doc.get("net_wt", ""), totals_calc.get("net", ""))
        if "Amount (USD)" in cols:
            show["Amount (USD)"] = decide(totals_doc.get("amount_usd", ""), totals_calc.get("amt", ""))

    return show


def build_cargo_blocks(extracted: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    blocks = {}
    for doc_key, doc_label in DOCS:
        payload = extracted.get(doc_key, {})
        cargo = payload.get("cargo", {}) if isinstance(payload, dict) else {}
        if not isinstance(cargo, dict):
            cargo = {}

        cols = DOC_COLS[doc_key]

        cargo_name = norm_spaces(str(cargo.get("cargo_name", "")))
        hs = norm_spaces(str(cargo.get("hs_code", "")))
        meas = norm_spaces(str(cargo.get("measurement_cbm", "")))

        # Measurement only from BL/SWB
        if doc_key != "BL_SWB":
            meas = ""

        totals = cargo.get("totals", {}) if isinstance(cargo, dict) else {}
        if not isinstance(totals, dict):
            totals = {}

        totals_qty = norm_spaces(str(totals.get("qty", "")))
        totals_g = to_mt(str(totals.get("gross_wt", "")))
        totals_n = to_mt(str(totals.get("net_wt", "")))
        totals_a = norm_spaces(str(totals.get("amount_usd", "")))

        # Proforma totals: if gross provided but net empty -> move to net
        if doc_key == "PROFORMA_INVOICE":
            if totals_g and not totals_n:
                totals_n = totals_g
                totals_g = ""

        # BL/Packing: no amount
        if doc_key in ["BL_SWB", "PACKING_LIST"]:
            totals_a = ""

        # CALC totals for check only
        totals_calc = compute_totals_from_lines(doc_key, cargo)

        # Total row must be "document totals 그대로" (no autofill)
        total_row = {c: "" for c in cols}
        if "QTY" in cols:
            total_row["QTY"] = totals_qty
        if "Gross WT (MT)" in cols:
            total_row["Gross WT (MT)"] = totals_g if doc_key != "PROFORMA_INVOICE" else ""
        if "Net WT (MT)" in cols:
            total_row["Net WT (MT)"] = totals_n
        if "Amount (USD)" in cols:
            total_row["Amount (USD)"] = totals_a
        if "Brand" in cols:
            total_row["Brand"] = ""
        if "Packing" in cols:
            total_row["Packing"] = ""
        if "Code" in cols:
            total_row["Code"] = ""

        totals_doc = {"qty": totals_qty, "gross_wt": totals_g, "net_wt": totals_n, "amount_usd": totals_a}
        check_row = build_total_check_row(doc_key, cols, totals_doc, totals_calc)

        blocks[doc_key] = {
            "doc_label": doc_label,
            "cols": cols,
            "cargo": cargo,
            "cargo_name": cargo_name,
            "hs": hs,
            "measurement": meas,
            "total_row": total_row,
            "check_row": check_row,
            "totals_calc": totals_calc,
        }
    return blocks


def compute_cargo_flags(blocks: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str, str], bool]:
    flags = {}

    # HS mismatch 6-digit
    hs_vals = {doc_key: hs6(blocks[doc_key]["hs"]) for doc_key, _ in DOCS}
    non_empty = [v for v in hs_vals.values() if v]
    hs_mis = (len(set(non_empty)) >= 2) if non_empty else False
    if hs_mis:
        base = non_empty[0]
        for doc_key, _ in DOCS:
            flags[("HS", doc_key, "Brand")] = (hs_vals[doc_key] != "" and hs_vals[doc_key] != base)
    else:
        for doc_key, _ in DOCS:
            flags[("HS", doc_key, "Brand")] = False

    # Total check row: show red if "no"
    for doc_key, _ in DOCS:
        for c in blocks[doc_key]["cols"]:
            v = norm_spaces(str(blocks[doc_key]["check_row"].get(c, ""))).lower()
            flags[("Check", doc_key, c)] = (v == "no")

    return flags


# =========================
# HTML rendering (NO Styler)
# =========================
def html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def render_shipment_html(headers, rows, flags) -> str:
    css = """
<style>
table.dc { border-collapse: collapse; width: 100%; table-layout: fixed; }
table.dc th, table.dc td { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px; }
table.dc th { background: #f8fafc; font-weight: 600; }
table.dc td.item { width: 220px; background: #fafafa; font-weight: 600; }
.red { color: #dc2626; font-weight: 700; }
</style>
"""
    h = [css, '<table class="dc">', "<thead><tr>"]
    for col in headers:
        h.append(f"<th>{html_escape(col)}</th>")
    h.append("</tr></thead><tbody>")

    for row_key, row_label, row_map in rows:
        h.append("<tr>")
        h.append(f'<td class="item">{html_escape(row_label)}</td>')
        for _, doc_label in DOCS:
            v = row_map.get(doc_label, "")
            is_red = flags.get((row_key, doc_label), False)
            cls = "red" if is_red else ""
            h.append(f'<td class="{cls}">{html_escape(v)}</td>')
        h.append("</tr>")

    h.append("</tbody></table>")
    return "\n".join(h)


def render_cargo_html(blocks, cargo_flags) -> str:
    """
    Two left columns: Group + Item
    Only Group column is merged for Line1~Line5 (Cargo detail).
    """
    css = """
<style>
table.dc2 { border-collapse: collapse; width: 100%; table-layout: fixed; }
table.dc2 th, table.dc2 td { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px; }
table.dc2 th { background: #f8fafc; font-weight: 700; text-align: center; }
table.dc2 td.group { width: 140px; background: #fafafa; font-weight: 800; }
table.dc2 td.item  { width: 180px; background: #fcfcfc; font-weight: 700; }
.red { color: #dc2626; font-weight: 800; }
</style>
"""

    r1 = ["<tr><th rowspan='2'>Group</th><th rowspan='2'>Item</th>"]
    r2 = ["<tr>"]
    for doc_key, doc_label in DOCS:
        cols = blocks[doc_key]["cols"]
        r1.append(f"<th colspan='{len(cols)}'>{html_escape(doc_label)}</th>")
        for c in cols:
            r2.append(f"<th>{html_escape(c)}</th>")
    r1.append("</tr>")
    r2.append("</tr>")

    def td(flag_key: Tuple[str, str, str], value: str) -> str:
        is_red = cargo_flags.get(flag_key, False)
        cls = "red" if is_red else ""
        return f'<td class="{cls}">{html_escape(value)}</td>'

    rows_html = []

    # Cargo name
    rows_html.append("<tr>")
    rows_html.append('<td class="group"></td>')
    rows_html.append('<td class="item">Cargo name</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = blocks[doc_key]["cargo_name"] if c == "Brand" else ""
            rows_html.append(td(("CargoName", doc_key, c), val))
    rows_html.append("</tr>")

    # HS
    rows_html.append("<tr>")
    rows_html.append('<td class="group"></td>')
    rows_html.append('<td class="item">HS code / pos tarif</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = blocks[doc_key]["hs"] if c == "Brand" else ""
            # red only when HS mismatch
            key = ("HS", doc_key, "Brand") if c == "Brand" else ("HSx", doc_key, c)
            rows_html.append(td(key, val))
    rows_html.append("</tr>")

    # Cargo detail (Group col merged only)
    for i in range(CARGO_LINE_COUNT):
        rows_html.append("<tr>")
        if i == 0:
            rows_html.append(f'<td class="group" rowspan="{CARGO_LINE_COUNT}">Cargo detail</td>')
        rows_html.append(f'<td class="item">Line {i+1}</td>')
        for doc_key, _ in DOCS:
            cargo = blocks[doc_key]["cargo"]
            cols = blocks[doc_key]["cols"]
            for c in cols:
                val = line_cell_value(doc_key, cargo, i, c)
                rows_html.append(td(("Line", doc_key, c), val))
        rows_html.append("</tr>")

    # Measurement
    rows_html.append("<tr>")
    rows_html.append('<td class="group"></td>')
    rows_html.append('<td class="item">Measurement (CBM)</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = blocks[doc_key]["measurement"] if c == "Brand" else ""
            rows_html.append(td(("CBM", doc_key, c), val))
    rows_html.append("</tr>")

    # Total (doc totals 그대로)
    rows_html.append("<tr>")
    rows_html.append('<td class="group"></td>')
    rows_html.append('<td class="item">Total</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = norm_spaces(str(blocks[doc_key]["total_row"].get(c, "")))
            rows_html.append(td(("Total", doc_key, c), val))
    rows_html.append("</tr>")

    # Total check (ok black, no red)
    rows_html.append("<tr>")
    rows_html.append('<td class="group"></td>')
    rows_html.append('<td class="item">Total check</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = norm_spaces(str(blocks[doc_key]["check_row"].get(c, "")))
            rows_html.append(td(("Check", doc_key, c), val))
    rows_html.append("</tr>")

    html = "\n".join(
        [css, '<table class="dc2">', "<thead>", "".join(r1), "".join(r2), "</thead>", "<tbody>"]
        + rows_html
        + ["</tbody></table>"]
    )
    return html


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload is minimized at top-right. Tables are rendered as HTML (no pandas Styler).")

top_left, top_right = st.columns([3.5, 1.5], gap="small")

with top_left:
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

# Always create extracted dict (even if missing)
extracted: Dict[str, Any] = {}
errors: List[str] = []
for doc_key, _ in DOCS:
    extracted[doc_key] = {"shipment": {}, "cargo": {}}

if run:
    key = get_api_key()
    if not key.startswith("sk-"):
        st.error("OPENAI_API_KEY is missing. Set it in Streamlit Cloud Secrets.")
    else:
        with st.spinner("Extracting PDFs and calling OpenAI (only for uploaded docs)..."):
            for doc_key, doc_label in DOCS:
                f = uploaded.get(doc_key)
                if not f:
                    continue
                try:
                    text = pdf_to_text(f)
                    if not text:
                        errors.append(f"[{doc_label}] No text extracted (scanned PDF?)")
                        continue
                    fp = file_fingerprint(f)
                    obj = extract_cached(doc_key, doc_label, fp, text)
                    if isinstance(obj, dict) and "_error" in obj:
                        errors.append(f"[{doc_label}] {obj['_error']}")
                        continue
                    extracted[doc_key] = obj
                except Exception as e:
                    errors.append(f"[{doc_label}] {type(e).__name__}: {str(e)[:200]}")

# Build views (always show tables even if no uploads)
ship_headers, ship_rows = build_shipment_matrix(extracted)
ship_flags = compute_shipment_mismatch_flags(ship_rows)

cargo_blocks = build_cargo_blocks(extracted)
cargo_flags = compute_cargo_flags(cargo_blocks)

st.divider()
k1, k2, k3, k4 = st.columns(4)
docs_uploaded_count = sum(1 for k, _ in DOCS if uploaded.get(k))
ship_mis_count = sum(1 for (rk, dl), v in ship_flags.items() if v and rk != "document_number")
k1.metric("Shipment mismatches", ship_mis_count)
k2.metric("HS mismatch (6-digit)", "YES" if any(cargo_flags.get(("HS", dk, "Brand"), False) for dk, _ in DOCS) else "NO")
k3.metric("Docs uploaded", docs_uploaded_count)
k4.metric("Extraction errors", len(errors))

if errors:
    with st.expander("⚠️ Extraction errors (details)", expanded=False):
        for e in errors:
            st.error(e)

st.subheader("Shipment Information")
st.markdown(render_shipment_html(ship_headers, ship_rows, ship_flags), unsafe_allow_html=True)

st.subheader("Cargo Information")
st.markdown(render_cargo_html(cargo_blocks, cargo_flags), unsafe_allow_html=True)

st.divider()
st.caption(
    "Applied rules: "
    "Doc number mismatch is ignored. "
    "POL compares city only. POD compares city only and treats Jakarta as Tanjung Priok. "
    "Vessel/Voy compares by ignoring punctuation; 'KMTC HOCHIMINH' + '2510S' is treated as equivalent. "
    "Container type/amount shows quantity only (container number => 1). "
    "HS comparison ignores punctuation and matches first 6 digits. "
    "KG is converted to MT. "
    "Total row shows document totals as-is. "
    "Total check compares sum(line items) vs document totals; ok=black, no=red. "
    "For Indonesian source text (e.g., BC 1.6), values are translated/normalized into English."
)