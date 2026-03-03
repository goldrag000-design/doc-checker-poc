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

# Shipment rows (left labels)
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

# Cargo structure (your requested layout)
CARGO_ROWS = [
    ("cargo_name", "Cargo name"),
    ("hs_code", "HS code / pos tarif"),
    ("cargo_detail", "Cargo detail"),       # merge Line 1~5 into one cell (multi-line)
    ("measurement_cbm", "Measurement (CBM)"),  # Line 6 renamed, value from BL/SWB
    ("total", "Total"),
    ("total_check", "Total check"),
]

# Per-document visible columns (Requirement #2 #3)
# - BL/SWB: remove Amount + Code
# - Packing List: remove Amount + Code
# - Proforma Invoice: remove Code
# - BC 1.6: keep Amount + Code
DOC_COLS = {
    "BL_SWB": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)"],
    "PACKING_LIST": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)"],
    "PROFORMA_INVOICE": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)", "Amount (USD)"],
    "CUSTOMS_BC16": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)", "Amount (USD)", "Code"],
}

MAX_TEXT_CHARS = 12000
MAX_LINE_ITEMS = 10  # extraction can bring more; we compress into Cargo detail


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
    # cache key: filename + size + sha1 of first chunk
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
        return None, f"{type(e)._name_}: {e}"


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
    # Keep first token before comma/paren/slash
    s = re.split(r"[,(/]", s, maxsplit=1)[0].strip()
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
    """
    Convert KG->MT. Keep 3 decimals (trim zeros).
    """
    s = norm_spaces(value)
    if not s:
        return ""
    low = s.lower()
    nums = re.findall(r"[-+]?\d[\d,]\.?\d", low)
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


# =========================
# Prompt (doc-specific rules included)
# =========================
def build_prompt(doc_key: str, doc_label: str, text: str) -> str:
    """
    Enforces your doc-specific interpretations:
    - BC 1.6: translate Indonesian -> English
    - BC 1.6: totals.gross_wt MUST be from 'berat kotor' (item 31) and line gross can be blank
    - Proforma: per-line 'WT' should be treated as NET WT (store into net_wt)
    - BL/SWB: measurement_cbm should be extracted (if any)
    """
    common_schema = """
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
Rules:
- If not found, use "".
- Keep company names and document numbers as-is.
- Weights: include unit if present (KG or MT).
- HS code: keep as written (we normalize later).
- Keep values short.
"""

    doc_rules = ""
    if doc_key == "CUSTOMS_BC16":
        doc_rules = """
Document-specific rules (Customs BC 1.6):
- Source text is Indonesian; translate extracted values into English (company names stay as-is).
- IMPORTANT: cargo.totals.gross_wt must be the value from 'Berat Kotor' (usually item 31).
- For BC 1.6 line_items: you may omit/leave gross_wt blank; net_wt and amount/code can be provided per line.
"""
    elif doc_key == "PROFORMA_INVOICE":
        doc_rules = """
Document-specific rules (Proforma Invoice):
- If the document shows a single weight per brand/line (e.g., "WT"), treat it as NET WT.
  Put it into line_items[].net_wt (NOT gross_wt).
- If totals for QTY are not explicitly written, you can still fill totals.qty if clearly inferable; otherwise leave "".
"""
    elif doc_key == "BL_SWB":
        doc_rules = """
Document-specific rules (BL/SWB):
- Extract measurement_cbm if shown (e.g., "Measurement (CBM)").
"""
    else:
        doc_rules = """
Document-specific rules (Packing List):
- Extract per line qty/gross/net if present.
"""

    return f"""
You extract structured data from shipping/customs documents.

{common_schema}
{doc_rules}

DOCUMENT TYPE: {doc_label}
TEXT:
-----
{text[:MAX_TEXT_CHARS]}
-----
"""


@st.cache_data(show_spinner=False)
def extract_cached(doc_key: str, doc_label: str, fp: str, text: str) -> Dict[str, Any]:
    """
    Cached by file fingerprint so repeated uploads are fast.
    """
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
    return {"error": f"{type(last_err).name_}: {str(last_err)[:250]}"}


# =========================
# Build Shipment table (with mismatch -> red text)
# =========================
def shipment_value_for_compare(row_key: str, v: str) -> str:
    v = norm_spaces(v)
    if not v:
        return ""
    if row_key == "pol":
        return city_only(v).lower()
    if row_key == "pod":
        return pod_equiv_key(v)
    if row_key == "document_number":
        # ignored mismatch
        return v.lower()
    return v.lower()


def build_shipment_matrix(extracted: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, Dict[str, str]]]]:
    """
    returns headers + rows
    rows: (row_label, {doc_label: value})
    """
    headers = ["ITEM"] + [label for _, label in DOCS]
    rows = []
    for row_key, row_label in SHIPMENT_ROWS:
        row_map = {}
        for doc_key, doc_label in DOCS:
            payload = extracted.get(doc_key, {})
            ship = payload.get("shipment", {}) if isinstance(payload, dict) else {}

            val = norm_spaces(str(ship.get(row_key, "")))

            # Requirement #1: BL/SWB PLB operator must always be blank
            if doc_key == "BL_SWB" and row_key == "plb_operator":
                val = ""

            # City-only view for display (pol/pod)
            if row_key == "pol":
                val = city_only(val).upper() if val else ""
            if row_key == "pod":
                val = city_only(val).upper() if val else ""

            row_map[doc_label] = val
        rows.append((row_key, row_label, row_map))
    return headers, rows


def compute_shipment_mismatch_flags(rows, ignore_doc_number=True) -> Dict[Tuple[str, str], bool]:
    """
    Returns flags for (row_key, doc_label) -> mismatch boolean.
    """
    flags = {}
    for row_key, row_label, row_map in rows:
        if ignore_doc_number and row_key == "document_number":
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

        # mismatch if at least 2 different compare keys
        mismatch = (len(set(non_empty)) >= 2)
        if mismatch:
            # mark only cells that are non-empty and differ from majority/first non-empty baseline
            baseline = non_empty[0]
            for idx, (_, doc_label) in enumerate(DOCS):
                c = comps[idx]
                flags[(row_key, doc_label)] = (c != "" and c != baseline)
        else:
            for _, doc_label in DOCS:
                flags[(row_key, doc_label)] = False
    return flags


# =========================
# Build Cargo table (custom layout + mismatch -> red text)
# =========================
def line_items_to_detail_lines(doc_key: str, cargo: dict) -> List[str]:
    """
    Merge Line 1~5 into a single multiline list like:
    STY | BUNDLE | 1 | 1.126 | 1.123 | 3248.84 | TIP
    but only include columns applicable to that doc.
    """
    cols = DOC_COLS[doc_key]
    li = cargo.get("line_items", []) if isinstance(cargo, dict) else []
    if not isinstance(li, list):
        li = []

    out = []
    for item in li[:5]:  # merge 1~5 only
        if not isinstance(item, dict):
            continue
        brand = norm_spaces(str(item.get("brand", "")))
        packing = norm_spaces(str(item.get("packing", "")))
        qty = norm_spaces(str(item.get("qty", "")))

        gross = to_mt(str(item.get("gross_wt", "")))
        net = to_mt(str(item.get("net_wt", "")))
        amt = norm_spaces(str(item.get("amount_usd", "")))
        code = norm_spaces(str(item.get("code", "")))

        # Requirement #8: Proforma "WT" treated as net_wt already by prompt; but if gross has value and net empty, move it.
        if doc_key == "PROFORMA_INVOICE":
            if gross and not net:
                net = gross
                gross = ""

        pieces = []
        # Build per visible columns
        for c in cols:
            if c == "Brand":
                pieces.append(brand)
            elif c == "Packing":
                pieces.append(packing)
            elif c == "QTY":
                pieces.append(qty)
            elif c == "Gross WT (MT)":
                pieces.append(gross)
            elif c == "Net WT (MT)":
                pieces.append(net)
            elif c == "Amount (USD)":
                pieces.append(amt)
            elif c == "Code":
                pieces.append(code)

        # Skip fully empty lines
        if any(p for p in pieces):
            out.append(" | ".join([p if p else "" for p in pieces]))
    return out


def compute_totals_from_lines(doc_key: str, cargo: dict) -> Dict[str, str]:
    """
    Requirement #6-9: totals.qty might not exist in docs -> calculate and fill.
    Also convert KG->MT.
    """
    li = cargo.get("line_items", []) if isinstance(cargo, dict) else []
    if not isinstance(li, list):
        li = []

    qty_sum = 0.0
    gross_sum = 0.0
    net_sum = 0.0
    amt_sum = 0.0
    has_any = False

    for item in li:
        if not isinstance(item, dict):
            continue
        q = to_float(str(item.get("qty", "")))
        g = to_float(to_mt(str(item.get("gross_wt", ""))))
        n = to_float(to_mt(str(item.get("net_wt", ""))))
        a = to_float(str(item.get("amount_usd", "")))

        # proforma: if gross filled but net empty, treat as net
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


def cargo_total_check(doc_key: str, totals_from_doc: Dict[str, str], totals_calc: Dict[str, str]) -> Dict[str, str]:
    """
    Requirement #6-9: Total check shown only under specified columns.
    We'll produce per-column ok/no strings (or "") and compare doc totals to calc totals when doc totals exist.
    If doc totals missing -> we fill totals with calc and mark ok.
    """
    def close(a: Optional[float], b: Optional[float], tol=0.001) -> bool:
        if a is None or b is None:
            return True  # if either missing, don't fail
        return abs(a - b) <= tol

    # Doc totals (raw)
    doc_qty = to_float(totals_from_doc.get("qty", ""))
    doc_g = to_float(to_mt(totals_from_doc.get("gross_wt", "")))
    doc_n = to_float(to_mt(totals_from_doc.get("net_wt", "")))
    doc_a = to_float(totals_from_doc.get("amount_usd", ""))

    calc_qty = to_float(totals_calc.get("qty", ""))
    calc_g = to_float(totals_calc.get("gross", ""))
    calc_n = to_float(totals_calc.get("net", ""))
    calc_a = to_float(totals_calc.get("amt", ""))

    # If doc missing qty, treat as ok after we fill it from calc
    ok_qty = close(doc_qty, calc_qty)
    ok_g = close(doc_g, calc_g)
    ok_n = close(doc_n, calc_n)
    ok_a = close(doc_a, calc_a)

    # Where to show check
    show = {c: "" for c in DOC_COLS[doc_key]}

    if doc_key in ["BL_SWB", "PACKING_LIST"]:
        # only QTY, Gross, Net
        show["QTY"] = "ok" if ok_qty else "no"
        show["Gross WT (MT)"] = "ok" if ok_g else "no"
        show["Net WT (MT)"] = "ok" if ok_n else "no"
    elif doc_key == "PROFORMA_INVOICE":
        # QTY, Net, Amount
        show["QTY"] = "ok" if ok_qty else "no"
        show["Net WT (MT)"] = "ok" if ok_n else "no"
        show["Amount (USD)"] = "ok" if ok_a else "no"
    elif doc_key == "CUSTOMS_BC16":
        # QTY, Gross, Net, Amount
        show["QTY"] = "ok" if ok_qty else "no"
        show["Gross WT (MT)"] = "ok" if ok_g else "no"
        show["Net WT (MT)"] = "ok" if ok_n else "no"
        show["Amount (USD)"] = "ok" if ok_a else "no"

    return show


def build_cargo_blocks(extracted: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns per doc block:
      {
        "Cargo name": {...},
        "HS": {...},
        "Cargo detail": "multiline",
        "Measurement (CBM)": "...",
        "Total": {col:value},
        "Total check": {col: ok/no}
      }
    """
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

        # Requirement #4: Measurement only from BL/SWB
        if doc_key != "BL_SWB":
            meas = ""

        # Totals from doc (raw)
        totals = cargo.get("totals", {}) if isinstance(cargo, dict) else {}
        if not isinstance(totals, dict):
            totals = {}

        totals_qty = norm_spaces(str(totals.get("qty", "")))
        totals_g = to_mt(str(totals.get("gross_wt", "")))
        totals_n = to_mt(str(totals.get("net_wt", "")))
        totals_a = norm_spaces(str(totals.get("amount_usd", "")))

        # Requirement #5: BC1.6 gross should be TOTAL ONLY (berat kotor item 31)
        # We enforce: do not show per-line gross; only Total row gross will be populated.
        # (Already handled by prompt; still keep safety)
        if doc_key == "CUSTOMS_BC16":
            # keep totals_g as-is; detail lines won't include gross if extracted; we won't show per-line gross in merged detail anyway.
            pass

        # Calculate totals from line items (to fill missing qty, etc.)
        totals_calc = compute_totals_from_lines(doc_key, cargo)

        # Requirement #6-9: If doc totals missing qty -> fill with calc
        if not totals_qty and totals_calc["qty"]:
            totals_qty = totals_calc["qty"]

        # For BL & Packing: amount not applicable -> ignore
        if doc_key in ["BL_SWB", "PACKING_LIST"]:
            totals_a = ""

        # For Proforma: gross shown in doc should be treated as net
        if doc_key == "PROFORMA_INVOICE":
            # If totals_g exists and totals_n empty, move gross->net
            if totals_g and not totals_n:
                totals_n = totals_g
                totals_g = ""

        # Build cargo detail lines (merged 1~5)
        detail_lines = line_items_to_detail_lines(doc_key, cargo)
        detail_text = "\n".join(detail_lines)

        # Total row values per visible cols
        total_row = {c: "" for c in cols}
        if "Brand" in cols:
            total_row["Brand"] = ""
        if "Packing" in cols:
            # show bundles if available in doc? If not, keep blank. We keep qty in QTY and weights in weight cols.
            total_row["Packing"] = totals_qty if ("Packing" in cols and doc_key in ["BL_SWB", "PACKING_LIST", "PROFORMA_INVOICE", "CUSTOMS_BC16"]) else ""

        # Better: Put qty in QTY column if present
        if "QTY" in cols:
            total_row["QTY"] = totals_qty

        if "Gross WT (MT)" in cols:
            # BC1.6: total gross only (berat kotor)
            total_row["Gross WT (MT)"] = totals_g if doc_key != "PROFORMA_INVOICE" else ""  # proforma gross not used
        if "Net WT (MT)" in cols:
            total_row["Net WT (MT)"] = totals_n
        if "Amount (USD)" in cols:
            total_row["Amount (USD)"] = totals_a
        if "Code" in cols:
            total_row["Code"] = ""

        # Total check row
        totals_from_doc = {
            "qty": totals_qty,
            "gross_wt": totals_g,
            "net_wt": totals_n,
            "amount_usd": totals_a,
        }
        check_row = cargo_total_check(doc_key, totals_from_doc, totals_calc)

        blocks[doc_key] = {
            "doc_label": doc_label,
            "cols": cols,
            "cargo_name": cargo_name,
            "hs": hs,
            "detail_text": detail_text,
            "measurement": meas,
            "total_row": total_row,
            "check_row": check_row,
            "totals_calc": totals_calc,
        }

    return blocks


def compute_cargo_mismatch_flags(blocks: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str, str], bool]:
    """
    Flags for cargo cells to show RED text:
      key: (row_name, doc_key, col_name) -> mismatch bool

    Requirement #10: mismatch among same items across docs -> red text.
    We apply it to:
      - HS code (6 digit)
      - Total row numeric cells (QTY, Gross, Net, Amount where applicable)
    """
    flags = {}

    # HS mismatch (6-digit)
    hs_vals = {}
    for doc_key, _ in DOCS:
        hs_vals[doc_key] = hs6(blocks[doc_key]["hs"])
    non_empty_hs = [v for v in hs_vals.values() if v]
    hs_mismatch = (len(set(non_empty_hs)) >= 2) if non_empty_hs else False
    if hs_mismatch:
        baseline = non_empty_hs[0]
        for doc_key, _ in DOCS:
            v = hs_vals[doc_key]
            flags[("HS", doc_key, "Brand")] = (v != "" and v != baseline)
    else:
        for doc_key, _ in DOCS:
            flags[("HS", doc_key, "Brand")] = False

    # Total row mismatches by column across docs (for shared metrics)
    # Collect totals per metric (normalized)
    metrics = ["QTY", "Gross WT (MT)", "Net WT (MT)", "Amount (USD)"]
    for metric in metrics:
        values = []
        per_doc = {}
        for doc_key, _ in DOCS:
            cols = blocks[doc_key]["cols"]
            if metric not in cols:
                per_doc[doc_key] = ""
                continue
            v = norm_spaces(str(blocks[doc_key]["total_row"].get(metric, "")))
            per_doc[doc_key] = v
            if v:
                values.append(v)

        if len([v for v in values if v]) <= 1:
            for doc_key, _ in DOCS:
                flags[("Total", doc_key, metric)] = False
            continue

        # Compare numeric if possible
        def norm_metric(m, s):
            if m in ["Gross WT (MT)", "Net WT (MT)"]:
                return to_float(s)
            if m == "QTY":
                return to_float(s)
            if m == "Amount (USD)":
                return to_float(s)
            return s

        normed = [norm_metric(metric, v) for v in values if v]
        # If numeric compare:
        if all(x is not None for x in normed):
            baseline = normed[0]
            for doc_key, _ in DOCS:
                v = per_doc[doc_key]
                if not v:
                    flags[("Total", doc_key, metric)] = False
                    continue
                x = norm_metric(metric, v)
                flags[("Total", doc_key, metric)] = (x is not None and abs(x - baseline) > 0.001)
        else:
            baseline = values[0]
            for doc_key, _ in DOCS:
                v = per_doc[doc_key]
                flags[("Total", doc_key, metric)] = (v != "" and v != baseline)

    # Total check row: if "no" then red (local flag)
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            v = norm_spaces(str(blocks[doc_key]["check_row"].get(c, ""))).lower()
            flags[("Check", doc_key, c)] = (v == "no")

    return flags


# =========================
# HTML Rendering (no Styler)
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
    """
    Simple HTML table, mismatch cells red.
    """
    css = """
<style>
table.dc { border-collapse: collapse; width: 100%; table-layout: fixed; }
table.dc th, table.dc td { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px; }
table.dc th { background: #f8fafc; font-weight: 600; }
table.dc td.item { width: 220px; background: #fafafa; font-weight: 600; }
.red { color: #dc2626; font-weight: 700; }
.small { font-size: 12px; color: #6b7280; }
</style>
"""
    h = [css, '<table class="dc">', "<thead><tr>"]
    for col in headers:
        h.append(f"<th>{html_escape(col)}</th>")
    h.append("</tr></thead><tbody>")

    for row_key, row_label, row_map in rows:
        h.append("<tr>")
        h.append(f'<td class="item">{html_escape(row_label)}</td>')
        for doc_key, doc_label in DOCS:
            v = row_map.get(doc_label, "")
            is_red = flags.get((row_key, doc_label), False)
            cls = "red" if is_red else ""
            h.append(f'<td class="{cls}">{html_escape(v)}</td>')
        h.append("</tr>")

    h.append("</tbody></table>")
    return "\n".join(h)


def render_cargo_html(blocks, cargo_flags) -> str:
    """
    Cargo: one big table with multiheaders (documents each have different subcolumns).
    Implements:
      - Cargo detail merged (1 cell) with multiline
      - Measurement row only BL
      - Total / Total check row show only under specified columns (already blanked in data)
      - Mismatch red text across docs (Requirement #10)
    """
    css = """
<style>
table.dc2 { border-collapse: collapse; width: 100%; table-layout: fixed; }
table.dc2 th, table.dc2 td { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px; }
table.dc2 th { background: #f8fafc; font-weight: 700; text-align: center; }
table.dc2 td.item { width: 220px; background: #fafafa; font-weight: 700; }
table.dc2 td { word-wrap: break-word; }
.red { color: #dc2626; font-weight: 800; }
pre.detail { margin: 0; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
</style>
"""
    # Build header rows:
    # Row1: ITEM + each doc label spanning its subcolumns count
    # Row2: subcolumns
    r1 = ["<tr><th rowspan='2'>ITEM</th>"]
    r2 = ["<tr>"]
    for doc_key, doc_label in DOCS:
        cols = blocks[doc_key]["cols"]
        r1.append(f"<th colspan='{len(cols)}'>{html_escape(doc_label)}</th>")
        for c in cols:
            r2.append(f"<th>{html_escape(c)}</th>")
    r1.append("</tr>")
    r2.append("</tr>")

    # Rows
    rows_html = []

    def cell(doc_key, col, value, flag_key):
        v = html_escape(value)
        is_red = cargo_flags.get(flag_key, False)
        cls = "red" if is_red else ""
        return f'<td class="{cls}">{v}</td>'

    # Cargo name row
    rows_html.append("<tr>")
    rows_html.append('<td class="item">Cargo name</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for i, c in enumerate(cols):
            val = blocks[doc_key]["cargo_name"] if c == "Brand" else ""
            rows_html.append(cell(doc_key, c, val, ("CargoName", doc_key, c)))
    rows_html.append("</tr>")

    # HS row
    rows_html.append("<tr>")
    rows_html.append('<td class="item">HS code / pos tarif</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = blocks[doc_key]["hs"] if c == "Brand" else ""
            rows_html.append(cell(doc_key, c, val, ("HS", doc_key, "Brand")))
    rows_html.append("</tr>")

    # Cargo detail row (merged content in Brand column; other cols blank)
    rows_html.append("<tr>")
    rows_html.append('<td class="item">Cargo detail</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            if c == "Brand":
                txt = blocks[doc_key]["detail_text"]
                is_red = False  # not applying mismatch on detail lines for now
                cls = "red" if is_red else ""
                rows_html.append(f'<td class="{cls}"><pre class="detail">{html_escape(txt)}</pre></td>')
            else:
                rows_html.append('<td></td>')
    rows_html.append("</tr>")

    # Measurement (CBM) row
    rows_html.append("<tr>")
    rows_html.append('<td class="item">Measurement (CBM)</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = blocks[doc_key]["measurement"] if c == "Brand" else ""
            rows_html.append(cell(doc_key, c, val, ("CBM", doc_key, c)))
    rows_html.append("</tr>")

    # Total row
    rows_html.append("<tr>")
    rows_html.append('<td class="item">Total</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = norm_spaces(str(blocks[doc_key]["total_row"].get(c, "")))
            rows_html.append(cell(doc_key, c, val, ("Total", doc_key, c)))
    rows_html.append("</tr>")

    # Total check row
    rows_html.append("<tr>")
    rows_html.append('<td class="item">Total check</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = norm_spaces(str(blocks[doc_key]["check_row"].get(c, "")))
            rows_html.append(cell(doc_key, c, val, ("Check", doc_key, c)))
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
st.caption("Upload is minimized at top-right. Tables are rendered as HTML for stability (no pandas Styler).")

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

# Default empty payloads
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
                    continue  # keep empty
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
                    errors.append(f"[{doc_label}] {type(e)._name_}: {str(e)[:200]}")

# Build shipment
ship_headers, ship_rows = build_shipment_matrix(extracted)
ship_flags = compute_shipment_mismatch_flags(ship_rows, ignore_doc_number=True)

# Build cargo
cargo_blocks = build_cargo_blocks(extracted)
cargo_flags = compute_cargo_mismatch_flags(cargo_blocks)

# KPI
docs_uploaded_count = sum(1 for k, _ in DOCS if uploaded.get(k))
st.divider()
k1, k2, k3, k4 = st.columns(4)
# shipment mismatch count
ship_mis_count = sum(1 for (rk, dl), v in ship_flags.items() if v and rk != "document_number")
k1.metric("Shipment mismatches", ship_mis_count)
k2.metric("HS mismatch (6-digit)", "YES" if any(cargo_flags.get(("HS", dk, "Brand"), False) for dk, _ in DOCS) else "NO")
k3.metric("Docs uploaded", docs_uploaded_count)
k4.metric("Extraction errors", len(errors))

if errors:
    with st.expander("⚠️ Extraction errors (details)", expanded=False):
        for e in errors:
            st.error(e)

# Render tables
st.subheader("Shipment Information")
st.markdown(render_shipment_html(ship_headers, ship_rows, ship_flags), unsafe_allow_html=True)

st.subheader("Cargo Information")
st.markdown(render_cargo_html(cargo_blocks, cargo_flags), unsafe_allow_html=True)

st.divider()
st.caption(
    "Applied rules: "
    "BL/SWB PLB operator is always blank. "
    "POL compares city only. POD compares city only and treats Jakarta as Tanjung Priok. "
    "HS comparison ignores punctuation and matches first 6 digits. "
    "KG is converted to MT. "
    "Total/QTY missing in documents are calculated from line items. "
    "For Indonesian source text (e.g., BC 1.6), values are translated/normalized into English. "
    "Mismatched cells are shown in red text."
)