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
APP_TITLE = "Doc Checker PoC — Shipment & Cargo Compare (auto-detect, single upload)"
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

# Column rules per doc:
# - BL/SWB & Packing List: remove Amount + Code
# - Proforma: remove Code
# - BC 1.6: keep Amount + Code
DOC_COLS = {
    "BL_SWB": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)"],
    "PACKING_LIST": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)"],
    "PROFORMA_INVOICE": ["Brand", "Packing", "QTY", "Gross WT (MT)", "Net WT (MT)", "Amount (USD)"],
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
# PDF extraction (page-level)
# =========================
def pdf_to_pages_text(file) -> List[str]:
    """KOR: PDF를 페이지별 텍스트 리스트로 추출"""
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        pages.append((p.extract_text() or "").strip())
    return pages


def fingerprint_from_inputs(uploaded_files: List, page_assignments: List[Tuple[str, int, str]]) -> str:
    """
    KOR:
    - 업로드 파일 내용 + (파일명/페이지/타입) 배정을 섞어서 캐시용 fingerprint 생성
    """
    h = hashlib.sha1()
    for f in uploaded_files:
        try:
            data = f.getvalue()
        except Exception:
            data = b""
        h.update(f.name.encode("utf-8", "ignore"))
        h.update(str(len(data)).encode("utf-8"))
        h.update(hashlib.sha1(data).digest())
    for fn, pno, dtype in page_assignments:
        h.update(f"{fn}:{pno}:{dtype}".encode("utf-8"))
    return h.hexdigest()


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
        s = s[start : end + 1]
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

    KOR:
    - "0"은 표시하지 않고 빈칸 반환 (요청 반영)
    """
    s = norm_spaces(v)
    if not s:
        return ""

    if CONTAINER_NO_RE.search(s.upper()):
        return "1"

    low = s.lower()

    m = re.search(r"x\s*(\d+)", low)
    if m:
        return "" if m.group(1) == "0" else m.group(1)

    m = re.search(r"(\d+)\s*x", low)
    if m:
        return "" if m.group(1) == "0" else m.group(1)

    m = re.search(r"(?:qty|quantity)\s*[:\-]?\s*(\d+)", low)
    if m:
        return "" if m.group(1) == "0" else m.group(1)

    m = re.search(r"\b(\d+)\b\s*(?:containers?|conts?|cont|units?|unit|ctn)\b", low)
    if m:
        return "" if m.group(1) == "0" else m.group(1)

    if re.fullmatch(r"\d+", s):
        return "" if s == "0" else s

    return ""


def loose_text_key(v: str) -> str:
    """
    KOR:
    Shipment 비교에서 '.', ',', '-', '_' 등 기호 차이는 무시.
    """
    s = norm_spaces(v).lower()
    s = re.sub(r"[.,\-_]", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# Cargo line mismatch helpers
# =========================
def norm_text_for_compare(v: str) -> str:
    s = norm_spaces(v).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_packing_for_compare(v: str) -> str:
    """
    - '1 BUNDLE'과 'BUNDLE' 동일 취급 (선행 숫자 제거)
    - bundles -> bundle
    """
    s = norm_spaces(v).lower()
    s = re.sub(r"^\d+\s+", "", s)
    s = s.replace("bundles", "bundle")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def value_key_for_line_compare(col: str, raw: str) -> Tuple[str, Optional[float], str]:
    if col in ("Gross WT (MT)", "Net WT (MT)"):
        f = to_float(raw)
        return ("num", f, "")
    if col == "Amount (USD)":
        f = to_float(raw)
        return ("num", f, "")
    if col == "QTY":
        f = to_float(raw)
        return ("num", f, "")
    if col == "Packing":
        return ("text", None, norm_packing_for_compare(raw))
    return ("text", None, norm_text_for_compare(raw))


def is_line_cell_mismatch(values_by_doc: Dict[str, str], col: str, tol: float = 0.002) -> Dict[str, bool]:
    """
    - 동일 line index(Line1끼리)에서 각 서류의 같은 컬럼 값 비교
    - 빈 값은 제외
    - 값이 2개 이상 있을 때 서로 다르면 mismatch
    - 숫자는 tol 허용오차
    """
    present = {dk: v for dk, v in values_by_doc.items() if norm_spaces(v)}
    if len(present) <= 1:
        return {dk: False for dk in values_by_doc.keys()}

    keys = {}
    for dk, v in present.items():
        kind, num, txt = value_key_for_line_compare(col, v)
        keys[dk] = (kind, num, txt)

    base_dk = next(iter(present.keys()))
    base_kind, base_num, base_txt = keys[base_dk]

    mismatch_map = {dk: False for dk in values_by_doc.keys()}

    for dk in present.keys():
        kind, num, txt = keys[dk]

        if kind != base_kind:
            mismatch_map[dk] = True
            continue

        if kind == "num":
            if base_num is None or num is None:
                mismatch_map[dk] = (norm_spaces(present[dk]) != norm_spaces(present[base_dk]))
            else:
                mismatch_map[dk] = (abs(num - base_num) > tol)
        else:
            mismatch_map[dk] = (txt != base_txt)

    if not any(mismatch_map.values()):
        mismatch_map = {dk: False for dk in mismatch_map.keys()}

    return mismatch_map


# =========================
# Document classifier (page-level)
# =========================
def contains_all(text: str, tokens: List[str]) -> bool:
    t = text.lower()
    return all(tok.lower() in t for tok in tokens)


def classify_page(text: str) -> Optional[str]:
    """
    Returns doc_key or None (unknown/ignore).
    Priority: BC1.6 > BL/SWB > Packing > Invoice
    """
    t = (text or "").lower()
    if not t.strip():
        return None

    # --- BC 1.6 ---
    bc16_long_tokens = [
        "pemberitahuan pabean pengeluaran barang",
        "dari kawasan pabean untuk ditimbun di pusat logistik berikat",
    ]
    if contains_all(t, bc16_long_tokens):
        return "CUSTOMS_BC16"
    if ("pemberitahuan pabean" in t) and ("pusat logistik berikat" in t):
        return "CUSTOMS_BC16"
    if "bc 1.6" in t or "bc1.6" in t.replace(" ", ""):
        return "CUSTOMS_BC16"

    # --- BL / SWB ---
    if ("bill of lading" in t) or ("sea waybill" in t) or ("seawaybill" in t) or ("waybill" in t) or ("b/l" in t):
        return "BL_SWB"

    # --- Packing List ---
    if "packing list" in t:
        return "PACKING_LIST"

    # --- Invoice ---
    if "invoice" in t:
        return "PROFORMA_INVOICE"

    return None


def split_and_group_pages(uploaded_files: List) -> Tuple[Dict[str, str], List[Tuple[str, int, str]]]:
    """
    Returns:
    - grouped_text_by_doc_key: doc_key -> combined text
    - assignments: (filename, page_index, assigned_doc_key)
    continuation:
    - unknown page => assign to last_type (if exists), else ignore
    """
    grouped_pages: Dict[str, List[str]] = {k: [] for k, _ in DOCS}
    assignments: List[Tuple[str, int, str]] = []
    last_type: Optional[str] = None

    for f in uploaded_files:
        pages = pdf_to_pages_text(f)
        for idx, page_text in enumerate(pages):
            dtype = classify_page(page_text)
            if dtype is None:
                if last_type is None:
                    continue
                dtype = last_type
            else:
                last_type = dtype

            if dtype not in grouped_pages:
                continue

            grouped_pages[dtype].append(page_text)
            assignments.append((getattr(f, "name", "file.pdf"), idx, dtype))

    grouped_text = {k: "\n\n".join(v).strip() for k, v in grouped_pages.items()}
    return grouped_text, assignments


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
- If a container field is not applicable, return "" (do NOT force 0).
"""
    elif doc_key == "PROFORMA_INVOICE":
        doc_rules += """
Document-specific rules (Invoice):
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
    """
    - Shipment 비교: 기호 차이 무시(.,-_ 등)
    - POL/POD/Vessel/Container 특수 정규화 유지
    """
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
    return loose_text_key(v)


def build_shipment_matrix(extracted: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str, Dict[str, str]]]]:
    headers = ["ITEM"] + [label for _, label in DOCS]
    rows = []
    for row_key, row_label in SHIPMENT_ROWS:
        row_map = {}
        for doc_key, doc_label in DOCS:
            payload = extracted.get(doc_key, {})
            ship = payload.get("shipment", {}) if isinstance(payload, dict) else {}
            val = norm_spaces(str(ship.get(row_key, "")))

            # BL/SWB PLB operator is always blank
            if doc_key == "BL_SWB" and row_key == "plb_operator":
                val = ""

            # Display rules
            if row_key == "pol":
                val = city_only(val).upper() if val else ""
            if row_key == "pod":
                val = city_only(val).upper() if val else ""

            # Containers: show qty only; hide 0 (applies to ALL docs incl. BC1.6)
            if row_key in ("container_20", "container_40"):
                val = container_qty_only(val)

            row_map[doc_label] = val
        rows.append((row_key, row_label, row_map))
    return headers, rows


def compute_shipment_mismatch_flags(rows) -> Dict[Tuple[str, str], bool]:
    flags = {}
    for row_key, _, row_map in rows:
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

    # Invoice: treat WT as NET if gross filled but net empty
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


def build_total_check_row(
    doc_key: str,
    cols: List[str],
    totals_doc: Dict[str, str],
    totals_calc: Dict[str, str],
) -> Dict[str, str]:
    """
    Total row = doc totals (unchanged).
    Total check compares (sum of line items) vs (doc totals).
    - ok in black
    - no in red
    If doc total is blank => check cell blank.
    """
    def close(a: Optional[float], b: Optional[float], tol=0.001) -> bool:
        if a is None or b is None:
            return False
        return abs(a - b) <= tol

    def decide(doc_val: str, calc_val: str) -> str:
        if not norm_spaces(doc_val):
            return ""
        da = to_float(doc_val)
        cb = to_float(calc_val)
        if da is None or cb is None:
            return "ok" if norm_spaces(doc_val) == norm_spaces(calc_val) else "no"
        return "ok" if close(da, cb) else "no"

    show = {c: "" for c in cols}

    if doc_key in ["BL_SWB", "PACKING_LIST"]:
        if "QTY" in cols:
            show["QTY"] = decide(totals_doc.get("qty", ""), totals_calc.get("qty", ""))
        if "Gross WT (MT)" in cols:
            show["Gross WT (MT)"] = decide(totals_doc.get("gross_wt", ""), totals_calc.get("gross", ""))
        if "Net WT (MT)" in cols:
            show["Net WT (MT)"] = decide(totals_doc.get("net_wt", ""), totals_calc.get("net", ""))
    elif doc_key == "PROFORMA_INVOICE":
        if "QTY" in cols:
            show["QTY"] = decide(totals_doc.get("qty", ""), totals_calc.get("qty", ""))
        if "Net WT (MT)" in cols:
            show["Net WT (MT)"] = decide(totals_doc.get("net_wt", ""), totals_calc.get("net", ""))
        if "Amount (USD)" in cols:
            show["Amount (USD)"] = decide(totals_doc.get("amount_usd", ""), totals_calc.get("amt", ""))
    elif doc_key == "CUSTOMS_BC16":
        if "QTY" in cols:
            show["QTY"] = decide(totals_doc.get("qty", ""), totals_calc.get("qty", ""))
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

        if doc_key != "BL_SWB":
            meas = ""

        totals = cargo.get("totals", {}) if isinstance(cargo, dict) else {}
        if not isinstance(totals, dict):
            totals = {}

        totals_qty = norm_spaces(str(totals.get("qty", "")))
        totals_g = to_mt(str(totals.get("gross_wt", "")))
        totals_n = to_mt(str(totals.get("net_wt", "")))
        totals_a = norm_spaces(str(totals.get("amount_usd", "")))

        if doc_key == "PROFORMA_INVOICE":
            if totals_g and not totals_n:
                totals_n = totals_g
                totals_g = ""

        if doc_key in ["BL_SWB", "PACKING_LIST"]:
            totals_a = ""

        totals_calc = compute_totals_from_lines(doc_key, cargo)

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
        }
    return blocks


def compute_cargo_flags(
    blocks: Dict[str, Dict[str, Any]],
    bc16_shipper: str,
    bc16_cargo_owner: str,
) -> Dict[Tuple[str, str, str], bool]:
    """
    Flags key convention used by renderer:
      ("HS", doc_key, "Brand") for HS row (Brand column)
      ("Check", doc_key, col) for total-check row
      (f"Line{n}", doc_key, col) for line-cell mismatch (Line1~5)
      (f"Line{n}", "CUSTOMS_BC16", "Code") can also be forced red by TIP-rule
    """
    flags: Dict[Tuple[str, str, str], bool] = {}

    # 1) HS mismatch (6-digit) -> mark HS row Brand cell red
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

    # 2) Total check row: red only when "no"
    for doc_key, _ in DOCS:
        for c in blocks[doc_key]["cols"]:
            v = norm_spaces(str(blocks[doc_key]["check_row"].get(c, ""))).lower()
            flags[("Check", doc_key, c)] = (v == "no")

    # 3) Cargo line mismatch (Line1~Line5) cell-level
    all_cols = set()
    for doc_key, _ in DOCS:
        for c in blocks[doc_key]["cols"]:
            all_cols.add(c)

    for line_idx in range(CARGO_LINE_COUNT):
        for col in sorted(all_cols):
            values_by_doc: Dict[str, str] = {}
            for doc_key, _ in DOCS:
                if col not in blocks[doc_key]["cols"]:
                    values_by_doc[doc_key] = ""
                    continue
                cargo = blocks[doc_key]["cargo"]
                values_by_doc[doc_key] = line_cell_value(doc_key, cargo, line_idx, col)

            mis_map = is_line_cell_mismatch(values_by_doc, col)
            for doc_key, _ in DOCS:
                flags[(f"Line{line_idx+1}", doc_key, col)] = bool(mis_map.get(doc_key, False))

    # 4) TIP rule (Customs BC 1.6 Code)
    shipper_k = loose_text_key(bc16_shipper)
    owner_k = loose_text_key(bc16_cargo_owner)
    tip_should_be_red = bool(owner_k) and (owner_k == shipper_k)

    if tip_should_be_red:
        doc_key = "CUSTOMS_BC16"
        if "Code" in blocks[doc_key]["cols"]:
            cargo = blocks[doc_key]["cargo"]
            for line_idx in range(CARGO_LINE_COUNT):
                code_val = line_cell_value(doc_key, cargo, line_idx, "Code")
                if norm_spaces(code_val).lower() == "tip":
                    flags[(f"Line{line_idx+1}", doc_key, "Code")] = True

    return flags


# =========================
# HTML rendering (horizontal + scroll)
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
.scrollx { overflow-x: auto; width: 100%; }
table.dc { border-collapse: collapse; width: max-content; min-width: 100%; }
table.dc th, table.dc td {
  border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px;
  white-space: nowrap;
}
table.dc th { background: #f8fafc; font-weight: 600; }
table.dc td.item { width: 220px; background: #fafafa; font-weight: 600; }
.red { color: #dc2626; font-weight: 700; }
</style>
"""
    h = [css, '<div class="scrollx">', '<table class="dc">', "<thead><tr>"]
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

    h.append("</tbody></table></div>")
    return "\n".join(h)


def render_cargo_html(blocks, cargo_flags) -> str:
    """
    Two left columns: Group + Item
    Only Group column is merged for Line1~Line5 (Cargo detail).
    """
    css = """
<style>
.scrollx { overflow-x: auto; width: 100%; }
table.dc2 { border-collapse: collapse; width: max-content; min-width: 100%; }
table.dc2 th, table.dc2 td {
  border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px;
  white-space: nowrap;
}
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
            key = ("HS", doc_key, "Brand") if c == "Brand" else ("HSx", doc_key, c)
            rows_html.append(td(key, val))
    rows_html.append("</tr>")

    # Cargo detail (merge ONLY group column)
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
                rows_html.append(td((f"Line{i+1}", doc_key, c), val))
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

    # Total
    rows_html.append("<tr>")
    rows_html.append('<td class="group"></td>')
    rows_html.append('<td class="item">Total</td>')
    for doc_key, _ in DOCS:
        cols = blocks[doc_key]["cols"]
        for c in cols:
            val = norm_spaces(str(blocks[doc_key]["total_row"].get(c, "")))
            rows_html.append(td(("Total", doc_key, c), val))
    rows_html.append("</tr>")

    # Total check
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
        [css, '<div class="scrollx">', '<table class="dc2">', "<thead>", "".join(r1), "".join(r2), "</thead>", "<tbody>"]
        + rows_html
        + ["</tbody></table></div>"]
    )
    return html


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Single upload (multiple PDFs) → auto-detect BL/Invoice/Packing/BC 1.6 → extract & compare. Horizontal scroll for readability.")

top_left, top_right = st.columns([3.5, 1.5], gap="small")

with top_left:
    with st.expander("Debug / Status", expanded=False):
        key = get_api_key()
        st.write("API key loaded:", "✅ Yes" if key.startswith("sk-") else "❌ No (OPENAI_API_KEY missing)")
        st.write("Model:", MODEL)
        st.write("Auto-detect rules:", "BL/SWB, Invoice, Packing List, Customs BC 1.6 (page keywords + continuation)")

with top_right:
    with st.expander("📤 Upload (single) — PDFs (merged OK)", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files (merged PDFs are OK).",
            type=["pdf"],
            accept_multiple_files=True,
            key="u_multi",
        )
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
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Classifying pages → grouping by doc type → calling OpenAI (only for detected doc types)..."):
                grouped_text, assignments = split_and_group_pages(uploaded_files)
                fp_all = fingerprint_from_inputs(uploaded_files, assignments)

                for doc_key, doc_label in DOCS:
                    text = grouped_text.get(doc_key, "")
                    if not text:
                        continue
                    try:
                        obj = extract_cached(doc_key, doc_label, f"{fp_all}:{doc_key}", text)
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

# TIP rule needs Shipment(Bc1.6) shipper/cargo_owner
bc16_shipper = ""
bc16_cargo_owner = ""
try:
    bc16_shipper = norm_spaces(str(extracted.get("CUSTOMS_BC16", {}).get("shipment", {}).get("shipper", "")))
    bc16_cargo_owner = norm_spaces(str(extracted.get("CUSTOMS_BC16", {}).get("shipment", {}).get("cargo_owner", "")))
except Exception:
    pass

cargo_flags = compute_cargo_flags(cargo_blocks, bc16_shipper, bc16_cargo_owner)

st.divider()
k1, k2, k3, k4 = st.columns(4)

docs_uploaded_count = len(uploaded_files) if uploaded_files else 0
ship_mis_count = sum(1 for (rk, _dl), v in ship_flags.items() if v and rk != "document_number")

k1.metric("Shipment mismatches", ship_mis_count)
k2.metric(
    "HS mismatch (6-digit)",
    "YES" if any(cargo_flags.get(("HS", dk, "Brand"), False) for dk, _ in DOCS) else "NO",
)
k3.metric("PDF files uploaded", docs_uploaded_count)
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
    "Rules applied: Doc number mismatch ignored. "
    "Shipment compare ignores punctuation differences (.,-_ etc). "
    "POL compares city only. POD compares city only and treats Jakarta as Tanjung Priok. "
    "Vessel/Voy ignores punctuation; KMTC HOCHIMINH + 2510S is treated as equivalent. "
    "Container type/amount shows quantity only (container number => 1); value '0' is displayed as blank. "
    "HS comparison ignores punctuation and matches first 6 digits. "
    "KG is converted to MT. "
    "Total row shows document totals as-is. "
    "Total check compares sum(line items) vs document totals; ok=black, no=red. "
    "Cargo line (Line1~5) cell mismatch is highlighted in red. "
    "TIP in Customs BC 1.6 Code is highlighted red ONLY when (BC1.6 Cargo owner == BC1.6 Shipper) and Cargo owner is not blank."
)