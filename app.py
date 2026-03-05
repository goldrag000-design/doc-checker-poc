import os
import re
import json
import time
import csv
import hashlib
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter

import certifi
import httpx
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =========================
# App config
# =========================
APP_TITLE = "DOC checker for PLB incoming"
SUBTITLE_TEXT = "(PoC_ver.1_developed by Brad Ha)"
SUBTITLE_COLOR = "#1d4ed8"  # blue
LOGO_PATH = "logo.png"

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

# Cargo Summary (UI only): 8 fields + BC 1.6 code row
CARGO_SUMMARY_ROWS = [
    ("cargo_name", "Cargo name"),
    ("hs6", "HS code (6-digit)"),
    ("packing_type", "Packing type"),
    ("total_bundles", "Total bundles"),
    ("total_pcs", "Total quantity (PCS)"),
    ("gross_mt", "Total Gross WT (MT)"),
    ("net_mt", "Total Net WT (MT)"),
    ("measurement_cbm", "Measurement (CBM)"),
    ("bc16_code", "Customs (BC 1.6) code"),
]

MAX_TEXT_CHARS = 12000
HS_RULES_CSV = "hs_rules.csv"


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
    """KOR: 업로드 파일 내용 + (파일명/페이지/타입) 배정을 섞어서 캐시용 fingerprint 생성"""
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
    """
    KOR:
    - KG면 /1000 해서 MT로 통일
    - 소수점 3자리까지 표준화 표시 (뒤 0 제거)
    """
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


def round3_str(num: Optional[float]) -> str:
    if num is None:
        return ""
    return f"{round(num, 3):.3f}".rstrip("0").rstrip(".")


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
    - "0"은 표시하지 않고 빈칸 반환
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


def norm_text_strict_key(v: str) -> str:
    """
    KOR:
    Cargo name 비교용: 소문자 + 기호 제거 + 중복공백 제거
    (철자는 느슨하게 처리하지 않음 = 이 정규화 후 문자열이 1글자라도 다르면 mismatch)
    """
    s = norm_spaces(v).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def packing_type_key(v: str) -> str:
    """
    KOR:
    Packing type 비교/표시 통일용 정규화:
    1) BC1.6 포함 아래 표현 모두 BUNDLE로:
       BUNDLE, BUNDLE (BE), BDL, BE, B/E  -> bundle
    2) 아래 표현 모두 PCS로:
       PCS, PC, PIECE, PIECES, PCE, PCS (PCE) -> pcs
    """
    s = norm_spaces(v).lower()

    # "1.0000 BUNDLE (BE)" 같은 경우 숫자 접두 제거
    s = re.sub(r"^\d+(\.\d+)?\s+", "", s)

    # 기호 제거 (괄호 등) + 공백 정리
    s = re.sub(r"[^\w\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Bundle 계열
    if ("bundle" in s) or (s in ("bdl", "be", "b/e")):
        return "bundle"

    # PCS 계열
    if ("pcs" in s) or (s in ("pc", "pce", "piece", "pieces")):
        return "pcs"

    return s


def int_only_str(v: str) -> str:
    """KOR: 숫자만 추출하여 정수로 표시"""
    f = to_float(v)
    if f is None:
        return ""
    return str(int(round(f)))


def cbm_key(v: str) -> str:
    """KOR: CBM 비교: 숫자 비교 + 소수점 셋째자리 반올림(표준화)"""
    f = to_float(v)
    if f is None:
        return ""
    return round3_str(f)


def mt_key(v: str) -> str:
    """KOR: MT 비교: KG면 /1000 후, 소수점 셋째자리 반올림(표준화)"""
    s = norm_spaces(v)
    if not s:
        return ""
    low = s.lower()
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", low)
    if not nums:
        return ""
    n = nums[0].replace(",", "")
    try:
        x = float(n)
    except Exception:
        return ""
    if "kg" in low:
        x = x / 1000.0
    return round3_str(x)


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

    # --- Invoice (typo tolerant: NVOICE) ---
    if "invoice" in t or "nvoice" in t:
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
- HS code: keep as written (we normalize later to 6-digit).
- Keep values short.

Shipment rules:
- container_20 and container_40 MUST be quantity ONLY (a number).
- If only a container number is shown, use 1.
"""

    if doc_key == "CUSTOMS_BC16":
        doc_rules = """
Document-specific rules (Customs BC 1.6):
- Source text is Indonesian; translate extracted values into English (company names stay as-is).
- IMPORTANT: cargo.totals.gross_wt must be the value from 'Berat Kotor' (usually item 31).
- For BC 1.6 line_items: fill hs_code per line if present; also fill line_items[].code with the value shown as 'Kode: TIP' or 'Kode: KON' (only TIP/KON).
- If a container field is not applicable, return "" (do NOT force 0).
"""
    elif doc_key == "PROFORMA_INVOICE":
        doc_rules = """
Document-specific rules (Invoice):
- If the document shows a single weight per brand/line (e.g., "WT"), treat it as NET WT.
  Put it into line_items[].net_wt (NOT gross_wt).
"""
    elif doc_key == "BL_SWB":
        doc_rules = """
Document-specific rules (BL/SWB):
- Extract measurement_cbm if shown (Measurement/CBM).
"""
    else:
        doc_rules = """
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
        # Document number differences do NOT trigger red highlight
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
# Cargo Summary (UI v2)
# =========================
def get_cargo(payload: dict) -> dict:
    cargo = payload.get("cargo", {}) if isinstance(payload, dict) else {}
    return cargo if isinstance(cargo, dict) else {}


def get_line_items(cargo: dict) -> List[dict]:
    li = cargo.get("line_items", []) if isinstance(cargo, dict) else []
    if not isinstance(li, list):
        return []
    out = []
    for it in li:
        if isinstance(it, dict):
            out.append(it)
    return out


def invoice_net_rule(gross: Optional[float], net: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Invoice rule: If only gross exists per line, treat it as net."""
    if gross is not None and net is None:
        return None, gross
    return gross, net


def sum_qty_by_packing(line_items: List[dict], target_pack_key: str) -> Optional[float]:
    """KOR: packing_type_key(packing) == target_pack_key 인 라인만 qty 합산"""
    s = 0.0
    has = False
    for it in line_items:
        p = packing_type_key(str(it.get("packing", "")))
        if not p or p != target_pack_key:
            continue
        q = to_float(str(it.get("qty", "")))
        if q is None:
            continue
        s += q
        has = True
    return s if has else None


def sum_weights_from_lines(doc_key: str, line_items: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (gross_mt_sum, net_mt_sum) as floats (MT).
    KOR:
    - gross/net 값이 KG면 /1000
    - Invoice는 gross만 있으면 net으로 이동
    """
    g_sum = 0.0
    n_sum = 0.0
    g_has = False
    n_has = False

    for it in line_items:
        g_raw = str(it.get("gross_wt", ""))
        n_raw = str(it.get("net_wt", ""))

        g_num = None
        if norm_spaces(g_raw):
            gk = mt_key(g_raw)
            g_num = to_float(gk) if gk else None

        n_num = None
        if norm_spaces(n_raw):
            nk = mt_key(n_raw)
            n_num = to_float(nk) if nk else None

        if doc_key == "PROFORMA_INVOICE":
            g_num, n_num = invoice_net_rule(g_num, n_num)

        if g_num is not None:
            g_sum += g_num
            g_has = True
        if n_num is not None:
            n_sum += n_num
            n_has = True

    return (g_sum if g_has else None, n_sum if n_has else None)


def dominant_packing_type(line_items: List[dict]) -> str:
    """KOR: packing type 후보를 normalize해서 가장 많이 나온 값을 선택"""
    keys = []
    for it in line_items:
        pk = packing_type_key(str(it.get("packing", "")))
        if pk:
            keys.append(pk)
    if not keys:
        return ""
    c = Counter(keys)
    return c.most_common(1)[0][0]


def dominant_bc16_code(line_items: List[dict]) -> str:
    """
    KOR:
    - BC 1.6 line_items[].code 에서 TIP/KON을 수집
    - 가장 많이 나온 값을 선택
    - TIP/KON 외 값은 무시
    """
    keys = []
    for it in line_items:
        c = norm_spaces(str(it.get("code", ""))).upper()
        if c in ("TIP", "KON"):
            keys.append(c)
    if not keys:
        return ""
    return Counter(keys).most_common(1)[0][0]


def build_cargo_summary_matrix(extracted: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str, Dict[str, str]]]]:
    headers = ["ITEM"] + [label for _, label in DOCS]
    rows: List[Tuple[str, str, Dict[str, str]]] = []

    for row_key, row_label in CARGO_SUMMARY_ROWS:
        row_map: Dict[str, str] = {}

        for doc_key, doc_label in DOCS:
            payload = extracted.get(doc_key, {})
            cargo = get_cargo(payload)
            line_items = get_line_items(cargo)

            cargo_name = norm_spaces(str(cargo.get("cargo_name", "")))
            hs = norm_spaces(str(cargo.get("hs_code", "")))
            meas = norm_spaces(str(cargo.get("measurement_cbm", "")))

            # 기존 로직 유지: BL/SWB만 measurement 보이도록
            if doc_key != "BL_SWB":
                meas = ""

            totals = cargo.get("totals", {}) if isinstance(cargo, dict) else {}
            totals = totals if isinstance(totals, dict) else {}

            totals_qty_raw = norm_spaces(str(totals.get("qty", "")))
            totals_g_raw = norm_spaces(str(totals.get("gross_wt", "")))
            totals_n_raw = norm_spaces(str(totals.get("net_wt", "")))

            # packing type
            pack_dom = dominant_packing_type(line_items)  # normalized key
            pack_disp = pack_dom.upper() if pack_dom else ""

            # BC 1.6 code (TIP/KON) — only show on BC 1.6 column
            bc16_code = dominant_bc16_code(line_items) if doc_key == "CUSTOMS_BC16" else ""

            # total bundles / pcs (line_items 기반)
            bundles_sum = sum_qty_by_packing(line_items, "bundle")
            pcs_sum = sum_qty_by_packing(line_items, "pcs")  # ✅ pc → pcs (정규화와 일치)

            bundles_disp = ""
            if bundles_sum is not None:
                bundles_disp = str(int(round(bundles_sum)))
            elif totals_qty_raw:
                bundles_disp = int_only_str(totals_qty_raw)

            pcs_disp = ""
            if pcs_sum is not None:
                pcs_disp = str(int(round(pcs_sum)))

            # weights: totals 우선, 없으면 line_items 합계
            gross_disp = to_mt(totals_g_raw) if totals_g_raw else ""
            net_disp = to_mt(totals_n_raw) if totals_n_raw else ""

            if doc_key == "PROFORMA_INVOICE":
                if gross_disp and not net_disp:
                    net_disp = gross_disp
                    gross_disp = ""

            if (not gross_disp) and (not net_disp):
                g_sum, n_sum = sum_weights_from_lines(doc_key, line_items)
                gross_disp = round3_str(g_sum) if g_sum is not None and doc_key != "PROFORMA_INVOICE" else ""
                net_disp = round3_str(n_sum) if n_sum is not None else ""

            meas_disp = ""
            if meas:
                f = to_float(meas)
                meas_disp = round3_str(f) if f is not None else meas

            if row_key == "cargo_name":
                val = cargo_name
            elif row_key == "hs6":
                val = hs
            elif row_key == "packing_type":
                val = pack_disp
            elif row_key == "total_bundles":
                val = bundles_disp
            elif row_key == "total_pcs":
                val = pcs_disp
            elif row_key == "gross_mt":
                val = gross_disp
            elif row_key == "net_mt":
                val = net_disp
            elif row_key == "measurement_cbm":
                val = meas_disp
            elif row_key == "bc16_code":
                val = bc16_code if doc_key == "CUSTOMS_BC16" else ""
            else:
                val = ""

            row_map[doc_label] = norm_spaces(val)

        rows.append((row_key, row_label, row_map))

    return headers, rows


def cargo_value_for_compare(row_key: str, display_value: str) -> str:
    v = norm_spaces(display_value)
    if not v:
        return ""

    if row_key == "cargo_name":
        return norm_text_strict_key(v)

    if row_key == "hs6":
        return hs6(v)

    if row_key == "packing_type":
        return packing_type_key(v)

    if row_key in ("total_bundles", "total_pcs"):
        return int_only_str(v)

    if row_key in ("gross_mt", "net_mt"):
        return mt_key(v)

    if row_key == "measurement_cbm":
        return cbm_key(v)

    if row_key == "bc16_code":
        return ""

    return norm_text_strict_key(v)


def compute_bc16_party_same(extracted: Dict[str, Any]) -> Tuple[bool, str, str]:
    bc16 = extracted.get("CUSTOMS_BC16", {}) if isinstance(extracted, dict) else {}
    ship = bc16.get("shipment", {}) if isinstance(bc16, dict) else {}
    shipper = norm_spaces(str(ship.get("shipper", "")))
    owner = norm_spaces(str(ship.get("cargo_owner", "")))

    sk = loose_text_key(shipper)
    ok = loose_text_key(owner)
    same = bool(sk) and (sk == ok)
    return same, shipper, owner


def compute_cargo_summary_mismatch_flags(
    rows: List[Tuple[str, str, Dict[str, str]]],
    extracted: Dict[str, Any],
) -> Dict[Tuple[str, str], bool]:
    flags: Dict[Tuple[str, str], bool] = {}
    bc16_same, _, _ = compute_bc16_party_same(extracted)
    bc16_label = dict(DOCS).get("CUSTOMS_BC16", "Customs BC 1.6")

    for row_key, _row_label, row_map in rows:
        if row_key == "bc16_code":
            for _, doc_label in DOCS:
                flags[(row_key, doc_label)] = False

            code_disp = norm_spaces(row_map.get(bc16_label, "")).upper()

            if code_disp == "TIP":
                flags[(row_key, bc16_label)] = bool(code_disp) and bc16_same
            elif code_disp == "KON":
                flags[(row_key, bc16_label)] = bool(code_disp) and (not bc16_same)
            else:
                flags[(row_key, bc16_label)] = False

            continue

        comps = []
        for _, doc_label in DOCS:
            comps.append(cargo_value_for_compare(row_key, row_map.get(doc_label, "")))

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
# HS Rules Lookup (hs_rules.csv)
# =========================
@st.cache_data(show_spinner=False)
def load_hs_rules(path: str) -> Dict[str, Dict[str, str]]:
    """
    Robust CSV loader:
    - Auto-detect delimiter: , ; \t |
    - Normalizes headers (strip/lower)
    - Works with your headers:
        HS Code, Duty, VAT / WHT, Restriction (import), Restriction (export)
    """
    rules: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        return rules

    def digits(x: Any) -> str:
        return re.sub(r"\D", "", str(x or ""))

    def hs6_from_value(v: Any) -> str:
        s = digits(v)
        return s[:6] if len(s) >= 6 else ""

    def norm_header(h: str) -> str:
        return re.sub(r"\s+", " ", (h or "")).strip().lower()

    def get_by_keys(row: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            if k in row and row[k] is not None:
                return str(row[k])
        return ""

    raw_text = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                raw_text = f.read()
            break
        except Exception:
            continue

    if not raw_text:
        return rules

    sample = raw_text[:5000]
    delim = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        delim = dialect.delimiter
    except Exception:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {d: first_line.count(d) for d in [",", ";", "\t", "|"]}
        delim = max(counts, key=counts.get) if counts else ","

    reader = csv.reader(raw_text.splitlines(), delimiter=delim)
    rows = list(reader)
    if not rows:
        return rules

    headers_raw = rows[0]
    headers = [norm_header(h) for h in headers_raw]

    for r in rows[1:]:
        if not any(str(x).strip() for x in r):
            continue
        row = {}
        for i, h in enumerate(headers):
            row[h] = r[i] if i < len(r) else ""

        hs = hs6_from_value(get_by_keys(row, ["hs6", "hs code", "hs_code", "hs"]))
        if not hs:
            joined = " ".join(str(v) for v in row.values())
            m = re.search(r"\b(\d{6})\b", joined)
            hs = m.group(1) if m else ""
        if not hs:
            continue

        duty = norm_spaces(get_by_keys(row, ["duty"]))
        vat_wht = norm_spaces(get_by_keys(row, ["vat/wht", "vat / wht", "vat_wht"]))

        r_import = norm_spaces(get_by_keys(row, ["restriction (import)", "restriction import", "restriction"]))
        r_export = norm_spaces(get_by_keys(row, ["restriction (export)", "restriction export"]))

        rules[hs] = {
            "duty": duty,
            "vat_wht": vat_wht,
            "restriction_import": r_import,
            "restriction_export": r_export,
        }

    return rules


def build_tariff_rows_from_cargo_summary(
    cargo_rows: List[Tuple[str, str, Dict[str, str]]],
    hs_rules: Dict[str, Dict[str, str]],
) -> Tuple[bool, List[Dict[str, str]]]:
    hs_row_map: Dict[str, str] = {}
    for row_key, _label, row_map in cargo_rows:
        if row_key == "hs6":
            hs_row_map = row_map
            break

    doc_hs: List[Tuple[str, str]] = []
    for _, doc_label in DOCS:
        raw = norm_spaces(hs_row_map.get(doc_label, ""))
        h = hs6(raw)
        if h:
            doc_hs.append((doc_label, h))

    if not doc_hs:
        return True, []

    distinct = sorted({h for _, h in doc_hs})
    is_single = len(distinct) == 1

    def lookup(h: str) -> Dict[str, str]:
        r = hs_rules.get(h, {})
        if not r:
            return {
                "duty": "",
                "vat_wht": "",
                "restriction_import": "",
                "restriction_export": "",
                "status": "NOT_FOUND",
            }
        return {
            "duty": r.get("duty", ""),
            "vat_wht": r.get("vat_wht", ""),
            "restriction_import": r.get("restriction_import", ""),
            "restriction_export": r.get("restriction_export", ""),
            "status": "OK",
        }

    out: List[Dict[str, str]] = []
    if is_single:
        h = distinct[0]
        r = lookup(h)
        out.append(
            {
                "document": "All documents",
                "hs6": h,
                "duty": r["duty"],
                "vat_wht": r["vat_wht"],
                "restriction_import": r["restriction_import"],
                "restriction_export": r["restriction_export"],
                "status": r["status"],
            }
        )
        return True, out

    for doc_label, h in doc_hs:
        r = lookup(h)
        out.append(
            {
                "document": doc_label,
                "hs6": h,
                "duty": r["duty"],
                "vat_wht": r["vat_wht"],
                "restriction_import": r["restriction_import"],
                "restriction_export": r["restriction_export"],
                "status": r["status"],
            }
        )
    return False, out


# =========================
# HTML rendering (shared)
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


def render_tariff_table_html(is_single: bool, rows: List[Dict[str, str]]) -> str:
    css = """
<style>
.tariff-wrap { overflow-x: auto; width: 100%; }
table.tariff { border-collapse: collapse; width: max-content; min-width: 100%; }
table.tariff th, table.tariff td {
  border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px;
}
table.tariff th { background: #f8fafc; font-weight: 600; white-space: nowrap; }
.muted { color: #6b7280; }
.warn { color: #b45309; font-weight: 700; }
</style>
"""
    if not rows:
        return css + "<div class='muted'>No HS code (6-digit) found in Cargo Summary.</div>"

    cols = ["HS Code", "Duty", "VAT / WHT", "Restriction (import)", "Restriction (export)"]
    if not is_single:
        cols = ["Document"] + cols

    h = [css, "<div class='tariff-wrap'>", "<table class='tariff'>", "<thead><tr>"]
    for c in cols:
        h.append(f"<th>{html_escape(c)}</th>")
    h.append("</tr></thead><tbody>")

    for r in rows:
        h.append("<tr>")
        if not is_single:
            h.append(f"<td>{html_escape(r.get('document',''))}</td>")
        h.append(f"<td>{html_escape(r.get('hs6',''))}</td>")

        status = r.get("status", "OK")
        if status == "NOT_FOUND":
            h.append("<td class='warn'>Not found</td>")
            h.append("<td class='warn'>Not found</td>")
            h.append("<td class='warn'>Not found</td>")
            h.append("<td class='warn'>Not found</td>")
        else:
            h.append(f"<td>{html_escape(r.get('duty',''))}</td>")
            h.append(f"<td>{html_escape(r.get('vat_wht',''))}</td>")
            h.append(f"<td>{html_escape(r.get('restriction_import',''))}</td>")
            h.append(f"<td>{html_escape(r.get('restriction_export',''))}</td>")

        h.append("</tr>")

    h.append("</tbody></table></div>")
    return "\n".join(h)


def render_simple_compare_html(headers, rows, flags) -> str:
    css = """
<style>
.scrollx { overflow-x: auto; width: 100%; }
table.dc { border-collapse: collapse; width: max-content; min-width: 100%; }
table.dc th, table.dc td {
  border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px;
  white-space: nowrap;
}
table.dc th { background: #f8fafc; font-weight: 600; }
table.dc td.item { width: 240px; background: #fafafa; font-weight: 600; }
.red { color: #dc2626; font-weight: 800; }
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


# =========================
# UI (Final layout + Professional header)
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Professional header styles
st.markdown(
    f"""
    <style>
      .app-header {{
        display:flex;
        align-items:center;
        gap:14px;
        padding:10px 6px 12px 6px;
        border-bottom:1px solid #e5e7eb;
        margin-bottom:10px;
      }}
      .app-header img {{
        height:48px;
        width:auto;
      }}
      .app-title {{
        display:flex;
        align-items:baseline;
        gap:10px;
        flex-wrap:wrap;
      }}
      .app-title .t {{
        font-size:28px;
        font-weight:800;
        letter-spacing:0.2px;
        margin:0;
        line-height:1.15;
        color:#111827;
      }}
      .app-title .s {{
        font-size:22px;
        font-weight:600;
        color:{SUBTITLE_COLOR};
        margin:0;
        line-height:1.15;
        white-space:nowrap;
      }}
      .app-subcap {{
        color:#6b7280;
        font-size:13px;
        margin:6px 0 12px 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header bar (logo + title)
logo_html = ""
if os.path.exists(LOGO_PATH):
    import base64

    with open(LOGO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f"<img src='data:image/png;base64,{b64}' />"

st.markdown(
    f"""
    <div class="app-header">
      {logo_html}
      <div class="app-title">
        <div class="t">{html_escape(APP_TITLE)}</div>
        <div class="s">{html_escape(SUBTITLE_TEXT)}</div>
      </div>
    </div>
    <div class="app-subcap">
      Upload PDFs (merged OK) → auto-detect BL/Invoice/Packing/BC 1.6 → extract & compare.
      Cargo Information is Cargo Summary (8 fields) + BC 1.6 code.
    </div>
    """,
    unsafe_allow_html=True,
)

# Default extracted dict (always defined)
extracted: Dict[str, Any] = {doc_key: {"shipment": {}, "cargo": {}} for doc_key, _ in DOCS}
errors: List[str] = []
uploaded_files = None
run = False

# KPI row (top dashboard)
k1, k2, k3, k4 = st.columns(4)
# initialize KPI with current empty state
k1.metric("Shipment mismatches", 0)
k2.metric("Cargo summary mismatches", 0)
k3.metric("PDF files uploaded", 0)
k4.metric("Extraction errors", 0)

# Upload + Debug row (final layout)
upload_col, debug_col = st.columns([3.2, 1.0], gap="medium")

with upload_col:
    with st.expander("📤 Upload PDFs (merged OK)", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files (merged PDFs are OK).",
            type=["pdf"],
            accept_multiple_files=True,
            key="u_multi",
        )
        run = st.button("Extract & Compare", use_container_width=True)

with debug_col:
    with st.expander("Debug / Status", expanded=False):
        key = get_api_key()
        st.write("API key loaded:", "✅ Yes" if key.startswith("sk-") else "❌ No (OPENAI_API_KEY missing)")
        st.write("Model:", MODEL)
        st.write("HS rules file:", HS_RULES_CSV, "✅" if os.path.exists(HS_RULES_CSV) else "❌ Missing")
        st.write("Auto-detect rules:", "BL/SWB, Invoice, Packing List, Customs BC 1.6 (page keywords + continuation)")

# Run extraction
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

# Build tables
ship_headers, ship_rows = build_shipment_matrix(extracted)
ship_flags = compute_shipment_mismatch_flags(ship_rows)

cargo_headers, cargo_rows = build_cargo_summary_matrix(extracted)
cargo_flags = compute_cargo_summary_mismatch_flags(cargo_rows, extracted)

# HS tariff lookup (Option A UI below cargo summary; mismatch policy: per-doc rows)
hs_rules = load_hs_rules(HS_RULES_CSV)
tariff_is_single, tariff_rows = build_tariff_rows_from_cargo_summary(cargo_rows, hs_rules)

# Update KPI row after computation
docs_uploaded_count = len(uploaded_files) if uploaded_files else 0
ship_mis_count = sum(1 for (rk, _dl), v in ship_flags.items() if v and rk != "document_number")
cargo_mis_count = sum(1 for (_rk, _dl), v in cargo_flags.items() if v)

# Re-render KPI with final values (Streamlit runs top-to-bottom; so show a new KPI row)
st.markdown("---")
k1b, k2b, k3b, k4b = st.columns(4)
k1b.metric("Shipment mismatches", ship_mis_count)
k2b.metric("Cargo summary mismatches", cargo_mis_count)
k3b.metric("PDF files uploaded", docs_uploaded_count)
k4b.metric("Extraction errors", len(errors))

if errors:
    with st.expander("⚠️ Extraction errors (details)", expanded=False):
        for e in errors:
            st.error(e)

st.subheader("Shipment Information")
st.markdown(render_simple_compare_html(ship_headers, ship_rows, ship_flags), unsafe_allow_html=True)

st.subheader("Cargo Information (Summary)")
st.markdown(render_simple_compare_html(cargo_headers, cargo_rows, cargo_flags), unsafe_allow_html=True)

st.subheader("Tariff & Restrictions (from HS 6-digit)")
if not os.path.exists(HS_RULES_CSV):
    st.warning(f"Missing lookup file: {HS_RULES_CSV}. Place it in the project folder.")
else:
    st.markdown(render_tariff_table_html(tariff_is_single, tariff_rows), unsafe_allow_html=True)

st.divider()
st.caption(
    "Rules applied: Doc number mismatch ignored. "
    "Shipment compare ignores punctuation differences (.,-_ etc). "
    "POL compares city only. POD compares city only and treats Jakarta as Tanjung Priok. "
    "Vessel/Voy ignores punctuation; KMTC HOCHIMINH + 2510S is treated as equivalent. "
    "Container type/amount shows quantity only (container number => 1); value '0' is displayed as blank. "
    "Cargo Summary compare: empty values ignored; compare only when 2+ values exist; "
    "Cargo name uses strict text normalization (lower + symbol/extra spaces removed); "
    "HS compares first 6 digits (numbers only); "
    "Packing type normalizes BC1.6 codes (BE/BDL/B/E/BUNDLE(BE)→BUNDLE) and PCS variants (PC/PCE/PIECE→PCS); "
    "Totals (bundles/pcs) compare as integers; "
    "Gross/Net/CBM compare after unit normalization and rounding to 3 decimals. "
    "Customs (BC 1.6) code: show TIP/KON only on BC 1.6 column; "
    "TIP is red when Shipper == Cargo owner; KON is red when Shipper != Cargo owner."
)