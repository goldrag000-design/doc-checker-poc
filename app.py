import os
import re
import json
import time

import certifi
import httpx
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI


FIELDS = [
    "shipper",
    "consignee",
    "cargo_name",
    "hs_code",
    "package_type",
    "quantity",
    "container_types_and_qty",
    "gross_weight",
    "net_weight",
    "cargo_value",
]


def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts)


def norm(v: str) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v)
    v2 = re.sub(r"[.\s]", "", v)
    if v2.isdigit() and 4 <= len(v2) <= 12:
        return v2
    return v


def call_openai_once(model: str, prompt: str) -> str:
    """
    ✅ 안정성 우선:
    - 매 요청마다 새 httpx.Client 생성 -> Streamlit rerun/keep-alive 꼬임 회피
    """
    with httpx.Client(
        verify=certifi.where(),
        timeout=httpx.Timeout(connect=20.0, read=60.0, write=60.0, pool=60.0),
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
    ) as http_client:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=http_client,
        )
        r = client.responses.create(model=model, input=prompt)
        return (r.output_text or "").strip()


def ping_openai() -> str:
    return call_openai_once("gpt-4.1-mini", "Reply with exactly: PONG")


def extract_fields_with_openai(doc_name: str, text: str) -> dict:
    text = text[:6000]  # payload 줄이기

    prompt = f"""
Return JSON only. Keys must be exactly:
{FIELDS}

Rules:
- If not found, use "".
- hs_code: digits only (remove dots/spaces).
- container_types_and_qty: like "20GP x1; 40HC x2".
- Keep values short. No explanations.

DOCUMENT: {doc_name}
TEXT:
{text}
"""

    last_err = None
    for attempt in range(1, 5):  # 1s,2s,4s,8s
        try:
            raw = call_openai_once("gpt-4.1-mini", prompt)
            parsed = json.loads(raw)
            data = {k: "" for k in FIELDS}
            for k in FIELDS:
                data[k] = norm(str(parsed.get(k, "")))
            return data
        except Exception as e:
            last_err = e
            time.sleep(2 ** (attempt - 1))

    data = {k: "" for k in FIELDS}
    data["cargo_name"] = f"[API_CONNECTION_ERROR] {type(last_err).__name__}: {str(last_err)[:200]}"
    return data


# ---------------- UI ----------------
st.set_page_config(page_title="Doc Checker PoC", layout="wide")
st.title("Doc Checker PoC - Step 1 (Stable single doc extraction)")

with st.expander("Debug / Status", expanded=True):
    key = os.getenv("OPENAI_API_KEY") or ""
    st.write("API key loaded:", "✅ Yes" if key.startswith("sk-") else "❌ No (OPENAI_API_KEY missing)")
    if st.button("Ping OpenAI (should return PONG)"):
        try:
            pong = ping_openai()
            st.success(f"Ping result: {pong}")
        except Exception as e:
            st.error(f"Ping failed: {type(e).__name__}: {str(e)[:300]}")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Upload 1 PDF")
    f = st.file_uploader("Drag & drop", type=["pdf"], accept_multiple_files=False)
    show_text = st.checkbox("Show extracted text preview", value=True)
    run = st.button("Extract fields")

with right:
    st.subheader("Extracted Fields")
    if not f:
        st.info("Upload 1 PDF on the left.")
    else:
        text = pdf_to_text(f)
        if show_text:
            st.text_area("Preview (first 2000 chars)", value=text[:2000], height=220)

        if run:
            with st.spinner("Calling OpenAI (stable mode)..."):
                data = extract_fields_with_openai(f.name, text)
            st.json(data)