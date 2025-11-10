# -*- coding: utf-8 -*-
import os, datetime
import streamlit as st
import cv2, numpy as np

from modules.preprocessing import preprocess_document, save_outputs
from modules.deepseek_md import image_to_markdown

st.set_page_config(page_title="Image → Markdown (DeepSeek 4-bit)", layout="wide")
st.title("🖼️➡️📝 Ảnh sang văn bản (Markdown) — DeepSeekOCR 4-bit")
st.caption("Tiền xử lý (rectify + B/W) ➜ DeepSeekOCR ➜ Lưu .txt/.md")

uploaded = st.file_uploader("Tải ảnh tài liệu (jpg/png)", type=["jpg","jpeg","png"])
c1, c2 = st.columns([1,1])

if uploaded:
    buf = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    with st.spinner("Tiền xử lý..."):
        pp = preprocess_document(img, bw_method="adaptive")

    with c1:
        st.subheader("Rectified")
        st.image(cv2.cvtColor(pp["warped_bgr"], cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        st.image(pp["bw"], caption="B/W", use_column_width=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", "output", ts)
    os.makedirs(out_dir, exist_ok=True)
    paths = save_outputs(pp, img, out_dir=out_dir, prefix="pre_")

    # Use B/W image for OCR as requested
    ocr_image = paths["bw"]

    with st.spinner("DeepSeekOCR đang xử lý..."):
        res = image_to_markdown(ocr_image, out_dir)

    md = res.get("markdown", "") or ""
    txt = res.get("text", "") or ""

    with c2:
        st.subheader("Markdown")
        if md.strip():
            st.code(md[:8000], language="markdown")
        elif txt.strip():
            st.code(txt[:8000], language="text")
        else:
            st.warning("Model không trả về nội dung.")

        # Save
        md_path = os.path.join(out_dir, "output.md")
        txt_path = os.path.join(out_dir, "output.txt")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md if md else txt)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt if txt else md)

        st.success("Đã lưu kết quả.")
        st.download_button("Tải output.md", data=(md if md else txt), file_name="output.md", mime="text/markdown")
        st.download_button("Tải output.txt", data=(txt if txt else md), file_name="output.txt", mime="text/plain")
else:
    st.info("⬆️ Tải ảnh để bắt đầu.")
