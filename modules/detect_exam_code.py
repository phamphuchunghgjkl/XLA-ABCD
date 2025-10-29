# -*- coding: utf-8 -*-
"""
Detect exam code (Mã đề / Đề) ANYWHERE on the page (no fixed ROI).

Usage:
    python -m modules.detect_exam_code --img "data/samples/0001.jpg" --show
"""

import cv2
import re
import numpy as np
import pytesseract
import unicodedata
import argparse
from pytesseract import Output
from typing import Tuple, Optional, Dict, List

# ===== (0) Chỉ định tesseract.exe nếu cần (Windows) =====
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt để regex ổn định (ma de -> mã đề)."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()

def merge_boxes(boxes: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
    xs = [x for x, y, w, h in boxes]
    ys = [y for x, y, w, h in boxes]
    x2s = [x + w for x, y, w, h in boxes]
    y2s = [y + h for x, y, w, h in boxes]
    x1, y1, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)
    return (x1, y1, x2 - x1, y2 - y1)

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Tiền xử lý nhẹ (phù hợp ảnh chụp đa dạng)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # cân sáng cục bộ tránh bóng
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # làm mượt nhưng giữ biên chữ
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
    return gray

# --------------------------------------------------------
# Core: tìm theo NGỮ CẢNH DÒNG (ổn định nhất)
# --------------------------------------------------------
# các biến thể từ khóa (đã bỏ dấu khi so sánh)
KEYS = (
    "ma de", "ma de:", "ma de-", "ma de.", "made",   # mã đề
    "de", "de:", "de-",                               # đề
    "ma de thi", "ma de bai thi", "ma de so",         # một số form khác
)
# mã đề cho phép: 1-8 ký tự A-Z hoặc số (ví dụ: 0122, ABC, 1A2B)
CODE_RE = re.compile(r"^[A-Z0-9]{1,8}$")

def _group_lines(data: Dict) -> Dict:
    """Gom indices theo (block, paragraph, line)."""
    n = len(data["text"])
    lines = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    # sort words theo word_num
    for k in list(lines.keys()):
        lines[k] = sorted(lines[k], key=lambda j: data["word_num"][j])
    return lines

def _normalize_token(tok: str) -> str:
    """Chuẩn hóa 1 token để kiểm tra mã đề: bỏ kí tự thừa, upper-case."""
    t = tok.strip().strip(":").strip(".").strip(")").strip("(").replace("–","-")
    t = t.replace("O", "0")  # O -> 0 (hay nhầm)
    return t.upper()

def find_exam_code_by_lines(gray: np.ndarray) -> Tuple[Optional[str], Optional[Tuple[int,int,int,int]], Dict]:
    """
    Quét toàn ảnh theo dòng, tìm '... mã đề/đề ... <CODE>'.
    Trả về (code, bbox, info).
    """
    data = pytesseract.image_to_data(
        gray,
        lang="vie+eng",
        config="--oem 1 --psm 6 -c preserve_interword_spaces=1",
        output_type=Output.DICT,
    )

    lines = _group_lines(data)
    best = {"code": None, "bbox": None, "conf": -1.0, "source": "line-context"}

    for key, idxs in lines.items():
        words = [data["text"][j].strip() for j in idxs]
        boxes = [(data["left"][j], data["top"][j], data["width"][j], data["height"][j]) for j in idxs]
        confs = [float(data["conf"][j]) if str(data["conf"][j]).replace('.','',1).isdigit() else -1 for j in idxs]

        line_raw = " ".join(words)
        line_norm = strip_accents(line_raw)

        # chỉ xét dòng có chứa từ khóa
        if not any(k in line_norm for k in KEYS):
            continue

        # tìm vị trí token sau "mã đề" hoặc sau "đề"
        # duyệt sliding window 1..3 token để cover các dấu : . -
        target_indices = []
        for j in range(len(words)):
            tk_j = strip_accents(words[j])
            # 'đề' đơn
            if tk_j in ("de", "de:", "de-", "de."):
                target_indices.append(j + 1)
            # 'mã đề' hai token
            if j+1 < len(words):
                pair = strip_accents(words[j] + " " + words[j+1])
                if "ma de" in pair:
                    target_indices.append(j + 2)

        # duyệt max 2 token sau vị trí tìm được để bắt code
        for idx_target in target_indices:
            for k in range(idx_target, min(idx_target + 3, len(words))):
                cand = _normalize_token(words[k])
                if CODE_RE.match(cand):
                    # tính bbox gộp quanh cụm chứa từ khóa + code (để debug hiển thị)
                    a = max(0, idx_target-2)
                    bbox = merge_boxes(boxes[a:k+1])

                    # confidence = trung bình các conf trong đoạn
                    conf_seg = [c for c in confs[a:k+1] if c >= 0]
                    conf_mean = float(np.mean(conf_seg)) if conf_seg else -1.0

                    if conf_mean > best["conf"]:
                        best = {"code": cand, "bbox": bbox, "conf": conf_mean, "source": "line-context"}
                    break

    return best["code"], best["bbox"], best

# --------------------------------------------------------
# Fallback: regex toàn ảnh (không có bbox chính xác)
# --------------------------------------------------------
FALLBACK_RE = re.compile(r"(?:ma\s*de|de)\s*[:\-\.\)]*\s*([A-Z0-9]{1,8})", re.IGNORECASE)

def fallback_regex(gray: np.ndarray):
    text = pytesseract.image_to_string(gray, lang="vie+eng", config="--oem 1 --psm 6")
    text_norm = strip_accents(text).upper()
    m = FALLBACK_RE.search(text_norm)
    if not m:
        return None, None, {"source": "global-regex", "raw": text[:200]}
    code = m.group(1).upper()
    return code, None, {"source": "global-regex", "raw": text[:200]}

# --------------------------------------------------------
# Public API
# --------------------------------------------------------
def read_exam_code_from_image(image_path: str) -> Dict:
    """
    Return:
      {
        'exam_code': '0122' | 'ABC' | None,
        'bbox': (x,y,w,h) | None,
        'source': 'line-context' | 'global-regex' | 'none',
        'confidence': float
      }
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)

    # scale về ~2200px chiều cao cho OCR ổn định
    h, w = bgr.shape[:2]
    if h < 1800:
        scale = 1800.0 / h
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    gray = preprocess_for_ocr(bgr)

    code, bbox, info = find_exam_code_by_lines(gray)
    if code is None:
        code, bbox, info2 = fallback_regex(gray)
        if code is None:
            return {"exam_code": None, "bbox": None, "source": "none", "confidence": -1}
        return {"exam_code": code, "bbox": bbox, "source": info2["source"], "confidence": 0.0}

    return {"exam_code": code, "bbox": bbox, "source": info["source"], "confidence": float(info["conf"])}

# --------------------------------------------------------
# CLI
# --------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to input image")
    ap.add_argument("--show", action="store_true", help="Show debug box if found")
    args = ap.parse_args()

    res = read_exam_code_from_image(args.img)
    print(f"[Exam code] -> {res['exam_code']}  (source={res['source']}, conf={res['confidence']:.1f})")

    if args.show and res["exam_code"] and res["bbox"]:
        bgr = cv2.imread(args.img)
        x,y,w,h = res["bbox"]
        cv2.rectangle(bgr, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(bgr, f"MA DE: {res['exam_code']}", (x, max(0,y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("exam-code", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
