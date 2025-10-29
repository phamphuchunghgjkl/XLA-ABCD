# -*- coding: utf-8 -*-
"""
Preprocess a document photo:
- detect page quad (contour + Hough fallback)
- perspective warp to a canonical size
- output gray and binarized (black & white)

Usage (standalone):
    python -m modules.preprocessing --img data/samples/0001.jpg --show
"""

import cv2
import numpy as np
import argparse
import os
import json


# ==========================
# Helpers
# ==========================
def _order_pts(pts):
    """Sort 4 points to TL, TR, BR, BL (clockwise)."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _resize_for_speed(img, max_side=1400):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale


# ==========================
# Page quad detection (Tier 1: contour)
# ==========================
def _find_quad_by_contour(bgr_small):
    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # làm nổi phần giấy sáng so nền
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (31,31)))
    thr = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 71, 2)
    edges = cv2.Canny(thr, 60, 180)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # thử 0.02–0.04
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    # fallback trên contour: minAreaRect
    rect = cv2.minAreaRect(cnts[0])
    quad = cv2.boxPoints(rect).astype("float32")
    return quad


# ==========================
# Page quad detection (Tier 2: Hough lines)
# ==========================
def _intersection(l1, l2):
    x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
    A1,B1 = y2-y1, x1-x2; C1 = A1*x1 + B1*y1
    A2,B2 = y4-y3, x3-x4; C2 = A2*x3 + B2*y3
    det = A1*B2 - A2*B1
    if abs(det) < 1e-6: return None
    x = (B2*C1 - B1*C2)/det; y = (A1*C2 - A2*C1)/det
    return (x, y)

def _find_quad_by_hough(bgr_small):
    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=min(bgr_small.shape[:2])//3, maxLineGap=20)
    if lines is None: 
        return None

    angs = []
    for L in lines[:,0]:
        x1,y1,x2,y2 = L
        ang = (np.degrees(np.arctan2(y2-y1, x2-x1)) + 180) % 180
        angs.append((L, ang))
    horiz = [L for L,a in angs if a<30 or a>150]
    vert  = [L for L,a in angs if 60<a<120]
    if len(horiz) < 2 or len(vert) < 2:
        return None

    def extremes(group, axis):
        pts = []
        for x1,y1,x2,y2 in group:
            pts += [(min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2))]
        arr = np.array(pts)
        i_min = np.argmin(arr[:,axis]); i_max = np.argmax(arr[:,axis])
        return group[i_min//2], group[i_max//2]

    top, bottom = extremes(horiz, axis=1)   # theo y
    left, right = extremes(vert, axis=0)    # theo x

    TL = _intersection(top, left)
    TR = _intersection(top, right)
    BR = _intersection(bottom, right)
    BL = _intersection(bottom, left)
    if None in (TL,TR,BR,BL):
        return None
    return np.array([TL,TR,BR,BL], dtype="float32")


# ==========================
# Public: detect + warp + binarize
# ==========================
def rectify_sheet(image_bgr, out_size=(2480, 3508)):
    """
    Detect page quad and warp to (W,H) (mặc định gần A4 @~300dpi).
    Return: (warped_bgr, quad_src_scaled)
    Raise RuntimeError nếu không tìm được tứ giác.
    """
    small, scale = _resize_for_speed(image_bgr, max_side=1400)

    quad = _find_quad_by_contour(small)
    if quad is None:
        quad = _find_quad_by_hough(small)
    if quad is None:
        raise RuntimeError("Không tìm được 4 góc trang")

    # scale quad về hệ toạ độ ảnh gốc
    quad = quad / scale
    src = _order_pts(quad)
    W, H = out_size
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M   = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (W, H))
    return warped, src


def to_gray_and_bw(warped_bgr, method="adaptive"):
    """
    Trả về (gray, bw) (bw = black & white).
    method:
      - "otsu":   Otsu global (nền đều)
      - "adaptive": Adaptive Gaussian (ảnh chụp, bóng)
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 10
        )
    return gray, bw


def preprocess_document(image_bgr, out_size=(2480,3508), bw_method="adaptive"):
    """
    One-call: detect page, warp, output gray & bw.
    Return dict:
      {
        "warped_bgr": ...,
        "gray": ...,
        "bw": ...,
        "quad_src": 4x2 float32 points (TL,TR,BR,BL)
      }
    """
    warped, quad = rectify_sheet(image_bgr, out_size=out_size)
    gray, bw = to_gray_and_bw(warped, method=bw_method)
    return {"warped_bgr": warped, "gray": gray, "bw": bw, "quad_src": quad}

def save_outputs(out_dict, original_bgr, out_dir="data/results", prefix="pre_"):
    os.makedirs(out_dir, exist_ok=True)

    path_warp = os.path.join(out_dir, f"{prefix}warped.jpg")
    path_gray = os.path.join(out_dir, f"{prefix}gray.png")
    path_bw   = os.path.join(out_dir, f"{prefix}bw.png")

    cv2.imwrite(path_warp, out_dict["warped_bgr"])
    cv2.imwrite(path_gray, out_dict["gray"])
    cv2.imwrite(path_bw,   out_dict["bw"])

    dbg = original_bgr.copy()
    q = out_dict["quad_src"].astype(int)
    for i in range(4):
        p1 = tuple(q[i]); p2 = tuple(q[(i+1)%4])
        cv2.line(dbg, p1, p2, (0,255,0), 3)
        cv2.circle(dbg, p1, 6, (0,0,255), -1)
    path_dbg = os.path.join(out_dir, f"{prefix}debug_quad.jpg")
    cv2.imwrite(path_dbg, dbg)

    meta = {
        "quad_src_TL_TR_BR_BL": out_dict["quad_src"].tolist(),
        "warped_size_wh": [int(out_dict["warped_bgr"].shape[1]),
                           int(out_dict["warped_bgr"].shape[0])]
    }
    path_meta = os.path.join(out_dir, f"{prefix}meta.json")
    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"warped": path_warp, "gray": path_gray, "bw": path_bw, "debug": path_dbg, "meta": path_meta}
# ==========================
# CLI (demo)
# ==========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to input image")
    ap.add_argument("--method", default="adaptive", choices=["adaptive","otsu"])
    ap.add_argument("--outdir", default="data/results", help="Where to save outputs")
    ap.add_argument("--prefix", default="pre_", help="Filename prefix")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true", help="Write outputs to disk")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    out = preprocess_document(img, bw_method=args.method)

    if args.write:
        paths = save_outputs(out, img, out_dir=args.outdir, prefix=args.prefix)
        print("[Saved]")
        for k,v in paths.items():
            print(f"  {k}: {v}")

    if args.show:
        cv2.imshow("warped - color", out["warped_bgr"])
        cv2.imshow("warped - gray",  out["gray"])
        cv2.imshow("warped - B/W",   out["bw"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
