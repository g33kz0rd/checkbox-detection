"""Find checkbox squares and decide which ones are marked."""

import cv2
import numpy as np

# Side length relative to page width. Tuned for small form boxes on a
# full-page scan, while still matching a synthetic page in the tests.
MIN_SIDE_RATIO = 0.008
MAX_SIDE_RATIO = 0.045
MIN_ASPECT = 0.8
MAX_ASPECT = 1.25
# Share of dark pixels inside the square, after the border is cropped away.
INK_THRESHOLD = 0.05
OVERLAP_THRESHOLD = 0.4


def detect_checkboxes(image_bgr: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = thresh.shape
    kernel_size = max(3, int(round(width * 0.0015)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_side = MIN_SIDE_RATIO * width
    max_side = MAX_SIDE_RATIO * width
    found = []
    for contour in contours:
        box = _square_box(contour, min_side, max_side)
        if box is None:
            continue
        x, y, w, h = box
        ink = _ink_ratio(thresh, x, y, w, h)
        found.append(
            {
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "is_checked": bool(ink >= INK_THRESHOLD),
            }
        )
    return _dedupe(found)


def _square_box(contour, min_side: float, max_side: float):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    x, y, w, h = cv2.boundingRect(approx)
    if w < min_side or h < min_side or w > max_side or h > max_side:
        return None
    aspect = w / float(h)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return None
    if cv2.contourArea(approx) < 0.7 * w * h:
        return None
    return x, y, w, h


def _ink_ratio(thresh, x: int, y: int, w: int, h: int) -> float:
    pad = max(2, int(min(w, h) * 0.2))
    if w <= pad * 2 or h <= pad * 2:
        return 0.0
    roi = thresh[y + pad : y + h - pad, x + pad : x + w - pad]
    if roi.size == 0:
        return 0.0
    return float(cv2.countNonZero(roi)) / float(roi.size)


def _iou(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _dedupe(boxes: list[dict]) -> list[dict]:
    ordered = sorted(
        boxes,
        key=lambda box: (box["bbox"][2] - box["bbox"][0]) * (box["bbox"][3] - box["bbox"][1]),
        reverse=True,
    )
    kept = []
    for box in ordered:
        if all(_iou(box["bbox"], other["bbox"]) < OVERLAP_THRESHOLD for other in kept):
            kept.append(box)
    kept.sort(key=lambda box: (box["bbox"][1], box["bbox"][0]))
    return kept
