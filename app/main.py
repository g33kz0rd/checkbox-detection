"""HTTP API and the one-page tester."""

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.cv import detect_checkboxes
from app.llm import VisionError, detect_with_local, detect_with_mock

STATIC = Path(__file__).resolve().parent.parent / "static" / "index.html"
METHODS = {"cv", "llm", "local"}

app = FastAPI(title="Checkbox detection")


@app.get("/")
def index():
    return FileResponse(STATIC)


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    method: str = Form("cv"),
):
    if method not in METHODS:
        raise HTTPException(status_code=400, detail="method must be cv, llm, or local")
    raw = await image.read()
    picture = _decode_image(raw)
    try:
        if method == "cv":
            boxes = detect_checkboxes(picture)
        elif method == "llm":
            boxes = detect_with_mock(picture)
        else:
            boxes = detect_with_local(picture)
    except VisionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return {"boxes": boxes}


def _decode_image(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file was empty")
    array = np.frombuffer(raw, dtype=np.uint8)
    picture = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if picture is None:
        raise HTTPException(status_code=400, detail="Could not read that file as an image")
    if picture.ndim == 2:
        return cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
    if picture.shape[2] == 4:
        return cv2.cvtColor(picture, cv2.COLOR_BGRA2BGR)
    return picture
