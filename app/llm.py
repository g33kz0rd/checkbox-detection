"""Vision-request client.

`detect_with_mock` answers the request in-process.
`detect_with_local` sends the same prompt to a local OpenAI-compatible
vision server. The default is Ollama on this machine.
"""

import base64
import json
import os
import re

import cv2
import httpx
import numpy as np

from app.cv import detect_checkboxes

# TODO: add an auth header when VISION_URL is not a local server.
# TODO: retry timeouts. A local model can be slow on the first call.
# TODO: validate the model JSON against a schema before returning it.
VISION_URL = os.environ.get("VISION_URL", "http://127.0.0.1:11434/v1/chat/completions")
VISION_MODEL = os.environ.get("VISION_MODEL", "moondream")
MAX_SIDE = 1280

PROMPT = (
    "You detect checkboxes in a document image. "
    "Return only JSON with this shape and no other text: "
    '{"boxes":[{"bbox":[x1,y1,x2,y2],"is_checked":true}]}. '
    "bbox is integer pixels of the image you see. "
    "x1,y1 is the top-left corner and x2,y2 is the bottom-right. "
    "Include every checkbox, whether it is filled or empty."
)


class VisionError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def detect_with_mock(image_bgr: np.ndarray) -> list[dict]:
    """Stand-in for the vision call. Same JSON shape, no network."""
    return detect_checkboxes(image_bgr)


def detect_with_local(image_bgr: np.ndarray) -> list[dict]:
    resized, scale = _downscale(image_bgr, MAX_SIDE)
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise VisionError(400, "Could not encode the image for the vision model")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    payload = {
        "model": VISION_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    try:
        response = httpx.post(VISION_URL, json=payload, timeout=180)
    except httpx.HTTPError as exc:
        raise VisionError(
            503,
            f"Local vision server is not reachable at {VISION_URL}. {exc}",
        ) from exc
    if response.status_code >= 400:
        raise VisionError(
            503,
            f"Local vision server returned {response.status_code}: {response.text[:300]}",
        )
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise VisionError(422, "Local vision server did not return JSON") from exc
    boxes = boxes_from_completion(body)
    return _scale_boxes(boxes, scale)


def boxes_from_completion(payload: dict) -> list[dict]:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionError(422, "Local model response did not include message text") from exc
    if not isinstance(content, str):
        raise VisionError(422, "Local model response text was not a string")
    return _parse_boxes(content)


def _downscale(image_bgr: np.ndarray, max_side: int):
    height, width = image_bgr.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image_bgr, 1.0
    scale = max_side / float(long_side)
    resized = cv2.resize(
        image_bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, resized.shape[1] / float(width)


def _parse_boxes(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match is None:
            raise VisionError(422, "Model did not return JSON")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VisionError(422, "Model JSON could not be parsed") from exc
    if not isinstance(data, dict) or not isinstance(data.get("boxes"), list):
        raise VisionError(422, "Model JSON did not contain a boxes list")
    boxes = []
    for item in data["boxes"]:
        if not isinstance(item, dict):
            raise VisionError(422, "A box in the model JSON was not an object")
        bbox = item.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            raise VisionError(422, "A box bbox was not four numbers")
        checked = item.get("is_checked")
        if not isinstance(checked, bool):
            raise VisionError(422, "A box is_checked was not true or false")
        boxes.append({"bbox": [int(round(value)) for value in bbox], "is_checked": checked})
    return boxes


def _scale_boxes(boxes: list[dict], scale: float) -> list[dict]:
    if scale <= 0:
        raise VisionError(422, "Could not scale model boxes back to the original image")
    factor = 1.0 / scale
    scaled = []
    for box in boxes:
        scaled.append(
            {
                "bbox": [int(round(value * factor)) for value in box["bbox"]],
                "is_checked": box["is_checked"],
            }
        )
    return scaled
