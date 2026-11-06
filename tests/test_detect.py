import io
import unittest
from unittest.mock import MagicMock, patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from app.cv import detect_checkboxes
from app.llm import VisionError, boxes_from_completion, detect_with_local, detect_with_mock
from app.main import app


def make_form():
    image = np.full((900, 1600, 3), 255, np.uint8)
    specs = [
        (100, 80, False),
        (400, 80, True),
        (100, 300, False),
        (400, 300, True),
    ]
    size = 40
    for x, y, checked in specs:
        cv2.rectangle(image, (x, y), (x + size - 1, y + size - 1), (0, 0, 0), 2)
        if checked:
            margin = 10
            cv2.line(
                image,
                (x + margin, y + margin),
                (x + size - 1 - margin, y + size - 1 - margin),
                (0, 0, 0),
                2,
            )
            cv2.line(
                image,
                (x + size - 1 - margin, y + margin),
                (x + margin, y + size - 1 - margin),
                (0, 0, 0),
                2,
            )
    return image


def png_bytes(image) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not encode the test image")
    return encoded.tobytes()


class DetectTests(unittest.TestCase):
    def test_cv_finds_checked_and_empty_squares(self):
        boxes = detect_checkboxes(make_form())
        self.assertEqual(len(boxes), 4)
        checked = [box for box in boxes if box["is_checked"]]
        empty = [box for box in boxes if not box["is_checked"]]
        self.assertEqual(len(checked), 2)
        self.assertEqual(len(empty), 2)
        self.assertTrue(all(box["bbox"][0] > 250 for box in checked))
        self.assertTrue(all(box["bbox"][0] < 250 for box in empty))

    def test_mock_matches_opencv(self):
        image = make_form()
        self.assertEqual(detect_with_mock(image), detect_checkboxes(image))

    def test_endpoint_cv_and_llm_and_page(self):
        client = TestClient(app)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Local LLM", page.text)

        image = png_bytes(make_form())
        for method in ("cv", "llm"):
            response = client.post(
                "/detect",
                files={"image": ("form.png", io.BytesIO(image), "image/png")},
                data={"method": method},
            )
            self.assertEqual(response.status_code, 200, response.text)
            boxes = response.json()["boxes"]
            self.assertEqual(len(boxes), 4)
            self.assertEqual(sum(box["is_checked"] for box in boxes), 2)

        rejected = client.post(
            "/detect",
            files={"image": ("form.png", io.BytesIO(image), "image/png")},
            data={"method": "nope"},
        )
        self.assertEqual(rejected.status_code, 400)

    def test_local_scales_boxes_back_to_the_original_image(self):
        # 1600px wide shrinks to 1280. A box reported on the small image
        # [80, 64, 112, 96] is [100, 80, 140, 120] on the original.
        content = '{"boxes":[{"bbox":[80,64,112,96],"is_checked":false}]}'
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        image = np.full((900, 1600, 3), 255, np.uint8)
        with patch("app.llm.httpx.post", return_value=response):
            boxes = detect_with_local(image)
        self.assertEqual(boxes, [{"bbox": [100, 80, 140, 120], "is_checked": False}])

    def test_local_reads_json_inside_a_fence(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"boxes":[{"bbox":[1,2,3,4],"is_checked":true}]}\n```'
                    }
                }
            ]
        }
        boxes = boxes_from_completion(payload)
        self.assertEqual(boxes, [{"bbox": [1, 2, 3, 4], "is_checked": True}])

    def test_local_reports_a_down_server(self):
        image = np.full((40, 40, 3), 255, np.uint8)
        with patch("app.llm.httpx.post", side_effect=httpx.ConnectError("refused")):
            with self.assertRaises(VisionError) as caught:
                detect_with_local(image)
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
