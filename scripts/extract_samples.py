"""Copy the document images out of the take-home PDF into samples/.

This is not part of setup and is not required to run the API.
Install the extra library into the project virtualenv first:

    .venv/Scripts/python -m pip install pymupdf

Then:

    .venv/Scripts/python scripts/extract_samples.py path\\to\\challenge.pdf
"""

import sys
from pathlib import Path

import fitz


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: extract_samples.py CHALLENGE.pdf")
        raise SystemExit(1)
    pdf_path = Path(sys.argv[1])
    destination = Path(__file__).resolve().parent.parent / "samples"
    destination.mkdir(exist_ok=True)
    document = fitz.open(pdf_path)
    saved = 0
    for page_index, page in enumerate(document, start=1):
        for image in page.get_images(full=True):
            pixmap = fitz.Pixmap(document, image[0])
            if pixmap.width < 400 or pixmap.height < 400:
                continue
            if pixmap.alpha:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            elif pixmap.n > 3:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            saved += 1
            output = destination / f"sample_{saved}.png"
            pixmap.save(output)
            print(f"wrote {output} ({pixmap.width}x{pixmap.height}, from page {page_index})")
    if saved == 0:
        print("No document images found in that PDF.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
