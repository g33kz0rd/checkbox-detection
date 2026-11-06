# Checkbox detection

A small API that finds checkboxes in a document image and marks each one filled or empty. A page in the same app lets you drop an image and read the JSON.

The response is always:

```json
{ "boxes": [ { "bbox": [x1, y1, x2, y2], "is_checked": true } ] }
```

`bbox` is the top-left and bottom-right corner in pixels of the uploaded image.

## Access

The running app is at [https://checkbox-detection-3614.onrender.com](https://checkbox-detection-3614.onrender.com).

Open that page, drop an image, and press Detect. OpenCV and Mock LLM both run on the server. Local LLM is not available there, because that mode calls a vision model on the same machine and this host does not have one.

From a terminal:

```bash
curl -F "image=@samples/sample_2.png" -F "method=cv" https://checkbox-detection-3614.onrender.com/detect
```

Use `method=llm` for the mock. The source is at [github.com/g33kz0rd/checkbox-detection](https://github.com/g33kz0rd/checkbox-detection). Deploys are listed in the [Render dashboard](https://dashboard.render.com/web/srv-dbn3q93rjlhs73diu650).

The free instance sleeps when nobody has used it for a while. The first request after that can take a minute while it wakes up.

## Setup

Do this yourself from the project folder. The script further down only repeats these steps.

1. Install Python 3.12.
   - From [python.org](https://www.python.org/downloads/), or with `winget install --id Python.Python.3.12 -e`.
   - The `python` command from the Windows Store is a stub, not an interpreter. Check with `py -3 --version`. You want 3.11 or newer.

2. Create a virtual environment:

```bash
py -3 -m venv .venv
```

3. Install the libraries listed in `requirements.txt`:

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

That installs FastAPI, Uvicorn, OpenCV, NumPy, and HTTPX.

4. Start the server:

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload
```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Docker

With Docker running:

```bash
docker compose up --build
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The container listens on `PORT` when that variable is set, otherwise 8000.

## Optional script

`setup.sh` is a shortcut for the steps above. Open the file and read it before you run it. It is short on purpose.

```bash
bash setup.sh
```

What it does, and nothing else:

- Looks for Python 3.11 or newer, trying `py -3`, then `python3`, then `python`. A command that is not a real interpreter is ignored.
- If none of those works, it runs `winget install --id Python.Python.3.12 -e`. That install is the only change outside this folder, and it happens only when Python is missing.
- Creates `.venv` in this folder if it is not already there.
- Runs `.venv/Scripts/python -m pip install -r requirements.txt`.
- Prints the Uvicorn command from step 4.

It does not install Ollama, download a model, change git settings, or edit files outside this project.

## Using the page

Drop an image, or choose a file. Pick a method and press Detect. The JSON appears next to a thumbnail. Click the thumbnail to enlarge it. Click the dark backdrop, or press Escape, to close it.

- **OpenCV** (`cv`) finds the boxes in code.
- **Mock LLM** (`llm`) builds a vision request and answers it in-process with the OpenCV result. No model and no network.
- **Local LLM** (`local`) sends that request to a vision model on this machine. If nothing is listening, the page shows that error.

The same call from a terminal:

```bash
curl -F "image=@samples/sample_1.png" -F "method=cv" http://127.0.0.1:8000/detect
```

Change `method` to `llm` or `local`.

## Optional local model

Skip this unless you want the Local LLM choice to call a real model. The app runs without it.

1. Install [Ollama](https://ollama.com/) and leave it running.
2. Pull a small vision model: `ollama pull moondream`.
3. Choose Local LLM on the page.

The client posts to `http://127.0.0.1:11434/v1/chat/completions` with model `moondream`. Point it somewhere else with `VISION_URL` and `VISION_MODEL` before you start Uvicorn. Any server that accepts an OpenAI-style chat completion with an image can sit behind those two variables.

The image is shrunk so the long side is at most 1280 pixels before that call. Boxes in the model reply are scaled back to the original image.

## Optional sample images

The take-home PDF is not required to run the app. To copy its document images into `samples/`:

```bash
.venv/Scripts/python -m pip install pymupdf
.venv/Scripts/python scripts/extract_samples.py "path\to\challenge.pdf"
```

`pymupdf` is only for that script. It is not in `requirements.txt`.

## Tests

```bash
.venv/Scripts/python -m unittest tests.test_detect
```

The tests draw a small form, check OpenCV and the mock, and stub the local HTTP call. They do not download a model.

## Approach

OpenCV turns the page into a black-and-white image, keeps contours that are near-squares of about checkbox size, and drops overlapping duplicates. A square is marked checked when the inside, with the border cropped off, has more ink than a fixed ratio.

The mock path exists so the vision-request code can be reviewed without a model. It returns the OpenCV boxes in the JSON a vision model is asked to produce. Local mode uses that same prompt against a server on this machine. Swapping `VISION_URL` is the step that would point the client at another host. Auth, retries, and a stricter schema check are left as `TODO`s in `app/llm.py`.

A blank form was not used. It helps only when it is the same form and the scans line up, and a plain pixel difference gets worse when the pages are shifted. The challenge images do not include blanks.

## Limitations

- A light or broken checkbox border is easy to miss.
- An X or check that touches the border can stop the outline from looking like a square, so that box is missed.
- A table cell or stamp close to checkbox size can be reported as a box.
- Rotated pages are not straightened.
- A local vision model is slow, and it often misses small boxes on a full-page scan. The mock does not have that problem because it never calls a model.
