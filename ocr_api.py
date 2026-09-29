"""
OlmOCR Production API Server
FastAPI endpoints for PDF/Image OCR processing.

Usage:
  python ocr_api.py --host 0.0.0.0 --port 8003
  python ocr_api.py --model ./models/olmOCR-2-7B-1025-Q8_0.gguf
"""

import os
import uuid
import time
import shutil
import logging
import argparse
from contextlib import asynccontextmanager
from typing import Optional, List
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ocr_engine import process_pdf, process_image, OCRResult

# -- LOGGING ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ocr_api")

# -- GLOBALS ------------------------------------------------------------------
# All paths are configurable via environment variables.
# Defaults are portable sample paths â€” override for your local setup.
UPLOAD_DIR = os.environ.get("OCR_UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.environ.get("OCR_OUTPUT_DIR", "./ocr_output")
MODEL_PATH = os.environ.get("OCR_MODEL_PATH",
    "./models/olmOCR-2-7B-1025-Q8_0.gguf")
MMPROJ_PATH = os.environ.get("OCR_MMPROJ_PATH",
    "./models/mmproj-olmOCR-2-7B-1025-f16.gguf")
CLI_PATH = os.environ.get("OCR_CLI_PATH",
    "llama-mtmd-cli")
GPU_LAYERS = int(os.environ.get("OCR_GPU_LAYERS", "999"))
MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "6144"))
CTX_SIZE = int(os.environ.get("OCR_CTX_SIZE", "8192"))
API_KEY = os.environ.get("OCR_API_KEY")


# -- SECURITY -----------------------------------------------------------------
def verify_api_key(request: Request):
    if not API_KEY:
        return None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
        if token == API_KEY:
            return token
    raise HTTPException(status_code=403, detail="Invalid API key")


# -- LIFESPAN -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("OlmOCR API started")
    logger.info(f"  Model:  {MODEL_PATH}")
    logger.info(f"  MMProj: {MMPROJ_PATH}")
    logger.info(f"  CLI:    {CLI_PATH}")
    yield
    logger.info("OlmOCR API shutting down")


# -- APP ----------------------------------------------------------------------
app = FastAPI(
    title="OlmOCR Production API",
    version="1.0.0",
    description="Production-grade Tamil & English legal document OCR",
    lifespan=lifespan,
)


# -- MODELS -------------------------------------------------------------------
class PageResult(BaseModel):
    page_num: int
    page_type: str
    dpi: int
    char_count: int
    elapsed_seconds: float
    text: str


class OCRResponse(BaseModel):
    job_id: str
    source_filename: str
    total_pages: int
    processed_pages: int
    total_elapsed_seconds: float
    pages: List[PageResult]
    combined_markdown: str
    output_dir: str


# -- ENDPOINTS ----------------------------------------------------------------
@app.get("/health")
async def health():
    model_ok = os.path.exists(MODEL_PATH)
    cli_ok = os.path.exists(CLI_PATH)
    return {
        "status": "ok" if (model_ok and cli_ok) else "degraded",
        "model_loaded": model_ok,
        "cli_available": cli_ok,
        "model_path": MODEL_PATH,
    }


@app.post("/v1/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(
    request: Request,
    file: UploadFile = File(...),
    pages: str = Form("all"),
):
    """Process a PDF file through the full OCR pipeline."""
    verify_api_key(request)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    try:
        # Save uploaded file
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        t0 = time.time()

        results = process_pdf(
            upload_path, job_dir, pages,
            cli=CLI_PATH, model=MODEL_PATH, mmproj=MMPROJ_PATH,
            gpu_layers=GPU_LAYERS, max_tokens=MAX_TOKENS, ctx_size=CTX_SIZE,
        )

        total_elapsed = time.time() - t0

        # Read combined markdown
        combined_path = os.path.join(job_dir, "combined_ocr_results.md")
        combined_md = ""
        if os.path.exists(combined_path):
            with open(combined_path, "r", encoding="utf-8") as f:
                combined_md = f.read()

        page_results = [
            PageResult(
                page_num=r.page_num,
                page_type=r.page_type,
                dpi=r.dpi,
                char_count=len(r.text),
                elapsed_seconds=round(r.elapsed_s, 2),
                text=r.text,
            ) for r in results
        ]

        return OCRResponse(
            job_id=job_id,
            source_filename=file.filename,
            total_pages=len(results),
            processed_pages=len(results),
            total_elapsed_seconds=round(total_elapsed, 2),
            pages=page_results,
            combined_markdown=combined_md,
            output_dir=job_dir,
        )

    except Exception as e:
        logger.error(f"PDF OCR failed: {e}")
        raise HTTPException(500, f"OCR processing failed: {e}")


@app.post("/v1/ocr/image", response_model=OCRResponse)
async def ocr_image(
    request: Request,
    file: UploadFile = File(...),
):
    """Process a single image through the OCR pipeline."""
    verify_api_key(request)

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    ext = Path(file.filename).suffix or ".png"
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        t0 = time.time()

        result = process_image(
            upload_path, job_dir,
            cli=CLI_PATH, model=MODEL_PATH, mmproj=MMPROJ_PATH,
            gpu_layers=GPU_LAYERS, max_tokens=MAX_TOKENS, ctx_size=CTX_SIZE,
        )

        total_elapsed = time.time() - t0

        if not result:
            raise HTTPException(500, "OCR returned no results")

        return OCRResponse(
            job_id=job_id,
            source_filename=file.filename,
            total_pages=1,
            processed_pages=1,
            total_elapsed_seconds=round(total_elapsed, 2),
            pages=[PageResult(
                page_num=1,
                page_type=result.page_type,
                dpi=result.dpi,
                char_count=len(result.text),
                elapsed_seconds=round(result.elapsed_s, 2),
                text=result.text,
            )],
            combined_markdown=result.text,
            output_dir=job_dir,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        raise HTTPException(500, f"OCR processing failed: {e}")


@app.get("/v1/ocr/result/{job_id}")
async def get_result(job_id: str, request: Request):
    """Retrieve combined markdown for a previous job."""
    verify_api_key(request)
    combined = os.path.join(OUTPUT_DIR, job_id, "combined_ocr_results.md")
    if not os.path.exists(combined):
        raise HTTPException(404, f"Job {job_id} not found")
    return FileResponse(combined, media_type="text/markdown")


@app.get("/v1/ocr/result/{job_id}/metadata")
async def get_metadata(job_id: str, request: Request):
    """Retrieve OCR metadata JSON for a previous job."""
    verify_api_key(request)
    meta = os.path.join(OUTPUT_DIR, job_id, "ocr_metadata.json")
    if not os.path.exists(meta):
        raise HTTPException(404, f"Job {job_id} not found")
    return FileResponse(meta, media_type="application/json")


# -- STARTUP ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OlmOCR Production API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--mmproj", default=MMPROJ_PATH)
    parser.add_argument("--cli", default=CLI_PATH)
    args = parser.parse_args()

    os.environ["OCR_MODEL_PATH"] = args.model
    os.environ["OCR_MMPROJ_PATH"] = args.mmproj
    os.environ["OCR_CLI_PATH"] = args.cli

    uvicorn.run("ocr_api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
