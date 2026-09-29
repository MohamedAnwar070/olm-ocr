#!/usr/bin/env python3
"""
OlmOCR CLI — Production wrapper around ocr_engine.
Usage:
  python run_olm_ocr.py --pdf doc.pdf --pages "1,3"
  python run_olm_ocr.py --image scan.png
"""

import argparse
import logging
import os
import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ocr_engine import process_pdf, process_image

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("olm_ocr_cli")


# -- ANSI Colors --------------------------------------------------------------
class C:
    G = "\033[92m"
    Y = "\033[93m"
    R = "\033[91m"
    B = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="OlmOCR Production Pipeline — Tamil & English Legal OCR"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", "-f", help="Input PDF path")
    src.add_argument("--image", "-i", help="Input image path")

    parser.add_argument("--output", "-o", default="./ocr_results",
                        help="Output directory (default: ./ocr_results)")
    parser.add_argument("--pages", default="all",
                        help="Pages to process: 'all', '1', '1-5', '3,5' (default: all)")

    # Model paths (override via flags or OCR_* env vars; see .env.example)
    parser.add_argument("--model",
                        default=os.environ.get("OCR_MODEL_PATH",
                                               "./models/olmOCR-2-7B-1025-Q8_0.gguf"))
    parser.add_argument("--mmproj",
                        default=os.environ.get("OCR_MMPROJ_PATH",
                                               "./models/mmproj-olmOCR-2-7B-1025-f16.gguf"))
    parser.add_argument("--cli",
                        default=os.environ.get("OCR_CLI_PATH", "llama-mtmd-cli"))

    # Inference
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=999)

    args = parser.parse_args()

    # Validate paths
    for label, path in [("Model", args.model), ("MMProj", args.mmproj), ("CLI", args.cli)]:
        if not os.path.exists(path):
            print(f"{C.R}[ERROR]{C.END} {label} not found: {path}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{C.BOLD}{'='*60}")
    print(f"  OlmOCR Production Pipeline")
    print(f"{'='*60}{C.END}\n")

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"{C.R}[ERROR]{C.END} PDF not found: {args.pdf}")
            sys.exit(1)

        print(f"{C.B}[*]{C.END} Source: {args.pdf}")
        print(f"{C.B}[*]{C.END} Pages:  {args.pages}")
        print(f"{C.B}[*]{C.END} Output: {args.output}\n")

        results = process_pdf(
            args.pdf,
            args.output,
            args.pages,
            cli=args.cli,
            model=args.model,
            mmproj=args.mmproj,
            gpu_layers=args.gpu_layers,
            max_tokens=args.max_tokens,
            ctx_size=args.ctx_size,
        )

        print(f"\n{C.BOLD}{'-'*60}{C.END}")
        print(f"{C.G}[✓]{C.END} Processed {len(results)} pages")
        for r in results:
            print(f"    Page {r.page_num:3d}  │  {r.page_type:6s}  │  {r.dpi} DPI  │  "
                  f"{len(r.text):5d} chars  │  {r.elapsed_s:.1f}s")
        print(f"\n{C.G}[✓]{C.END} Combined output: {args.output}/combined_ocr_results.md")
        print(f"{C.G}[✓]{C.END} Metadata:        {args.output}/ocr_metadata.json\n")

    elif args.image:
        if not os.path.exists(args.image):
            print(f"{C.R}[ERROR]{C.END} Image not found: {args.image}")
            sys.exit(1)

        print(f"{C.B}[*]{C.END} Source: {args.image}")
        print(f"{C.B}[*]{C.END} Output: {args.output}\n")

        result = process_image(
            args.image,
            args.output,
            cli=args.cli,
            model=args.model,
            mmproj=args.mmproj,
            gpu_layers=args.gpu_layers,
            max_tokens=args.max_tokens,
            ctx_size=args.ctx_size,
        )

        if result:
            print(f"\n{C.G}[OK]{C.END} Done -- {len(result.text)} chars in {result.elapsed_s:.1f}s")
        else:
            print(f"\n{C.R}[FAIL]{C.END} OCR failed")
            sys.exit(1)


if __name__ == "__main__":
    main()

