#!/usr/bin/env python3
"""
LabAlly Lab Notebook Parser — CLI Entry Point

Usage:
    python run.py --image path/to/notebook.jpg
    python run.py --image path/to/notebook.jpg --visualize
    python run.py --image path/to/notebook.jpg --vision-model minicpm-v --text-model mistral-nemo:latest
    python run.py --test-symbols   # Run symbol correction unit tests
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Parse handwritten chemistry lab notebook pages into structured JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--image', '-i', type=str,
        help='Path to lab notebook page image (JPG or PNG)')
    parser.add_argument('--vision-model', type=str, default='minicpm-v',
        help='Ollama vision model for zone extraction (default: minicpm-v)')
    parser.add_argument('--text-model', type=str, default='mistral-nemo:latest',
        help='Ollama text model for final assembly (default: mistral-nemo:latest)')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
        help='Directory for output files (default: output/)')
    parser.add_argument('--visualize', '-v', action='store_true',
        help='Save debug visualizations of each pipeline stage')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
        help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--test-symbols', action='store_true',
        help='Run symbol correction unit tests and exit')

    args = parser.parse_args()

    # Run symbol tests if requested
    if args.test_symbols:
        from stages.stage3_symbols import run_symbol_tests
        success = run_symbol_tests()
        sys.exit(0 if success else 1)

    # Require image for normal run
    if not args.image:
        parser.print_help()
        print("\nError: --image is required")
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Run pipeline
    from pipeline import run_pipeline

    result = run_pipeline(
        image_path=str(image_path),
        vision_model=args.vision_model,
        text_model=args.text_model,
        output_dir=args.output_dir,
        visualize=args.visualize,
        ollama_url=args.ollama_url
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    # result is a plain dict (model_dump() output from Pydantic)
    # Use .get() safely with fallbacks throughout
    print("\n── Extracted Experiment Summary ──")

    meta = result.get('metadata') or {}
    print(f"  Page:     {meta.get('page', '?')}")
    print(f"  Date:     {meta.get('date', '?')}")
    print(f"  Project:  {meta.get('project', '?')}")
    print(f"  Run ID:   {meta.get('run_id', '?')}")
    print(f"  Goal:     {result.get('goal', '?')}")

    quality = result.get('extraction_quality') or {}
    confidence  = quality.get('confidence', '?')
    zones_ok    = quality.get('zones_successful', '?')
    zones_total = quality.get('zones_extracted', '?')
    print(f"  Quality:  {confidence} ({zones_ok}/{zones_total} zones)")

    pipe_meta = result.get('_pipeline_meta') or {}
    score   = pipe_meta.get('completion_score', '?')
    elapsed = pipe_meta.get('elapsed_seconds', '?')
    print(f"  Score:    {score}")
    print(f"  Time:     {elapsed}s")

    print(f"\nFull result:  {args.output_dir}/result.json")
    print(f"Eval summary: {args.output_dir}/eval.json")


if __name__ == '__main__':
    main()
