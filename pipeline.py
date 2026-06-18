"""
Pipeline Orchestrator
---------------------
Wires all four stages together into a single callable pipeline.
"""

import json
import time
from pathlib import Path

from stages.stage0_preprocess import preprocess
from stages.stage1_segment import segment
from stages.stage2_extract import extract_all_zones
from stages.stage3_symbols import process_zones
from stages.stage4_assemble import assemble
from utils.ollama_client import OllamaClient


def run_pipeline(
    image_path: str,
    vision_model: str = "minicpm-v",
    text_model: str = "mistral-nemo:latest",
    output_dir: str = "output",
    visualize: bool = False,
    ollama_url: str = "http://localhost:11434"
) -> dict:
    """
    Run the full lab notebook parsing pipeline on a single image.

    Stage 1 now uses the vision model for classification (not hardcoded
    heuristics), so it generalises to any notebook page automatically.

    Args:
        image_path:    Path to notebook page image
        vision_model:  Ollama vision model — used in BOTH Stage 1 (classify)
                       and Stage 2 (extract). Needs image support.
        text_model:    Ollama text model for Stage 4 assembly only.
        output_dir:    Directory for output JSON and debug images.
        visualize:     Save intermediate stage images to output_dir/debug/.
        ollama_url:    Ollama API base URL.

    Returns:
        Structured experiment dict (also saved to output_dir/result.json)
    """
    t_start = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    debug_dir = str(out / "debug") if visualize else None

    print("=" * 60)
    print("LabAlly Lab Notebook Parser")
    print(f"Image:        {image_path}")
    print(f"Vision model: {vision_model}")
    print(f"Text model:   {text_model}")
    print(f"Output:       {output_dir}")
    print("=" * 60)

    # ── Preflight checks ───────────────────────────────────────────────────────
    client = OllamaClient(base_url=ollama_url)

    if not client.is_available():
        raise RuntimeError(
            f"Ollama not running at {ollama_url}. "
            "Start it with: ollama serve"
        )

    available = client.list_models()
    print(f"\nAvailable models: {available}")

    if not client.model_exists(vision_model):
        print(f"\nVision model '{vision_model}' not found locally.")
        print(f"Pulling now — this may take a few minutes...")
        client.pull_model(vision_model)

    # ── Warmup: load models into memory before pipeline starts ───────────────
    client.warmup(vision_model, text_model)

    # ── Stage 0: Preprocess ────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    preprocessed = preprocess(image_path, output_dir=debug_dir)

    # ── Stage 1: Segment ───────────────────────────────────────────────────────
    # Pass client + vision_model so Stage 1 uses LLM classification,
    # not hardcoded heuristics. This is what makes it generalisable.
    print("\n" + "─" * 40)
    zones = segment(
        preprocessed,
        client=client,
        vision_model=vision_model,
        output_dir=debug_dir
    )

    # ── Stage 2: Extract ───────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    zones = extract_all_zones(
        zones, client,
        model=vision_model,
        output_dir=debug_dir
    )

    # ── Stage 3: Symbol fix ────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    zones = process_zones(zones)

    # ── Stage 4: Assemble ──────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    result = assemble(zones, client, model=text_model)

    # ── Save output ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    result['_pipeline_meta'] = {
        "image_path": str(image_path),
        "vision_model": vision_model,
        "text_model": text_model,
        "elapsed_seconds": round(elapsed, 1)
    }

    out_path = out / "result.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Output saved to: {out_path}")
    print("=" * 60)

    return result
