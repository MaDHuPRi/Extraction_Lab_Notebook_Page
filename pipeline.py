"""
Pipeline Orchestrator
---------------------
Wires all stages together. Now includes:
- Pydantic validation of final output
- Optional MolScribe SMILES extraction for structure zones
- Swappable vision backend (Ollama or Claude API)
"""

import json
import time
from pathlib import Path
from typing import Optional

from stages.stage0_preprocess import preprocess
from stages.stage1_segment import segment
from stages.stage2_extract import extract_all_zones
from stages.stage3_symbols import process_zones
from stages.stage4_assemble import assemble
from utils.ollama_client import OllamaClient
from utils.schema import validate_output, ExperimentRecord
from utils.molscribe_client import (MolScribeClient,
                                     enrich_structures_with_smiles,
                                     is_molscribe_available)


def run_pipeline(
    image_path: str,
    vision_model: str = "minicpm-v",
    text_model: str = "mistral-nemo:latest",
    output_dir: str = "output",
    visualize: bool = False,
    ollama_url: str = "http://localhost:11434",
    use_molscribe: bool = True,
) -> dict:
    """
    Run the full lab notebook parsing pipeline.

    Args:
        image_path:     Path to notebook page image
        vision_model:   Ollama vision model (Stage 1 classify + Stage 2 extract)
        text_model:     Ollama text model (Stage 4 assembly)
        output_dir:     Directory for output files
        visualize:      Save debug images per stage
        ollama_url:     Ollama server URL
        use_molscribe:  If True and MolScribe installed, extract SMILES
                        from chemical structure zones

    Returns:
        Validated ExperimentRecord as dict
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
    print(f"MolScribe:    {'enabled' if use_molscribe else 'disabled'}")
    print(f"Output:       {output_dir}")
    print("=" * 60)

    # ── Preflight ──────────────────────────────────────────────────────────────
    client = OllamaClient(base_url=ollama_url)

    if not client.is_available():
        raise RuntimeError(
            f"Ollama not running at {ollama_url}. "
            "Start with: ollama serve"
        )

    available = client.list_models()
    print(f"\nAvailable models: {available}")

    if not client.model_exists(vision_model):
        print(f"\nPulling {vision_model}...")
        client.pull_model(vision_model)

    # ── Warmup ─────────────────────────────────────────────────────────────────
    client.warmup(vision_model, text_model)

    # ── Stage 0: Preprocess ────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    preprocessed = preprocess(image_path, output_dir=debug_dir)

    # ── Stage 1: Segment ───────────────────────────────────────────────────────
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

    # ── Stage 2b: MolScribe (optional) ─────────────────────────────────────────
    if use_molscribe and is_molscribe_available():
        print("\n" + "─" * 40)
        print("[MolScribe] Extracting SMILES from structure zones...")
        molscribe = MolScribeClient(device="cpu")
        zones = enrich_structures_with_smiles(zones, molscribe)
    elif use_molscribe and not is_molscribe_available():
        print("\n[MolScribe] Not installed — skipping SMILES extraction.")
        print("  Install with: pip install molscribe")

    # ── Stage 3: Symbol fix ────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    zones = process_zones(zones)

    # ── Stage 4: Assemble ──────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    raw_result = assemble(zones, client, model=text_model)

    # ── Pydantic validation ────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("[Validation] Running Pydantic schema validation...")
    record, errors = validate_output(raw_result)

    if errors:
        print(f"[Validation] {len(errors)} validation issue(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[Validation] All fields valid.")

    score = record.completion_score()
    print(f"[Validation] Completion score: {score:.0%} of key fields extracted")

    # ── Save outputs ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Full validated JSON
    result_dict = record.model_dump()
    result_dict['_pipeline_meta'] = {
        "image_path": str(image_path),
        "vision_model": vision_model,
        "text_model": text_model,
        "elapsed_seconds": round(elapsed, 1),
        "completion_score": score,
        "validation_errors": errors,
    }

    out_path = out / "result.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    # Flat evaluation dict for easy diffing
    eval_dict = record.to_evaluation_dict()
    eval_path = out / "eval.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_dict, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Completion score:   {score:.0%}")
    print(f"Full result:        {out_path}")
    print(f"Eval summary:       {eval_path}")
    print("=" * 60)

    return result_dict
