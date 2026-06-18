# LabAlly Lab Notebook Parser

A multi-stage pipeline for extracting structured, machine-readable data from handwritten chemistry lab notebook pages. Covers all four evaluation levels: plain text, scientific symbols, chemical structures, and full experiment understanding.

Runs entirely locally via Ollama — no external API calls, no data leaves your machine.

## Architecture

The core insight is that a single VLM pass over a full notebook page fails because it tries to do too many things at once: layout parsing, handwriting OCR, symbol recognition, chemical structure interpretation, and semantic understanding. This pipeline decomposes the problem into five independent stages.

```
Raw Image
    │
    ▼
Stage 0: Preprocessing
    Deskew, CLAHE contrast enhancement, denoise, binarize
    │
    ▼
Stage 1: Zone Segmentation
    OpenCV detects horizontal ruled lines → slices page into bands
    minicpm-v visually classifies each band:
    header | text | math | table | chemical_structure
    │
    ▼
Stage 2: Zone Extraction
    Each zone gets a type-specific prompt designed for that content
    Text zones, math zones, table zones, structure zones all handled differently
    │
    ▼
Stage 3: Symbol Post-Processing
    Deterministic rule engine corrects known VLM failure patterns:
    OCR confusion (O/0, l/1, S/5), Greek symbols, unit superscripts,
    scientific notation, chemistry notation — 15/15 unit tests passing
    │
    ▼
Stage 4: LLM Assembly
    Text LLM synthesises all zone outputs into a structured experiment record
    Pydantic schema validation with field-level completion scoring
    │
    ▼
output/result.json    ← full structured output
output/eval.json      ← flat evaluation dict for diffing
output/debug/         ← zone visualisations (with --visualize)
```

## Why this beats a single VLM pass

| Problem | Single-pass VLM | This pipeline |
|---|---|---|
| Symbol mangling (~50% failure rate) | Fails silently | Stage 3 rule engine catches and corrects |
| Chemical structures | Hallucinated or skipped | Isolated zone + structure-focused prompt |
| Math calculations | Merged with prose | Dedicated math zone extraction |
| Table parsing | Loses row/column alignment | Grid detection before extraction |
| New notebook pages | May fail on unseen layouts | Stage 1 uses VLM classification, not hardcoded rules |

## Strengths & Limitations

**Works on any notebook page:**
- Stage 0 preprocessing — deskew and CLAHE handles any phone photo angle or lighting
- Stage 1 segmentation — VLM classification means no hardcoded layout assumptions
- Stage 3 symbol rules — universal chemistry symbols, not tuned to a specific page
- Stage 4 assembly — pure LLM synthesis adapts to whatever zones it receives

**Known limitations:**
- Stage 2 prompts are tuned for electrochemistry — other scientific domains need a prompt swap (straightforward, no architectural change)
- Pipeline assumes ruled notebook paper — grid or unlined paper would need a different slicing strategy in Stage 1
- Math variable mapping (Q, n → charge_C, moles_deposited) is inconsistent — few-shot examples in the assembly prompt would fix this
- Table extraction can truncate at ~4 rows instead of 8 — prompt cap needs raising

## Fine-tuning roadmap

Each Stage 2 zone extractor is an independent fine-tuning target. The pipeline generates (zone_image_crop → extracted_json) pairs as it runs — exactly the training data format needed for QLoRA fine-tuning of specialist models per zone type. This is the same strategy LabAlly uses for XRD/NMR: train on real experimental data, not curated publications.

```
Phase 1 (now):   Prompt engineering + rule-based correction
Phase 2:         Collect (zone_crop → JSON) pairs, QLoRA fine-tune per zone type
Phase 3:         Swap LoRA adapter per zone at inference, Stage 3 rules become eval harness
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull the vision model (~5.5GB)
ollama pull minicpm-v

# Verify symbol correction rules
python run.py --test-symbols
```

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- minicpm-v for vision (Stages 1 + 2)
- Any text model for assembly (Stage 4) — llama3.2 recommended for speed

## Usage

```bash
# Basic run
python run.py --image path/to/notebook_page.jpg

# With debug visualisations
python run.py --image path/to/notebook_page.jpg --visualize

# Custom models
python run.py \
  --image path/to/notebook_page.jpg \
  --vision-model minicpm-v \
  --text-model llama3.2:latest \
  --output-dir output \
  --visualize

# Run symbol correction unit tests
python run.py --test-symbols
```

## Output format

```json
{
  "metadata": {
    "page": 57,
    "date": "June 4",
    "project": "Li electrodeposition – glyme electrolytes",
    "run_id": "240604-B1",
    "continued_from": "p3 56"
  },
  "goal": "Screen electrolyte 240604-B for stable Li plating at 30°C",
  "electrolyte": {
    "salt": "1M LiTFSI",
    "solvent": "diglyme:EtOH (4:1 v/v)",
    "additive": "5 mol% 12-crown-4",
    "volume_mL": 20,
    "preparation_notes": "Stir 20 min, glovebox H₂O < 1 ppm"
  },
  "conditions": {
    "temperature_C": 22.4,
    "atmosphere": "glovebox",
    "water_ppm": "< 1"
  },
  "electrodes": {
    "working": "glassy C RDE, 0.3 cm²",
    "counter": "Li foil",
    "reference": "Ag/AgCl"
  },
  "deposition": {
    "potential_V": -0.45,
    "duration_min": 90,
    "rotation_rpm": 1600,
    "current_density_mA_cm2": 0.5,
    "electrode_area_cm2": 0.3,
    "charge_C": 0.81,
    "moles_deposited": 8.4e-6,
    "mass_deposited_g": 5.8e-5
  },
  "chemical_structures": [
    {
      "label": "[Li(12-crown-4)]⁺",
      "type": "complex",
      "description": "Li⁺ coordinated inside 12-crown-4 macrocycle",
      "role": "additive complex in electrolyte"
    }
  ],
  "reactions": [
    {
      "equation": "Li⁺ + e⁻ → Li",
      "conditions": "-0.45V vs Ag/AgCl"
    }
  ],
  "observations": {
    "visual": "Film looks grey and dull",
    "xrd": {
      "peaks": [2.1, 4.7],
      "notes": "Weak peak at 2θ = 2.1°, shoulder at 2θ = 4.7°"
    }
  },
  "temperature_profile": [
    {"time": "0 min",  "temp_C": 22.4},
    {"time": "20 min", "temp_C": 30.1},
    {"time": "40 min", "temp_C": 31.5},
    {"time": "1 hr",   "temp_C": 32.0}
  ],
  "procedure_summary": "A 1M LiTFSI/diglyme:EtOH electrolyte with 5 mol% 12-crown-4 additive was prepared and used for Li electrodeposition at -0.45V vs Ag/AgCl for 90 min at 1600 rpm. The deposited film appeared grey and dull.",
  "experiment_interpretation": {
    "what_was_tested": "Effect of 12-crown-4 additive on Li plating stability in glyme electrolyte",
    "key_finding": "Li plating produced a grey, dull film with weak XRD signal, indicating poor deposit quality",
    "next_steps_implied": "Vary additive concentration or potential to improve film morphology"
  },
  "extraction_quality": {
    "zones_extracted": 11,
    "zones_successful": 11,
    "confidence": "high"
  },
  "_pipeline_meta": {
    "vision_model": "minicpm-v",
    "text_model": "llama3.2:latest",
    "elapsed_seconds": 421.9,
    "completion_score": 0.8
  }
}
```

## Project structure

```
labally_pipeline/
├── run.py                    # CLI entry point
├── pipeline.py               # Orchestrator — wires all stages together
├── requirements.txt
├── setup.py
├── stages/
│   ├── stage0_preprocess.py  # Image cleaning and normalisation
│   ├── stage1_segment.py     # Zone detection via ruled lines + VLM classification
│   ├── stage2_extract.py     # Vision LLM extraction with zone-specific prompts
│   ├── stage3_symbols.py     # Deterministic symbol post-processing rules
│   └── stage4_assemble.py    # LLM assembly into structured JSON
├── utils/
│   ├── ollama_client.py      # Ollama API wrapper (vision + text)
│   ├── schema.py             # Pydantic output schema + validation
│   └── molscribe_client.py   # MolScribe integration (SMILES extraction, optional)
├── output/                   # Results written here
│   ├── result.json           # Full structured output
│   ├── eval.json             # Flat evaluation dict
│   └── debug/                # Zone visualisations (--visualize flag)
└── tests/
    └── test_symbols.py       # Unit tests for symbol correction rules
```

## Sample performance on provided notebook page

- **Zones detected:** 11/11
- **Zones extracted:** 11/11 (0 errors)
- **Completion score:** 80% of key fields
- **Validation errors:** 0
- **Runtime:** ~7 minutes (Apple Silicon, local models)
- **Symbol corrections applied:** 6 zones

Fields reliably extracted: page metadata, project, goal, electrolyte composition, electrode setup, deposition conditions, chemical structures, experiment interpretation.

Fields with known gaps: full temperature profile (4/8 rows extracted), calculated values (charge, moles, mass — present in math zone but not always mapped through assembly).
