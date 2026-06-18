# LabAlly Lab Notebook Parser

A multi-stage pipeline for extracting structured, machine-readable data from handwritten chemistry lab notebook pages.

## Architecture

The core insight is that a single VLM pass over a full notebook page fails because it tries to do too many things at once: layout parsing, handwriting OCR, symbol recognition, chemical structure interpretation, and semantic understanding. This pipeline decomposes the problem into specialized stages.

```
Raw Image
    │
    ▼
Stage 0: Preprocessing       — deskew, denoise, contrast enhance
    │
    ▼
Stage 1: Zone Segmentation   — detect and classify page regions
    │                           (text | math | table | chemical structures)
    ├── text zones
    ├── math/equation zones
    ├── table zones
    └── chemical structure zones
    │
    ▼
Stage 2: Zone Extraction     — vision LLM (Ollama) with zone-specific prompts
    │                           each zone type gets a specialized prompt
    │                           that primes the model for what it will see
    ▼
Stage 3: Symbol Post-processing — rule-based cleanup for scientific notation
    │                             E-notation, °C, θ, ω, sub/superscripts,
    │                             mol%, v/v, cm², mA/cm²
    ▼
Stage 4: LLM Assembly        — text LLM synthesizes all zone outputs
    │                           into a structured experiment JSON record
    ▼
Output JSON
```

### Why this beats a single VLM pass

| Problem | Single-pass VLM | This pipeline |
|---|---|---|
| Symbol mangling (~50% failure) | Fails silently | Stage 3 rule engine catches and corrects |
| Chemical structures | Hallucinated or skipped | Isolated zone + structure-focused prompt |
| Math calculations | Merged with prose | Dedicated math zone extraction |
| Table parsing | Loses column alignment | Grid detection before extraction |
| Confidence | No signal | Per-zone success/failure flags in output |

### Fine-tuning targets (the LabAlly angle)

Each Stage 2 zone extractor is an independent fine-tuning target. The pipeline generates (image_crop → extracted_json) pairs per zone type — exactly the training data format needed to fine-tune specialist models. This is the same strategy LabAlly uses for XRD/NMR: train on real experimental data, not curated publications.

## Setup

```bash
pip install -r requirements.txt
ollama pull llava:13b        # vision model (or minicpm-v for better doc understanding)
```

## Usage

```bash
# Run on a single image
python run.py --image path/to/notebook_page.jpg

# Run with verbose zone visualization (useful for debugging)
python run.py --image path/to/notebook_page.jpg --visualize

# Output goes to output/result.json
```

## Output Format

```json
{
  "metadata": {
    "page": 57,
    "date": "June 4",
    "project": "Li electrodeposition – glyme electrolytes",
    "continued_from": "p3 56"
  },
  "goal": "Screen electrolyte 240604-B for stable Li plating at 30°C",
  "electrolyte": {
    "composition": "1M LiTFSI in diglyme:EtOH (4:1 v/v)",
    "volume_mL": 20,
    "additive": "5 mol% 12-crown-4",
    "preparation": "stir 20 min"
  },
  "conditions": {
    "temperature_C": 22.4,
    "atmosphere": "glovebox",
    "H2O_ppm": "<1"
  },
  "electrodes": {
    "working": "glassy C RDE, 0.3 cm²",
    "counter": "Li foil",
    "reference": "Ag/AgCl"
  },
  "deposition": {
    "run_id": "240604-B1",
    "potential_V": -0.45,
    "reference": "Ag/AgCl",
    "duration_min": 90,
    "rotation_rpm": 1600,
    "current_density_mA_cm2": 0.50,
    "area_cm2": 0.3,
    "charge_C": 0.81,
    "moles_Li": 8.4e-6,
    "mass_Li_g": 5.8e-5
  },
  "chemical_structures": [
    {
      "label": "[Li(12-crown-4)]+",
      "type": "crown_ether_complex",
      "description": "Li+ coordinated inside 12-crown-4 macrocycle",
      "role": "additive complex in electrolyte"
    },
    {
      "label": "LiTFSI",
      "type": "ionic_compound",
      "description": "Li+ with bis(trifluoromethanesulfonyl)imide anion, S-N-S backbone with =O and CF3 groups",
      "role": "electrolyte salt"
    }
  ],
  "reaction": "Li+ + e- → Li at -0.45V vs Ag/AgCl",
  "temperature_profile": [
    {"time": "0 min", "temp_C": 22.4},
    {"time": "1 min", "temp_C": 23.1},
    {"time": "5 min", "temp_C": 25.6},
    {"time": "10 min", "temp_C": 27.9},
    {"time": "20 min", "temp_C": 30.1},
    {"time": "40 min", "temp_C": 31.5},
    {"time": "1 hr", "temp_C": 32.0},
    {"time": "1 hr 30 min", "temp_C": 32.6}
  ],
  "observations": [
    "Film looks grey and dull",
    "XRD min peak at 2θ = 2.1° (low intensity)",
    "Shoulder at 2θ = 4.7°"
  ],
  "zones_extracted": {
    "text": {"count": 3, "confidence": "high"},
    "math": {"count": 2, "confidence": "high"},
    "table": {"count": 1, "confidence": "high"},
    "chemical_structures": {"count": 2, "confidence": "medium"}
  }
}
```

## Project Structure

```
labally_pipeline/
├── run.py                   # CLI entry point
├── pipeline.py              # Orchestrator — wires all stages together
├── requirements.txt
├── stages/
│   ├── stage0_preprocess.py # Image cleaning and normalization
│   ├── stage1_segment.py    # Zone detection and classification
│   ├── stage2_extract.py    # Vision LLM extraction per zone
│   ├── stage3_symbols.py    # Symbol post-processing rules
│   └── stage4_assemble.py   # LLM assembly into structured JSON
├── utils/
│   ├── ollama_client.py     # Ollama API wrapper
│   ├── image_utils.py       # Crop, annotate, save zone images
│   └── schema.py            # Output JSON schema + validation
├── output/                  # Results written here
└── tests/
    └── test_symbols.py      # Unit tests for symbol rules
```
