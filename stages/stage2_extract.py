"""
Stage 2: Zone Extraction
------------------------
Sends each detected zone to a vision LLM (via Ollama) with a
zone-type-specific prompt designed to maximize accuracy for that
content type.

The key architectural insight: the same VLM performs very differently
depending on what it's primed to look for. A single generic prompt
misses symbols, misreads structures, and loses table alignment.
Zone-specific prompts dramatically improve accuracy because the model
knows exactly what kind of content it's looking at.

Prompt design principles used here:
1. Tell the model what TYPE of content it will see
2. Enumerate the specific symbols/formats to watch for
3. Give explicit output format to prevent hallucination
4. For structures: describe what to look for, not what to produce
5. Always ask for explicit uncertainty flags

Fine-tuning note: Each prompt here is a fine-tuning target.
The (zone_image → expected_output) pairs this pipeline generates
are exactly the training data LabAlly would use to train specialist models.
"""

import base64
import json
import re
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from stages.stage1_segment import Zone, ZoneType
from utils.ollama_client import OllamaClient


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at reading handwritten chemistry lab notebooks.
You extract information with extreme precision, paying special attention to:
- Scientific symbols: °C, °, θ, λ, ω, α, β, Δ
- Units: mA/cm², mol%, v/v, rpm, ppm, mL, mA, cm², μL
- Scientific notation: written as e.g. "1.5E-4" or "1.5×10⁻⁴"  
- Subscripts and superscripts in chemical formulas: Li⁺, e⁻, H₂O, cm²
- Fractions and ratios written inline: 4:1 v/v, 0.5 mA/cm²
- Crossed-out text indicates corrections — note both the original and correction
Return only what is asked. Never invent content you cannot see."""


PROMPTS = {
    ZoneType.HEADER: """This is the header region of a handwritten chemistry lab notebook page.
Extract ALL of the following if present:
- Page number
- Date (exact as written)
- Project title
- Any "continued from" or cross-reference notes

Return as JSON:
{
  "page_number": "...",
  "date": "...", 
  "project": "...",
  "cross_reference": "..."
}
Use null for any field not present.""",

    ZoneType.TEXT: """This is a text section from a handwritten chemistry lab notebook.
Extract ALL written content EXACTLY as written, including:
- Any crossed-out text (mark as "[CROSSED OUT: text]")
- Abbreviations (keep as written: elec., ref., conc., etc.)
- Chemical names and formulas inline (LiTFSI, diglyme, EtOH, etc.)
- Units exactly as written (mL, tot, mol%, v/v, ppm)
- Numbers with their units

Return as JSON:
{
  "raw_text": "...",
  "key_value_pairs": {"key": "value"},
  "chemicals_mentioned": ["..."],
  "uncertainties": ["anything you're not sure about"]
}""",

    ZoneType.MATH: """This is a mathematical calculation section from a handwritten chemistry lab notebook.
Extract EVERY equation, calculation, and numerical result. Pay extreme attention to:
- Exponents and subscripts: cm², mA, Li⁺, e⁻
- Scientific notation: "1.5E-4" means 1.5×10⁻⁴
- Units at each step of the calculation
- The final numerical result with units
- Any intermediate steps shown

Return as JSON:
{
  "equations": [
    {
      "expression": "...",
      "result": "...",
      "result_units": "...",
      "notes": "..."
    }
  ],
  "variables_defined": {"variable": "value with units"},
  "uncertainties": ["anything you're not sure about"]
}""",

    ZoneType.TABLE: """This is a data table from a handwritten chemistry lab notebook.
Extract the COMPLETE table preserving the exact structure:
- Column headers (exactly as written, with units)
- Every row of data
- Any observations or notes in the rightmost columns
- Watch for merged cells or multi-line entries

Return as JSON:
{
  "headers": ["col1", "col2", "..."],
  "rows": [
    ["val1", "val2", "..."],
    ...
  ],
  "units": {"col1": "unit", "col2": "unit"},
  "notes_column": ["observation1", "observation2", "..."],
  "uncertainties": ["anything you're not sure about"]
}""",

    ZoneType.STRUCTURE: """This is a chemical structure region from a handwritten chemistry lab notebook.
This region may contain:
- Hand-drawn molecular structures (rings, bonds, atoms labeled)
- Reaction arrows with conditions written above/below
- Ion charges (+ or - superscripts)
- Labels underneath structures
- Structural formulas

For each structure or reaction you can identify:
1. Describe the structure systematically (what atoms, what connectivity, what functional groups)
2. Note any labels, charges, or subscripts
3. Identify if it's a reaction scheme (→ arrow present)
4. Note any conditions written on arrows

Return as JSON:
{
  "structures": [
    {
      "description": "...",
      "label": "...",
      "charge": "...",
      "type": "molecule|ion|complex|reaction_scheme"
    }
  ],
  "reactions": [
    {
      "reactants": ["..."],
      "products": ["..."],
      "conditions": "...",
      "arrow_label": "..."
    }
  ],
  "uncertainties": ["anything you're not sure about"]
}"""
}


# ── Extraction ─────────────────────────────────────────────────────────────────

def encode_image_b64(image: np.ndarray) -> str:
    """Encode numpy image array to base64 string for Ollama API."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')


def extract_zone(zone: Zone, client: OllamaClient,
                 model: str = "llava:13b",
                 retry_count: int = 2) -> dict:
    """
    Extract content from a single zone using the vision LLM.

    Args:
        zone: Zone object with crop image
        client: Ollama client
        model: Vision model to use
        retry_count: Number of retries on parse failure

    Returns:
        Parsed dict from LLM response, with 'raw_response' and 'parse_error' fields
    """
    prompt = PROMPTS.get(zone.zone_type, PROMPTS[ZoneType.TEXT])
    image_b64 = encode_image_b64(zone.crop)

    for attempt in range(retry_count + 1):
        try:
            response = client.chat_with_image(
                model=model,
                system=SYSTEM_PROMPT,
                prompt=prompt,
                image_b64=image_b64
            )

            raw = response.strip()

            # Parse JSON from response
            parsed = extract_json_from_response(raw)
            parsed['_raw_response'] = raw
            parsed['_parse_error'] = None
            parsed['_zone_type'] = zone.zone_type.value
            return parsed

        except json.JSONDecodeError as e:
            if attempt < retry_count:
                print(f"  [Stage 2] JSON parse failed (attempt {attempt+1}), retrying...")
                time.sleep(1)
                continue
            else:
                print(f"  [Stage 2] JSON parse failed after {retry_count+1} attempts")
                return {
                    '_raw_response': response if 'response' in dir() else '',
                    '_parse_error': str(e),
                    '_zone_type': zone.zone_type.value
                }
        except Exception as e:
            print(f"  [Stage 2] Error extracting zone: {e}")
            return {
                '_raw_response': '',
                '_parse_error': str(e),
                '_zone_type': zone.zone_type.value
            }


def extract_json_from_response(text: str) -> dict:
    """
    Robustly parse JSON from LLM response.
    LLMs often wrap JSON in markdown code blocks — strip those.
    """
    # Remove markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first JSON object in the response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())

    # If nothing works, return the text as-is in a wrapper
    return {"raw_text": text}


def extract_all_zones(zones: List[Zone],
                       client: OllamaClient,
                       model: str = "llava:13b",
                       output_dir: Optional[str] = None) -> List[Zone]:
    """
    Run extraction on all zones. Updates zone.extraction_result in place.

    Args:
        zones: List of Zone objects from Stage 1
        client: Ollama client
        model: Vision model name
        output_dir: If set, saves each zone crop as a debug image

    Returns:
        Same list of zones, with extraction_result populated
    """
    print(f"[Stage 2] Extracting {len(zones)} zones with model '{model}'...")

    for i, zone in enumerate(zones):
        print(f"[Stage 2] Zone {i+1}/{len(zones)}: {zone.zone_type.value} "
              f"(bbox y={zone.bbox[1]}, h={zone.bbox[3]})")

        if zone.crop is None or zone.crop.size == 0:
            print(f"  [Stage 2] Empty crop, skipping")
            zone.extraction_result = {'_parse_error': 'empty crop',
                                       '_zone_type': zone.zone_type.value}
            continue

        # Save zone crop for debugging
        if output_dir:
            crop_path = Path(output_dir) / f"zone_{i:02d}_{zone.zone_type.value}.jpg"
            cv2.imwrite(str(crop_path), zone.crop)

        result = extract_zone(zone, client, model)
        zone.extraction_result = result

        status = "OK" if not result.get('_parse_error') else f"ERROR: {result['_parse_error'][:50]}"
        print(f"  [Stage 2] {status}")

    successes = sum(1 for z in zones if not z.extraction_result.get('_parse_error'))
    print(f"[Stage 2] Done. {successes}/{len(zones)} zones extracted successfully.")
    return zones
