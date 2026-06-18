"""
Stage 4: LLM Assembly
---------------------
Takes all zone extraction results (from Stages 2+3) and synthesizes
them into a single, coherent, structured experiment record.

This stage uses a TEXT-only LLM (no vision needed) — we've already
done the hard visual extraction work. The LLM here does semantic work:
understanding what the experiment was about, relating the pieces, and
producing a clean, schema-conformant output.

Why a separate assembly stage vs just asking Stage 2 to do everything?
- Zone extraction is local: each zone sees only its crop
- Assembly is global: needs to understand relationships across zones
- Example: the moles calculation in Zone 3 references the area from Zone 1
- A text LLM reading structured intermediate output is far more reliable
  than a vision LLM trying to interpret a full complex page at once

Model choice: any Ollama text model works — llama3.2, mistral-nemo, qwen3.5.
llama3.2 is fast; mistral-nemo is more instruction-following for JSON.
"""

import json
import re
from typing import List, Optional
from stages.stage1_segment import Zone, ZoneType
from utils.ollama_client import OllamaClient


ASSEMBLY_SYSTEM_PROMPT = """You are a chemistry lab data extraction expert.
You will be given extracted text and data from different regions of a 
handwritten lab notebook page. Your job is to synthesize this into a 
single structured JSON record representing the full experiment.

Rules:
- Use ONLY information present in the extracted zones. Never invent data.
- Preserve exact values (temperatures, concentrations, potentials)
- If a value appears in multiple zones, use the most specific/clear one
- Mark genuinely uncertain values with a trailing "?" in the value string
- Chemical formulas should use standard notation: LiTFSI, not Li TFSI
- All temperatures in °C unless otherwise noted
- Return ONLY valid JSON. No preamble, no explanation, no markdown fences."""


ASSEMBLY_PROMPT_TEMPLATE = """Below are the extracted contents of each region of a handwritten chemistry lab notebook page.
Synthesize these into a single structured experiment record.

=== EXTRACTED ZONE DATA ===
{zone_data}

=== OUTPUT FORMAT ===
Return a JSON object with these fields (omit fields where data is absent):

{{
  "metadata": {{
    "page": <page number as int>,
    "date": "<date string>",
    "project": "<project title>",
    "run_id": "<experiment run ID>",
    "continued_from": "<cross-reference if present>"
  }},
  "goal": "<one sentence describing what this experiment aimed to do>",
  "electrolyte": {{
    "salt": "<salt name and concentration>",
    "solvent": "<solvent name and ratio if applicable>",
    "additive": "<additive name, amount, and purpose>",
    "volume_mL": <total volume as number>,
    "preparation_notes": "<how it was prepared>"
  }},
  "conditions": {{
    "temperature_C": <number>,
    "atmosphere": "<glovebox/air/N2/etc>",
    "water_ppm": "<value or threshold>",
    "other": "<any other relevant conditions>"
  }},
  "electrodes": {{
    "working": "<material, type, area>",
    "counter": "<material>",
    "reference": "<reference electrode>"
  }},
  "deposition": {{
    "potential_V": <number>,
    "reference": "<reference electrode>",
    "duration_min": <number>,
    "duration_s": <number>,
    "rotation_rpm": <number or null>,
    "current_density_mA_cm2": <number>,
    "electrode_area_cm2": <number>,
    "current_A": <number>,
    "charge_C": <number>,
    "moles_deposited": <number>,
    "mass_deposited_g": <number>
  }},
  "chemical_structures": [
    {{
      "label": "<name or formula>",
      "type": "<molecule|ion|complex|salt>",
      "description": "<structural description>",
      "role": "<role in experiment>"
    }}
  ],
  "reactions": [
    {{
      "equation": "<reaction equation>",
      "conditions": "<potential, temperature, etc>"
    }}
  ],
  "observations": {{
    "visual": "<visual observations of film/product>",
    "xrd": {{
      "peaks": [<2theta values as numbers>],
      "notes": "<peak description>"
    }},
    "other": ["<other observations>"]
  }},
  "temperature_profile": [
    {{"time": "<time string>", "temp_C": <number>}}
  ],
  "extraction_quality": {{
    "zones_extracted": <total zones>,
    "zones_successful": <zones with no errors>,
    "confidence": "<high|medium|low>",
    "uncertain_fields": ["<field names where data was ambiguous>"]
  }}
}}"""


def format_zones_for_assembly(zones: List[Zone]) -> str:
    """
    Format all zone extraction results as a readable string for the LLM.
    Groups by zone type and cleans up internal metadata fields.
    """
    lines = []

    for i, zone in enumerate(zones):
        if not zone.extraction_result:
            continue

        result = zone.extraction_result.copy()

        # Remove internal metadata fields before sending to LLM
        raw = result.pop('_raw_response', '')
        error = result.pop('_parse_error', None)
        zone_type = result.pop('_zone_type', zone.zone_type.value)

        lines.append(f"\n--- Zone {i+1}: {zone_type.upper()} ---")

        if error:
            # If parsing failed, include the raw response so LLM can still try
            lines.append(f"[Parse failed, raw text follows]")
            lines.append(raw[:2000] if raw else "[empty]")
        else:
            lines.append(json.dumps(result, indent=2, ensure_ascii=False))

    return '\n'.join(lines)


def compute_extraction_quality(zones: List[Zone]) -> dict:
    """Compute quality metrics across all zones."""
    total = len(zones)
    successful = sum(
        1 for z in zones
        if z.extraction_result and not z.extraction_result.get('_parse_error')
    )

    if successful == total:
        confidence = "high"
    elif successful >= total * 0.7:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "zones_extracted": total,
        "zones_successful": successful,
        "confidence": confidence
    }


def assemble(zones: List[Zone],
             client: OllamaClient,
             model: str = "llama3.2:latest") -> dict:
    """
    Synthesize all zone results into a structured experiment record.

    Args:
        zones: List of Zone objects with populated extraction_results
        client: Ollama client
        model: Text model to use for assembly (no vision needed)

    Returns:
        Structured experiment dict, or partial dict with error info on failure
    """
    print(f"[Stage 4] Assembling experiment record using '{model}'...")

    zone_data = format_zones_for_assembly(zones)
    quality = compute_extraction_quality(zones)

    prompt = ASSEMBLY_PROMPT_TEMPLATE.format(zone_data=zone_data)

    try:
        response = client.chat(
            model=model,
            system=ASSEMBLY_SYSTEM_PROMPT,
            prompt=prompt
        )

        raw = response.strip()

        # Strip any markdown fences the model added despite instructions
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()

        result = json.loads(raw)

        # Inject quality metrics
        if 'extraction_quality' not in result:
            result['extraction_quality'] = {}
        result['extraction_quality'].update(quality)

        print(f"[Stage 4] Assembly complete. "
              f"Confidence: {quality['confidence']} "
              f"({quality['zones_successful']}/{quality['zones_extracted']} zones OK)")

        return result

    except json.JSONDecodeError as e:
        print(f"[Stage 4] JSON parse failed: {e}")
        print(f"[Stage 4] Raw response (first 500 chars): {raw[:500]}")

        # Return partial result with the raw response for manual inspection
        return {
            "_assembly_error": str(e),
            "_raw_assembly_response": raw,
            "extraction_quality": {
                **quality,
                "confidence": "low",
                "assembly_failed": True
            }
        }

    except Exception as e:
        print(f"[Stage 4] Assembly failed: {e}")
        return {
            "_assembly_error": str(e),
            "extraction_quality": {
                **quality,
                "confidence": "low",
                "assembly_failed": True
            }
        }
