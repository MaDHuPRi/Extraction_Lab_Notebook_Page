"""
Stage 1: Zone Segmentation (Robust Version)
--------------------------------------------
Previous version used brittle OpenCV heuristics to classify zones
(circular contour detection, fixed y-position thresholds, hardcoded
band heights). These were tuned to the sample page and would fail
on unseen notebook pages in a live demo.

New strategy — two-step approach:
  Step A: Structural slicing (OpenCV)
      Use ONLY horizontal ruled lines to slice the page into bands.
      This is purely mechanical — no classification, no heuristics.
      Ruled notebook lines are a reliable signal on ANY notebook page.

  Step B: Visual classification (minicpm-v)
      Send each band crop to the vision LLM with a short classification
      prompt. Let the model decide what type of content it's looking at.
      The vision model is far better at this than OpenCV heuristics.

Why this is more robust:
  - No assumptions about where math/structures appear on the page
  - No assumptions about handwriting scale or line spacing
  - No circular contour detection that fires on noise
  - Works on portrait/landscape, dense/sparse, any subject matter
  - The only OpenCV work is line detection — which is highly reliable

Tradeoff: slightly slower (one extra LLM call per band for classification)
but classification calls are fast (short prompt, small crop, ~1-2s each).
"""

import cv2
import numpy as np
import base64
import json
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ZoneType(Enum):
    HEADER = "header"
    TEXT = "text"
    MATH = "math"
    TABLE = "table"
    STRUCTURE = "chemical_structure"
    UNKNOWN = "unknown"


@dataclass
class Zone:
    """A detected region on the notebook page."""
    zone_type: ZoneType
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in pixels
    confidence: float                  # 0-1
    crop: np.ndarray = field(default=None, repr=False)
    extraction_result: dict = field(default=None)

    @property
    def area(self):
        return self.bbox[2] * self.bbox[3]

    @property
    def aspect_ratio(self):
        return self.bbox[2] / max(self.bbox[3], 1)

    def to_dict(self):
        return {
            "zone_type": self.zone_type.value,
            "bbox": {"x": self.bbox[0], "y": self.bbox[1],
                     "w": self.bbox[2], "h": self.bbox[3]},
            "confidence": round(self.confidence, 2),
            "extraction_result": self.extraction_result
        }


# ── Step A: Structural slicing ─────────────────────────────────────────────────

def detect_ruled_lines(binary: np.ndarray,
                        min_length_ratio: float = 0.3) -> List[int]:
    """
    Find y-coordinates of horizontal ruled lines on the notebook page.

    Uses morphological opening with a wide horizontal kernel — this
    isolates pixels that are part of long horizontal strokes, filtering
    out handwriting (which is short and irregular).

    min_length_ratio: line must span at least this fraction of page width.
    Lowered from 0.4 → 0.3 to catch partial lines near page edges.
    """
    h, w = binary.shape
    min_length = int(w * min_length_ratio)

    # Wide horizontal kernel extracts only long horizontal strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find rows with significant horizontal line coverage
    row_sums = np.sum(horizontal == 255, axis=1)
    threshold = min_length * 0.25
    line_rows = np.where(row_sums > threshold)[0]

    if len(line_rows) == 0:
        return []

    # Cluster adjacent rows into single line y-coordinates
    lines = []
    cluster_start = line_rows[0]
    prev = line_rows[0]

    for y in line_rows[1:]:
        if y - prev > 8:
            lines.append((cluster_start + prev) // 2)
            cluster_start = y
        prev = y
    lines.append((cluster_start + prev) // 2)

    return lines


def slice_into_bands(binary: np.ndarray,
                      enhanced: np.ndarray,
                      min_band_height: int = 25) -> List[dict]:
    """
    Slice the page into horizontal bands using ruled lines as dividers.

    Returns list of band dicts:
    {
        'y_start': int,
        'y_end': int,
        'binary_crop': np.ndarray,
        'color_crop': np.ndarray,
        'rel_y': float   # relative position on page (0=top, 1=bottom)
    }

    min_band_height: skip bands thinner than this (the ruled lines themselves).
    """
    h, w = binary.shape
    dividers = detect_ruled_lines(binary)

    print(f"[Stage 1] Found {len(dividers)} ruled lines → "
          f"{len(dividers)+1} candidate bands")

    boundaries = [0] + dividers + [h]
    bands = []

    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]

        # Skip bands that are just the ruled lines themselves
        if (y_end - y_start) < min_band_height:
            continue

        # Small padding so we don't clip ascenders/descenders
        pad = 4
        ys = max(0, y_start - pad)
        ye = min(h, y_end + pad)

        bands.append({
            'y_start': ys,
            'y_end': ye,
            'binary_crop': binary[ys:ye, :],
            'color_crop': enhanced[ys:ye, :],
            'rel_y': y_start / h
        })

    return bands


def merge_thin_bands(bands: List[dict],
                      min_height: int = 60) -> List[dict]:
    """
    Merge consecutive bands that are too thin to classify reliably.

    Very thin bands (single lines of text/math) get merged with their
    neighbor so the vision model has enough context to classify correctly.
    Merges upward (into previous band) preferentially.
    """
    if not bands:
        return bands

    merged = []
    i = 0

    while i < len(bands):
        band = bands[i]
        height = band['y_end'] - band['y_start']

        # If too thin and there's a next band, merge forward
        if height < min_height and i + 1 < len(bands):
            next_band = bands[i + 1]
            combined = {
                'y_start': band['y_start'],
                'y_end': next_band['y_end'],
                'binary_crop': np.vstack([band['binary_crop'],
                                           next_band['binary_crop']]),
                'color_crop': np.vstack([band['color_crop'],
                                          next_band['color_crop']]),
                'rel_y': band['rel_y']
            }
            merged.append(combined)
            i += 2  # Skip next band since we absorbed it
        else:
            merged.append(band)
            i += 1

    return merged


# ── Step B: Visual classification ─────────────────────────────────────────────

CLASSIFICATION_PROMPT = """Look at this crop from a handwritten chemistry lab notebook page.
Classify what type of content this region contains.

Choose EXACTLY ONE of these types:
- header: page number, date, project title, cross-references
- text: prose notes, goals, observations, experimental descriptions
- math: equations, calculations, unit conversions, numerical results
- table: structured rows and columns of data (time/temperature/measurements)
- chemical_structure: hand-drawn molecular structures, reaction arrows, bond diagrams

Rules:
- If the region is mostly blank/whitespace, pick the closest match to any content present
- A region can have BOTH text and a formula — pick whichever dominates
- Reaction arrows (→) with structures = chemical_structure
- Numbered calculations with units = math

Respond with ONLY a JSON object, nothing else:
{"type": "<one of the five types above>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}"""


def encode_crop_b64(crop: np.ndarray) -> str:
    """Encode a numpy image crop to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def classify_band_with_llm(band: dict,
                             client,
                             vision_model: str,
                             band_index: int,
                             total_bands: int) -> Tuple[ZoneType, float, str]:
    """
    Ask the vision LLM to classify a single band.

    Falls back to heuristic classification if LLM call fails,
    so a network/model error never crashes the whole pipeline.

    Returns: (ZoneType, confidence, reason)
    """
    # Fast heuristic: top 8% of page is almost always a header
    if band['rel_y'] < 0.08:
        return ZoneType.HEADER, 0.90, "top of page heuristic"

    # Fast heuristic: nearly empty band
    binary = band['binary_crop']
    dark_ratio = np.sum(binary < 128) / max(binary.size, 1)
    if dark_ratio < 0.005:
        return ZoneType.TEXT, 0.40, "nearly empty band"

    try:
        image_b64 = encode_crop_b64(band['color_crop'])

        response = client.chat_with_image(
            model=vision_model,
            prompt=CLASSIFICATION_PROMPT,
            image_b64=image_b64,
            temperature=0.0  # Deterministic classification
        )

        # Parse JSON response
        raw = response.strip()
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)

        # Find JSON object in response
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            zone_type_str = data.get('type', 'text').lower()
            confidence = float(data.get('confidence', 0.7))
            reason = data.get('reason', '')

            # Map string to ZoneType enum
            type_map = {
                'header': ZoneType.HEADER,
                'text': ZoneType.TEXT,
                'math': ZoneType.MATH,
                'table': ZoneType.TABLE,
                'chemical_structure': ZoneType.STRUCTURE,
            }
            zone_type = type_map.get(zone_type_str, ZoneType.TEXT)
            return zone_type, confidence, reason

    except Exception as e:
        print(f"  [Stage 1] Classification LLM failed for band {band_index}: {e}")

    # Fallback: simple heuristic based on position
    return _heuristic_fallback(band)


def _heuristic_fallback(band: dict) -> Tuple[ZoneType, float, str]:
    """
    Simple fallback classification when LLM is unavailable.
    Much simpler than before — just position + density.
    """
    rel_y = band['rel_y']
    height = band['y_end'] - band['y_start']
    binary = band['binary_crop']
    dark_ratio = np.sum(binary < 128) / max(binary.size, 1)

    if rel_y < 0.08:
        return ZoneType.HEADER, 0.75, "fallback: top position"
    if height > 150 and dark_ratio > 0.04:
        return ZoneType.STRUCTURE, 0.50, "fallback: tall dense region"
    if dark_ratio > 0.02 and height < 100:
        return ZoneType.MATH, 0.50, "fallback: single line with content"
    return ZoneType.TEXT, 0.60, "fallback: default"


# ── Post-processing ────────────────────────────────────────────────────────────

def merge_consecutive_zones(zones: List[Zone],
                              enhanced: np.ndarray) -> List[Zone]:
    """
    Merge consecutive zones of the same type that are close together.

    Rationale: a multi-line paragraph gets sliced into many bands by
    the ruled lines. We want to reunite them into one TEXT zone so
    Stage 2 gets full context rather than one line at a time.

    Never merges STRUCTURE or TABLE zones — these need their own
    spatial context to extract correctly.
    """
    NO_MERGE_TYPES = {ZoneType.STRUCTURE, ZoneType.TABLE}

    if not zones:
        return zones

    merged = [zones[0]]

    for zone in zones[1:]:
        prev = merged[-1]
        gap = zone.bbox[1] - (prev.bbox[1] + prev.bbox[3])

        should_merge = (
            zone.zone_type == prev.zone_type
            and zone.zone_type not in NO_MERGE_TYPES
            and prev.zone_type not in NO_MERGE_TYPES
            and gap < 80  # Within ~80px = same logical block
        )

        if should_merge:
            new_y = prev.bbox[1]
            new_h = (zone.bbox[1] + zone.bbox[3]) - prev.bbox[1]
            merged[-1].bbox = (0, new_y, prev.bbox[2], new_h)
            merged[-1].crop = enhanced[new_y:new_y + new_h, :]
            # Average confidence, weighted toward lower (more uncertain merged zones)
            merged[-1].confidence = min(prev.confidence, zone.confidence)
        else:
            merged.append(zone)

    return merged


# ── Main entry point ───────────────────────────────────────────────────────────

def segment(preprocessed: dict,
            client=None,
            vision_model: str = "minicpm-v",
            output_dir: str = None) -> List[Zone]:
    """
    Main segmentation function.

    Args:
        preprocessed:  Output dict from Stage 0 preprocess()
        client:        OllamaClient instance (required for LLM classification)
        vision_model:  Vision model name for classification
        output_dir:    If set, saves annotated debug image

    Returns:
        List of Zone objects sorted top-to-bottom, ready for Stage 2
    """
    binary = preprocessed['binary']
    enhanced = preprocessed['enhanced']
    h, w = preprocessed['shape']

    print(f"[Stage 1] Segmenting {w}x{h} image...")

    # Step A: Slice into bands using ruled lines only
    bands = slice_into_bands(binary, enhanced)
    bands = merge_thin_bands(bands, min_height=60)
    print(f"[Stage 1] {len(bands)} bands after merging thin slices")

    # Step B: Classify each band with vision LLM
    if client is None:
        print("[Stage 1] No client provided — using heuristic fallback for all bands")

    zones = []
    for i, band in enumerate(bands):
        print(f"[Stage 1] Classifying band {i+1}/{len(bands)} "
              f"(y={band['y_start']}, h={band['y_end']-band['y_start']})...",
              end=' ')

        if client is not None:
            zone_type, confidence, reason = classify_band_with_llm(
                band, client, vision_model, i, len(bands)
            )
        else:
            zone_type, confidence, reason = _heuristic_fallback(band)

        print(f"→ {zone_type.value} ({confidence:.2f}) [{reason[:50]}]")

        bw = w
        bh = band['y_end'] - band['y_start']

        zone = Zone(
            zone_type=zone_type,
            bbox=(0, band['y_start'], bw, bh),
            confidence=confidence,
            crop=band['color_crop']
        )
        zones.append(zone)

    # Merge consecutive same-type zones
    zones = merge_consecutive_zones(zones, enhanced)

    print(f"\n[Stage 1] Final zones ({len(zones)} total):")
    for z in zones:
        print(f"  [{z.zone_type.value:18s}] y={z.bbox[1]:4d}  "
              f"h={z.bbox[3]:4d}  conf={z.confidence:.2f}")

    if output_dir:
        _save_visualization(enhanced, zones, output_dir)

    return zones


def _save_visualization(image: np.ndarray,
                         zones: List[Zone],
                         output_dir: str):
    """Save annotated zone visualization for debugging."""
    import os
    from pathlib import Path

    colors = {
        ZoneType.HEADER:    (255, 165,   0),
        ZoneType.TEXT:      (  0, 200,   0),
        ZoneType.MATH:      (  0,   0, 255),
        ZoneType.TABLE:     (255,   0, 255),
        ZoneType.STRUCTURE: (  0, 200, 200),
        ZoneType.UNKNOWN:   (128, 128, 128),
    }

    vis = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, zone in enumerate(zones):
        x, y, w, h = zone.bbox
        color = colors.get(zone.zone_type, (128, 128, 128))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        label = f"{i+1}. {zone.zone_type.value} ({zone.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, font, 0.65, 2)
        cv2.rectangle(vis, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(vis, label, (x + 3, y - 4), font, 0.65, (0, 0, 0), 2)

    out_path = Path(output_dir) / "stage1_zones.jpg"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"[Stage 1] Saved zone visualization → {out_path}")
