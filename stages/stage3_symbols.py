"""
Stage 3: Symbol Post-Processing
--------------------------------
Rule-based cleanup of extracted text to fix the specific symbol mangling
patterns that vision LLMs consistently get wrong on scientific content.

This is the stage that directly addresses LabAlly's stated problem:
"frontier VLMs are only ~50% accurate on critical symbol extraction."

The fix isn't just better prompts — it's a deterministic rule layer
that catches and corrects predictable failure modes AFTER the LLM runs.

Failure mode categories addressed:
  1. Scientific notation: "1.5E-4" vs "1.5e-4" vs "1.5×10-4" vs "1.5·10⁻⁴"
  2. Degree symbol: "30C" → "30°C", "22.4C" → "22.4°C"
  3. Greek letters: "theta" → "θ", "omega" → "ω", "lambda" → "λ"
  4. Superscripts: "cm2" → "cm²", "mA/cm2" → "mA/cm²", "Li+" → "Li⁺"
  5. Subscripts in formulas: "H2O" → "H₂O", "LiTFSI" stays as-is
  6. Ratio notation: "4:1 v/v" preserved, "mol %" → "mol%"
  7. XRD-specific: "2theta" → "2θ", "20 = 2.1" → "2θ = 2.1°"

Each rule is a (pattern, replacement) pair applied in order.
Order matters: more specific patterns before more general ones.
"""

import re
import json
from typing import Any, Union


# ── Rule Definitions ───────────────────────────────────────────────────────────

# Scientific notation normalization
# All variants → standard Python float string: e.g. "1.5E-4"
SCIENTIFIC_NOTATION_RULES = [
    # "1.5×10-4" or "1.5×10⁻⁴" → "1.5E-4"
    (r'(\d+\.?\d*)\s*[×x]\s*10\s*[\^]?\s*[-−](\d+)', r'\1E-\2'),
    (r'(\d+\.?\d*)\s*[×x]\s*10\s*⁻(\d+)', r'\1E-\2'),
    (r'(\d+\.?\d*)\s*[×x]\s*10\s*\^(\d+)', r'\1E+\2'),
    # "8.4 E-6" → "8.4E-6" (remove space)
    (r'(\d+\.?\d*)\s+[Ee]\s*([+-]?\d+)', r'\1E\2'),
    # Normalize case: "8.4e-6" → "8.4E-6"
    (r'(\d+\.?\d*)e([+-]\d+)', r'\1E\2'),
]

# Degree symbol rules
DEGREE_RULES = [
    # "30C" or "30 C" after a number → "30°C" (but not "mA/cm2C")
    (r'(\d+\.?\d*)\s*°?\s*C\b(?!/)', r'\1°C'),
    # "22.4C" where C is temperature
    (r'(\d+\.?\d+)\s*C\b', r'\1°C'),
    # "30 degrees C" → "30°C"
    (r'(\d+\.?\d*)\s+degrees?\s+C', r'\1°C'),
    # Standalone degree: "°" already correct, skip
]

# Greek letter rules (text → symbol)
GREEK_RULES = [
    # 2θ patterns first (XRD context)
    (r'\b2\s*[Tt]heta\b', '2θ'),
    (r'\b2\s*[Tt]h\b', '2θ'),
    (r'\b20\s*=', '2θ ='),   # Common OCR error: "20" for "2θ"
    # Standalone Greek
    (r'\b[Tt]heta\b', 'θ'),
    (r'\b[Oo]mega\b', 'ω'),
    (r'\b[Ll]ambda\b', 'λ'),
    (r'\b[Aa]lpha\b', 'α'),
    (r'\b[Bb]eta\b', 'β'),
    (r'\b[Dd]elta\b', 'Δ'),
    # Shorthand
    (r'\bw\s*=\s*(\d)', r'ω = \1'),   # "w = 1600" → "ω = 1600"
]

# Superscript/subscript rules for units and chemistry
SUPER_SUB_RULES = [
    # Units with superscripts
    (r'\bcm\s*\^?\s*2\b', 'cm²'),
    (r'\bcm\s*2\b', 'cm²'),
    (r'\bm\s*2\b(?!mol)', 'm²'),
    # Combined units
    (r'mA\s*/\s*cm\s*\^?\s*2', 'mA/cm²'),
    (r'mA\s*/\s*cm2\b', 'mA/cm²'),
    (r'A\s*/\s*cm\s*\^?\s*2', 'A/cm²'),
    # Ion charges
    (r'\bLi\s*\+', 'Li⁺'),
    (r'\be\s*-(?=\s|$|\))', 'e⁻'),
    (r'\bAg\s*\+', 'Ag⁺'),
    # Chemical formula subscripts
    (r'\bH\s*2\s*O\b', 'H₂O'),
    (r'\bCO\s*2\b', 'CO₂'),
]

# Chemistry-specific notation rules
CHEM_RULES = [
    # Ratio notation cleanup
    (r'mol\s+%', 'mol%'),
    (r'v\s*/\s*v\b', 'v/v'),
    (r'(\d+)\s*:\s*(\d+)\s*v\s*/\s*v', r'\1:\2 v/v'),
    # Concentration
    (r'\b(\d+\.?\d*)\s*M\b(?!Hz|Pa|ol)', r'\1 M'),  # 1M → 1 M (molarity)
    # Reference electrode
    (r'Ag\s*/\s*Ag\s*Cl', 'Ag/AgCl'),
    # Common abbreviations
    (r'\bvs\.\s*', 'vs '),
]

# XRD-specific rules
XRD_RULES = [
    (r'2\s*[Tt]heta\s*=\s*(\d+\.?\d*)\s*°?', r'2θ = \1°'),
    (r'2θ\s*=\s*(\d+\.?\d*)\b(?!°)', r'2θ = \1°'),
    (r'[Ss]houlder\s+at\s+2θ', 'shoulder at 2θ'),
    (r'[Pp]eak\s+at\s+2θ', 'peak at 2θ'),
    (r'\(low\s+int(?:ens?\.?|ensity)?\)', '(low intensity)'),
]

# OCR confusion fixes — must run BEFORE other rules
# Vision models confuse: O/0, l/1, S/5, I/1
# Use a lambda for numeric-string fixes to handle multiple simultaneous confusions
import re as _re

def _fix_decimal(m):
    """Fix O->0 and S->5 in a decimal number string like O.SO -> 0.50"""
    s = m.group(0)
# OCR confusion fixes — must run BEFORE other rules
# Vision models confuse: O/0, l/1, S/5, I/1

def _fix_decimal(m):
    """Fix O->0 and S->5 in a decimal number string like O.SO -> 0.50"""
    return m.group(0).replace('O', '0').replace('S', '5')

OCR_FIX_RULES = [
    (r'[OS][.][0-9OS]+', _fix_decimal),    # O.SO->0.50, O.45->0.45
    (r'(?<=\d)O(?=\d)', '0'),              # 1O5 -> 105
    (r'<\s*l\s*ppm', '< 1 ppm'),           # < l ppm -> < 1 ppm
    (r'\blM\b', '1M'),                     # lM -> 1M (before space rule)
    (r'\bl\s+M\b', '1 M'),                 # l M -> 1 M
    (r'\bS\s*mol\s*%', '5 mol%'),          # S mol% -> 5 mol%
    (r'\bS%', '5%'),                       # S% -> 5%
    (r'(\d{6}-[A-Z]+)I\b', r'\g<1>1'),     # 240604-BI -> 240604-B1
]

# All rules in application order — OCR fixes first
ALL_RULES = (
    OCR_FIX_RULES +
    SCIENTIFIC_NOTATION_RULES +
    DEGREE_RULES +
    GREEK_RULES +
    SUPER_SUB_RULES +
    CHEM_RULES +
    XRD_RULES
)


# ── Application ────────────────────────────────────────────────────────────────

def fix_symbols(text: str) -> str:
    """
    Apply all symbol correction rules to a string.
    Returns corrected string.
    """
    if not isinstance(text, str):
        return text

    result = text
    for pattern, replacement in ALL_RULES:
        try:
            result = re.sub(pattern, replacement, result)
        except re.error:
            continue  # Skip malformed patterns

    return result


def fix_dict(obj: Any) -> Any:
    """
    Recursively apply symbol fixes to all string values in a dict/list.
    Preserves structure, only modifies string leaf values.
    Keys that start with '_' (internal metadata) are skipped.
    """
    if isinstance(obj, str):
        return fix_symbols(obj)
    elif isinstance(obj, dict):
        return {
            k: (v if k.startswith('_') else fix_dict(v))
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [fix_dict(item) for item in obj]
    else:
        return obj  # int, float, None, bool — pass through


def process_zones(zones: list) -> list:
    """
    Apply symbol post-processing to all zone extraction results.
    Updates zone.extraction_result in place.

    Args:
        zones: List of Zone objects from Stage 2

    Returns:
        Same list with corrected extraction_results
    """
    print(f"[Stage 3] Applying symbol corrections to {len(zones)} zones...")

    corrections_total = 0

    for zone in zones:
        if not zone.extraction_result:
            continue

        original = json.dumps(zone.extraction_result)
        corrected = fix_dict(zone.extraction_result)
        zone.extraction_result = corrected

        # Count changes for reporting
        corrected_str = json.dumps(corrected)
        if original != corrected_str:
            corrections_total += 1

    print(f"[Stage 3] Applied corrections to {corrections_total} zones.")
    return zones


# ── Testing ────────────────────────────────────────────────────────────────────

def run_symbol_tests():
    """
    Sanity check the rules against known LLM failure patterns.
    Run directly: python -m stages.stage3_symbols
    """
    test_cases = [
        # (input, expected_output)
        ("temperature: 30C",           "temperature: 30°C"),
        ("J = 0.5 mA/cm2",            "J = 0.5 mA/cm²"),
        ("n = 8.4 E-6 mol",           "n = 8.4E-6 mol"),
        ("1.5e-4 A",                   "1.5E-4 A"),
        ("8.4×10-6 mol",               "8.4E-6 mol"),
        ("w = 1600 rpm",               "ω = 1600 rpm"),
        ("2theta = 2.1",              "2θ = 2.1°"),
        ("20 = 4.7",                  "2θ = 4.7°"),   # OCR: "20" → "2θ"
        ("H2O < 1 ppm",               "H₂O < 1 ppm"),
        ("Li+",                        "Li⁺"),
        ("e-",                         "e⁻"),
        ("Ag/AgCl",                    "Ag/AgCl"),     # Should be preserved
        ("4:1 v/v",                    "4:1 v/v"),     # Should be preserved
        ("5 mol %",                    "5 mol%"),
        ("T @ 22.4C",                 "T @ 22.4°C"),
    ]

    passed = 0
    failed = 0

    print("Running symbol correction tests...\n")
    for input_text, expected in test_cases:
        result = fix_symbols(input_text)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] '{input_text}'")
        if not ok:
            print(f"         Expected: '{expected}'")
            print(f"         Got:      '{result}'")

    print(f"\n{passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    run_symbol_tests()
