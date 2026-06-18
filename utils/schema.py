"""
Pydantic Schema + Validation
-----------------------------
Typed output schema for the experiment record.
Gives us:
- Automatic type coercion (string "0.81" → float 0.81)
- Field-level validation with clear error messages
- A contract between pipeline stages
- Free diff-based evaluation harness

This is also the structured output target we'd pass to a VLM's
structured output mode (e.g. Claude API with response_format).
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re


class Metadata(BaseModel):
    page: Optional[int] = None
    date: Optional[str] = None
    project: Optional[str] = None
    run_id: Optional[str] = None
    continued_from: Optional[str] = None


class Electrolyte(BaseModel):
    salt: Optional[str] = None
    solvent: Optional[str] = None
    additive: Optional[str] = None
    volume_mL: Optional[float] = None
    preparation_notes: Optional[str] = None


class Conditions(BaseModel):
    temperature_C: Optional[float] = None
    atmosphere: Optional[str] = None
    water_ppm: Optional[str] = None
    other: Optional[str] = None


class Electrodes(BaseModel):
    working: Optional[str] = None
    counter: Optional[str] = None
    reference: Optional[str] = None


class Deposition(BaseModel):
    potential_V: Optional[float] = None
    reference: Optional[str] = None
    duration_min: Optional[float] = None
    duration_s: Optional[float] = None
    rotation_rpm: Optional[float] = None
    current_density_mA_cm2: Optional[float] = None
    electrode_area_cm2: Optional[float] = None
    current_A: Optional[float] = None
    charge_C: Optional[float] = None
    moles_deposited: Optional[float] = None
    mass_deposited_g: Optional[float] = None

    @field_validator('current_density_mA_cm2', 'charge_C',
                     'moles_deposited', 'mass_deposited_g',
                     'potential_V', mode='before')
    @classmethod
    def coerce_numeric_string(cls, v):
        """
        Handle OCR artifacts in numeric fields.
        Converts strings like '0.50?' or 'O.81' to floats.
        Returns None if unparseable.
        """
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            # Strip uncertainty markers
            cleaned = v.replace('?', '').strip()
            # Fix common OCR: O->0, S->5
            cleaned = cleaned.replace('O', '0').replace('S', '5')
            try:
                return float(cleaned)
            except ValueError:
                return None
        return v


class ChemicalStructure(BaseModel):
    label: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    smiles: Optional[str] = None        # Filled by MolScribe if available
    role: Optional[str] = None


class Reaction(BaseModel):
    equation: Optional[str] = None
    conditions: Optional[str] = None


class XRD(BaseModel):
    peaks: Optional[List[float]] = None
    notes: Optional[str] = None

    @field_validator('peaks', mode='before')
    @classmethod
    def coerce_peaks(cls, v):
        """Convert list of mixed types to list of floats."""
        if v is None:
            return None
        if isinstance(v, list):
            result = []
            for item in v:
                try:
                    result.append(float(item))
                except (ValueError, TypeError):
                    pass
            return result if result else None
        return v


class Observations(BaseModel):
    visual: Optional[str] = None
    xrd: Optional[XRD] = None
    other: Optional[List[str]] = None


class TemperaturePoint(BaseModel):
    time: Optional[str] = None
    temp_C: Optional[float] = None

    @field_validator('temp_C', mode='before')
    @classmethod
    def coerce_temp(cls, v):
        if isinstance(v, str):
            cleaned = v.replace('°C', '').replace('C', '').strip()
            cleaned = cleaned.replace('O', '0')
            try:
                return float(cleaned)
            except ValueError:
                return None
        return v


class ExperimentInterpretation(BaseModel):
    what_was_tested: Optional[str] = None
    key_finding: Optional[str] = None
    next_steps_implied: Optional[str] = None


class ExtractionQuality(BaseModel):
    zones_extracted: Optional[int] = None
    zones_successful: Optional[int] = None
    confidence: Optional[str] = None
    uncertain_fields: Optional[List[str]] = None
    assembly_failed: Optional[bool] = None


class ExperimentRecord(BaseModel):
    """
    Top-level structured output for a parsed lab notebook page.
    All fields optional — partial extraction is still valuable.
    """
    metadata: Optional[Metadata] = None
    goal: Optional[str] = None
    electrolyte: Optional[Electrolyte] = None
    conditions: Optional[Conditions] = None
    electrodes: Optional[Electrodes] = None
    deposition: Optional[Deposition] = None
    chemical_structures: Optional[List[ChemicalStructure]] = None
    reactions: Optional[List[Reaction]] = None
    observations: Optional[Observations] = None
    temperature_profile: Optional[List[TemperaturePoint]] = None
    procedure_summary: Optional[str] = None
    experiment_interpretation: Optional[ExperimentInterpretation] = None
    extraction_quality: Optional[ExtractionQuality] = None

    def completion_score(self) -> float:
        """
        Returns fraction of key fields that were successfully extracted.
        Useful for evaluation and for flagging pages that need human review.
        """
        key_fields = [
            self.metadata and self.metadata.page,
            self.metadata and self.metadata.date,
            self.goal,
            self.electrolyte and self.electrolyte.salt,
            self.conditions and self.conditions.temperature_C,
            self.electrodes and self.electrodes.working,
            self.deposition and self.deposition.potential_V,
            self.deposition and self.deposition.charge_C,
            self.temperature_profile and len(self.temperature_profile) > 0,
            self.observations and self.observations.xrd,
        ]
        filled = sum(1 for f in key_fields if f)
        return round(filled / len(key_fields), 2)

    def to_evaluation_dict(self) -> dict:
        """
        Flat dict of key fields for evaluation harness.
        Makes it easy to diff against ground truth.
        """
        return {
            "page": self.metadata.page if self.metadata else None,
            "date": self.metadata.date if self.metadata else None,
            "goal": self.goal,
            "salt": self.electrolyte.salt if self.electrolyte else None,
            "solvent": self.electrolyte.solvent if self.electrolyte else None,
            "temp_C": self.conditions.temperature_C if self.conditions else None,
            "potential_V": self.deposition.potential_V if self.deposition else None,
            "charge_C": self.deposition.charge_C if self.deposition else None,
            "moles": self.deposition.moles_deposited if self.deposition else None,
            "temp_profile_rows": len(self.temperature_profile) if self.temperature_profile else 0,
            "completion_score": self.completion_score(),
        }


def validate_output(raw_dict: dict) -> tuple[ExperimentRecord, list]:
    """
    Validate and coerce raw LLM output dict into typed ExperimentRecord.

    Returns:
        (record, errors) — record is always returned even if partial,
        errors is a list of validation issues for logging
    """
    errors = []
    try:
        record = ExperimentRecord.model_validate(raw_dict)
        return record, errors
    except Exception as e:
        errors.append(str(e))
        # Try field by field to get partial record
        partial = {}
        for field_name in ExperimentRecord.model_fields:
            try:
                if field_name in raw_dict:
                    partial[field_name] = raw_dict[field_name]
            except Exception as fe:
                errors.append(f"{field_name}: {fe}")
        try:
            record = ExperimentRecord.model_validate(partial)
        except Exception:
            record = ExperimentRecord()
        return record, errors
