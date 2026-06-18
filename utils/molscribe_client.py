"""
MolScribe Integration
---------------------
MolScribe converts hand-drawn molecular structure images → SMILES strings.
Developed at MIT: https://github.com/thomas0809/MolScribe

This is plugged into Stage 2 as a specialist handler for STRUCTURE zones,
running in parallel with the VLM's prose description. The VLM describes
what it sees; MolScribe gives the machine-readable SMILES.

Installation (requires torch, needs ~2GB):
    pip install molscribe

Model download (~500MB, happens automatically on first call):
    from molscribe import MolScribe
    model = MolScribe("swin_base_char_aux_1m680k", device="cpu")

Why this matters:
    VLM output:    "A crown ether ring with Li+ coordinated inside"
    MolScribe:     "[Li+].C1COCCOCCOCCO1"   ← actually machine-readable
    
    The SMILES string can be:
    - Validated with RDKit (is it a real molecule?)
    - Searched in PubChem/ChemDraw
    - Used to compute molecular properties
    - Fed into downstream ML models
"""

import numpy as np
from typing import Optional
import cv2


def is_molscribe_available() -> bool:
    """Check if MolScribe and its dependencies are installed and loadable."""
    try:
        import molscribe
        return True
    except (ImportError, OSError):
        # OSError catches torch shared library errors on some systems
        return False


class MolScribeClient:
    """
    Wrapper around MolScribe for SMILES extraction from structure images.
    
    Lazy-loads the model on first use — it's ~500MB so we don't want
    to load it if no structure zones are detected on the page.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def _load_model(self):
        """Load MolScribe model on first use."""
        if self._model is not None:
            return
        try:
            from molscribe import MolScribe
            print("[MolScribe] Loading model (first use — downloads ~500MB)...")
            self._model = MolScribe("swin_base_char_aux_1m680k",
                                     device=self.device)
            print("[MolScribe] Model ready.")
        except ImportError:
            raise RuntimeError(
                "MolScribe not installed. Run: pip install molscribe\n"
                "Requires PyTorch — install torch first."
            )

    def predict_smiles(self, image: np.ndarray) -> Optional[str]:
        """
        Convert a chemical structure image crop to a SMILES string.

        Args:
            image: BGR numpy array of the structure region

        Returns:
            SMILES string, or None if prediction fails/confidence too low
        """
        self._load_model()

        try:
            from PIL import Image as PILImage

            # MolScribe expects PIL Image in RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            output = self._model.predict_image(pil_img)

            smiles = output.get('smiles', '')
            confidence = output.get('confidence', 0.0)

            print(f"[MolScribe] SMILES: {smiles} (confidence: {confidence:.2f})")

            # Only return if confidence is reasonable
            if confidence > 0.3 and smiles and smiles != '[C]':
                return smiles
            else:
                print(f"[MolScribe] Low confidence ({confidence:.2f}), skipping")
                return None

        except Exception as e:
            print(f"[MolScribe] Prediction failed: {e}")
            return None

    def predict_smiles_batch(self,
                              images: list) -> list:
        """
        Predict SMILES for multiple structure images.
        Batching is more efficient than individual calls.
        """
        self._load_model()
        results = []
        for img in images:
            results.append(self.predict_smiles(img))
        return results


def enrich_structures_with_smiles(zones: list,
                                   molscribe: MolScribeClient) -> list:
    """
    For all STRUCTURE zones that have been extracted, attempt to
    add SMILES strings via MolScribe.

    Updates zone.extraction_result['structures'][i]['smiles'] in place.
    Non-destructive: if MolScribe fails, existing data is unchanged.
    """
    from stages.stage1_segment import ZoneType

    structure_zones = [z for z in zones
                       if z.zone_type == ZoneType.STRUCTURE
                       and z.crop is not None]

    if not structure_zones:
        return zones

    print(f"[MolScribe] Processing {len(structure_zones)} structure zones...")

    for zone in structure_zones:
        smiles = molscribe.predict_smiles(zone.crop)

        if smiles and zone.extraction_result:
            # Add SMILES to the first structure in the extraction result
            structures = zone.extraction_result.get('structures', [])
            if structures:
                structures[0]['smiles'] = smiles
            else:
                zone.extraction_result['smiles'] = smiles

    return zones
