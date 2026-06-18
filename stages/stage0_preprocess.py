"""
Stage 0: Preprocessing
----------------------
Cleans and normalizes the raw notebook image before any extraction.

Operations:
- Deskew (correct camera tilt)
- Adaptive contrast enhancement (handles uneven lighting from phone photos)
- Denoise (remove JPEG artifacts)
- Normalize to consistent DPI-equivalent resolution
- Binarize a copy for layout detection (Stage 1 uses this)

Returns both the enhanced color image (for vision LLM) and the
binarized image (for OpenCV contour/line detection).
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    """Load image from path, handle both JPEG and PNG."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct skew introduced by camera angle when photographing notebook.
    Uses Hough line detection to find dominant horizontal lines (notebook rules)
    and rotates to align them.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines via Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    if lines is None:
        return image  # Can't detect lines, return as-is

    # Collect angles of near-horizontal lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Near-horizontal lines: theta close to 0 or pi
        angle_deg = np.degrees(theta)
        if angle_deg < 20 or angle_deg > 160:
            # Convert to deviation from horizontal
            if angle_deg > 90:
                angles.append(angle_deg - 180)
            else:
                angles.append(angle_deg)

    if not angles:
        return image

    # Use median angle to avoid outlier influence
    skew_angle = np.median(angles)

    # Only correct if skew is meaningful (>0.5°) and not extreme (>15°)
    if abs(skew_angle) < 0.5 or abs(skew_angle) > 15:
        return image

    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Adaptive contrast enhancement for uneven phone-photo lighting.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the L channel in LAB color space — preserves color while
    boosting local contrast where handwriting is faint.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE: tile size 8x8, clip limit 2.0 (prevents noise amplification)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Remove JPEG compression artifacts and noise.
    Uses Non-Local Means Denoising — slower than Gaussian blur
    but preserves edges (critical for handwriting strokes).
    h=10 is conservative: reduces noise without blurring thin strokes.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10,
                                            templateWindowSize=7,
                                            searchWindowSize=21)


def normalize_size(image: np.ndarray, target_width: int = 2000) -> np.ndarray:
    """
    Normalize to consistent width for reliable zone detection.
    Vision models also perform better with consistent input sizes.
    Upscale if smaller, downscale if larger — always preserve aspect ratio.
    """
    h, w = image.shape[:2]
    if w == target_width:
        return image

    scale = target_width / w
    new_h = int(h * scale)
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (target_width, new_h), interpolation=interpolation)


def binarize(image: np.ndarray) -> np.ndarray:
    """
    Create a clean black-on-white binary image for layout detection.
    Uses Sauvola-style adaptive thresholding (via cv2.adaptiveThreshold)
    which handles local lighting variations better than global Otsu.

    This binary image is used by Stage 1 for contour/line detection —
    NOT sent to the vision LLM (which gets the enhanced color image).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: 51x51 block size, C=15 offset
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=15
    )
    return binary


def preprocess(image_path: str, output_dir: str = None) -> dict:
    """
    Full preprocessing pipeline.

    Args:
        image_path: Path to raw notebook image
        output_dir: If provided, saves intermediate images here for debugging

    Returns:
        {
            'enhanced': np.ndarray,   # Color enhanced image → sent to vision LLM
            'binary': np.ndarray,     # B&W binary → used by Stage 1 segmenter
            'original': np.ndarray,   # Original loaded image
            'shape': (h, w)
        }
    """
    print(f"[Stage 0] Loading image: {image_path}")
    original = load_image(image_path)
    print(f"[Stage 0] Original size: {original.shape[1]}x{original.shape[0]}")

    print("[Stage 0] Normalizing size...")
    normalized = normalize_size(original)

    print("[Stage 0] Deskewing...")
    deskewed = deskew(normalized)

    print("[Stage 0] Enhancing contrast...")
    enhanced = enhance_contrast(deskewed)

    print("[Stage 0] Denoising...")
    denoised = denoise(enhanced)

    print("[Stage 0] Binarizing for layout detection...")
    binary = binarize(denoised)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "stage0_enhanced.jpg"), denoised)
        cv2.imwrite(str(out / "stage0_binary.jpg"), binary)
        print(f"[Stage 0] Saved debug images to {output_dir}")

    print(f"[Stage 0] Done. Output size: {denoised.shape[1]}x{denoised.shape[0]}")

    return {
        'enhanced': denoised,
        'binary': binary,
        'original': original,
        'shape': (denoised.shape[0], denoised.shape[1])
    }
