"""
Method 4: FFT-based Registration (Phase Correlation)
Author: [Team Member 4]
Date: 2025

This module implements frequency-domain image registration using phase correlation.
Fast and efficient for rigid transformations (translation, rotation).
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
import time


class FFTRegistration:
    """
    FFT-based registration using phase correlation.

    This method works in the frequency domain to find translation offsets.
    Very fast but limited to rigid transformations.
    """

    def __init__(
        self,
        upsample_factor: int = 10,
        handle_rotation: bool = True,
        max_rotation: float = 45.0
    ):
        """
        Initialize FFT-based registration.

        Args:
            upsample_factor: Upsampling factor for sub-pixel accuracy (1-100)
            handle_rotation: Whether to search for rotation (slower but more robust)
            max_rotation: Maximum rotation angle to search in degrees
        """
        self.upsample_factor = upsample_factor
        self.handle_rotation = handle_rotation
        self.max_rotation = max_rotation

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize to [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)

        return image

    def compute_rotation_fft(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Estimate rotation angle using FFT-based log-polar transform.

        Args:
            image1: Reference image
            image2: Image to align

        Returns:
            Rotation angle in degrees
        """
        # Convert to log-polar coordinates
        h, w = image1.shape
        center = (w // 2, h // 2)
        radius = min(center)

        # Create log-polar transform
        def to_log_polar(img):
            # Simple approach: convert to polar, then take log of radius
            y, x = np.indices(img.shape)
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            theta = np.arctan2(y - center[1], x - center[0])

            # Avoid log(0)
            r = np.clip(r, 1, radius)

            # Map to log-polar
            log_r = np.log(r)

            # Normalize angles to [0, 2π]
            theta = (theta + np.pi) / (2 * np.pi)

            return theta, log_r

        # Try rotation estimation (simplified version)
        # Full implementation would use FFT on log-polar transform
        # For now, try discrete angles
        best_angle = 0
        best_score = -np.inf

        for angle in np.arange(-self.max_rotation, self.max_rotation, 1.0):
            # Rotate image2
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image2, M, (w, h))

            # Compute similarity (normalized cross-correlation)
            score = np.sum(image1 * rotated) / (np.linalg.norm(image1) * np.linalg.norm(rotated))

            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle

    def estimate_translation(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Estimate translation using phase correlation.

        Args:
            image1: Reference image (fixed)
            image2: Moving image

        Returns:
            Tuple of (shift vector, correlation score, stats dict)
        """
        # Normalize images
        img1 = self.normalize_image(image1)
        img2 = self.normalize_image(image2)

        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Use scikit-image phase correlation
        # Call without return_error first, handle result
        result = phase_cross_correlation(
            img1, img2,
            upsample_factor=self.upsample_factor
        )

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            # Newer version returns (shift, error, diffphase)
            shift, error, diffphase = result
            correlation = 1.0 - error
        elif isinstance(result, tuple) and len(result) == 2:
            # Some versions return (shift, error)
            shift, error = result
            correlation = 1.0 - error
        else:
            # Older version returns just shift
            shift = result
            # Estimate correlation manually
            h, w = img1.shape

            # Extract shift values carefully
            if hasattr(shift, '__len__') and len(shift) >= 2:
                shift_y = float(shift[0])
                shift_x = float(shift[1])
            else:
                shift_y = 0.0
                shift_x = 0.0

            M = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
            shifted = cv2.warpAffine(img2, M, (w, h))

            # Simple correlation measure
            mask = (img1 > 0) & (shifted > 0)
            if mask.sum() > 100:
                diff = np.sum((img1[mask] - shifted[mask]) ** 2)
                max_diff = np.sum(img1[mask] ** 2)
                correlation = 1.0 - (diff / (max_diff + 1e-6))
                correlation = max(0.0, min(1.0, correlation))
            else:
                correlation = 0.0
            error = 1.0 - correlation

        # Extract scalar values from shift array safely
        if hasattr(shift, '__len__') and len(shift) >= 2:
            shift_y = float(shift[0])
            shift_x = float(shift[1])
        else:
            shift_y = 0.0
            shift_x = 0.0

        stats = {
            'shift_y': shift_y,
            'shift_x': shift_x,
            'correlation': float(correlation),
            'error': float(error)
        }

        return shift, correlation, stats

    def register_images(
        self,
        fixed: np.ndarray,
        moving: np.ndarray
    ) -> Dict:
        """
        Register moving image to fixed image using phase correlation.

        Args:
            fixed: Reference image
            moving: Image to align

        Returns:
            Dictionary containing:
                - registered_image: Aligned image
                - shift: Translation vector
                - rotation: Rotation angle (if enabled)
                - correlation: Similarity score
                - transform: 2x3 affine transformation matrix
                - success: Whether registration succeeded
        """
        # Handle rotation if enabled
        rotation_angle = 0.0
        preprocessed = moving.copy()

        if self.handle_rotation:
            rotation_angle = self.compute_rotation_fft(fixed, moving)

            # Apply rotation
            h, w = moving.shape
            center = (w // 2, h // 2)
            M_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            preprocessed = cv2.warpAffine(moving, M_rot, (w, h))

        # Estimate translation
        shift, correlation, stats = self.estimate_translation(fixed, preprocessed)

        # Create transformation matrix
        h, w = fixed.shape

        # Combined rotation + translation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        M[0, 2] += shift[1]  # x translation
        M[1, 2] += shift[0]  # y translation

        # Apply transformation
        registered = cv2.warpAffine(
            moving, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Check if registration succeeded (high correlation)
        success = correlation > 0.3

        return {
            'registered_image': registered,
            'shift': shift,
            'rotation': rotation_angle,
            'correlation': correlation,
            'transform': M,
            'stats': stats,
            'success': success
        }

    def find_best_atlas_slice(
        self,
        query_image: np.ndarray,
        atlas_dir: str,
        top_k: int = 5,
        verbose: bool = True
    ):
        """
        Find best matching atlas slice using phase correlation.

        Args:
            query_image: Query histology image
            atlas_dir: Directory containing atlas slices
            top_k: Number of top matches to return
            verbose: Print progress

        Returns:
            List of (filename, correlation_score, stats) tuples
        """
        atlas_dir = Path(atlas_dir) if isinstance(atlas_dir, str) else atlas_dir
        atlas_paths = list(atlas_dir.glob("*.png")) + list(atlas_dir.glob("*.jpg"))

        if len(atlas_paths) == 0:
            raise ValueError(f"No atlas images found in {atlas_dir}")

        results = []

        for i, atlas_path in enumerate(atlas_paths):
            # Load atlas slice
            atlas_img = cv2.imread(str(atlas_path), cv2.IMREAD_GRAYSCALE)
            if atlas_img is None:
                continue

            # Resize to match query size
            if atlas_img.shape != query_image.shape:
                atlas_img = cv2.resize(
                    atlas_img,
                    (query_image.shape[1], query_image.shape[0])
                )

            # Compute phase correlation
            shift, correlation, stats = self.estimate_translation(atlas_img, query_image)

            results.append((atlas_path.name, correlation, stats))

            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(atlas_paths)} slices...")

        # Sort by correlation (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        if verbose:
            print(f"\n📊 Top {min(top_k, len(results))} matches:")
            for rank, (filename, corr, stats) in enumerate(results[:top_k], 1):
                print(f"  {rank}. {filename}: correlation={corr:.4f}")

        return results[:top_k]


def compute_alignment_metrics(fixed: np.ndarray, moving: np.ndarray) -> Dict[str, float]:
    """
    Compute alignment quality metrics.

    Args:
        fixed: Reference image
        moving: Registered image

    Returns:
        Dictionary with NMI, SSIM, MSE
    """
    from skimage.metrics import structural_similarity

    # Ensure same shape
    if fixed.shape != moving.shape:
        moving = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]))

    # Create mask
    mask = (fixed > 0) & (moving > 0)

    # Extract foreground
    fixed_fg = fixed[mask].astype(np.float32)
    moving_fg = moving[mask].astype(np.float32)

    # Normalized Mutual Information
    if len(fixed_fg) > 0:
        hist, _, _ = np.histogram2d(fixed_fg, moving_fg, bins=64)
        pxy = hist / hist.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)

        eps = 1e-12
        hx = -np.sum(px * np.log(px + eps))
        hy = -np.sum(py * np.log(py + eps))
        hxy = -np.sum(pxy * np.log(pxy + eps))

        nmi = (hx + hy) / max(hxy, eps)
    else:
        nmi = 0.0

    # SSIM
    ssim = structural_similarity(fixed, moving, data_range=fixed.max() - fixed.min())

    # MSE
    mse = np.mean((fixed.astype(float) - moving.astype(float)) ** 2)

    return {
        'nmi': float(nmi),
        'ssim': float(ssim),
        'mse': float(mse)
    }
