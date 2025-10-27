"""
Method 1: Intensity-Based Registration (Mutual Information)
Author: [Team Member 1]
Date: 2025

Uses SimpleITK's Mattes Mutual Information for image registration.
"""

import cv2
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Dict, Optional, List
from pathlib import Path


class MutualInformationRegistration:
    """
    Intensity-based registration using Mutual Information.

    Uses SimpleITK's Mattes MI metric with multi-resolution optimization.
    """

    def __init__(
        self,
        downsample_factor: int = 2,
        hist_bins: int = 64,
        sample_percent: float = 0.2,
        max_iter: int = 150,
        rot_range_deg: float = 20
    ):
        """
        Initialize MI registration.

        Args:
            downsample_factor: Downsample images for speed (>1 means smaller)
            hist_bins: Number of bins for MI histogram
            sample_percent: Sampling ratio for MI computation
            max_iter: Maximum iterations per registration
            rot_range_deg: Random rotation perturbation range (degrees)
        """
        self.downsample_factor = downsample_factor
        self.hist_bins = hist_bins
        self.sample_percent = sample_percent
        self.max_iter = max_iter
        self.rot_range_deg = rot_range_deg

    def to_sitk_gray(self, img_np: np.ndarray, down: int = 1) -> sitk.Image:
        """Convert numpy array to SimpleITK image with optional downsampling."""
        if down > 1:
            img_np = cv2.resize(
                img_np,
                (img_np.shape[1] // down, img_np.shape[0] // down),
                interpolation=cv2.INTER_AREA
            )
        return sitk.GetImageFromArray(img_np.astype(np.float32))

    def make_mask(self, img_np: np.ndarray, down: int = 1,
                  thresh: int = 0, min_pixels: int = 50) -> Optional[sitk.Image]:
        """Generate binary mask to ignore pure black background."""
        if down > 1:
            img_np = cv2.resize(
                img_np,
                (img_np.shape[1] // down, img_np.shape[0] // down),
                interpolation=cv2.INTER_NEAREST
            )
        mask = (img_np > thresh).astype(np.uint8)
        if mask.sum() < min_pixels:
            return None
        return sitk.GetImageFromArray(mask)

    def normalized_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bins: int = 64
    ) -> float:
        """
        Compute Normalized Mutual Information.

        NMI = (H(X) + H(Y)) / H(X,Y)
        Higher values indicate better alignment.
        """
        x = x.astype(np.float32).ravel()
        y = y.astype(np.float32).ravel()

        if mask is not None:
            m = mask.astype(bool).ravel()
            if m.sum() == 0:
                return -np.inf
            x = x[m]
            y = y[m]

        # Adaptive intensity range (exclude extreme background)
        x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
        y_min, y_max = np.percentile(y, 1), np.percentile(y, 99)
        x = np.clip(x, x_min, x_max)
        y = np.clip(y, y_min, y_max)

        # Joint histogram
        H, _, _ = np.histogram2d(x, y, bins=bins)
        Pxy = H / np.maximum(H.sum(), 1.0)
        Px = Pxy.sum(axis=1, keepdims=True)
        Py = Pxy.sum(axis=0, keepdims=True)

        # Entropies
        eps = 1e-12
        Hx = -np.sum(Px * np.log(Px + eps))
        Hy = -np.sum(Py * np.log(Py + eps))
        Hxy = -np.sum(Pxy * np.log(Pxy + eps))

        return (Hx + Hy) / max(Hxy, eps)

    def register_to_atlas(
        self,
        fixed_np: np.ndarray,
        moving_np: np.ndarray
    ) -> Tuple[float, sitk.Transform, np.ndarray, np.ndarray]:
        """
        Perform MI registration between fixed and moving images.

        Args:
            fixed_np: Reference/atlas image
            moving_np: Query image to register

        Returns:
            Tuple of (nmi_score, transform, registered_image, downsampled_fixed)
        """
        down = self.downsample_factor

        # Convert to SimpleITK
        fixed_img = self.to_sitk_gray(fixed_np, down=down)
        moving_img = self.to_sitk_gray(moving_np, down=down)

        # Create masks
        fixed_mask = self.make_mask(fixed_np, down=down, thresh=0)
        moving_mask = self.make_mask(moving_np, down=down, thresh=0)

        def _run_registration(mask_fixed, mask_moving, sampling='RANDOM'):
            """Internal registration function with different sampling strategies."""
            reg = sitk.ImageRegistrationMethod()

            # Metric: Mattes Mutual Information
            reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.hist_bins)

            if sampling == 'RANDOM':
                reg.SetMetricSamplingStrategy(reg.RANDOM)
                reg.SetMetricSamplingPercentage(self.sample_percent)
            else:
                reg.SetMetricSamplingStrategy(reg.NONE)

            # Set masks
            if mask_fixed is not None:
                reg.SetMetricFixedMask(mask_fixed)
            if mask_moving is not None:
                reg.SetMetricMovingMask(mask_moving)

            # Interpolator
            reg.SetInterpolator(sitk.sitkLinear)

            # Multi-resolution pyramid
            reg.SetShrinkFactorsPerLevel([4, 2, 1])
            reg.SetSmoothingSigmasPerLevel([2, 1, 0])
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Initialize transform (Similarity2D: translation + rotation + scale)
            initial_tx = sitk.CenteredTransformInitializer(
                fixed_img,
                moving_img,
                sitk.Similarity2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )

            # Add random rotation perturbation to escape local minima
            if self.rot_range_deg > 0:
                angle_rad = np.deg2rad(np.random.uniform(-self.rot_range_deg, self.rot_range_deg))
                initial_tx.SetAngle(initial_tx.GetAngle() + float(angle_rad))

            # Optimizer: Regular Step Gradient Descent
            reg.SetOptimizerAsRegularStepGradientDescent(
                learningRate=2.0,
                minStep=1e-3,
                numberOfIterations=self.max_iter,
                relaxationFactor=0.5,
                gradientMagnitudeTolerance=1e-6
            )
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetInitialTransform(initial_tx, inPlace=False)

            # Execute registration
            final_tx = reg.Execute(fixed_img, moving_img)

            # Apply transformation
            moved = sitk.Resample(
                moving_img,
                fixed_img,
                final_tx,
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32
            )

            return final_tx, moved

        # Try with mask + random sampling (preferred)
        try:
            final_tx, moved = _run_registration(fixed_mask, moving_mask, sampling='RANDOM')
        except Exception:
            # Fallback: no mask
            try:
                final_tx, moved = _run_registration(None, None, sampling='RANDOM')
            except Exception:
                # Final fallback: no sampling
                final_tx, moved = _run_registration(None, None, sampling='NONE')

        # Convert results back to numpy
        moved_np = sitk.GetArrayFromImage(moved)
        fixed_np_ds = sitk.GetArrayFromImage(fixed_img)

        # Compute NMI on registered result
        joint_mask = ((moved_np > 0) & (fixed_np_ds > 0)).astype(np.uint8)
        nmi = self.normalized_mutual_information(
            fixed_np_ds, moved_np,
            mask=joint_mask,
            bins=self.hist_bins
        )

        return nmi, final_tx, moved_np, fixed_np_ds

    def find_best_atlas_slice(
        self,
        query_image: np.ndarray,
        atlas_dir: str,
        top_k: int = 5,
        verbose: bool = True,
        coarse_search: bool = True,
        coarse_stride: int = 4
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find best matching atlas slice using MI registration.

        Args:
            query_image: Query histology image
            atlas_dir: Directory containing atlas slices
            top_k: Number of top matches to return
            verbose: Print progress
            coarse_search: Use coarse-to-fine search (much faster!)
            coarse_stride: Check every Nth slice in coarse search

        Returns:
            List of (filename, nmi_score, stats) tuples sorted by NMI
        """
        atlas_dir = Path(atlas_dir) if isinstance(atlas_dir, str) else atlas_dir
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        atlas_paths = sorted([p for p in atlas_dir.glob("**/*") if p.suffix.lower() in exts])

        if len(atlas_paths) == 0:
            raise ValueError(f"No atlas images found in {atlas_dir}")

        if verbose:
            print(f"✓ Found {len(atlas_paths)} atlas slices")

        # Ensure query is uint8
        query_np = query_image.copy()
        if query_np.dtype != np.uint8:
            query_np = query_np.astype(np.uint8)

        # Coarse-to-fine search strategy
        if coarse_search and len(atlas_paths) > 20:
            # Phase 1: Coarse search - check every Nth slice
            coarse_paths = atlas_paths[::coarse_stride]
            if verbose:
                print(f"  Phase 1: Coarse search ({len(coarse_paths)} slices)...")

            coarse_results = []
            for i, atlas_path in enumerate(coarse_paths):
                try:
                    atlas_np = cv2.imread(str(atlas_path), cv2.IMREAD_GRAYSCALE)
                    if atlas_np is None:
                        continue

                    # Quick resize
                    scale = min(query_np.shape[0] / atlas_np.shape[0], query_np.shape[1] / atlas_np.shape[1])
                    if scale < 0.5 or scale > 2.0:
                        new_wh = (max(8, int(atlas_np.shape[1] * scale)), max(8, int(atlas_np.shape[0] * scale)))
                        atlas_resized = cv2.resize(atlas_np, new_wh, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                    else:
                        atlas_resized = atlas_np

                    nmi, _, _, _ = self.register_to_atlas(atlas_resized, query_np)
                    coarse_results.append((i, nmi, atlas_path))
                except:
                    continue

            if len(coarse_results) == 0:
                # Fallback to full search
                if verbose:
                    print("  Coarse search failed, using full search...")
                paths_to_check = atlas_paths
            else:
                # Phase 2: Fine search - check neighbors of best coarse matches
                coarse_results.sort(key=lambda x: x[1], reverse=True)
                best_coarse_idx = coarse_results[0][0]

                # Get actual index in full list
                best_full_idx = best_coarse_idx * coarse_stride

                # Check neighbors: ±coarse_stride*2 around best match
                start_idx = max(0, best_full_idx - coarse_stride * 2)
                end_idx = min(len(atlas_paths), best_full_idx + coarse_stride * 2 + 1)
                paths_to_check = atlas_paths[start_idx:end_idx]

                if verbose:
                    print(f"  Phase 2: Fine search ({len(paths_to_check)} slices around slice {best_full_idx})...")
        else:
            paths_to_check = atlas_paths

        results = []

        for i, atlas_path in enumerate(paths_to_check):
            # Load atlas slice
            atlas_np = cv2.imread(str(atlas_path), cv2.IMREAD_GRAYSCALE)
            if atlas_np is None:
                continue

            # Resize atlas to roughly match query size
            scale = min(
                query_np.shape[0] / atlas_np.shape[0],
                query_np.shape[1] / atlas_np.shape[1]
            )

            if scale < 0.5 or scale > 2.0:
                new_wh = (
                    max(8, int(atlas_np.shape[1] * scale)),
                    max(8, int(atlas_np.shape[0] * scale))
                )
                atlas_resized = cv2.resize(
                    atlas_np, new_wh,
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                )
            else:
                atlas_resized = atlas_np

            # Run registration
            try:
                nmi, transform, moved_np, fixed_np_ds = self.register_to_atlas(
                    atlas_resized, query_np
                )

                stats = {
                    'nmi': nmi,
                    'scale': scale
                }

                results.append((atlas_path.name, nmi, stats))

            except Exception as e:
                if verbose:
                    print(f"  Skipped {atlas_path.name}: {e}")
                continue

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(atlas_paths)} slices...")

        # Sort by NMI (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        if verbose:
            print(f"\n📊 Top {min(top_k, len(results))} matches:")
            for rank, (filename, nmi, stats) in enumerate(results[:top_k], 1):
                print(f"  {rank}. {filename}: NMI={nmi:.4f}")

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
