"""
Method 5: Hybrid Registration (Feature-based + Intensity-based)
Author: [Team Member 5]
Date: 2025

This module combines Method 2 (ORB features) for fast initial alignment
with Method 1 (Mutual Information) for fine refinement.

Pipeline:
1. Phase 1: ORB feature matching for Z-level search (fast, ~1s)
2. Phase 2: RANSAC transformation estimation
3. Phase 3: MI-based refinement for sub-pixel accuracy (optional)
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import time
import SimpleITK as sitk

# Import Method 2 (Feature-based)
from method2_features import FeatureBasedRegistration, compute_alignment_metrics


class HybridRegistration:
    """
    Hybrid registration combining feature-based and intensity-based methods.

    Strategy:
    - Use ORB for fast coarse alignment (handles large rotations/translations)
    - Refine with MI for sub-pixel accuracy (handles intensity variations)
    """

    def __init__(
        self,
        use_orb: bool = True,
        nfeatures: int = 2000,
        enable_mi_refinement: bool = True,
        mi_iterations: int = 200
    ):
        """
        Initialize hybrid registration.

        Args:
            use_orb: Use ORB (True) or SIFT (False) for feature detection
            nfeatures: Number of features to detect
            enable_mi_refinement: Whether to refine with MI registration
            mi_iterations: Maximum iterations for MI optimization
        """
        # Initialize feature-based registrator (Method 2)
        self.feature_reg = FeatureBasedRegistration(
            detector_type='ORB' if use_orb else 'SIFT',
            nfeatures=nfeatures,
            ratio_threshold=0.75,
            ransac_threshold=5.0
        )

        self.enable_mi_refinement = enable_mi_refinement
        self.mi_iterations = mi_iterations

    def refine_with_mi(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        initial_transform: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Refine alignment using Mutual Information registration.

        Args:
            fixed: Reference image
            moving: Image to align (possibly pre-aligned)
            initial_transform: Initial 2x3 affine transform (optional)

        Returns:
            Tuple of (refined_transform, registered_image, stats)
        """
        # Convert to SimpleITK images
        fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
        moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

        # Initialize registration method
        registration = sitk.ImageRegistrationMethod()

        # Metric: Mattes Mutual Information
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.2)

        # Optimizer: Regular Step Gradient Descent
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=self.mi_iterations,
            relaxationFactor=0.5
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        # Interpolator
        registration.SetInterpolator(sitk.sitkLinear)

        # Multi-resolution strategy
        registration.SetShrinkFactorsPerLevel([2, 1])
        registration.SetSmoothingSigmasPerLevel([1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Initialize transform
        if initial_transform is not None:
            # Convert 2x3 OpenCV affine to SimpleITK
            # SimpleITK uses Similarity2DTransform (4 DOF: rotation, scale, translation)
            initial_tx = sitk.Similarity2DTransform()

            # Extract parameters from OpenCV affine matrix
            # [a -b tx]
            # [b  a ty]
            a = initial_transform[0, 0]
            b = initial_transform[1, 0]
            tx = initial_transform[0, 2]
            ty = initial_transform[1, 2]

            scale = np.sqrt(a**2 + b**2)
            angle = np.arctan2(b, a)

            # Set center to image center
            center_x = fixed.shape[1] / 2.0
            center_y = fixed.shape[0] / 2.0

            initial_tx.SetCenter([center_x, center_y])
            initial_tx.SetAngle(angle)
            initial_tx.SetScale(scale)
            initial_tx.SetTranslation([tx, ty])
        else:
            # Use centered transform initializer
            initial_tx = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.Similarity2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )

        registration.SetInitialTransform(initial_tx, inPlace=False)

        # Execute registration
        try:
            final_tx = registration.Execute(fixed_sitk, moving_sitk)

            # Apply transformation
            registered_sitk = sitk.Resample(
                moving_sitk,
                fixed_sitk,
                final_tx,
                sitk.sitkLinear,
                0.0,
                sitk.sitkFloat32
            )

            registered = sitk.GetArrayFromImage(registered_sitk)

            # Convert SimpleITK transform back to 2x3 OpenCV affine
            # Handle different transform types
            try:
                params = final_tx.GetParameters()
                angle = final_tx.GetAngle()
                scale = params[0]
                tx, ty = final_tx.GetTranslation()

                # Build 2x3 matrix
                cos_a = np.cos(angle) * scale
                sin_a = np.sin(angle) * scale

                refined_transform = np.array([
                    [cos_a, -sin_a, tx],
                    [sin_a, cos_a, ty]
                ], dtype=np.float32)
            except (AttributeError, IndexError):
                # CompositeTransform or other type - just use initial transform
                if initial_transform is not None:
                    refined_transform = initial_transform
                    registered = sitk.GetArrayFromImage(registered_sitk)
                else:
                    # Return None to signal failure
                    return None, moving, {'error': 'Cannot extract transform parameters'}

            # Get final metric value
            final_metric = registration.GetMetricValue()

            stats = {
                'final_metric': float(final_metric),
                'iterations': registration.GetOptimizerIteration(),
                'stop_condition': registration.GetOptimizerStopConditionDescription()
            }

            return refined_transform, registered, stats

        except Exception as e:
            print(f"⚠️  MI refinement failed: {e}")
            # Return initial result if refinement fails
            if initial_transform is not None:
                h, w = fixed.shape
                registered = cv2.warpAffine(moving, initial_transform, (w, h))
                return initial_transform, registered, {'error': str(e)}
            else:
                return None, moving, {'error': str(e)}

    def find_best_atlas_slice(
        self,
        query_image: np.ndarray,
        atlas_dir: str,
        top_k: int = 5,
        verbose: bool = True
    ):
        """
        Find best atlas slice using feature matching (Phase 1).

        Args:
            query_image: Query histology image
            atlas_dir: Directory with atlas slices
            top_k: Number of top candidates
            verbose: Print progress

        Returns:
            List of (filename, match_count, stats) tuples
        """
        if verbose:
            print("Phase 1: ORB-based Z-level search...")

        return self.feature_reg.find_best_atlas_slice(
            query_image, atlas_dir, top_k, verbose
        )

    def register_to_atlas(
        self,
        query_image: np.ndarray,
        atlas_image: np.ndarray,
        nfeatures_fine: int = 4000
    ) -> Dict:
        """
        Register query to atlas using hybrid approach.

        Pipeline:
        1. Feature-based coarse alignment (ORB + RANSAC)
        2. MI-based refinement (optional)

        Args:
            query_image: Moving image
            atlas_image: Fixed/reference image
            nfeatures_fine: Features for fine registration

        Returns:
            Dictionary with registration results
        """
        # Phase 2: Feature-based registration
        feature_result = self.feature_reg.register_to_atlas(
            query_image, atlas_image, nfeatures_fine
        )

        if not feature_result['success']:
            return feature_result

        # Phase 3: MI refinement (optional)
        if self.enable_mi_refinement:
            refined_transform, refined_image, mi_stats = self.refine_with_mi(
                fixed=atlas_image,
                moving=query_image,
                initial_transform=feature_result['transform']
            )

            if refined_transform is not None:
                # Update with refined results
                feature_result['transform'] = refined_transform
                feature_result['registered_image'] = refined_image
                feature_result['mi_stats'] = mi_stats
                feature_result['refinement'] = 'MI'
            else:
                feature_result['refinement'] = 'Failed'
        else:
            feature_result['refinement'] = 'None'

        return feature_result


def compute_alignment_metrics_hybrid(fixed: np.ndarray, moving: np.ndarray) -> Dict[str, float]:
    """Wrapper for alignment metrics."""
    return compute_alignment_metrics(fixed, moving)
