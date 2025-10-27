"""
Method 2: Feature-Based Registration (ORB/SIFT + RANSAC)
Author: Aditya Maniar
Date: 2025

This module implements keypoint-based image registration for brain slice alignment.
Uses ORB or SIFT features with RANSAC for robust transformation estimation.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import time


class FeatureBasedRegistration:
    """
    Feature-based registration using ORB or SIFT keypoints with RANSAC.

    Attributes:
        detector_type: 'ORB' or 'SIFT'
        nfeatures: Number of keypoints to detect
        ratio_threshold: Lowe's ratio test threshold (0.75 recommended)
        ransac_threshold: RANSAC reprojection error threshold (pixels)
    """

    def __init__(
        self,
        detector_type: str = 'ORB',
        nfeatures: int = 2000,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 5.0,
        ransac_max_iters: int = 2000,
        ransac_confidence: float = 0.995
    ):
        """
        Initialize feature-based registration.

        Args:
            detector_type: 'ORB' (fast) or 'SIFT' (accurate)
            nfeatures: Number of keypoints to detect
            ratio_threshold: Lowe's ratio test threshold (0.7-0.8)
            ransac_threshold: Max pixel error for RANSAC inliers
            ransac_max_iters: Maximum RANSAC iterations
            ransac_confidence: RANSAC confidence level (0.99-0.999)
        """
        self.detector_type = detector_type.upper()
        self.nfeatures = nfeatures
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence

        # Initialize detector and matcher
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()

    def _create_detector(self):
        """Create feature detector (ORB or SIFT)."""
        if self.detector_type == 'ORB':
            return cv2.ORB_create(
                nfeatures=self.nfeatures,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=10,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
        elif self.detector_type == 'SIFT':
            return cv2.SIFT_create(
                nfeatures=0,  # Detect all features
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def _create_matcher(self):
        """Create feature matcher based on detector type."""
        if self.detector_type == 'ORB':
            # Use Hamming distance for binary descriptors
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # SIFT
            # Use FLANN for floating-point descriptors
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute descriptors.

        Args:
            image: Grayscale image (uint8)

        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Array of shape (n_keypoints, descriptor_dim)
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        apply_ratio_test: bool = True
    ) -> List[cv2.DMatch]:
        """
        Match features between two descriptor sets.

        Args:
            desc1: Descriptors from image 1
            desc2: Descriptors from image 2
            apply_ratio_test: Whether to apply Lowe's ratio test

        Returns:
            List of good matches (cv2.DMatch objects)
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        # Find k=2 nearest neighbors for ratio test
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:  # Need 2 neighbors for ratio test
                m, n = match_pair
                if not apply_ratio_test or m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1 and not apply_ratio_test:
                # Only one match found, accept if not using ratio test
                good_matches.append(match_pair[0])

        return good_matches

    def estimate_transform(
        self,
        kp1: List,
        kp2: List,
        matches: List[cv2.DMatch],
        min_inliers: int = 15
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Estimate affine transformation using RANSAC.

        Args:
            kp1: Keypoints from image 1 (moving)
            kp2: Keypoints from image 2 (fixed/atlas)
            matches: Good matches from match_features()
            min_inliers: Minimum number of inliers required

        Returns:
            transform: 2x3 affine transformation matrix (or None if failed)
            inliers: Boolean mask of inlier matches
            stats: Dictionary with matching statistics
        """
        stats = {
            'n_matches': len(matches),
            'n_inliers': 0,
            'inlier_ratio': 0.0,
            'success': False
        }

        if len(matches) < min_inliers:
            return None, None, stats

        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate affine transform with RANSAC
        try:
            transform, inliers = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=self.ransac_max_iters,
                confidence=self.ransac_confidence
            )
        except cv2.error:
            return None, None, stats

        if transform is None or inliers is None:
            return None, None, stats

        # Update statistics
        n_inliers = np.sum(inliers)
        stats['n_inliers'] = int(n_inliers)
        stats['inlier_ratio'] = n_inliers / len(matches)
        stats['success'] = n_inliers >= min_inliers

        return transform, inliers, stats

    def find_best_atlas_slice(
        self,
        query_image: np.ndarray,
        atlas_dir: Path,
        top_k: int = 5,
        verbose: bool = True
    ) -> List[Tuple[str, int, Dict]]:
        """
        Find the most similar atlas slices based on feature matching.

        Phase 1: Z-Level Search

        Args:
            query_image: Input histology image (grayscale)
            atlas_dir: Directory containing atlas slice images
            top_k: Number of top candidates to return
            verbose: Whether to print progress

        Returns:
            List of (filename, n_matches, stats) tuples, sorted by match count
        """
        # Detect features in query image
        query_kp, query_desc = self.detect_and_compute(query_image)

        if query_desc is None or len(query_kp) < 10:
            print(f"⚠️ Too few features in query image: {len(query_kp) if query_kp else 0}")
            # Try enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(query_image)
            query_kp, query_desc = self.detect_and_compute(enhanced)

            if query_desc is None or len(query_kp) < 10:
                print("❌ Feature detection failed even after enhancement")
                return []

        if verbose:
            print(f"✓ Detected {len(query_kp)} features in query image")

        # Load atlas slices - convert to Path if string
        atlas_dir = Path(atlas_dir) if isinstance(atlas_dir, str) else atlas_dir
        atlas_paths = list(atlas_dir.glob("*.png")) + list(atlas_dir.glob("*.jpg"))

        if len(atlas_paths) == 0:
            raise ValueError(f"No atlas images found in {atlas_dir}")

        # Match against each atlas slice
        results = []

        for i, atlas_path in enumerate(atlas_paths):
            # Read atlas slice
            atlas_img = cv2.imread(str(atlas_path), cv2.IMREAD_GRAYSCALE)
            if atlas_img is None:
                continue

            # Detect features in atlas
            atlas_kp, atlas_desc = self.detect_and_compute(atlas_img)

            if atlas_desc is None or len(atlas_kp) < 10:
                continue

            # Match features
            matches = self.match_features(query_desc, atlas_desc)

            # Store result
            stats = {
                'n_query_features': len(query_kp),
                'n_atlas_features': len(atlas_kp),
                'n_matches': len(matches)
            }

            results.append((atlas_path.name, len(matches), stats))

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(atlas_paths)} slices...")

        # Sort by number of matches (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        if verbose:
            print(f"\n📊 Top {min(top_k, len(results))} matches:")
            for rank, (filename, n_matches, stats) in enumerate(results[:top_k], 1):
                print(f"  {rank}. {filename}: {n_matches} matches")

        return results[:top_k]

    def register_to_atlas(
        self,
        query_image: np.ndarray,
        atlas_image: np.ndarray,
        nfeatures_fine: int = 4000
    ) -> Dict:
        """
        Register query image to atlas image using feature matching + RANSAC.

        Phase 2: Fine Registration

        Args:
            query_image: Moving image (grayscale)
            atlas_image: Fixed/reference image (grayscale)
            nfeatures_fine: Number of features for fine registration

        Returns:
            Dictionary containing:
                - transform: 2x3 affine matrix
                - inliers: Boolean mask
                - matches: List of good matches
                - keypoints_query: Detected keypoints in query
                - keypoints_atlas: Detected keypoints in atlas
                - stats: Matching statistics
                - registered_image: Transformed query image
        """
        # Temporarily increase features for fine registration
        original_nfeatures = self.nfeatures
        self.nfeatures = nfeatures_fine
        self.detector = self._create_detector()

        # Detect features
        kp_query, desc_query = self.detect_and_compute(query_image)
        kp_atlas, desc_atlas = self.detect_and_compute(atlas_image)

        # Restore original nfeatures
        self.nfeatures = original_nfeatures
        self.detector = self._create_detector()

        # Match features
        matches = self.match_features(desc_query, desc_atlas)

        # Estimate transformation
        transform, inliers, stats = self.estimate_transform(
            kp_query, kp_atlas, matches
        )

        # Apply transformation
        registered_image = None
        if transform is not None:
            h, w = atlas_image.shape
            registered_image = cv2.warpAffine(
                query_image, transform, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return {
            'transform': transform,
            'inliers': inliers,
            'matches': matches,
            'keypoints_query': kp_query,
            'keypoints_atlas': kp_atlas,
            'stats': stats,
            'registered_image': registered_image,
            'success': stats['success']
        }


def compute_alignment_metrics(
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute alignment quality metrics.

    Args:
        fixed: Fixed/reference image
        moving: Registered moving image
        mask: Optional binary mask for foreground

    Returns:
        Dictionary with NMI, SSIM, MSE
    """
    from skimage.metrics import structural_similarity

    # Ensure same shape
    if fixed.shape != moving.shape:
        moving = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]))

    # Create mask if not provided
    if mask is None:
        mask = (fixed > 0) & (moving > 0)

    # Extract foreground pixels
    fixed_fg = fixed[mask].astype(np.float32)
    moving_fg = moving[mask].astype(np.float32)

    if len(fixed_fg) == 0:
        return {'nmi': 0.0, 'ssim': 0.0, 'mse': np.inf}

    # Normalized Mutual Information
    def normalized_mutual_information(x, y, bins=64):
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
        pxy = hist_2d / np.maximum(hist_2d.sum(), 1.0)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)

        eps = 1e-12
        hx = -np.sum(px * np.log(px + eps))
        hy = -np.sum(py * np.log(py + eps))
        hxy = -np.sum(pxy * np.log(pxy + eps))

        return (hx + hy) / max(hxy, eps)

    nmi = normalized_mutual_information(fixed_fg, moving_fg)

    # SSIM
    ssim = structural_similarity(fixed, moving, data_range=255)

    # MSE
    mse = np.mean((fixed_fg - moving_fg) ** 2)

    return {
        'nmi': float(nmi),
        'ssim': float(ssim),
        'mse': float(mse)
    }


def draw_matches_overlay(
    img1: np.ndarray,
    kp1: List,
    img2: np.ndarray,
    kp2: List,
    matches: List[cv2.DMatch],
    inliers: Optional[np.ndarray] = None,
    max_matches: int = 50
) -> np.ndarray:
    """
    Draw feature matches between two images.

    Args:
        img1: First image
        kp1: Keypoints from first image
        img2: Second image
        kp2: Keypoints from second image
        matches: List of matches
        inliers: Optional boolean mask (inliers=True)
        max_matches: Maximum number of matches to draw

    Returns:
        Visualization image with matches drawn
    """
    # Select subset of matches to draw
    if len(matches) > max_matches:
        if inliers is not None:
            # Prioritize inliers
            inlier_indices = np.where(inliers.ravel())[0]
            if len(inlier_indices) > 0:
                selected = np.random.choice(
                    inlier_indices,
                    size=min(max_matches, len(inlier_indices)),
                    replace=False
                )
                matches_to_draw = [matches[i] for i in selected]
            else:
                matches_to_draw = matches[:max_matches]
        else:
            matches_to_draw = matches[:max_matches]
    else:
        matches_to_draw = matches

    # Determine match colors
    if inliers is not None:
        match_colors = [
            (0, 255, 0) if inliers[i] else (0, 0, 255)
            for i in range(len(matches_to_draw))
        ]
    else:
        match_colors = None

    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_to_draw, None,
        matchColor=match_colors,
        singlePointColor=(255, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return img_matches
