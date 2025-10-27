
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time
import re
import os

# Import our custom module
sys.path.insert(0, str(Path.cwd() / 'src'))
from method2_features import FeatureBasedRegistration, compute_alignment_metrics, draw_matches_overlay

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")

print(f"Current directory: {CURR_DIR}")
print(f"Base directory: {BASE_DIR}")
print(f"Atlas directory: {RAW_ATLAS_DIR}")
print(f"Test directory: {SYNTHETIC_TEST_DIR}")

print("="*60)
print("Method 2: Feature-Based Registration - Quick Test")
print("="*60)

# Check if directories exist
if not RAW_ATLAS_DIR:
    print(f"❌ Error: Atlas directory not found: {RAW_ATLAS_DIR}")
    print("Please update BASE_DIR path in the script.")
    sys.exit(1)

if not SYNTHETIC_TEST_DIR:
    print(f"❌ Error: Test directory not found: {SYNTHETIC_TEST_DIR}")
    print("Please update BASE_DIR path in the script.")
    sys.exit(1)

# Load ground truth
# gt_df = pd.read_csv(SYNTHETIC_TEST_DIR / "ground_truth.csv")
gt_df = pd.read_csv(r"C:\UTS\3\IMPR\Project\AllenBrain\project_data\test_synthetic\ground_truth.csv")
print(f"✅ Found {len(gt_df)} synthetic test images")
print(f"✅ Found atlas directory: {RAW_ATLAS_DIR}")

# Initialize feature-based registration with ORB
print("\n" + "="*60)
print("Initializing ORB detector...")
print("="*60)
registrator = FeatureBasedRegistration(
    detector_type='ORB',
    nfeatures=2000,
    ratio_threshold=0.75,
    ransac_threshold=5.0
)
print("✅ ORB detector initialized")

# Test on first image
print("\n" + "="*60)
print("Testing on first synthetic image...")
print("="*60)

test_row = gt_df.iloc[0]
test_image_path = str(SYNTHETIC_TEST_DIR) + "/images" + f"/{test_row['synthetic_file']}"
test_image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)

if test_image is None:
    print(f"❌ Error: Cannot load test image: {test_image_path}")
    sys.exit(1)

print(f"Test image: {test_row['synthetic_file']}")
print(f"Ground truth: {test_row['source_file']}")
print(f"Image shape: {test_image.shape}")

# Phase 1: Find best matching atlas slice
print("\n--- Phase 1: Z-Level Search ---")
start_time = time.time()

try:
    top_matches = registrator.find_best_atlas_slice(
        query_image=test_image,
        atlas_dir=RAW_ATLAS_DIR,
        top_k=5
    )
    phase1_time = time.time() - start_time

    # Unpack best result: (filename, n_matches, stats)
    best_filename, best_match_count, best_stats = top_matches[0]

    print(f"✅ Best match: {best_filename}")
    print(f"✅ Match count: {best_match_count} good matches")
    print(f"✅ Time: {phase1_time:.2f}s")

    # Check if prediction is correct
    is_correct = (best_filename == test_row['source_file'])
    if is_correct:
        print(f"✅ CORRECT prediction!")
    else:
        print(f"❌ INCORRECT - Expected: {test_row['source_file']}")

except Exception as e:
    print(f"❌ Error in Phase 1: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Phase 2: Fine registration with RANSAC
print("\n--- Phase 2: Fine Registration ---")
start_time = time.time()

try:
    # Load the best matching atlas slice
    atlas_image_path = os.path.join(RAW_ATLAS_DIR, best_filename)
    atlas_image = cv2.imread(atlas_image_path, cv2.IMREAD_GRAYSCALE)
    reg_result = registrator.register_to_atlas(
        query_image=test_image,
        atlas_image=atlas_image,
        nfeatures_fine=4000
    )
    phase2_time = time.time() - start_time

    if reg_result['success']:
        print(f"✅ Registration successful!")
        n_matches = len(reg_result['matches'])
        n_inliers = reg_result['stats']['n_inliers']
        print(f"✅ Total matches: {n_matches}")
        print(f"✅ Inliers: {n_inliers} ({n_inliers/n_matches*100:.1f}%)")
        print(f"✅ Time: {phase2_time:.2f}s")

        # Compute metrics
        metrics = compute_alignment_metrics(atlas_image, reg_result['registered_image'])
        print(f"\n--- Alignment Metrics ---")
        print(f"NMI:  {metrics['nmi']:.4f}")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"MSE:  {metrics['mse']:.2f}")
    else:
        print(f"❌ Registration failed")

except Exception as e:
    print(f"❌ Error in Phase 2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print(f"Total pipeline time: {phase1_time + phase2_time:.2f}s")
print("="*60)

# Ask if user wants to run batch evaluation
print("\n" + "="*60)
print("Single image test completed successfully!")
print("="*60)
print("\nNext steps:")
print("1. Run batch evaluation: python run_method2_batch.py")
print("2. Or manually edit NUM_TEST_IMAGES in run_method2_batch.py")
print("\nTest completed successfully! ✅")
