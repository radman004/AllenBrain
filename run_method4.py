"""
Method 4: FFT Phase Correlation - Quick Test Script
Author: [Team Member 4]
Date: 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import time
import os

# Import our custom module
sys.path.insert(0, str(Path.cwd() / 'src'))
from method4_fft import FFTRegistration, compute_alignment_metrics

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")

print("="*60)
print("Method 4: FFT Phase Correlation - Quick Test")
print("="*60)

# Load ground truth
gt_csv_path = os.path.join(SYNTHETIC_TEST_DIR, "ground_truth.csv")
gt_df = pd.read_csv(gt_csv_path)
print(f"✅ Found {len(gt_df)} synthetic test images")

# Initialize FFT registration
print("\n" + "="*60)
print("Initializing FFT registration...")
print("="*60)
registrator = FFTRegistration(
    upsample_factor=10,
    handle_rotation=True,
    max_rotation=45.0
)
print("✅ FFT registration initialized")

# Test on first image
print("\n" + "="*60)
print("Testing on first synthetic image...")
print("="*60)

test_row = gt_df.iloc[0]
test_image_path = os.path.join(SYNTHETIC_TEST_DIR, "images", test_row["synthetic_file"])
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

if test_image is None:
    print(f"❌ Error: Cannot load test image: {test_image_path}")
    sys.exit(1)

print(f"Test image: {test_row['synthetic_file']}")
print(f"Ground truth: {test_row['source_file']}")
print(f"Image shape: {test_image.shape}")

# Phase 1: Find best matching atlas slice
print("\n--- Phase 1: FFT-based Z-Level Search ---")
start_time = time.time()

try:
    top_matches = registrator.find_best_atlas_slice(
        query_image=test_image,
        atlas_dir=RAW_ATLAS_DIR,
        top_k=5
    )
    phase1_time = time.time() - start_time

    # Unpack best result: (filename, correlation, stats)
    best_filename, best_correlation, best_stats = top_matches[0]

    print(f"✅ Best match: {best_filename}")
    print(f"✅ Correlation: {best_correlation:.4f}")
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

# Phase 2: Fine registration
print("\n--- Phase 2: FFT Registration ---")
start_time = time.time()

try:
    # Load the best matching atlas slice
    atlas_image_path = os.path.join(RAW_ATLAS_DIR, best_filename)
    atlas_image = cv2.imread(atlas_image_path, cv2.IMREAD_GRAYSCALE)

    reg_result = registrator.register_images(
        fixed=atlas_image,
        moving=test_image
    )
    phase2_time = time.time() - start_time

    if reg_result['success']:
        print(f"✅ Registration successful!")
        print(f"✅ Correlation: {reg_result['correlation']:.4f}")
        print(f"✅ Translation: ({reg_result['stats']['shift_x']:.1f}, {reg_result['stats']['shift_y']:.1f})")
        print(f"✅ Rotation: {reg_result['rotation']:.2f}°")
        print(f"✅ Time: {phase2_time:.2f}s")

        # Compute metrics
        metrics = compute_alignment_metrics(atlas_image, reg_result['registered_image'])
        print(f"\n--- Alignment Metrics ---")
        print(f"NMI:  {metrics['nmi']:.4f}")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"MSE:  {metrics['mse']:.2f}")
    else:
        print(f"❌ Registration failed (low correlation)")

except Exception as e:
    print(f"❌ Error in Phase 2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print(f"Total pipeline time: {phase1_time + phase2_time:.2f}s")
print("="*60)

print("\n" + "="*60)
print("Single image test completed successfully!")
print("="*60)
print("\nNext steps:")
print("1. Run batch evaluation: python run_method4_batch.py")
print("\nTest completed successfully! ✅")
