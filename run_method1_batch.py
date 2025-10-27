"""
Method 1: Mutual Information Registration - Batch Evaluation
Author: [Team Member 1]
Date: 2025

Evaluates MI-based registration on synthetic test set.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import time
import re
import os
from tqdm import tqdm

# Import our custom module
sys.path.insert(0, str(Path.cwd() / 'src'))
from method1_mi import MutualInformationRegistration, compute_alignment_metrics

# Configuration
NUM_TEST_IMAGES = 720  # Change to smaller number for quick testing

CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")
RESULTS_DIR = os.path.join(SYNTHETIC_TEST_DIR, "results")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print(f"Method 1: Mutual Information Registration - Batch Evaluation")
print("="*60)
print(f"Testing on {NUM_TEST_IMAGES} images")
print(f"Results will be saved to: {RESULTS_DIR}")
print("="*60)

# Load ground truth
gt_csv_path = os.path.join(SYNTHETIC_TEST_DIR, "ground_truth.csv")
gt_df = pd.read_csv(gt_csv_path)

# Initialize registrator with FAST settings
registrator = MutualInformationRegistration(
    downsample_factor=4,      # Increased from 2 → 4x faster
    hist_bins=32,             # Reduced from 64 → 2x faster
    sample_percent=0.1,       # Reduced from 0.2 → faster
    max_iter=50,              # Reduced from 150 → 3x faster
    rot_range_deg=0           # Disabled random rotation → faster
)

print(f"✅ Initialized MI registration")
print(f"   Downsample: {registrator.downsample_factor}×")
print(f"   Histogram bins: {registrator.hist_bins}")
print(f"   Sample percent: {registrator.sample_percent}")
print(f"   Max iterations: {registrator.max_iter}")

# Results storage
results = []
correct_count = 0
total_count = 0
total_time = 0

# Process each test image
for i, row in tqdm(gt_df.head(NUM_TEST_IMAGES).iterrows(), total=NUM_TEST_IMAGES, desc="Processing"):
    # Load test image
    test_path = os.path.join(SYNTHETIC_TEST_DIR, "images", row["synthetic_file"])
    test_image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if test_image is None:
        continue

    # Skip mostly black images
    if (np.sum(test_image < 15) / test_image.size > 0.90):
        continue

    # Extract ground truth slice filename
    gt_source = Path(row['source_file']).name

    # Extract slice numbers
    gt_match = re.search(r'\d+', gt_source)
    if not gt_match:
        continue
    gt_slice_num = int(gt_match.group())

    # Run MI registration
    start_time = time.time()
    try:
        # Find best atlas slice (with fast coarse-to-fine search)
        top_matches = registrator.find_best_atlas_slice(
            query_image=test_image,
            atlas_dir=RAW_ATLAS_DIR,
            top_k=1,
            verbose=False,
            coarse_search=True,   # Enable fast coarse-to-fine search
            coarse_stride=4       # Check every 4th slice first
        )

        if not top_matches or len(top_matches) == 0:
            continue

        # Unpack best result
        best_filename, best_nmi, best_stats = top_matches[0]

        # Load atlas slice and register
        atlas_image_path = os.path.join(RAW_ATLAS_DIR, best_filename)
        atlas_image = cv2.imread(atlas_image_path, cv2.IMREAD_GRAYSCALE)

        if atlas_image is None:
            continue

        # Resize atlas to match query
        scale = min(
            test_image.shape[0] / atlas_image.shape[0],
            test_image.shape[1] / atlas_image.shape[1]
        )

        if scale < 0.5 or scale > 2.0:
            new_wh = (
                max(8, int(atlas_image.shape[1] * scale)),
                max(8, int(atlas_image.shape[0] * scale))
            )
            atlas_resized = cv2.resize(
                atlas_image, new_wh,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            )
        else:
            atlas_resized = atlas_image

        # Perform registration to get registered image
        nmi_score, transform, registered_image, _ = registrator.register_to_atlas(
            atlas_resized, test_image
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        # Extract predicted slice number
        pred_match = re.search(r'\d+', best_filename)
        if not pred_match:
            continue
        pred_slice_num = int(pred_match.group())

        # Compute additional metrics (SSIM) on full-resolution if possible
        # Upsample registered image back to original size for metric computation
        if registered_image.shape != atlas_image.shape:
            registered_full = cv2.resize(
                registered_image,
                (atlas_image.shape[1], atlas_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            registered_full = registered_image

        metrics = compute_alignment_metrics(atlas_image, registered_full)

        # Check accuracy (within ±4 slices)
        slice_error = abs(gt_slice_num - pred_slice_num)
        is_correct = slice_error <= 4

        if is_correct:
            correct_count += 1
        total_count += 1

        # Store results
        results.append({
            'synthetic_file': row['synthetic_file'],
            'gt_source_file': gt_source,
            'gt_slice_num': gt_slice_num,
            'pred_slice_name': best_filename,
            'pred_slice_num': pred_slice_num,
            'slice_error': slice_error,
            'correct': is_correct,
            'nmi': nmi_score,
            'ssim': metrics['ssim'],
            'mse': metrics['mse'],
            'time_sec': elapsed,
            'success': True
        })

    except Exception as e:
        print(f"\n❌ Error processing {row['synthetic_file']}: {e}")
        continue

# Print summary
print("\n" + "="*60)
print(f"BATCH EVALUATION RESULTS (Method 1 - MI)")
print("="*60)
print(f"Total processed: {total_count}/{NUM_TEST_IMAGES}")
print(f"Z-level accuracy (±4 slices): {correct_count}/{total_count} = {correct_count/total_count*100:.2f}%")
print(f"Average time per image: {total_time/total_count:.2f}s")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
output_path = os.path.join(RESULTS_DIR, "method1_mi_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

# Compute detailed statistics
print(f"\n" + "="*60)
print("Detailed Statistics:")
print("="*60)
print(f"Mean NMI:  {results_df['nmi'].mean():.4f} ± {results_df['nmi'].std():.4f}")
print(f"Mean SSIM: {results_df['ssim'].mean():.4f} ± {results_df['ssim'].std():.4f}")
print(f"Mean MSE:  {results_df['mse'].mean():.2f} ± {results_df['mse'].std():.2f}")

# Error analysis
print(f"\n" + "="*60)
print("Error Analysis:")
print("="*60)
print(f"Exact match (error=0): {len(results_df[results_df['slice_error'] == 0])} ({len(results_df[results_df['slice_error'] == 0])/len(results_df)*100:.1f}%)")
print(f"Within ±1 slice:       {len(results_df[results_df['slice_error'] <= 1])} ({len(results_df[results_df['slice_error'] <= 1])/len(results_df)*100:.1f}%)")
print(f"Within ±2 slices:      {len(results_df[results_df['slice_error'] <= 2])} ({len(results_df[results_df['slice_error'] <= 2])/len(results_df)*100:.1f}%)")
print(f"Within ±4 slices:      {len(results_df[results_df['slice_error'] <= 4])} ({len(results_df[results_df['slice_error'] <= 4])/len(results_df)*100:.1f}%)")

# Show first 10 results
print(f"\n" + "="*60)
print("First 10 Results:")
print("="*60)
print(results_df[['synthetic_file', 'gt_slice_num', 'pred_slice_num', 'slice_error', 'correct', 'nmi', 'time_sec']].head(10).to_string(index=False))

print("\n" + "="*60)
print("✅ Batch evaluation completed!")
print("="*60)
