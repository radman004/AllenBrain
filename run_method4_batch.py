"""
Method 4: FFT Phase Correlation - Batch Evaluation
Author: [Team Member 4]
Date: 2025
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
from method4_fft import FFTRegistration, compute_alignment_metrics

# Configuration
NUM_TEST_IMAGES = 720  # Change to smaller number for quick testing
HANDLE_ROTATION = True  # Set to False for faster processing

CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")
RESULTS_DIR = os.path.join(SYNTHETIC_TEST_DIR, "results")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print(f"Method 4: FFT Phase Correlation - Batch Evaluation")
print("="*60)
print(f"Testing on {NUM_TEST_IMAGES} images")
print(f"Rotation handling: {'Enabled' if HANDLE_ROTATION else 'Disabled'}")
print(f"Results will be saved to: {RESULTS_DIR}")
print("="*60)

# Load ground truth
gt_csv_path = os.path.join(SYNTHETIC_TEST_DIR, "ground_truth.csv")
gt_df = pd.read_csv(gt_csv_path)

# Initialize registrator
registrator = FFTRegistration(
    upsample_factor=10,
    handle_rotation=HANDLE_ROTATION,
    max_rotation=45.0
)

print(f"✅ Initialized FFT registration")

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

    # Run registration pipeline
    start_time = time.time()
    try:
        # Phase 1: Find best atlas slice
        top_matches = registrator.find_best_atlas_slice(
            query_image=test_image,
            atlas_dir=RAW_ATLAS_DIR,
            top_k=5,
            verbose=False
        )

        if not top_matches or len(top_matches) == 0:
            continue

        # Unpack best result: (filename, correlation, stats)
        best_filename, best_correlation, best_stats = top_matches[0]

        # Phase 2: Fine registration
        atlas_image_path = os.path.join(RAW_ATLAS_DIR, best_filename)
        atlas_image = cv2.imread(atlas_image_path, cv2.IMREAD_GRAYSCALE)

        if atlas_image is None:
            continue

        reg_result = registrator.register_images(
            fixed=atlas_image,
            moving=test_image
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        # Extract predicted slice number
        pred_match = re.search(r'\d+', best_filename)
        if not pred_match:
            continue
        pred_slice_num = int(pred_match.group())

        # Compute metrics if registration succeeded
        nmi = ssim = mse = None
        if reg_result['success'] and reg_result['registered_image'] is not None:
            metrics = compute_alignment_metrics(atlas_image, reg_result['registered_image'])
            nmi = metrics['nmi']
            ssim = metrics['ssim']
            mse = metrics['mse']

        # Check accuracy (within ±4 slices)
        is_correct = abs(gt_slice_num - pred_slice_num) <= 4
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
            'slice_error': abs(gt_slice_num - pred_slice_num),
            'correct': is_correct,
            'correlation': best_correlation,
            'shift_x': reg_result['stats']['shift_x'],
            'shift_y': reg_result['stats']['shift_y'],
            'rotation': reg_result['rotation'],
            'nmi': nmi,
            'ssim': ssim,
            'mse': mse,
            'time_sec': elapsed,
            'success': reg_result['success']
        })

    except Exception as e:
        print(f"\n❌ Error processing {row['synthetic_file']}: {e}")
        continue

# Print summary
print("\n" + "="*60)
print(f"BATCH EVALUATION RESULTS (FFT)")
print("="*60)
print(f"Total processed: {total_count}/{NUM_TEST_IMAGES}")
print(f"Z-level accuracy (±4 slices): {correct_count}/{total_count} = {correct_count/total_count*100:.2f}%")
print(f"Average time per image: {total_time/total_count:.2f}s")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
output_path = os.path.join(RESULTS_DIR, "method4_fft_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

# Compute detailed statistics
successful_reg = results_df[results_df['success'] == True]
print(f"\n" + "="*60)
print("Detailed Statistics:")
print("="*60)
print(f"Successful registrations: {len(successful_reg)}/{len(results_df)} ({len(successful_reg)/len(results_df)*100:.1f}%)")
if len(successful_reg) > 0:
    print(f"Mean NMI:  {successful_reg['nmi'].mean():.4f} ± {successful_reg['nmi'].std():.4f}")
    print(f"Mean SSIM: {successful_reg['ssim'].mean():.4f} ± {successful_reg['ssim'].std():.4f}")
    print(f"Mean correlation: {results_df['correlation'].mean():.4f}")

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
print(results_df[['synthetic_file', 'gt_slice_num', 'pred_slice_num', 'slice_error', 'correct', 'correlation', 'time_sec']].head(10).to_string(index=False))

print("\n" + "="*60)
print("✅ Batch evaluation completed!")
print("="*60)
