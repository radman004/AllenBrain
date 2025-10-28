"""
Visual Side-by-Side Comparison: ORB vs SIFT
Shows actual registration outputs where they differ
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import re

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))
from method2_features import FeatureBasedRegistration, compute_alignment_metrics

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")
RESULTS_DIR = os.path.join(SYNTHETIC_TEST_DIR, "results")
VISUAL_DIR = os.path.join(RESULTS_DIR, "orb_vs_sift_comparison")

os.makedirs(VISUAL_DIR, exist_ok=True)

print("="*80)
print("ORB vs SIFT Visual Comparison")
print("="*80)

# Load results
orb_df = pd.read_csv(os.path.join(RESULTS_DIR, "method2_orb_results.csv"))
sift_df = pd.read_csv(os.path.join(RESULTS_DIR, "method2_sift_results.csv"))

print(f"\n✓ Loaded ORB results: {len(orb_df)} samples")
print(f"✓ Loaded SIFT results: {len(sift_df)} samples")

# Find interesting test cases
print("\n" + "="*80)
print("Selecting Test Cases")
print("="*80)

test_cases = []

# Case 1: Both correct (show ORB advantage in quality)
orb_correct = set(orb_df[orb_df['slice_error'] == 0]['synthetic_file'])
sift_correct = set(sift_df[sift_df['slice_error'] == 0]['synthetic_file'])
both_correct = orb_correct & sift_correct
if len(both_correct) > 0:
    case1 = list(both_correct)[0]
    test_cases.append(('Both Correct (Quality Comparison)', case1))
    print(f"✓ Case 1 (Both Correct): {case1}")

# Case 2: ORB correct, SIFT wrong (dramatic difference)
sift_wrong = set(sift_df[sift_df['slice_error'] > 4]['synthetic_file'])
orb_vs_sift = orb_correct & sift_wrong
if len(orb_vs_sift) > 0:
    case2 = list(orb_vs_sift)[0]
    test_cases.append(('ORB Correct, SIFT Wrong', case2))
    print(f"✓ Case 2 (ORB Wins): {case2}")

# Case 3: Both struggle but ORB better
orb_close = set(orb_df[(orb_df['slice_error'] > 0) & (orb_df['slice_error'] <= 2)]['synthetic_file'])
sift_far = set(sift_df[sift_df['slice_error'] > 2]['synthetic_file'])
orb_better = orb_close & sift_far
if len(orb_better) > 0:
    case3 = list(orb_better)[0]
    test_cases.append(('ORB Better (Challenging Case)', case3))
    print(f"✓ Case 3 (ORB Better): {case3}")

# Initialize registrators
print("\n" + "="*80)
print("Initializing ORB and SIFT Registrators")
print("="*80)

orb_reg = FeatureBasedRegistration(
    detector_type='ORB',
    nfeatures=2000,
    ratio_threshold=0.75,
    ransac_threshold=5.0
)
print("✓ ORB registrator ready")

sift_reg = FeatureBasedRegistration(
    detector_type='SIFT',
    nfeatures=2000,
    ratio_threshold=0.75,
    ransac_threshold=5.0
)
print("✓ SIFT registrator ready")

# Process each test case
gt_df = pd.read_csv(os.path.join(SYNTHETIC_TEST_DIR, "ground_truth.csv"))

for case_name, test_file in test_cases:
    print(f"\n{'='*80}")
    print(f"Case: {case_name} - {test_file}")
    print(f"{'='*80}")

    # Load test image
    test_path = os.path.join(SYNTHETIC_TEST_DIR, "images", test_file)
    test_image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if test_image is None:
        print(f"⚠ Could not load {test_file}")
        continue

    # Get ground truth
    gt_row = gt_df[gt_df['synthetic_file'] == test_file]
    if len(gt_row) == 0:
        print(f"⚠ No ground truth for {test_file}")
        continue

    gt_source = Path(gt_row.iloc[0]['source_file']).name
    gt_match = re.search(r'\d+', gt_source)
    gt_slice_num = int(gt_match.group()) if gt_match else 0

    print(f"Ground truth: {gt_source} (slice {gt_slice_num})")

    # ===================================================================
    # ORB Processing
    # ===================================================================
    print("\n1. Running ORB...")
    try:
        top_orb = orb_reg.find_best_atlas_slice(test_image, RAW_ATLAS_DIR, top_k=1, verbose=False)
        orb_file, _, _ = top_orb[0]
        orb_num = int(re.search(r'\d+', orb_file).group())

        atlas_orb = cv2.imread(os.path.join(RAW_ATLAS_DIR, orb_file), cv2.IMREAD_GRAYSCALE)
        reg_orb = orb_reg.register_to_atlas(test_image, atlas_orb, nfeatures_fine=4000)

        error_orb = abs(gt_slice_num - orb_num)
        metrics_orb = compute_alignment_metrics(atlas_orb, reg_orb['registered_image']) if reg_orb['success'] else {'nmi': 0, 'ssim': 0}

        print(f"   Predicted: slice {orb_num}, Error: {error_orb}")
        print(f"   NMI: {metrics_orb['nmi']:.3f}, SSIM: {metrics_orb['ssim']:.3f}")
        print(f"   Matches: {reg_orb['stats']['n_matches']}, Inliers: {reg_orb['stats']['n_inliers']} ({reg_orb['stats']['inlier_ratio']*100:.1f}%)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        orb_file = None
        reg_orb = None
        error_orb = 999
        metrics_orb = {'nmi': 0, 'ssim': 0}

    # ===================================================================
    # SIFT Processing
    # ===================================================================
    print("\n2. Running SIFT...")
    try:
        top_sift = sift_reg.find_best_atlas_slice(test_image, RAW_ATLAS_DIR, top_k=1, verbose=False)
        sift_file, _, _ = top_sift[0]
        sift_num = int(re.search(r'\d+', sift_file).group())

        atlas_sift = cv2.imread(os.path.join(RAW_ATLAS_DIR, sift_file), cv2.IMREAD_GRAYSCALE)
        reg_sift = sift_reg.register_to_atlas(test_image, atlas_sift, nfeatures_fine=4000)

        error_sift = abs(gt_slice_num - sift_num)
        metrics_sift = compute_alignment_metrics(atlas_sift, reg_sift['registered_image']) if reg_sift['success'] else {'nmi': 0, 'ssim': 0}

        print(f"   Predicted: slice {sift_num}, Error: {error_sift}")
        print(f"   NMI: {metrics_sift['nmi']:.3f}, SSIM: {metrics_sift['ssim']:.3f}")
        print(f"   Matches: {reg_sift['stats']['n_matches']}, Inliers: {reg_sift['stats']['n_inliers']} ({reg_sift['stats']['inlier_ratio']*100:.1f}%)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sift_file = None
        reg_sift = None
        error_sift = 999
        metrics_sift = {'nmi': 0, 'ssim': 0}

    # ===================================================================
    # Create Visualization
    # ===================================================================
    print("\n3. Creating visualization...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.4, wspace=0.3)

    # Row 0: Input and Ground Truth
    ax_input = fig.add_subplot(gs[0, 0:2])
    ax_input.imshow(test_image, cmap='gray')
    ax_input.set_title(f'Query Image: {test_file}', fontsize=14, fontweight='bold')
    ax_input.axis('off')

    ax_gt = fig.add_subplot(gs[0, 2:4])
    gt_atlas = cv2.imread(os.path.join(RAW_ATLAS_DIR, gt_source), cv2.IMREAD_GRAYSCALE)
    if gt_atlas is not None:
        ax_gt.imshow(gt_atlas, cmap='gray')
    ax_gt.set_title(f'Ground Truth: {gt_source} (slice {gt_slice_num})', fontsize=14, fontweight='bold')
    for spine in ax_gt.spines.values():
        spine.set_edgecolor('green')
        spine.set_linewidth(4)
    ax_gt.axis('off')

    # Comparison stats box
    ax_stats = fig.add_subplot(gs[0, 4])
    ax_stats.axis('off')
    stats_text = f"""COMPARISON

ORB:
  Error: {error_orb} slices
  NMI: {metrics_orb['nmi']:.3f}
  SSIM: {metrics_orb['ssim']:.3f}

SIFT:
  Error: {error_sift} slices
  NMI: {metrics_sift['nmi']:.3f}
  SSIM: {metrics_sift['ssim']:.3f}

Winner: {'ORB' if error_orb < error_sift else 'SIFT' if error_sift < error_orb else 'Tie'}
Δ Error: {abs(error_orb - error_sift)} slices
"""
    ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', va='center',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Row 1: ORB Results
    ax_orb1 = fig.add_subplot(gs[1, 0])
    ax_orb1.imshow(test_image, cmap='gray')
    ax_orb1.set_title('Query', fontsize=11)
    ax_orb1.axis('off')

    ax_orb2 = fig.add_subplot(gs[1, 1])
    if reg_orb and reg_orb['registered_image'] is not None:
        ax_orb2.imshow(reg_orb['registered_image'], cmap='gray')
    ax_orb2.set_title('Registered (ORB)', fontsize=11)
    ax_orb2.axis('off')

    ax_orb3 = fig.add_subplot(gs[1, 2])
    if orb_file and atlas_orb is not None:
        ax_orb3.imshow(atlas_orb, cmap='gray')
        color = 'green' if error_orb <= 4 else 'red'
        ax_orb3.set_title(f'Atlas: {orb_file}\nError: {error_orb} slices',
                         fontsize=11, color=color, fontweight='bold')
        for spine in ax_orb3.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    ax_orb3.axis('off')

    ax_orb4 = fig.add_subplot(gs[1, 3])
    if reg_orb and reg_orb['registered_image'] is not None and atlas_orb is not None:
        diff_orb = cv2.absdiff(reg_orb['registered_image'], atlas_orb)
        ax_orb4.imshow(diff_orb, cmap='hot')
        ax_orb4.set_title(f'Difference\nNMI: {metrics_orb["nmi"]:.3f}', fontsize=11)
    ax_orb4.axis('off')

    ax_orb5 = fig.add_subplot(gs[1, 4])
    ax_orb5.axis('off')
    orb_info = f"""ORB Details:

Matches: {reg_orb['stats']['n_matches'] if reg_orb else 0}
Inliers: {reg_orb['stats']['n_inliers'] if reg_orb else 0}
Ratio: {reg_orb['stats']['inlier_ratio']*100:.1f}% if reg_orb else 0

Binary descriptors
Hamming distance
FAST corners
"""
    ax_orb5.text(0.1, 0.5, orb_info, fontsize=10, family='monospace', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Row 2: SIFT Results
    ax_sift1 = fig.add_subplot(gs[2, 0])
    ax_sift1.imshow(test_image, cmap='gray')
    ax_sift1.set_title('Query', fontsize=11)
    ax_sift1.axis('off')

    ax_sift2 = fig.add_subplot(gs[2, 1])
    if reg_sift and reg_sift['registered_image'] is not None:
        ax_sift2.imshow(reg_sift['registered_image'], cmap='gray')
    ax_sift2.set_title('Registered (SIFT)', fontsize=11)
    ax_sift2.axis('off')

    ax_sift3 = fig.add_subplot(gs[2, 2])
    if sift_file and atlas_sift is not None:
        ax_sift3.imshow(atlas_sift, cmap='gray')
        color = 'green' if error_sift <= 4 else 'red'
        ax_sift3.set_title(f'Atlas: {sift_file}\nError: {error_sift} slices',
                          fontsize=11, color=color, fontweight='bold')
        for spine in ax_sift3.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    ax_sift3.axis('off')

    ax_sift4 = fig.add_subplot(gs[2, 3])
    if reg_sift and reg_sift['registered_image'] is not None and atlas_sift is not None:
        diff_sift = cv2.absdiff(reg_sift['registered_image'], atlas_sift)
        ax_sift4.imshow(diff_sift, cmap='hot')
        ax_sift4.set_title(f'Difference\nNMI: {metrics_sift["nmi"]:.3f}', fontsize=11)
    ax_sift4.axis('off')

    ax_sift5 = fig.add_subplot(gs[2, 4])
    ax_sift5.axis('off')
    sift_info = f"""SIFT Details:

Matches: {reg_sift['stats']['n_matches'] if reg_sift else 0}
Inliers: {reg_sift['stats']['n_inliers'] if reg_sift else 0}
Ratio: {reg_sift['stats']['inlier_ratio']*100:.1f}% if reg_sift else 0

Float descriptors
L2 distance
DoG blobs
"""
    ax_sift5.text(0.1, 0.5, sift_info, fontsize=10, family='monospace', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Add row labels
    fig.text(0.02, 0.62, 'ORB\n(Method 2)', ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    fig.text(0.02, 0.28, 'SIFT\n(Method 2)', ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.suptitle(f'ORB vs SIFT Comparison: {case_name}\n{test_file}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save
    safe_name = test_file.replace('.png', '').replace('.jpg', '')
    fig_path = os.path.join(VISUAL_DIR, f"orb_vs_sift_{safe_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fig_path}")
    plt.close()

print("\n" + "="*80)
print("✅ ORB vs SIFT visual comparison complete!")
print("="*80)
print(f"\nGenerated {len(test_cases)} comparison figures in: {VISUAL_DIR}")
print("\nEach figure shows:")
print("  • Query image and ground truth atlas")
print("  • ORB results: prediction, registration, difference, metrics")
print("  • SIFT results: prediction, registration, difference, metrics")
print("  • Side-by-side comparison highlighting where ORB outperforms SIFT")
