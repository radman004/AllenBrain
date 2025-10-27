"""
Visual Side-by-Side Method Output Comparison
Shows actual registration results from each method for the same test images
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
# Note: Method 1 (MI) implementation is in notebooks, not extracted yet

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")
RESULTS_DIR = os.path.join(SYNTHETIC_TEST_DIR, "results")
VISUAL_DIR = os.path.join(RESULTS_DIR, "visual_comparison")

os.makedirs(VISUAL_DIR, exist_ok=True)

print("="*80)
print("VISUAL METHOD OUTPUT COMPARISON")
print("="*80)

# Load results
method2_orb_path = os.path.join(RESULTS_DIR, "method2_orb_results.csv")
method2_sift_path = os.path.join(RESULTS_DIR, "method2_sift_results.csv")
method1_path = os.path.join(SYNTHETIC_TEST_DIR, "match_results.csv")

df_orb = pd.read_csv(method2_orb_path) if os.path.exists(method2_orb_path) else None
df_sift = pd.read_csv(method2_sift_path) if os.path.exists(method2_sift_path) else None
df_mi = pd.read_csv(method1_path) if os.path.exists(method1_path) else None

# Find interesting test cases:
# 1. One where all methods succeed
# 2. One where ORB succeeds but SIFT fails
# 3. One challenging case

print("\n" + "="*80)
print("Selecting Test Cases")
print("="*80)

test_cases = []

# Case 1: All methods correct
if df_orb is not None and df_sift is not None:
    # Find images where both are correct
    orb_correct = df_orb[df_orb['correct'] == True] if 'correct' in df_orb.columns else df_orb[df_orb['slice_error'] == 0]
    sift_correct = df_sift[df_sift['correct'] == True] if 'correct' in df_sift.columns else df_sift[df_sift['slice_error'] == 0]

    common_correct = set(orb_correct['synthetic_file']) & set(sift_correct['synthetic_file'])
    if len(common_correct) > 0:
        case1_file = list(common_correct)[0]
        test_cases.append(('All Correct', case1_file))
        print(f"✓ Case 1 (All Correct): {case1_file}")

# Case 2: ORB correct, SIFT wrong
if df_orb is not None and df_sift is not None:
    orb_correct = set(df_orb[df_orb['slice_error'] == 0]['synthetic_file'])
    sift_wrong = set(df_sift[df_sift['slice_error'] > 4]['synthetic_file'])
    orb_good_sift_bad = orb_correct & sift_wrong
    if len(orb_good_sift_bad) > 0:
        case2_file = list(orb_good_sift_bad)[0]
        test_cases.append(('ORB Success, SIFT Fail', case2_file))
        print(f"✓ Case 2 (ORB Success, SIFT Fail): {case2_file}")

# Case 3: Both struggle (high error but still within ±4)
if df_orb is not None:
    moderate_error = df_orb[(df_orb['slice_error'] > 0) & (df_orb['slice_error'] <= 4)]
    if len(moderate_error) > 0:
        case3_file = moderate_error.iloc[0]['synthetic_file']
        test_cases.append(('Challenging Case', case3_file))
        print(f"✓ Case 3 (Challenging): {case3_file}")

# If we don't have enough cases, just pick first few from ORB
if len(test_cases) < 3 and df_orb is not None:
    for i in range(min(3 - len(test_cases), len(df_orb))):
        if df_orb.iloc[i]['synthetic_file'] not in [c[1] for c in test_cases]:
            test_cases.append((f'Example {i+1}', df_orb.iloc[i]['synthetic_file']))

print(f"\n✓ Selected {len(test_cases)} test cases for visualization")

# ===================================================================
# Initialize registrators
# ===================================================================

print("\n" + "="*80)
print("Initializing Registrators")
print("="*80)

registrator_orb = FeatureBasedRegistration(
    detector_type='ORB',
    nfeatures=2000,
    ratio_threshold=0.75,
    ransac_threshold=5.0
)
print("✓ ORB registrator initialized")

registrator_sift = FeatureBasedRegistration(
    detector_type='SIFT',
    nfeatures=2000,
    ratio_threshold=0.75,
    ransac_threshold=5.0
)
print("✓ SIFT registrator initialized")

# ===================================================================
# Process each test case
# ===================================================================

print("\n" + "="*80)
print("Processing Test Cases")
print("="*80)

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
    gt_df = pd.read_csv(os.path.join(SYNTHETIC_TEST_DIR, "ground_truth.csv"))
    gt_row = gt_df[gt_df['synthetic_file'] == test_file]
    if len(gt_row) == 0:
        print(f"⚠ No ground truth for {test_file}")
        continue

    gt_source = Path(gt_row.iloc[0]['source_file']).name
    gt_match = re.search(r'\d+', gt_source)
    gt_slice_num = int(gt_match.group()) if gt_match else 0

    print(f"Ground truth: {gt_source} (slice {gt_slice_num})")

    # ===================================================================
    # Method 2 ORB
    # ===================================================================
    print("\n1. Running Method 2 (ORB)...")
    try:
        top_matches_orb = registrator_orb.find_best_atlas_slice(
            query_image=test_image,
            atlas_dir=RAW_ATLAS_DIR,
            top_k=1,
            verbose=False
        )
        best_orb_file, _, _ = top_matches_orb[0]
        pred_orb_num = int(re.search(r'\d+', best_orb_file).group())

        # Load atlas and register
        atlas_orb = cv2.imread(os.path.join(RAW_ATLAS_DIR, best_orb_file), cv2.IMREAD_GRAYSCALE)
        reg_orb = registrator_orb.register_to_atlas(test_image, atlas_orb, nfeatures_fine=4000)

        error_orb = abs(gt_slice_num - pred_orb_num)
        print(f"   Predicted: slice {pred_orb_num}, Error: {error_orb}")

        if reg_orb['success']:
            metrics_orb = compute_alignment_metrics(atlas_orb, reg_orb['registered_image'])
            print(f"   NMI: {metrics_orb['nmi']:.3f}, SSIM: {metrics_orb['ssim']:.3f}")
        else:
            print(f"   ⚠ Registration failed")
            metrics_orb = {'nmi': 0, 'ssim': 0}

    except Exception as e:
        print(f"   ❌ Error: {e}")
        best_orb_file = None
        reg_orb = None
        error_orb = 999
        metrics_orb = {'nmi': 0, 'ssim': 0}

    # ===================================================================
    # Method 2 SIFT
    # ===================================================================
    print("\n2. Running Method 2 (SIFT)...")
    try:
        top_matches_sift = registrator_sift.find_best_atlas_slice(
            query_image=test_image,
            atlas_dir=RAW_ATLAS_DIR,
            top_k=1,
            verbose=False
        )
        best_sift_file, _, _ = top_matches_sift[0]
        pred_sift_num = int(re.search(r'\d+', best_sift_file).group())

        # Load atlas and register
        atlas_sift = cv2.imread(os.path.join(RAW_ATLAS_DIR, best_sift_file), cv2.IMREAD_GRAYSCALE)
        reg_sift = registrator_sift.register_to_atlas(test_image, atlas_sift, nfeatures_fine=4000)

        error_sift = abs(gt_slice_num - pred_sift_num)
        print(f"   Predicted: slice {pred_sift_num}, Error: {error_sift}")

        if reg_sift['success']:
            metrics_sift = compute_alignment_metrics(atlas_sift, reg_sift['registered_image'])
            print(f"   NMI: {metrics_sift['nmi']:.3f}, SSIM: {metrics_sift['ssim']:.3f}")
        else:
            print(f"   ⚠ Registration failed")
            metrics_sift = {'nmi': 0, 'ssim': 0}

    except Exception as e:
        print(f"   ❌ Error: {e}")
        best_sift_file = None
        reg_sift = None
        error_sift = 999
        metrics_sift = {'nmi': 0, 'ssim': 0}

    # ===================================================================
    # Visualization
    # ===================================================================
    print("\n3. Creating visualization...")

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Row 0: Input and ground truth
    ax00 = fig.add_subplot(gs[0, 0:2])
    ax00.imshow(test_image, cmap='gray')
    ax00.set_title(f'Query Image: {test_file}', fontsize=13, fontweight='bold')
    ax00.axis('off')

    ax01 = fig.add_subplot(gs[0, 2:4])
    gt_atlas = cv2.imread(os.path.join(RAW_ATLAS_DIR, gt_source), cv2.IMREAD_GRAYSCALE)
    if gt_atlas is not None:
        ax01.imshow(gt_atlas, cmap='gray')
    ax01.set_title(f'Ground Truth: {gt_source} (slice {gt_slice_num})', fontsize=13, fontweight='bold')
    for spine in ax01.spines.values():
        spine.set_edgecolor('green')
        spine.set_linewidth(3)
    ax01.axis('off')

    # Row 1: Method 2 ORB
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(test_image, cmap='gray')
    ax10.set_title('Query', fontsize=11)
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[1, 1])
    if reg_orb and reg_orb['registered_image'] is not None:
        ax11.imshow(reg_orb['registered_image'], cmap='gray')
        ax11.set_title(f'Registered (ORB)', fontsize=11)
    else:
        ax11.text(0.5, 0.5, 'Failed', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Registered (ORB)', fontsize=11)
    ax11.axis('off')

    ax12 = fig.add_subplot(gs[1, 2])
    if best_orb_file and atlas_orb is not None:
        ax12.imshow(atlas_orb, cmap='gray')
        color = 'green' if error_orb <= 4 else 'red'
        ax12.set_title(f'Atlas: {best_orb_file}\nError: {error_orb}', fontsize=11, color=color, fontweight='bold')
        for spine in ax12.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    ax12.axis('off')

    ax13 = fig.add_subplot(gs[1, 3])
    if reg_orb and reg_orb['registered_image'] is not None and atlas_orb is not None:
        diff_orb = cv2.absdiff(reg_orb['registered_image'], atlas_orb)
        ax13.imshow(diff_orb, cmap='hot')
        ax13.set_title(f'Difference\nNMI: {metrics_orb["nmi"]:.3f}', fontsize=11)
    ax13.axis('off')

    # Row 2: Method 2 SIFT
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.imshow(test_image, cmap='gray')
    ax20.set_title('Query', fontsize=11)
    ax20.axis('off')

    ax21 = fig.add_subplot(gs[2, 1])
    if reg_sift and reg_sift['registered_image'] is not None:
        ax21.imshow(reg_sift['registered_image'], cmap='gray')
        ax21.set_title(f'Registered (SIFT)', fontsize=11)
    else:
        ax21.text(0.5, 0.5, 'Failed', ha='center', va='center', transform=ax21.transAxes)
        ax21.set_title('Registered (SIFT)', fontsize=11)
    ax21.axis('off')

    ax22 = fig.add_subplot(gs[2, 2])
    if best_sift_file and atlas_sift is not None:
        ax22.imshow(atlas_sift, cmap='gray')
        color = 'green' if error_sift <= 4 else 'red'
        ax22.set_title(f'Atlas: {best_sift_file}\nError: {error_sift}', fontsize=11, color=color, fontweight='bold')
        for spine in ax22.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    ax22.axis('off')

    ax23 = fig.add_subplot(gs[2, 3])
    if reg_sift and reg_sift['registered_image'] is not None and atlas_sift is not None:
        diff_sift = cv2.absdiff(reg_sift['registered_image'], atlas_sift)
        ax23.imshow(diff_sift, cmap='hot')
        ax23.set_title(f'Difference\nNMI: {metrics_sift["nmi"]:.3f}', fontsize=11)
    ax23.axis('off')

    # Add row labels
    fig.text(0.02, 0.68, 'Method 2\n(ORB)', ha='center', va='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    fig.text(0.02, 0.35, 'Method 2\n(SIFT)', ha='center', va='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.suptitle(f'Visual Method Comparison: {case_name}\n{test_file}',
                 fontsize=15, fontweight='bold', y=0.98)

    # Save
    safe_name = test_file.replace('.png', '').replace('.jpg', '')
    fig_path = os.path.join(VISUAL_DIR, f"comparison_{safe_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {fig_path}")
    plt.close()

print("\n" + "="*80)
print("✅ Visual comparison complete!")
print("="*80)
print(f"\nVisualizations saved to: {VISUAL_DIR}")
print(f"\nGenerated {len(test_cases)} comparison figures showing:")
print("  - Query image vs ground truth")
print("  - ORB registration results (Z-level prediction, registered image, difference)")
print("  - SIFT registration results (Z-level prediction, registered image, difference)")
print("  - Color-coded borders: Green = correct (≤4 slices), Red = incorrect")
