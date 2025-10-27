"""
Create Visual Pipeline Diagrams for Method 2
Shows: Input → Process → Output with actual images

This script generates figures showing:
1. Feature detection on query image
2. Feature matching (query vs atlas)
3. RANSAC inliers/outliers
4. Registration result (before/after)
5. Quality metrics visualization
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))
from method2_features import FeatureBasedRegistration, compute_alignment_metrics

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")
RESULTS_DIR = os.path.join(SYNTHETIC_TEST_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "pipeline_figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

print("="*60)
print("Creating Pipeline Visualizations")
print("="*60)

# Load ground truth
gt_csv_path = os.path.join(SYNTHETIC_TEST_DIR, "ground_truth.csv")
gt_df = pd.read_csv(gt_csv_path)

# Select a good example (one with exact match)
# Let's use the first successful one
test_row = gt_df.iloc[41]  # synthetic_00042.png

# Load test image
test_path = os.path.join(SYNTHETIC_TEST_DIR, "images", test_row["synthetic_file"])
test_image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

# Extract ground truth
gt_source = Path(test_row['source_file']).name
import re
gt_match = re.search(r'\d+', gt_source)
gt_slice_num = int(gt_match.group())

print(f"\nProcessing example: {test_row['synthetic_file']}")
print(f"Ground truth: {gt_source} (slice {gt_slice_num})")

# Initialize registrator
registrator = FeatureBasedRegistration(
    detector_type='ORB',
    nfeatures=2000,
    ratio_threshold=0.75,
    ransac_threshold=5.0
)

print("\n" + "="*60)
print("PHASE 1: Feature Detection")
print("="*60)

# Detect features in query
query_kp, query_desc = registrator.detect_and_compute(test_image)
print(f"Detected {len(query_kp)} keypoints in query image")

# Visualize feature detection
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original image
axes[0].imshow(test_image, cmap='gray')
axes[0].set_title('Input: Query Brain Slice\n(synthetic_00042.png)', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Image with keypoints
img_with_kp = cv2.drawKeypoints(
    test_image, query_kp, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(0, 255, 0)
)
axes[1].imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Feature Detection (ORB)\n{len(query_kp)} keypoints detected', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "step1_feature_detection.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*60)
print("PHASE 2: Z-Level Search")
print("="*60)

# Find best atlas slice
top_matches = registrator.find_best_atlas_slice(
    query_image=test_image,
    atlas_dir=RAW_ATLAS_DIR,
    top_k=5,
    verbose=False
)

print(f"\nTop 5 matches:")
for i, (filename, match_count, stats) in enumerate(top_matches, 1):
    print(f"  {i}. {filename}: {match_count} matches")

best_filename, best_match_count, best_stats = top_matches[0]
pred_match = re.search(r'\d+', best_filename)
pred_slice_num = int(pred_match.group())

is_correct = abs(gt_slice_num - pred_slice_num) <= 4
print(f"\nPrediction: {best_filename} (slice {pred_slice_num})")
print(f"Ground truth: {gt_source} (slice {gt_slice_num})")
print(f"Slice error: {abs(gt_slice_num - pred_slice_num)}")
print(f"Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

# Load atlas image
atlas_image_path = os.path.join(RAW_ATLAS_DIR, best_filename)
atlas_image = cv2.imread(atlas_image_path, cv2.IMREAD_GRAYSCALE)

# Visualize Z-level search results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Query
axes[0, 0].imshow(test_image, cmap='gray')
axes[0, 0].set_title('Query Image\n(Unknown Z-level)', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Top 5 matches
for i, (filename, match_count, stats) in enumerate(top_matches[:5]):
    row = (i + 1) // 3
    col = (i + 1) % 3

    atlas_path = os.path.join(RAW_ATLAS_DIR, filename)
    atlas_img = cv2.imread(atlas_path, cv2.IMREAD_GRAYSCALE)

    axes[row, col].imshow(atlas_img, cmap='gray')

    # Extract slice number
    slice_match = re.search(r'\d+', filename)
    slice_num = int(slice_match.group()) if slice_match else 0

    # Color based on correctness
    border_color = 'green' if i == 0 and is_correct else 'red' if i == 0 else 'gray'
    for spine in axes[row, col].spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3 if i == 0 else 1)

    title = f"Rank {i+1}: {filename}\n{match_count} matches"
    if i == 0:
        title += "\n✓ BEST MATCH"
    axes[row, col].set_title(title, fontsize=10, fontweight='bold' if i == 0 else 'normal')
    axes[row, col].axis('off')

plt.suptitle('PHASE 1: Z-Level Search Results', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(FIGURES_DIR, "step2_zlevel_search.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*60)
print("PHASE 3: Fine Registration with RANSAC")
print("="*60)

# Fine registration
reg_result = registrator.register_to_atlas(
    query_image=test_image,
    atlas_image=atlas_image,
    nfeatures_fine=4000
)

print(f"Registration success: {reg_result['success']}")
print(f"Total matches: {reg_result['stats']['n_matches']}")
print(f"Inliers: {reg_result['stats']['n_inliers']}")
print(f"Inlier ratio: {reg_result['stats']['inlier_ratio']:.3f}")

# Extract transformation parameters from matrix
M = reg_result['transform']
if M is not None:
    angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    tx = M[0, 2]
    ty = M[1, 2]
    print(f"Rotation: {angle:.2f}°")
    print(f"Scale: {scale:.3f}")
    print(f"Translation: ({tx:.1f}, {ty:.1f}) pixels")
else:
    angle = scale = tx = ty = 0.0

# Visualize feature matching
# Re-detect with 4000 features to get keypoints
registrator.detector.setMaxFeatures(4000)
query_kp_fine, query_desc_fine = registrator.detect_and_compute(test_image)
atlas_kp_fine, atlas_desc_fine = registrator.detect_and_compute(atlas_image)
matches = registrator.match_features(query_desc_fine, atlas_desc_fine)

# Estimate transform to get inliers
M, inlier_mask, stats = registrator.estimate_transform(
    query_kp_fine, atlas_kp_fine, matches, min_inliers=15
)

# Draw matches (showing only inliers in green, outliers in red)
matches_img = np.zeros((
    max(test_image.shape[0], atlas_image.shape[0]),
    test_image.shape[1] + atlas_image.shape[1],
    3
), dtype=np.uint8)

# Place images side by side
matches_img[:test_image.shape[0], :test_image.shape[1], :] = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
matches_img[:atlas_image.shape[0], test_image.shape[1]:, :] = cv2.cvtColor(atlas_image, cv2.COLOR_GRAY2BGR)

# Draw matches
np.random.seed(42)
sample_indices = np.random.choice(len(matches), min(50, len(matches)), replace=False)

for idx in sample_indices:
    m = matches[idx]
    is_inlier = inlier_mask[idx]

    pt1 = tuple(map(int, query_kp_fine[m.queryIdx].pt))
    pt2 = tuple(map(int, atlas_kp_fine[m.trainIdx].pt))
    pt2 = (pt2[0] + test_image.shape[1], pt2[1])

    color = (0, 255, 0) if is_inlier else (255, 0, 0)  # Green for inliers, red for outliers
    thickness = 2 if is_inlier else 1

    cv2.line(matches_img, pt1, pt2, color, thickness)
    cv2.circle(matches_img, pt1, 3, color, -1)
    cv2.circle(matches_img, pt2, 3, color, -1)

fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
ax.set_title(f'Feature Matching & RANSAC\nTotal: {len(matches)} matches | Inliers: {stats["n_inliers"]} ({stats["inlier_ratio"]*100:.1f}%)',
             fontsize=13, fontweight='bold')
ax.axis('off')

# Add legend
green_patch = mpatches.Patch(color='green', label=f'Inliers ({stats["n_inliers"]})')
red_patch = mpatches.Patch(color='red', label=f'Outliers ({len(matches) - stats["n_inliers"]})')
ax.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=11)

# Add labels
ax.text(test_image.shape[1]//2, 20, 'QUERY', ha='center', color='yellow', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
ax.text(test_image.shape[1] + atlas_image.shape[1]//2, 20, 'ATLAS', ha='center', color='yellow', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "step3_feature_matching.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*60)
print("PHASE 4: Registration Result")
print("="*60)

# Compute metrics
metrics = compute_alignment_metrics(atlas_image, reg_result['registered_image'])
print(f"NMI:  {metrics['nmi']:.4f}")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"MSE:  {metrics['mse']:.2f}")

# Visualize registration result
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Before registration
axes[0, 0].imshow(test_image, cmap='gray')
axes[0, 0].set_title('Query Image\n(Before Registration)', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(atlas_image, cmap='gray')
axes[0, 1].set_title(f'Atlas Reference\n({best_filename})', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Difference before
diff_before = cv2.absdiff(test_image, atlas_image)
axes[0, 2].imshow(diff_before, cmap='hot')
axes[0, 2].set_title('Difference (Before)\nHigh error', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# Row 2: After registration
axes[1, 0].imshow(reg_result['registered_image'], cmap='gray')
axes[1, 0].set_title('Query Image\n(After Registration)', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(atlas_image, cmap='gray')
axes[1, 1].set_title(f'Atlas Reference\n({best_filename})', fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Difference after
diff_after = cv2.absdiff(reg_result['registered_image'], atlas_image)
axes[1, 2].imshow(diff_after, cmap='hot')
axes[1, 2].set_title('Difference (After)\nLow error', fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

# Add metrics text
metrics_text = f"""Registration Quality:
NMI:  {metrics['nmi']:.4f}
SSIM: {metrics['ssim']:.4f}
MSE:  {metrics['mse']:.2f}

Transformation:
Rotation: {angle:.2f}°
Scale: {scale:.3f}
Translation: ({tx:.1f}, {ty:.1f})px
Inliers: {reg_result['stats']['n_inliers']}/{reg_result['stats']['n_matches']} ({reg_result['stats']['inlier_ratio']*100:.1f}%)"""

plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('PHASE 2 & 3: Registration Result (Before vs After)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.12, 1, 0.96])
fig_path = os.path.join(FIGURES_DIR, "step4_registration_result.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*60)
print("PHASE 5: Overlay Visualization")
print("="*60)

# Create overlay (checkerboard pattern)
h, w = atlas_image.shape
overlay = np.zeros((h, w, 3), dtype=np.uint8)

# Red channel: Atlas
# Green channel: Registered query
overlay[:, :, 2] = atlas_image  # Red for atlas
overlay[:, :, 1] = reg_result['registered_image']  # Green for registered

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overlay
axes[0].imshow(overlay)
axes[0].set_title('Overlay Visualization\nRed=Atlas, Green=Query, Yellow=Overlap', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Checkerboard
checker_size = 32
checker = np.zeros_like(atlas_image)
for i in range(0, h, checker_size):
    for j in range(0, w, checker_size):
        if ((i // checker_size) + (j // checker_size)) % 2 == 0:
            checker[i:i+checker_size, j:j+checker_size] = atlas_image[i:i+checker_size, j:j+checker_size]
        else:
            checker[i:i+checker_size, j:j+checker_size] = reg_result['registered_image'][i:i+checker_size, j:j+checker_size]

axes[1].imshow(checker, cmap='gray')
axes[1].set_title('Checkerboard Visualization\nAlternating between Atlas and Registered Query', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.suptitle(f'Registration Quality Assessment (NMI={metrics["nmi"]:.3f}, SSIM={metrics["ssim"]:.3f})',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(FIGURES_DIR, "step5_overlay_visualization.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*60)
print("Creating Complete Pipeline Summary")
print("="*60)

# Create summary figure with all steps
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# Step 1: Input
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(test_image, cmap='gray')
ax1.set_title('1. INPUT\nQuery Brain Slice', fontsize=11, fontweight='bold')
ax1.axis('off')

# Step 2: Feature detection
ax2 = fig.add_subplot(gs[0, 1])
img_kp = cv2.drawKeypoints(test_image, query_kp[:100], None, color=(0, 255, 0))  # Show 100 keypoints for clarity
ax2.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
ax2.set_title(f'2. FEATURE DETECTION\n{len(query_kp)} ORB keypoints', fontsize=11, fontweight='bold')
ax2.axis('off')

# Step 3: Best atlas match
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(atlas_image, cmap='gray')
ax3.set_title(f'3. Z-LEVEL SEARCH\nBest: {best_filename}\n{best_match_count} matches', fontsize=11, fontweight='bold')
for spine in ax3.spines.values():
    spine.set_edgecolor('green')
    spine.set_linewidth(3)
ax3.axis('off')

# Step 4: Feature matching (small version)
ax4 = fig.add_subplot(gs[1, :])
# Use a smaller sample for summary
sample_matches_img = matches_img.copy()
ax4.imshow(cv2.cvtColor(sample_matches_img, cv2.COLOR_BGR2RGB))
ax4.set_title(f'4. FEATURE MATCHING & RANSAC\n{stats["n_inliers"]}/{len(matches)} inliers ({stats["inlier_ratio"]*100:.1f}%)',
              fontsize=11, fontweight='bold')
ax4.axis('off')

# Step 5: Transformation
ax5 = fig.add_subplot(gs[2, 0])
# Draw transformation visualization
transform_vis = np.ones((200, 200, 3), dtype=np.uint8) * 255
cv2.putText(transform_vis, 'AFFINE', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(transform_vis, 'TRANSFORM', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
cv2.putText(transform_vis, f'Rot: {angle:.1f}deg', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(transform_vis, f'Scale: {scale:.2f}', (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(transform_vis, f'Tx: {tx:.1f}px', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(transform_vis, f'Ty: {ty:.1f}px', (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
ax5.imshow(transform_vis)
ax5.set_title('5. TRANSFORMATION\nParameters', fontsize=11, fontweight='bold')
ax5.axis('off')

# Step 6: Registered image
ax6 = fig.add_subplot(gs[2, 1])
ax6.imshow(reg_result['registered_image'], cmap='gray')
ax6.set_title('6. REGISTRATION\nAligned Query', fontsize=11, fontweight='bold')
ax6.axis('off')

# Step 7: Difference
ax7 = fig.add_subplot(gs[2, 2])
ax7.imshow(diff_after, cmap='hot')
ax7.set_title('7. DIFFERENCE\nLow Error', fontsize=11, fontweight='bold')
ax7.axis('off')

# Step 8: Metrics
ax8 = fig.add_subplot(gs[3, :])
ax8.axis('off')
metrics_summary = f"""
═══════════════════════════════════════════════════════════════════════════════════════
                                    FINAL RESULTS
═══════════════════════════════════════════════════════════════════════════════════════

Query Image: {test_row['synthetic_file']}
Ground Truth: {gt_source} (slice {gt_slice_num})
Prediction: {best_filename} (slice {pred_slice_num})
Slice Error: {abs(gt_slice_num - pred_slice_num)} slices

Status: {'✓ EXACT MATCH!' if abs(gt_slice_num - pred_slice_num) == 0 else '✓ CORRECT (within ±4)' if is_correct else '✗ INCORRECT'}

Quality Metrics:
  • NMI (Normalized Mutual Information): {metrics['nmi']:.4f}   [Target: >1.6]
  • SSIM (Structural Similarity):        {metrics['ssim']:.4f}  [Target: >0.8]
  • MSE (Mean Squared Error):            {metrics['mse']:.2f}   [Target: <500]

Registration Performance:
  • Total matches: {reg_result['stats']['n_matches']}
  • Inliers: {reg_result['stats']['n_inliers']} ({reg_result['stats']['inlier_ratio']*100:.1f}%)
  • Rotation: {angle:.2f}°
  • Scale: {scale:.3f}
  • Translation: ({tx:.1f}, {ty:.1f}) pixels

═══════════════════════════════════════════════════════════════════════════════════════
"""
ax8.text(0.5, 0.5, metrics_summary, ha='center', va='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if is_correct else 'lightcoral', alpha=0.8))

plt.suptitle('Complete Pipeline: Feature-Based Registration (Method 2 - ORB)',
             fontsize=15, fontweight='bold', y=0.98)
fig_path = os.path.join(FIGURES_DIR, "complete_pipeline_summary.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*60)
print("✅ All visualizations created successfully!")
print("="*60)
print(f"\nGenerated figures:")
print(f"  1. {os.path.join(FIGURES_DIR, 'step1_feature_detection.png')}")
print(f"  2. {os.path.join(FIGURES_DIR, 'step2_zlevel_search.png')}")
print(f"  3. {os.path.join(FIGURES_DIR, 'step3_feature_matching.png')}")
print(f"  4. {os.path.join(FIGURES_DIR, 'step4_registration_result.png')}")
print(f"  5. {os.path.join(FIGURES_DIR, 'step5_overlay_visualization.png')}")
print(f"  6. {os.path.join(FIGURES_DIR, 'complete_pipeline_summary.png')}")
print(f"\nAll figures saved to: {FIGURES_DIR}")
print("\n✅ Ready to use in your report!")
