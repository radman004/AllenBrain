"""
Comprehensive Method Comparison and Visual Output Analysis
Compares Methods 1 (MI), 2 (ORB), 2 (SIFT), and 5 (ORB Enhanced)
Shows visual side-by-side outputs
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import re

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RAW_ATLAS_DIR = os.path.join(BASE_DIR, "raw_atlas_slices")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")
RESULTS_DIR = os.path.join(SYNTHETIC_TEST_DIR, "results")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "method_comparison")

os.makedirs(COMPARISON_DIR, exist_ok=True)

print("="*80)
print("COMPREHENSIVE METHOD COMPARISON")
print("="*80)

# ===================================================================
# PART 1: Load All Results
# ===================================================================

print("\n" + "="*80)
print("Loading Results from All Methods")
print("="*80)

methods = {}

# Method 1: Intensity-based (MI)
method1_path = os.path.join(RESULTS_DIR, "method1_mi_results.csv")
print(f"Method 1 path: {method1_path}")
if os.path.exists(method1_path):
    df1 = pd.read_csv(method1_path)
    methods['Method 1 (MI)'] = df1
    print(f"✓ Method 1 (MI): {len(df1)} samples")
    print(f"  Columns: {list(df1.columns)}")
else:
    print(f"⚠ Method 1 not found")

# Method 2: ORB
method2_orb_path = os.path.join(RESULTS_DIR, "method2_orb_results.csv")
if os.path.exists(method2_orb_path):
    df2_orb = pd.read_csv(method2_orb_path)
    methods['Method 2 (ORB)'] = df2_orb
    print(f"✓ Method 2 (ORB): {len(df2_orb)} samples")
    print(f"  Columns: {list(df2_orb.columns)}")
else:
    print(f"⚠ Method 2 ORB not found")

# Method 2: SIFT
method2_sift_path = os.path.join(RESULTS_DIR, "method2_sift_results.csv")
if os.path.exists(method2_sift_path):
    df2_sift = pd.read_csv(method2_sift_path)
    methods['Method 2 (SIFT)'] = df2_sift
    print(f"✓ Method 2 (SIFT): {len(df2_sift)} samples")
    print(f"  Columns: {list(df2_sift.columns)}")
else:
    print(f"⚠ Method 2 SIFT not found")

# Method 5: ORB Enhanced
method5_path = os.path.join(RESULTS_DIR, "method5_orb_only_results.csv")
if os.path.exists(method5_path):
    df5 = pd.read_csv(method5_path)
    methods['Method 5 (ORB Enhanced)'] = df5
    print(f"✓ Method 5 (ORB Enhanced): {len(df5)} samples")
    print(f"  Columns: {list(df5.columns)}")
else:
    print(f"⚠ Method 5 not found")

if len(methods) == 0:
    print("\n❌ No results found!")
    sys.exit(1)

print(f"\n✓ Loaded {len(methods)} methods for comparison")

# ===================================================================
# PART 2: Unified Comparison Analysis
# ===================================================================

print("\n" + "="*80)
print("Comparative Analysis")
print("="*80)

comparison_data = []

for method_name, df in methods.items():
    n_samples = len(df)

    # Standardize column names - different methods use different formats
    # Method 1 uses 'correct', others use 'slice_error'
    if 'slice_error' in df.columns:
        exact_acc = (df['slice_error'] == 0).sum() / n_samples * 100
        within_4 = (df['slice_error'] <= 4).sum() / n_samples * 100
        mean_error = df['slice_error'].mean()
        median_error = df['slice_error'].median()
    elif 'correct' in df.columns:
        # Method 1 format
        within_4 = (df['correct'].sum() / n_samples) * 100
        # Try to compute exact accuracy from predictions
        if 'gt_slice_num' in df.columns and 'pred_slice_num' in df.columns:
            slice_error = abs(df['gt_slice_num'] - df['pred_slice_num'])
            exact_acc = (slice_error == 0).sum() / n_samples * 100
            mean_error = slice_error.mean()
            median_error = slice_error.median()
        else:
            exact_acc = 0
            mean_error = 0
            median_error = 0
    else:
        exact_acc = within_4 = mean_error = median_error = 0

    # Time
    avg_time = df['time_sec'].mean() if 'time_sec' in df.columns else 0
    total_time = df['time_sec'].sum() if 'time_sec' in df.columns else 0

    # Quality metrics (NMI, SSIM)
    # Some methods compute these, others don't
    if 'nmi' in df.columns:
        success_df = df[df['success'] == True] if 'success' in df.columns else df
        avg_nmi = success_df['nmi'].mean() if len(success_df) > 0 else 0
        avg_ssim = success_df['ssim'].mean() if 'ssim' in success_df.columns and len(success_df) > 0 else 0
    else:
        avg_nmi = avg_ssim = 0

    comparison_data.append({
        'Method': method_name,
        'Samples': n_samples,
        'Exact Match (%)': f"{exact_acc:.1f}",
        'Within ±4 (%)': f"{within_4:.1f}",
        'Mean Error (slices)': f"{mean_error:.2f}",
        'Median Error (slices)': f"{median_error:.1f}",
        'Avg NMI': f"{avg_nmi:.3f}",
        'Avg SSIM': f"{avg_ssim:.3f}",
        'Avg Time (s)': f"{avg_time:.2f}",
        'Total Time (min)': f"{total_time/60:.1f}",
        # Store numeric values for ranking
        '_exact_acc': exact_acc,
        '_within4': within_4,
        '_time': avg_time,
        '_nmi': avg_nmi
    })

comparison_df = pd.DataFrame(comparison_data)

# Print table
print("\n📊 METHOD COMPARISON TABLE")
print("="*80)
display_cols = ['Method', 'Exact Match (%)', 'Within ±4 (%)', 'Mean Error (slices)',
                'Avg NMI', 'Avg SSIM', 'Avg Time (s)', 'Total Time (min)']
print(comparison_df[display_cols].to_string(index=False))

# Save comparison
comparison_csv = os.path.join(COMPARISON_DIR, "all_methods_comparison.csv")
comparison_df.to_csv(comparison_csv, index=False)
print(f"\n✓ Saved: {comparison_csv}")

# ===================================================================
# PART 3: Ranking and Best Method
# ===================================================================

print("\n" + "="*80)
print("Method Rankings")
print("="*80)

print("\n🏆 ACCURACY RANKING (Exact Match):")
sorted_by_acc = comparison_df.sort_values('_exact_acc', ascending=False)
for i, row in enumerate(sorted_by_acc.iterrows(), 1):
    idx, r = row
    print(f"  {i}. {r['Method']}: {r['Exact Match (%)']}%")

print("\n⚡ SPEED RANKING (Fastest to Slowest):")
sorted_by_speed = comparison_df.sort_values('_time', ascending=True)
for i, row in enumerate(sorted_by_speed.iterrows(), 1):
    idx, r = row
    print(f"  {i}. {r['Method']}: {r['Avg Time (s)']}s")

print("\n🎯 QUALITY RANKING (NMI):")
sorted_by_nmi = comparison_df.sort_values('_nmi', ascending=False)
for i, row in enumerate(sorted_by_nmi.iterrows(), 1):
    idx, r = row
    print(f"  {i}. {r['Method']}: {r['Avg NMI']}")

# Determine overall winner
print("\n" + "="*80)
print("🏅 OVERALL ASSESSMENT")
print("="*80)

# Score each method (accuracy=50%, speed=30%, quality=20%)
scores = []
for idx, row in comparison_df.iterrows():
    acc_score = (row['_exact_acc'] / 100) * 50
    speed_score = (1 / (row['_time'] + 0.1)) * 30 / 10  # Normalized
    quality_score = (row['_nmi'] / 2.0) * 20  # NMI typically 0-2
    total = acc_score + speed_score + quality_score
    scores.append({
        'Method': row['Method'],
        'Accuracy Score': acc_score,
        'Speed Score': speed_score,
        'Quality Score': quality_score,
        'Total Score': total
    })

scores_df = pd.DataFrame(scores).sort_values('Total Score', ascending=False)
print(scores_df.to_string(index=False))

winner = scores_df.iloc[0]
print(f"\n🏆 WINNER: {winner['Method']} (Score: {winner['Total Score']:.2f})")

# ===================================================================
# PART 4: Visual Comparison Plots
# ===================================================================

print("\n" + "="*80)
print("Generating Comparison Plots")
print("="*80)

# Plot 1: Accuracy Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

method_names = comparison_df['Method'].tolist()
exact_accs = comparison_df['_exact_acc'].tolist()
within4_accs = comparison_df['_within4'].tolist()
avg_times = comparison_df['_time'].tolist()

# Exact accuracy
axes[0].barh(method_names, exact_accs, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Exact Match Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Exact Match Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlim([0, 100])
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(exact_accs):
    axes[0].text(v + 2, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')

# Within ±4 accuracy
axes[1].barh(method_names, within4_accs, color='coral', edgecolor='black')
axes[1].set_xlabel('Accuracy Within ±4 Slices (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Accuracy (Within ±4 Slices)', fontsize=13, fontweight='bold')
axes[1].set_xlim([0, 100])
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(within4_accs):
    axes[1].text(v + 2, i, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')

# Runtime
axes[2].barh(method_names, avg_times, color='mediumseagreen', edgecolor='black')
axes[2].set_xlabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
axes[2].set_title('Runtime Comparison', fontsize=13, fontweight='bold')
axes[2].grid(axis='x', alpha=0.3)
for i, v in enumerate(avg_times):
    axes[2].text(v + 0.1, i, f'{v:.2f}s', va='center', fontsize=10, fontweight='bold')

plt.suptitle('Method Comparison: Accuracy and Speed', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(COMPARISON_DIR, "all_methods_comparison.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

# Plot 2: Error Distribution Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (method_name, df) in enumerate(methods.items()):
    if idx >= 4:
        break

    ax = axes[idx]

    # Get error data
    if 'slice_error' in df.columns:
        errors = df['slice_error']
    elif 'gt_slice_num' in df.columns and 'pred_slice_num' in df.columns:
        errors = abs(df['gt_slice_num'] - df['pred_slice_num'])
    else:
        continue

    # Histogram
    bins = range(0, int(errors.max()) + 2)
    ax.hist(errors, bins=bins, edgecolor='black', color='skyblue', alpha=0.7)
    ax.axvline(x=4, color='red', linestyle='--', linewidth=2, label='±4 threshold')
    ax.set_xlabel('Z-Level Error (slices)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{method_name}\nMean Error: {errors.mean():.2f}, Median: {errors.median():.1f}',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

plt.suptitle('Error Distribution Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = os.path.join(COMPARISON_DIR, "error_distributions.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*80)
print("✅ Comparison analysis complete!")
print("="*80)
print(f"\nFiles saved to: {COMPARISON_DIR}")
