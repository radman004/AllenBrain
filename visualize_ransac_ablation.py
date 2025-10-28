"""
RANSAC Ablation Study Visualization
Shows the impact of RANSAC geometric filtering on Method 2 (ORB)
Dedicated figure for demonstrating RANSAC's contribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
SYNTHETIC_TEST_DIR = os.path.join(BASE_DIR, "test_synthetic")
RESULTS_DIR = os.path.join(SYNTHETIC_TEST_DIR, "results")
ABLATION_DIR = os.path.join(RESULTS_DIR, "ransac_ablation")

os.makedirs(ABLATION_DIR, exist_ok=True)

print("="*80)
print("RANSAC ABLATION STUDY VISUALIZATION")
print("="*80)

# Load ablation results
with_ransac_path = os.path.join(RESULTS_DIR, "method2_with_ransac_ablation.csv")
without_ransac_path = os.path.join(RESULTS_DIR, "method2_without_ransac_ablation.csv")

if not os.path.exists(with_ransac_path):
    print(f"\n❌ With RANSAC results not found: {with_ransac_path}")
    print("Run: python run_method2_ablation.py")
    exit(1)

if not os.path.exists(without_ransac_path):
    print(f"\n❌ Without RANSAC results not found: {without_ransac_path}")
    print("Run: python run_method2_ablation.py")
    exit(1)

df_with = pd.read_csv(with_ransac_path)
df_without = pd.read_csv(without_ransac_path)

print(f"\n✓ Loaded With RANSAC: {len(df_with)} samples")
print(f"✓ Loaded Without RANSAC: {len(df_without)} samples")

# Compute metrics
metrics = {
    'With RANSAC': {
        'exact': (df_with['slice_error'] == 0).sum() / len(df_with) * 100,
        'within_1': (df_with['slice_error'] <= 1).sum() / len(df_with) * 100,
        'within_2': (df_with['slice_error'] <= 2).sum() / len(df_with) * 100,
        'within_4': (df_with['slice_error'] <= 4).sum() / len(df_with) * 100,
        'mean_error': df_with['slice_error'].mean(),
        'inlier_ratio': df_with['inlier_ratio'].mean(),
        'nmi': df_with['nmi'].mean(),
        'time': df_with['time_sec'].mean(),
        'errors': df_with['slice_error']
    },
    'Without RANSAC': {
        'exact': (df_without['slice_error'] == 0).sum() / len(df_without) * 100,
        'within_1': (df_without['slice_error'] <= 1).sum() / len(df_without) * 100,
        'within_2': (df_without['slice_error'] <= 2).sum() / len(df_without) * 100,
        'within_4': (df_without['slice_error'] <= 4).sum() / len(df_without) * 100,
        'mean_error': df_without['slice_error'].mean(),
        'inlier_ratio': df_without['inlier_ratio'].mean(),
        'nmi': df_without['nmi'].mean(),
        'time': df_without['time_sec'].mean(),
        'errors': df_without['slice_error']
    }
}

# Print summary
print("\n" + "="*80)
print("ABLATION STUDY RESULTS")
print("="*80)

for variant, m in metrics.items():
    print(f"\n{variant}:")
    print(f"  Exact match:  {m['exact']:.1f}%")
    print(f"  Within ±1:    {m['within_1']:.1f}%")
    print(f"  Within ±2:    {m['within_2']:.1f}%")
    print(f"  Within ±4:    {m['within_4']:.1f}%")
    print(f"  Mean error:   {m['mean_error']:.2f} slices")
    print(f"  Inlier ratio: {m['inlier_ratio']:.1%}")
    print(f"  Avg NMI:      {m['nmi']:.3f}")
    print(f"  Avg time:     {m['time']:.2f}s")

improvement = metrics['With RANSAC']['within_4'] - metrics['Without RANSAC']['within_4']
print(f"\n🎯 RANSAC Improvement: +{improvement:.1f}% accuracy (within ±4)")
print(f"   Outlier rejection: {100 - metrics['With RANSAC']['inlier_ratio']*100:.1f}% of matches filtered")

# ===================================================================
# Create Comprehensive Ablation Figure
# ===================================================================

print("\n" + "="*80)
print("Generating RANSAC Ablation Figure")
print("="*80)

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# ===================================================================
# Plot 1: Accuracy Comparison (Bar Chart)
# ===================================================================
ax1 = fig.add_subplot(gs[0, 0])

categories = ['Exact\nMatch', 'Within\n±1', 'Within\n±2', 'Within\n±4']
with_vals = [metrics['With RANSAC']['exact'], metrics['With RANSAC']['within_1'],
             metrics['With RANSAC']['within_2'], metrics['With RANSAC']['within_4']]
without_vals = [metrics['Without RANSAC']['exact'], metrics['Without RANSAC']['within_1'],
                metrics['Without RANSAC']['within_2'], metrics['Without RANSAC']['within_4']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, with_vals, width, label='With RANSAC', color='steelblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, without_vals, width, label='Without RANSAC', color='coral', edgecolor='black')

ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10)
ax1.set_ylim([0, 105])
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ===================================================================
# Plot 2: Error Distribution Comparison
# ===================================================================
ax2 = fig.add_subplot(gs[0, 1])

bins = range(0, 15)
ax2.hist(metrics['With RANSAC']['errors'], bins=bins, alpha=0.6, label='With RANSAC',
         color='steelblue', edgecolor='black')
ax2.hist(metrics['Without RANSAC']['errors'], bins=bins, alpha=0.6, label='Without RANSAC',
         color='coral', edgecolor='black')
ax2.axvline(x=4, color='red', linestyle='--', linewidth=2, label='±4 threshold')
ax2.set_xlabel('Z-Level Error (slices)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Error Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# ===================================================================
# Plot 3: Inlier Ratio Comparison
# ===================================================================
ax3 = fig.add_subplot(gs[0, 2])

variants = ['With\nRANSAC', 'Without\nRANSAC']
inlier_ratios = [metrics['With RANSAC']['inlier_ratio'] * 100,
                 metrics['Without RANSAC']['inlier_ratio'] * 100]
colors = ['steelblue', 'coral']

bars = ax3.bar(variants, inlier_ratios, color=colors, edgecolor='black', width=0.6)
ax3.set_ylabel('Inlier Ratio (%)', fontsize=12, fontweight='bold')
ax3.set_title('Match Quality (Inlier Ratio)', fontsize=13, fontweight='bold')
ax3.set_ylim([0, 110])
ax3.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# ===================================================================
# Plot 4: Quality Metrics (NMI, SSIM)
# ===================================================================
ax4 = fig.add_subplot(gs[1, 0])

quality_metrics = ['NMI', 'Mean Error\n(slices)']
with_quality = [metrics['With RANSAC']['nmi'], metrics['With RANSAC']['mean_error']]
without_quality = [metrics['Without RANSAC']['nmi'], metrics['Without RANSAC']['mean_error']]

x = np.arange(len(quality_metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, with_quality, width, label='With RANSAC', color='steelblue', edgecolor='black')
bars2 = ax4.bar(x + width/2, without_quality, width, label='Without RANSAC', color='coral', edgecolor='black')

ax4.set_ylabel('Value', fontsize=10, fontweight='bold')
ax4.set_title('Registration Quality Metrics', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(quality_metrics, fontsize=10)
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ===================================================================
# Plot 5: Summary Text Box
# ===================================================================
ax5 = fig.add_subplot(gs[1, 1:])
ax5.axis('off')

summary_text = f"""
RANSAC ABLATION STUDY SUMMARY

Test Set: {len(df_with)} images from Allen Brain Atlas synthetic dataset

═══════════════════════════════════════════════════════════════════════

With RANSAC (Standard Method 2):
  • Exact Match:       {metrics['With RANSAC']['exact']:.1f}%
  • Within ±4 slices:  {metrics['With RANSAC']['within_4']:.1f}%
  • Mean Error:        {metrics['With RANSAC']['mean_error']:.2f} slices
  • Inlier Ratio:      {metrics['With RANSAC']['inlier_ratio']:.1%} (outliers rejected!)
  • Avg NMI:           {metrics['With RANSAC']['nmi']:.3f}

Without RANSAC (Least Squares on All Matches):
  • Exact Match:       {metrics['Without RANSAC']['exact']:.1f}%
  • Within ±4 slices:  {metrics['Without RANSAC']['within_4']:.1f}%
  • Mean Error:        {metrics['Without RANSAC']['mean_error']:.2f} slices
  • Inlier Ratio:      100.0% (NO rejection - all matches used!)
  • Avg NMI:           {metrics['Without RANSAC']['nmi']:.3f}

═══════════════════════════════════════════════════════════════════════

RANSAC's Contribution:
  • Accuracy Improvement:  +{improvement:.1f}% (within ±4 slices)
  • Error Reduction:       -{(metrics['Without RANSAC']['mean_error'] - metrics['With RANSAC']['mean_error']):.2f} slices
  • Quality Improvement:   +{(metrics['With RANSAC']['nmi'] - metrics['Without RANSAC']['nmi']):.3f} NMI
  • Outlier Filtering:     {100 - metrics['With RANSAC']['inlier_ratio']*100:.1f}% of matches rejected

Conclusion: RANSAC geometric filtering is CRITICAL for robust registration. Without RANSAC, outlier matches
corrupt the transformation estimation, leading to reduced accuracy and alignment quality.
"""

ax5.text(0.05, 0.5, summary_text, fontsize=10, family='monospace', va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=2))

plt.suptitle('Method 2 (ORB) - RANSAC Ablation Study', fontsize=16, fontweight='bold', y=0.98)

# Save figure
fig_path = os.path.join(ABLATION_DIR, "ransac_ablation_study.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {fig_path}")
plt.close()

print("\n" + "="*80)
print("✅ RANSAC ablation visualization complete!")
print("="*80)
print(f"\nFigure saved to: {ABLATION_DIR}")
print("\nUse this figure in your report to demonstrate the importance of")
print("RANSAC geometric filtering for robust feature-based registration.")
