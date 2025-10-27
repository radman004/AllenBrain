"""
Results Visualization and Comparison Script
Author: Team
Date: 2025

Compares all methods and generates figures for A2 report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configuration
CURR_DIR = Path.cwd()
BASE_DIR = os.path.join(CURR_DIR, "project_data")
RESULTS_DIR = os.path.join(BASE_DIR, "test_synthetic", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

print("="*60)
print("Method Comparison & Visualization")
print("="*60)

# Load results from all methods
methods = {}

# Method 1: Intensity-based (MI)
method1_path = os.path.join(RESULTS_DIR, "method1_mi_results.csv")
if os.path.exists(method1_path):
    methods['Method 1 (MI)'] = pd.read_csv(method1_path)
    print(f"✅ Loaded Method 1 results ({len(methods['Method 1 (MI)'])} samples)")
else:
    print(f"⚠️  Method 1 results not found")

# Method 2: Feature-based (ORB)
method2_orb_path = os.path.join(RESULTS_DIR, "method2_orb_results.csv")
if os.path.exists(method2_orb_path):
    methods['Method 2 (ORB)'] = pd.read_csv(method2_orb_path)
    print(f"✅ Loaded Method 2 ORB results ({len(methods['Method 2 (ORB)'])} samples)")
else:
    print(f"⚠️  Method 2 ORB results not found")

# Method 2: Feature-based (SIFT)
method2_sift_path = os.path.join(RESULTS_DIR, "method2_sift_results.csv")
if os.path.exists(method2_sift_path):
    methods['Method 2 (SIFT)'] = pd.read_csv(method2_sift_path)
    print(f"✅ Loaded Method 2 SIFT results ({len(methods['Method 2 (SIFT)'])} samples)")
else:
    print(f"⚠️  Method 2 SIFT results not found")

# Method 3: Edge-based
method3_path = os.path.join(RESULTS_DIR, "method3_edge_results.csv")
if os.path.exists(method3_path):
    methods['Method 3 (Edge)'] = pd.read_csv(method3_path)
    print(f"✅ Loaded Method 3 results ({len(methods['Method 3 (Edge)'])} samples)")
else:
    print(f"⚠️  Method 3 results not found")

# Method 4: FFT Phase Correlation
method4_path = os.path.join(RESULTS_DIR, "method4_fft_results.csv")
if os.path.exists(method4_path):
    methods['Method 4 (FFT)'] = pd.read_csv(method4_path)
    print(f"✅ Loaded Method 4 results ({len(methods['Method 4 (FFT)'])} samples)")
else:
    print(f"⚠️  Method 4 results not found")

# Method 5: Hybrid (try both filenames)
method5_path = os.path.join(RESULTS_DIR, "method5_hybrid_results.csv")
method5_orb_path = os.path.join(RESULTS_DIR, "method5_orb_only_results.csv")
if os.path.exists(method5_path):
    methods['Method 5 (Hybrid)'] = pd.read_csv(method5_path)
    print(f"✅ Loaded Method 5 results ({len(methods['Method 5 (Hybrid)'])} samples)")
elif os.path.exists(method5_orb_path):
    methods['Method 5 (ORB Enhanced)'] = pd.read_csv(method5_orb_path)
    print(f"✅ Loaded Method 5 ORB-only results ({len(methods['Method 5 (ORB Enhanced)'])} samples)")
else:
    print(f"⚠️  Method 5 results not found")

if len(methods) == 0:
    print("\n❌ No results found! Please run batch evaluations first.")
    exit(1)

print(f"\nTotal methods loaded: {len(methods)}")

# ===================================================================
# Figure 1: Method Comparison Summary Table
# ===================================================================

print("\n" + "="*60)
print("Generating Summary Table...")
print("="*60)

summary_data = []
for method_name, df in methods.items():
    # Calculate metrics
    n_samples = len(df)
    exact_acc = (df['slice_error'] == 0).sum() / n_samples * 100 if 'slice_error' in df.columns else 0
    within_4 = (df['slice_error'] <= 4).sum() / n_samples * 100 if 'slice_error' in df.columns else \
               (df['correct'].sum() / n_samples * 100 if 'correct' in df.columns else 0)

    avg_time = df['time_sec'].mean() if 'time_sec' in df.columns else 0

    # NMI/SSIM from successful registrations
    success_df = df[df['success'] == True] if 'success' in df.columns else df
    avg_nmi = success_df['nmi'].mean() if 'nmi' in success_df.columns and len(success_df) > 0 else 0
    avg_ssim = success_df['ssim'].mean() if 'ssim' in success_df.columns and len(success_df) > 0 else 0

    summary_data.append({
        'Method': method_name,
        'Exact Match (%)': f"{exact_acc:.1f}",
        'Within ±4 (%)': f"{within_4:.1f}",
        'Avg NMI': f"{avg_nmi:.3f}",
        'Avg SSIM': f"{avg_ssim:.3f}",
        'Avg Time (s)': f"{avg_time:.2f}",
        'Samples': n_samples
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_path = os.path.join(RESULTS_DIR, "method_comparison_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n✅ Summary table saved to: {summary_path}")

# ===================================================================
# Figure 2: Accuracy Comparison Bar Chart
# ===================================================================

print("\n" + "="*60)
print("Generating Accuracy Comparison...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Exact match accuracy
method_names = [s['Method'] for s in summary_data]
exact_accs = [float(s['Exact Match (%)']) for s in summary_data]
within4_accs = [float(s['Within ±4 (%)']) for s in summary_data]

axes[0].bar(range(len(method_names)), exact_accs, color='steelblue', edgecolor='black')
axes[0].set_xticks(range(len(method_names)))
axes[0].set_xticklabels(method_names, rotation=45, ha='right')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Exact Match Accuracy')
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 100])

# Add value labels
for i, v in enumerate(exact_accs):
    axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

# Plot 2: Within ±4 slices accuracy
axes[1].bar(range(len(method_names)), within4_accs, color='coral', edgecolor='black')
axes[1].set_xticks(range(len(method_names)))
axes[1].set_xticklabels(method_names, rotation=45, ha='right')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy (Within ±4 Slices)')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0, 100])

# Add value labels
for i, v in enumerate(within4_accs):
    axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
accuracy_path = os.path.join(FIGURES_DIR, "accuracy_comparison.png")
plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
print(f"✅ Accuracy comparison saved to: {accuracy_path}")
plt.close()

# ===================================================================
# Figure 3: Runtime Comparison
# ===================================================================

print("\n" + "="*60)
print("Generating Runtime Comparison...")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 6))

avg_times = [float(s['Avg Time (s)']) for s in summary_data]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = ax.bar(range(len(method_names)), avg_times, color=colors[:len(method_names)], edgecolor='black')

ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=45, ha='right')
ax.set_ylabel('Average Runtime (seconds)')
ax.set_title('Method Runtime Comparison')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(avg_times):
    ax.text(i, v + max(avg_times)*0.02, f'{v:.2f}s', ha='center', fontsize=10)

plt.tight_layout()
runtime_path = os.path.join(FIGURES_DIR, "runtime_comparison.png")
plt.savefig(runtime_path, dpi=300, bbox_inches='tight')
print(f"✅ Runtime comparison saved to: {runtime_path}")
plt.close()

# ===================================================================
# Figure 4: Accuracy vs Runtime Trade-off
# ===================================================================

print("\n" + "="*60)
print("Generating Accuracy vs Runtime Trade-off...")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

for i, (method_name, exact_acc, runtime) in enumerate(zip(method_names, within4_accs, avg_times)):
    ax.scatter(runtime, exact_acc, s=200, alpha=0.7, color=colors[i], edgecolor='black', linewidth=2, label=method_name)
    ax.annotate(method_name, (runtime, exact_acc), xytext=(10, 10), textcoords='offset points', fontsize=10)

ax.set_xlabel('Average Runtime (seconds)', fontsize=12)
ax.set_ylabel('Accuracy Within ±4 Slices (%)', fontsize=12)
ax.set_title('Accuracy vs Runtime Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# Add diagonal lines showing "efficiency" zones
ax.axhline(y=70, color='red', linestyle='--', alpha=0.3, label='70% threshold')
ax.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='90% threshold')

ax.legend(loc='lower right')

plt.tight_layout()
tradeoff_path = os.path.join(FIGURES_DIR, "accuracy_runtime_tradeoff.png")
plt.savefig(tradeoff_path, dpi=300, bbox_inches='tight')
print(f"✅ Trade-off plot saved to: {tradeoff_path}")
plt.close()

# ===================================================================
# Figure 5: Quality Metrics Comparison (NMI & SSIM)
# ===================================================================

print("\n" + "="*60)
print("Generating Quality Metrics Comparison...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

avg_nmis = [float(s['Avg NMI']) for s in summary_data]
avg_ssims = [float(s['Avg SSIM']) for s in summary_data]

# NMI comparison
axes[0].bar(range(len(method_names)), avg_nmis, color='mediumseagreen', edgecolor='black')
axes[0].set_xticks(range(len(method_names)))
axes[0].set_xticklabels(method_names, rotation=45, ha='right')
axes[0].set_ylabel('NMI')
axes[0].set_title('Normalized Mutual Information')
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(avg_nmis):
    axes[0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)

# SSIM comparison
axes[1].bar(range(len(method_names)), avg_ssims, color='orange', edgecolor='black')
axes[1].set_xticks(range(len(method_names)))
axes[1].set_xticklabels(method_names, rotation=45, ha='right')
axes[1].set_ylabel('SSIM')
axes[1].set_title('Structural Similarity Index')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0, 1])

for i, v in enumerate(avg_ssims):
    axes[1].text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
quality_path = os.path.join(FIGURES_DIR, "quality_metrics_comparison.png")
plt.savefig(quality_path, dpi=300, bbox_inches='tight')
print(f"✅ Quality metrics comparison saved to: {quality_path}")
plt.close()

# ===================================================================
# Figure 6: Error Distribution (for one method - example with Method 2 ORB)
# ===================================================================

if 'Method 2 (ORB)' in methods:
    print("\n" + "="*60)
    print("Generating Error Distribution (Method 2 ORB)...")
    print("="*60)

    df_orb = methods['Method 2 (ORB)']

    if 'slice_error' in df_orb.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        errors = df_orb['slice_error']
        ax.hist(errors, bins=range(0, int(errors.max()) + 2), edgecolor='black', color='skyblue', alpha=0.7)

        ax.axvline(x=4, color='red', linestyle='--', linewidth=2, label='±4 slice threshold')

        ax.set_xlabel('Z-Level Error (slices)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Method 2 (ORB): Z-Level Error Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

        plt.tight_layout()
        error_dist_path = os.path.join(FIGURES_DIR, "method2_error_distribution.png")
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        print(f"✅ Error distribution saved to: {error_dist_path}")
        plt.close()

# ===================================================================
# Summary
# ===================================================================

print("\n" + "="*60)
print("Visualization Complete!")
print("="*60)
print(f"\nGenerated figures:")
print(f"  1. {accuracy_path}")
print(f"  2. {runtime_path}")
print(f"  3. {tradeoff_path}")
print(f"  4. {quality_path}")
if 'Method 2 (ORB)' in methods and 'slice_error' in methods['Method 2 (ORB)'].columns:
    print(f"  5. {error_dist_path}")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print(f"Summary table: {summary_path}")
print("\n✅ Ready for A2 report!")
