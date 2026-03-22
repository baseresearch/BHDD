"""
BHDD (Burmese Handwritten Digit Dataset) — Deep Dataset Exploration
====================================================================
Generates publication-quality figures and statistics for the dataset paper.
All outputs go to paper/figures/ and paper/*.csv.
"""

import os
import sys
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
np.random.seed(42)
warnings.filterwarnings('ignore', category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data.pkl')
FIG_DIR = os.path.join(ROOT, 'paper', 'figures')
PAPER_DIR = os.path.join(ROOT, 'paper')
os.makedirs(FIG_DIR, exist_ok=True)

BURMESE_DIGITS = ['၀', '၁', '၂', '၃', '၄', '၅', '၆', '၇', '၈', '၉']
NUM_CLASSES = 10

# Publication color palette — colorblind-friendly (adapted from Tableau 10)
COLORS = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
    '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
]

DPI = 300

# ── Find a font that can render Myanmar/Burmese glyphs ────────────────────────
MYANMAR_FONT = None
_candidates = [
    os.path.expanduser('~/Library/Fonts/Padauk-Regular.ttf'),
    '/System/Library/Fonts/NotoSansMyanmar.ttc',
    '/System/Library/Fonts/NotoSerifMyanmar.ttc',
    '/System/Library/Fonts/Supplemental/Myanmar MN.ttc',
    '/System/Library/Fonts/Supplemental/Myanmar Sangam MN.ttc',
]
for _p in _candidates:
    if os.path.exists(_p):
        MYANMAR_FONT = fm.FontProperties(fname=_p)
        print(f"[INFO] Using Myanmar font: {_p}")
        break

if MYANMAR_FONT is None:
    print("[WARN] No Myanmar font found — Burmese labels will render as boxes.")
    MYANMAR_FONT = fm.FontProperties()

# ── Matplotlib global styling ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND VERIFY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. LOADING AND VERIFYING DATA STRUCTURE")
print("=" * 70)

with open(DATA_PATH, 'rb') as f:
    dataset = pickle.load(f)

train_data = dataset['trainDataset']
test_data = dataset['testDataset']

print(f"  Top-level keys : {list(dataset.keys())}")
print(f"  Train samples  : {len(train_data):,}")
print(f"  Test samples   : {len(test_data):,}")
print(f"  Total          : {len(train_data) + len(test_data):,}")

sample = train_data[0]
print(f"  Sample keys    : {list(sample.keys())}")
print(f"  Image shape    : {sample['image'].shape}")
print(f"  Image dtype    : {sample['image'].dtype}")
print(f"  Label dtype    : {type(sample['label']).__name__} ({sample['label'].dtype if hasattr(sample['label'], 'dtype') else 'N/A'})")

# Collect all into arrays for efficient computation
train_images = np.array([d['image'] for d in train_data], dtype=np.float32)
train_labels = np.array([d['label'] for d in train_data], dtype=np.int32)
test_images = np.array([d['image'] for d in test_data], dtype=np.float32)
test_labels = np.array([d['label'] for d in test_data], dtype=np.int32)

print(f"  Train images   : shape={train_images.shape}, dtype={train_images.dtype}")
print(f"  Train labels   : shape={train_labels.shape}, range=[{train_labels.min()}, {train_labels.max()}]")
print(f"  Test images    : shape={test_images.shape}, dtype={test_images.dtype}")
print(f"  Test labels    : shape={test_labels.shape}, range=[{test_labels.min()}, {test_labels.max()}]")
print(f"  Pixel range    : [{train_images.min():.0f}, {train_images.max():.0f}]")

# Verify expected sizes
assert len(train_data) == 60000, f"Expected 60000 train, got {len(train_data)}"
assert len(test_data) == 27561, f"Expected 27561 test, got {len(test_data)}"
print("  [OK] Train/test sizes match expected (60,000 / 27,561)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PER-CLASS SAMPLE COUNTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. PER-CLASS SAMPLE COUNTS")
print("=" * 70)

train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
test_counts = np.bincount(test_labels, minlength=NUM_CLASSES)

header = f"{'Class':>6} {'Burmese':>8} {'Train':>8} {'Test':>8} {'Total':>8} {'Train%':>8} {'Test%':>8}"
print(header)
print("-" * len(header))
for c in range(NUM_CLASSES):
    total = train_counts[c] + test_counts[c]
    tr_pct = 100 * train_counts[c] / len(train_data)
    te_pct = 100 * test_counts[c] / len(test_data)
    print(f"{c:>6} {BURMESE_DIGITS[c]:>8} {train_counts[c]:>8,} {test_counts[c]:>8,} {total:>8,} {tr_pct:>7.2f}% {te_pct:>7.2f}%")
print("-" * len(header))
print(f"{'Total':>15} {train_counts.sum():>8,} {test_counts.sum():>8,} {train_counts.sum()+test_counts.sum():>8,}")

# Save CSV
csv_path = os.path.join(PAPER_DIR, 'class_distribution.csv')
with open(csv_path, 'w') as f:
    f.write("class,burmese_digit,train_count,test_count,total,train_pct,test_pct\n")
    for c in range(NUM_CLASSES):
        total = train_counts[c] + test_counts[c]
        tr_pct = 100 * train_counts[c] / len(train_data)
        te_pct = 100 * test_counts[c] / len(test_data)
        f.write(f"{c},{BURMESE_DIGITS[c]},{train_counts[c]},{test_counts[c]},{total},{tr_pct:.4f},{te_pct:.4f}\n")
print(f"  [SAVED] {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASS DISTRIBUTION BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. CLASS DISTRIBUTION BAR CHART")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(NUM_CLASSES)
width = 0.35

bars_train = ax.bar(x - width/2, train_counts, width, label='Train (60,000)',
                    color=COLORS[0], edgecolor='white', linewidth=0.5)
bars_test = ax.bar(x + width/2, test_counts, width, label='Test (27,561)',
                   color=COLORS[1], edgecolor='white', linewidth=0.5)

ax.set_xlabel('Digit Class', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('BHDD Class Distribution: Train vs Test Split', fontsize=14, fontweight='bold')
ax.set_xticks(x)
# Two-line labels: Burmese digit on top, Arabic numeral below
xlabels = [f'{BURMESE_DIGITS[i]}\n({i})' for i in range(NUM_CLASSES)]
ax.set_xticklabels(xlabels, fontproperties=MYANMAR_FONT, fontsize=11)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')

# Add value labels on bars
for bar in bars_train:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 50, f'{int(h):,}',
            ha='center', va='bottom', fontsize=7, color='#333333')
for bar in bars_test:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 50, f'{int(h):,}',
            ha='center', va='bottom', fontsize=7, color='#333333')

ax.set_ylim(0, max(train_counts.max(), test_counts.max()) * 1.15)
fig.tight_layout()
path = os.path.join(FIG_DIR, 'class_distribution.png')
fig.savefig(path)
plt.close(fig)
print(f"  [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAMPLE VISUALIZATION GRID (10 rows x 5 cols)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. SAMPLE VISUALIZATION GRID")
print("=" * 70)

n_cols = 5
fig, axes = plt.subplots(NUM_CLASSES, n_cols, figsize=(6, 12))
fig.suptitle('Sample Images per Digit Class (BHDD Training Set)',
             fontsize=13, fontweight='bold', y=0.995)

for cls in range(NUM_CLASSES):
    indices = np.where(train_labels == cls)[0]
    chosen = np.random.choice(indices, size=n_cols, replace=False)
    for j, idx in enumerate(chosen):
        ax = axes[cls, j]
        ax.imshow(train_images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.5)
    # Row label
    axes[cls, 0].set_ylabel(f'{BURMESE_DIGITS[cls]}  ({cls})',
                            fontproperties=MYANMAR_FONT,
                            fontsize=12, rotation=0, labelpad=35, va='center')

fig.tight_layout(rect=[0.06, 0, 1, 0.98], h_pad=0.3, w_pad=0.3)
path = os.path.join(FIG_DIR, 'sample_grid.png')
fig.savefig(path)
plt.close(fig)
print(f"  [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PIXEL INTENSITY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. PIXEL INTENSITY STATISTICS (per class, training set)")
print("=" * 70)

header = (f"{'Class':>6} {'Burmese':>8} {'MeanPx':>8} {'StdPx':>8} "
          f"{'MinPx':>6} {'MaxPx':>6} {'%NonZero':>9}")
print(header)
print("-" * len(header))

pixel_stats = []
for c in range(NUM_CLASSES):
    mask = train_labels == c
    imgs = train_images[mask]
    mean_px = imgs.mean()
    std_px = imgs.std()
    min_px = imgs.min()
    max_px = imgs.max()
    nonzero_pct = 100.0 * (imgs > 0).sum() / imgs.size
    pixel_stats.append({
        'class': c,
        'burmese': BURMESE_DIGITS[c],
        'mean': mean_px,
        'std': std_px,
        'min': min_px,
        'max': max_px,
        'nonzero_pct': nonzero_pct,
        'n_samples': int(mask.sum()),
    })
    print(f"{c:>6} {BURMESE_DIGITS[c]:>8} {mean_px:>8.2f} {std_px:>8.2f} "
          f"{min_px:>6.0f} {max_px:>6.0f} {nonzero_pct:>8.2f}%")

# Overall
print("-" * len(header))
mean_all = train_images.mean()
std_all = train_images.std()
nz_all = 100.0 * (train_images > 0).sum() / train_images.size
print(f"{'ALL':>15} {mean_all:>8.2f} {std_all:>8.2f} "
      f"{train_images.min():>6.0f} {train_images.max():>6.0f} {nz_all:>8.2f}%")

csv_path = os.path.join(PAPER_DIR, 'pixel_stats.csv')
with open(csv_path, 'w') as f:
    f.write("class,burmese_digit,n_samples,mean_pixel,std_pixel,min_pixel,max_pixel,nonzero_pct\n")
    for s in pixel_stats:
        f.write(f"{s['class']},{s['burmese']},{s['n_samples']},"
                f"{s['mean']:.4f},{s['std']:.4f},{s['min']:.0f},{s['max']:.0f},{s['nonzero_pct']:.4f}\n")
print(f"  [SAVED] {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIXEL INTENSITY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. PIXEL INTENSITY DISTRIBUTION (per class)")
print("=" * 70)

from scipy.ndimage import gaussian_filter1d

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: overall distribution — all pixels (including background)
# Show as stepped histograms; useful for understanding the bimodal nature
bins_all = np.arange(0, 257, 2)
for c in range(NUM_CLASSES):
    mask = train_labels == c
    pixels = train_images[mask].ravel()
    ax1.hist(pixels, bins=bins_all, density=True, alpha=0.7, color=COLORS[c],
             label=f'{BURMESE_DIGITS[c]} ({c})', histtype='step', linewidth=1.3)

ax1.set_xlabel('Pixel Intensity', fontsize=11)
ax1.set_ylabel('Density (log scale)', fontsize=11)
ax1.set_title('Full Pixel Intensity Distribution', fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.set_xlim(-5, 260)
ax1.legend(ncol=2, loc='upper right', frameon=True, framealpha=0.9,
           edgecolor='#cccccc', prop=MYANMAR_FONT, fontsize=8)

# Right panel: foreground-only (pixels > 20 to exclude near-black noise),
# smoothed KDE-like curves — shows the "ink intensity" profile per class
THRESHOLD = 20
fine_bins = np.linspace(THRESHOLD, 255, 200)
bin_centers = (fine_bins[:-1] + fine_bins[1:]) / 2
for c in range(NUM_CLASSES):
    mask = train_labels == c
    pixels = train_images[mask].ravel()
    pixels = pixels[pixels > THRESHOLD]
    counts, _ = np.histogram(pixels, bins=fine_bins, density=True)
    smoothed = gaussian_filter1d(counts, sigma=5)
    ax2.plot(bin_centers, smoothed, color=COLORS[c], linewidth=1.8, alpha=0.85,
             label=f'{BURMESE_DIGITS[c]} ({c})')

ax2.set_xlabel(f'Pixel Intensity (foreground, >{THRESHOLD})', fontsize=11)
ax2.set_ylabel('Density (smoothed)', fontsize=11)
ax2.set_title('Ink Intensity Profiles per Digit Class', fontsize=12, fontweight='bold')
ax2.set_xlim(THRESHOLD - 5, 260)
ax2.legend(ncol=2, loc='upper right', frameon=True, framealpha=0.9,
           edgecolor='#cccccc', prop=MYANMAR_FONT, fontsize=8)

fig.tight_layout()
path = os.path.join(FIG_DIR, 'pixel_intensity_distribution.png')
fig.savefig(path)
plt.close(fig)
print(f"  [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MORPHOLOGICAL DIVERSITY VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. MORPHOLOGICAL DIVERSITY VISUALIZATION")
print("=" * 70)

# Select 4 classes that likely show high diversity
diversity_classes = [0, 3, 5, 8]
n_samples_div = 14  # samples per class

fig, axes = plt.subplots(len(diversity_classes), n_samples_div,
                         figsize=(14, 4.5))
fig.suptitle('Intra-class Morphological Diversity in BHDD',
             fontsize=13, fontweight='bold', y=1.02)

for row, cls in enumerate(diversity_classes):
    indices = np.where(train_labels == cls)[0]
    chosen = np.random.choice(indices, size=n_samples_div, replace=False)
    for col, idx in enumerate(chosen):
        ax = axes[row, col]
        ax.imshow(train_images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#dddddd')
            spine.set_linewidth(0.3)
    axes[row, 0].set_ylabel(f'{BURMESE_DIGITS[cls]}  ({cls})',
                            fontproperties=MYANMAR_FONT,
                            fontsize=12, rotation=0, labelpad=35, va='center')

fig.tight_layout(rect=[0.05, 0, 1, 0.97], h_pad=0.4, w_pad=0.2)
path = os.path.join(FIG_DIR, 'morphological_diversity.png')
fig.savefig(path)
plt.close(fig)
print(f"  [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. VISUALLY SIMILAR PAIRS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. VISUALLY SIMILAR / CONFUSABLE PAIRS")
print("=" * 70)

# Pairs driven by actual CNN confusion matrix results (all pairs with 3+ errors):
#   ၀ → ၁: 21 errors (closed circle vs open circle — top confusion by far)
#   ၃ → ၁:  7 errors (curved strokes misread as open circle)
#   ၀ → ၈:  5 errors (circular vs looped shapes)
#   ၁ → ၀:  3 errors (reverse of top pair — bidirectional confusion)
#   ၁ → ၈:  3 errors (open circle vs spiral)
#   ၅ → ၈:  3 errors (looped shapes)
#   ၈ → ၀:  3 errors (reverse circular confusion)
# We show the top 4 confusion directions grouped by theme:
confusable_pairs = [
    (0, 1, 'closed vs open circle\n(24 errors both ways)'),
    (3, 1, 'curved stroke vs open circle\n(7 errors)'),
    (0, 8, 'circle vs spiral\n(8 errors both ways)'),
    (5, 8, 'looped shape ambiguity\n(3 errors)'),
]

n_pair_samples = 5
n_pairs = len(confusable_pairs)
fig, axes = plt.subplots(n_pairs, 2 * n_pair_samples + 1,
                         figsize=(12, 5.6),
                         gridspec_kw={'width_ratios':
                                      [1]*n_pair_samples + [0.3] + [1]*n_pair_samples})
fig.suptitle('Visually Confusable Digit Pairs in Burmese Script',
             fontsize=13, fontweight='bold', y=1.02)

for row, (clsA, clsB, desc) in enumerate(confusable_pairs):
    idxA = np.where(train_labels == clsA)[0]
    idxB = np.where(train_labels == clsB)[0]
    chosenA = np.random.choice(idxA, size=n_pair_samples, replace=False)
    chosenB = np.random.choice(idxB, size=n_pair_samples, replace=False)

    # Class A samples (left side)
    for j, idx in enumerate(chosenA):
        ax = axes[row, j]
        ax.imshow(train_images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS[0])
            spine.set_linewidth(1.2)
        if row == 0:
            ax.set_title(f'{BURMESE_DIGITS[clsA]} ({clsA})' if j == n_pair_samples // 2 else '',
                         fontproperties=MYANMAR_FONT, fontsize=10, color=COLORS[0])

    # Separator column
    sep_ax = axes[row, n_pair_samples]
    sep_ax.set_xticks([])
    sep_ax.set_yticks([])
    sep_ax.set_facecolor('white')
    for spine in sep_ax.spines.values():
        spine.set_visible(False)
    sep_ax.text(0.5, 0.5, 'vs', ha='center', va='center',
                fontsize=10, fontweight='bold', color='#888888',
                transform=sep_ax.transAxes)

    # Class B samples (right side)
    for j, idx in enumerate(chosenB):
        col = n_pair_samples + 1 + j
        ax = axes[row, col]
        ax.imshow(train_images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS[1])
            spine.set_linewidth(1.2)
        if row == 0:
            ax.set_title(f'{BURMESE_DIGITS[clsB]} ({clsB})' if j == n_pair_samples // 2 else '',
                         fontproperties=MYANMAR_FONT, fontsize=10, color=COLORS[1])

    # Row label on the far left — use default font for description text
    axes[row, 0].set_ylabel(desc,
                            fontsize=8, rotation=0, labelpad=130, va='center',
                            linespacing=1.4)

fig.tight_layout(rect=[0.16, 0, 1, 0.96], h_pad=0.6, w_pad=0.3)
path = os.path.join(FIG_DIR, 'similar_pairs.png')
fig.savefig(path)
plt.close(fig)
print(f"  [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. MEAN DIGIT IMAGES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("9. MEAN DIGIT IMAGES")
print("=" * 70)

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
fig.suptitle('Mean Image per Digit Class (BHDD Training Set)',
             fontsize=13, fontweight='bold')

mean_images = []
for c in range(NUM_CLASSES):
    mask = train_labels == c
    mean_img = train_images[mask].mean(axis=0)
    mean_images.append(mean_img)

    row, col = divmod(c, 5)
    ax = axes[row, col]
    im = ax.imshow(mean_img, cmap='inferno', vmin=0, vmax=mean_img.max())
    ax.set_title(f'{BURMESE_DIGITS[c]}  ({c})', fontproperties=MYANMAR_FONT,
                 fontsize=12, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout(rect=[0, 0, 0.92, 0.93])
# Shared colorbar
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Mean Pixel Intensity')
path = os.path.join(FIG_DIR, 'mean_digits.png')
fig.savefig(path)
plt.close(fig)
print(f"  [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. PER-CLASS VARIANCE HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("10. PER-CLASS VARIANCE HEATMAPS")
print("=" * 70)

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
fig.suptitle('Pixel-wise Variance per Digit Class (BHDD Training Set)',
             fontsize=13, fontweight='bold')

var_images = []
global_vmax = 0
for c in range(NUM_CLASSES):
    mask = train_labels == c
    var_img = train_images[mask].var(axis=0)
    var_images.append(var_img)
    if var_img.max() > global_vmax:
        global_vmax = var_img.max()

for c in range(NUM_CLASSES):
    row, col = divmod(c, 5)
    ax = axes[row, col]
    im = ax.imshow(var_images[c], cmap='hot', vmin=0, vmax=global_vmax)
    ax.set_title(f'{BURMESE_DIGITS[c]}  ({c})', fontproperties=MYANMAR_FONT,
                 fontsize=12, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout(rect=[0, 0, 0.92, 0.93])
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Pixel Variance')
path = os.path.join(FIG_DIR, 'variance_heatmaps.png')
fig.savefig(path)
plt.close(fig)
print(f"  [SAVED] {path}")

# Print variance summary
print("\n  Per-class mean variance (higher = more writer variability):")
for c in range(NUM_CLASSES):
    print(f"    Class {c} ({BURMESE_DIGITS[c]}): mean_var={var_images[c].mean():.2f}, max_var={var_images[c].max():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPLORATION COMPLETE — Summary of outputs")
print("=" * 70)
print(f"  Figures directory : {FIG_DIR}")
for fname in sorted(os.listdir(FIG_DIR)):
    fpath = os.path.join(FIG_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"    {fname:40s}  ({size_kb:,.1f} KB)")
print(f"  CSV files:")
for fname in ['class_distribution.csv', 'pixel_stats.csv']:
    fpath = os.path.join(PAPER_DIR, fname)
    if os.path.exists(fpath):
        print(f"    {fname}")
print("  Done.")
