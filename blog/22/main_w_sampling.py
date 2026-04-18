"""
Improper Model Comparison Example
===================================
Demonstrates how training data selection affects model evaluation,
leading to misleading comparisons between classifiers.

Setup:
- 4 Gaussians in 2D with means (-1,1), (1,1), (1,-1), (-1,-1), variance=0.25
- Labels: 0, 1, 2, 3
- Three training sets with different compositions
- One shared test set (200 points, all 4 classes)
- Naive Bayes classifier on each training set

New (v2):
- sample_from_gnb(): extracts learned means/variances from a fitted GaussianNB
  and draws samples from the model's generative distribution.
- Figure 4: side-by-side comparison of what each model "thinks" the world
  looks like (model samples) vs. the true population ellipses.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ── Reproducibility ────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── Gaussian parameters ────────────────────────────────────────────────────────
MEANS    = [(-1,  1), ( 1,  1), ( 1, -1), (-1, -1)]
VARIANCE = 0.25
STD      = np.sqrt(VARIANCE)
LABELS   = [0, 1, 2, 3]
COLORS   = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]  # one per class
NAMES    = [f"Gaussian {i}" for i in LABELS]


def sample(class_idx: int, n: int) -> np.ndarray:
    """Return n samples from true Gaussian `class_idx`."""
    mu = MEANS[class_idx]
    return rng.normal(loc=mu, scale=STD, size=(n, 2))


# ── Generate datasets ──────────────────────────────────────────────────────────
# Training set 1 – only classes 0 and 1, 50 each
X_tr1 = np.vstack([sample(0, 50), sample(1, 50)])
y_tr1 = np.hstack([np.zeros(50, dtype=int), np.ones(50, dtype=int)])

# Training set 2 – all 4 classes, 10 each
X_tr2 = np.vstack([sample(i, 10) for i in LABELS])
y_tr2 = np.hstack([np.full(10, i, dtype=int) for i in LABELS])

# Training set 3 – all 4 classes, 100 each
X_tr3 = np.vstack([sample(i, 100) for i in LABELS])
y_tr3 = np.hstack([np.full(100, i, dtype=int) for i in LABELS])

# Test set – all 4 classes, 200 each (800 total)
X_test = np.vstack([sample(i, 200) for i in LABELS])
y_test = np.hstack([np.full(200, i, dtype=int) for i in LABELS])


# ── Train Naive Bayes models (with Laplace smoothing via var_smoothing) ────────
# var_smoothing adds a small fraction of the largest feature variance to all
# variances, acting as a form of Laplace / additive smoothing.
SMOOTHING = 1e-2

clf1 = GaussianNB(var_smoothing=SMOOTHING).fit(X_tr1, y_tr1)
clf2 = GaussianNB(var_smoothing=SMOOTHING).fit(X_tr2, y_tr2)
clf3 = GaussianNB(var_smoothing=SMOOTHING).fit(X_tr3, y_tr3)

acc1 = accuracy_score(y_test, clf1.predict(X_test))
acc2 = accuracy_score(y_test, clf2.predict(X_test))
acc3 = accuracy_score(y_test, clf3.predict(X_test))

print(f"Model 1 accuracy (train: 50+50 from classes 0&1): {acc1:.3f}")
print(f"Model 2 accuracy (train: 10 per class):           {acc2:.3f}")
print(f"Model 3 accuracy (train: 100 per class):          {acc3:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Sampling from a fitted GaussianNB
# ══════════════════════════════════════════════════════════════════════════════

def sample_from_gnb(clf: GaussianNB, n_per_class: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from the generative distribution learned by a fitted GaussianNB.

    GaussianNB models each class c and each feature f as an independent
    Gaussian N(mu_{c,f}, sigma^2_{c,f}).  The fitted parameters are stored in:

        clf.theta_   – shape (n_classes, n_features): per-class feature means
        clf.var_     – shape (n_classes, n_features): per-class feature variances
                       (already includes the var_smoothing epsilon)
        clf.classes_ – the class labels in the order they were fitted

    For each class we sample n_per_class points by drawing each feature
    independently from its learned Gaussian and stacking them as columns.
    This respects the Naive Bayes conditional-independence assumption.

    Parameters
    ----------
    clf          : fitted GaussianNB instance
    n_per_class  : number of samples to draw for each class the model knows
    rng          : numpy random Generator (for reproducibility)

    Returns
    -------
    X : float array of shape (n_classes * n_per_class, n_features)
    y : int   array of shape (n_classes * n_per_class,)
    """
    X_chunks, y_chunks = [], []

    for class_idx, label in enumerate(clf.classes_):
        means = clf.theta_[class_idx]          # (n_features,)
        stds  = np.sqrt(clf.var_[class_idx])   # (n_features,)  var already smoothed

        # Sample each feature independently (Naive Bayes assumption), then stack.
        features = [
            rng.normal(loc=means[f], scale=stds[f], size=n_per_class)
            for f in range(len(means))
        ]
        X_chunks.append(np.column_stack(features))
        y_chunks.append(np.full(n_per_class, int(label), dtype=int))

    return np.vstack(X_chunks), np.hstack(y_chunks)


# Draw 50 samples per class from each model
N_MODEL_SAMPLES = 50
X_s1, y_s1 = sample_from_gnb(clf1, N_MODEL_SAMPLES, rng)
X_s2, y_s2 = sample_from_gnb(clf2, N_MODEL_SAMPLES, rng)
X_s3, y_s3 = sample_from_gnb(clf3, N_MODEL_SAMPLES, rng)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def draw_ellipse(ax, mean, std, color, alpha=0.18, lw=2, linestyle="solid"):
    """Draw a filled 1-sigma confidence ellipse (isotropic: width == height)."""
    ax.add_patch(Ellipse(xy=mean, width=2*std, height=2*std,
                         edgecolor=color, facecolor=color,
                         linewidth=lw, alpha=alpha))
    ax.add_patch(Ellipse(xy=mean, width=2*std, height=2*std,
                         edgecolor=color, facecolor="none",
                         linewidth=lw, linestyle=linestyle))


def draw_gnb_ellipses(ax, clf: GaussianNB):
    """
    Draw axis-aligned 1-sigma ellipses from the GaussianNB's learned parameters.

    Because GaussianNB uses a diagonal covariance matrix (one variance per
    feature per class), each ellipse can have different width and height —
    unlike the true isotropic Gaussians.  The dashed border makes them easy
    to distinguish from the solid true-population ellipses.
    """
    for class_idx, label in enumerate(clf.classes_):
        mu   = clf.theta_[class_idx]           # (n_features,)
        stds = np.sqrt(clf.var_[class_idx])    # (n_features,)
        col  = COLORS[int(label)]

        # Filled region
        ax.add_patch(Ellipse(xy=(mu[0], mu[1]),
                             width=2*stds[0], height=2*stds[1],
                             edgecolor=col, facecolor=col,
                             linewidth=2, alpha=0.18))
        # Dashed border to signal "model belief"
        ax.add_patch(Ellipse(xy=(mu[0], mu[1]),
                             width=2*stds[0], height=2*stds[1],
                             edgecolor=col, facecolor="none",
                             linewidth=2, linestyle="--"))


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

AXIS_LIM = (-3.5, 3.5)


def style_ax(ax, title):
    ax.set_xlim(*AXIS_LIM)
    ax.set_ylim(*AXIS_LIM)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – The four Gaussians (population view)
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(6, 6))
fig1.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

for i, (mu, col) in enumerate(zip(MEANS, COLORS)):
    draw_ellipse(ax, mu, STD, col, alpha=0.22)
    ax.scatter(*mu, color=col, s=120, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.text(mu[0], mu[1] + STD + 0.12, NAMES[i],
            ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")

style_ax(ax, "Population: Four 2-D Gaussians")
ax.legend(handles=[mpatches.Patch(color=c, label=n) for c, n in zip(COLORS, NAMES)],
          loc="upper right", fontsize=8, framealpha=0.9)

fig1.tight_layout()
fig1.savefig("fig1_gaussians.png", dpi=150, bbox_inches="tight")
print("Saved fig1_gaussians.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Gaussians + three training sets
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_DATA = [
    (X_tr1, y_tr1, "Training Set 1\n(50 pts cls 0 & 1 only)"),
    (X_tr2, y_tr2, "Training Set 2\n(10 pts per class)"),
    (X_tr3, y_tr3, "Training Set 3\n(100 pts per class)"),
]
MARKER_STYLES = ["o", "s", "^"]

fig2, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig2.patch.set_facecolor("#FAFAFA")

for ax, (X_tr, y_tr, title), mk in zip(axes, TRAIN_DATA, MARKER_STYLES):
    ax.set_facecolor("#FAFAFA")
    for mu, col in zip(MEANS, COLORS):
        draw_ellipse(ax, mu, STD, col, alpha=0.14)
    for i, col in enumerate(COLORS):
        mask = y_tr == i
        if mask.any():
            ax.scatter(X_tr[mask, 0], X_tr[mask, 1],
                       c=col, marker=mk, s=55, edgecolors="white",
                       linewidths=0.8, zorder=4, label=NAMES[i])
    style_ax(ax, title)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

fig2.suptitle("Gaussians + Training Sets", fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig("fig2_training_sets.png", dpi=150, bbox_inches="tight")
print("Saved fig2_training_sets.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Classification accuracy bar chart
# ══════════════════════════════════════════════════════════════════════════════
MODEL_LABELS = [
    "Model 1\n(cls 0&1 only\n50 pts each)",
    "Model 2\n(all classes\n10 pts each)",
    "Model 3\n(all classes\n100 pts each)",
]
ACCURACIES = [acc1, acc2, acc3]
BAR_COLORS = ["#E63946", "#457B9D", "#2A9D8F"]

fig3, ax = plt.subplots(figsize=(7, 5))
fig3.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

bars = ax.bar(MODEL_LABELS, ACCURACIES, color=BAR_COLORS,
              width=0.5, edgecolor="white", linewidth=1.2, zorder=3)
for bar, acc in zip(bars, ACCURACIES):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{acc:.1%}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.axhline(0.25, color="grey", linestyle="--", linewidth=1.2, label="Chance (25%)")
ax.set_ylim(0, 1.12)
ax.set_ylabel("Test Accuracy (all 4 classes, 800 pts)", fontsize=10)
ax.set_title("Naive Bayes – Test Accuracy by Training Set\n"
             "(same test set for all three models)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)

fig3.tight_layout()
fig3.savefig("fig3_accuracy.png", dpi=150, bbox_inches="tight")
print("Saved fig3_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – What each model "thinks" the world looks like
#
# Each panel shows:
#   • Solid thin ellipses  – true population (ground truth)
#   • Dashed thick ellipses – what the model learned (from clf.theta_ / clf.var_)
#   • Diamond markers      – points sampled from the model via sample_from_gnb()
#
# Model 1 is the most instructive: it only learned two classes, so you see
# only two dashed ellipses and two clusters of samples, while the solid
# ellipses for classes 2 & 3 still appear in the background as a reminder
# of what the model was never shown.
# ══════════════════════════════════════════════════════════════════════════════
MODEL_SAMPLE_DATA = [
    (clf1, X_s1, y_s1, f"Model 1 samples\n(test acc = {acc1:.1%})"),
    (clf2, X_s2, y_s2, f"Model 2 samples\n(test acc = {acc2:.1%})"),
    (clf3, X_s3, y_s3, f"Model 3 samples\n(test acc = {acc3:.1%})"),
]

fig4, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig4.patch.set_facecolor("#FAFAFA")

for ax, (clf, X_s, y_s, title) in zip(axes, MODEL_SAMPLE_DATA):
    ax.set_facecolor("#FAFAFA")

    # True population ellipses (solid, very light) ── ground truth backdrop
    for mu, col in zip(MEANS, COLORS):
        draw_ellipse(ax, mu, STD, col, alpha=0.09, lw=1.4)

    # Model-learned ellipses (dashed) ── what the model has internalized
    draw_gnb_ellipses(ax, clf)

    # Samples drawn from the model's generative distribution (diamonds)
    for label in clf.classes_:
        mask = y_s == label
        ax.scatter(X_s[mask, 0], X_s[mask, 1],
                   c=COLORS[int(label)], marker="D", s=42,
                   edgecolors="white", linewidths=0.6,
                   zorder=4, label=NAMES[int(label)])

    style_ax(ax, title)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

# Shared figure legend explaining the visual grammar
solid_patch  = mpatches.Patch(facecolor="#BBBBBB", edgecolor="#555555",
                               linewidth=1.4, label="True population (solid ellipse)")
dashed_line  = plt.Line2D([0], [0], color="#555555", linewidth=2,
                           linestyle="--", label="Model-learned ellipse (dashed)")
sample_point = plt.Line2D([0], [0], marker="D", color="w",
                           markerfacecolor="#555555", markersize=7,
                           label="Samples drawn from model (◆)")
fig4.legend(handles=[solid_patch, dashed_line, sample_point],
            loc="lower center", ncol=3, fontsize=9,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.07))

fig4.suptitle(
    "Model Generative View: What Each Model Has Learned\n"
    "(solid = true population · dashed = model's belief · ◆ = model samples)",
    fontsize=13, fontweight="bold", y=1.03)
fig4.tight_layout()
fig4.savefig("fig4_model_samples.png", dpi=150, bbox_inches="tight")
print("Saved fig4_model_samples.png")

plt.show()
print("\nDone.")