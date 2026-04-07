from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual tutorial: project high-dimensional digits with PCA.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pca_tutorial.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def create_figure(output_path: Path, show: bool) -> None:
    digits = load_digits()
    x = digits.data
    y = digits.target
    images = digits.images

    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=0)
    embedding = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#f4f0e8")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.1)
    fig.suptitle(
        "Sklearn Tutorial: PCA for Handwritten Digits",
        fontsize=18,
        fontweight="bold",
    )

    sample_indices = [3, 17, 42, 91, 128, 256]
    image_gs = gs[0, 0].subgridspec(2, 2, wspace=0.25, hspace=0.12)
    for i, idx in enumerate(sample_indices):
        if i >= 4:
            break
        ax = fig.add_subplot(image_gs[i // 2, i % 2])
        ax.imshow(images[idx], cmap="magma_r")
        ax.set_title(f"Digit {y[idx]}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_scatter = fig.add_subplot(gs[0, 1])
    scatter = ax_scatter.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="tab10",
        s=26,
        alpha=0.72,
        edgecolor="none",
    )
    ax_scatter.set_title("2D PCA Projection", fontsize=14, pad=10)
    ax_scatter.set_xlabel("principal component 1")
    ax_scatter.set_ylabel("principal component 2")
    ax_scatter.grid(alpha=0.2)

    for digit in range(10):
        mask = y == digit
        center = np.mean(embedding[mask], axis=0)
        ax_scatter.text(
            center[0],
            center[1],
            str(digit),
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.25", "fc": "white", "ec": "black", "alpha": 0.8},
        )

    cbar = fig.colorbar(scatter, ax=ax_scatter, ticks=range(10), pad=0.03, fraction=0.055)
    cbar.set_label("digit label")

    explained = pca.explained_variance_ratio_
    fig.text(
        0.5,
        0.03,
        "NumPy/PCA summary: "
        f"embedding mean = ({np.mean(embedding[:, 0]):+.2f}, {np.mean(embedding[:, 1]):+.2f}), "
        f"variance captured = {100 * explained[0]:.1f}% + {100 * explained[1]:.1f}% "
        f"= {100 * np.sum(explained):.1f}%",
        ha="center",
        fontsize=11,
    )

    fig.subplots_adjust(left=0.05, right=0.93, top=0.9, bottom=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"saved figure to {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    create_figure(output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
