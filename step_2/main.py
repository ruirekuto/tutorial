from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual tutorial: discover clusters with KMeans.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kmeans_tutorial.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def build_dataset(random_state: int = 12) -> np.ndarray:
    x, _ = make_blobs(
        n_samples=450,
        centers=[(-3, -1), (0.5, 2.8), (3.2, -1.4), (0.0, -3.2)],
        cluster_std=[0.9, 0.7, 1.0, 0.8],
        random_state=random_state,
    )
    transform = np.array([[0.75, -0.35], [0.25, 1.1]])
    warped = x @ transform
    return StandardScaler().fit_transform(warped)


def scatter_plain(ax: plt.Axes, x: np.ndarray) -> None:
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c="#34495e",
        s=28,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title("Raw Data")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def scatter_clusters(
    ax: plt.Axes,
    x: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    title: str,
) -> None:
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=labels,
        cmap="Set2",
        s=30,
        alpha=0.82,
        edgecolor="black",
        linewidth=0.25,
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="#111111",
        marker="X",
        s=220,
        linewidth=1.0,
        edgecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_regions(ax: plt.Axes, model: KMeans, x: np.ndarray) -> None:
    margin = 0.8
    x_min, x_max = x[:, 0].min() - margin, x[:, 0].max() + margin
    y_min, y_max = x[:, 1].min() - margin, x[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    region_labels = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, region_labels, cmap="Set2", alpha=0.32)
    ax.contour(xx, yy, region_labels, colors="white", linewidths=1.0, alpha=0.9)
    scatter_clusters(
        ax,
        x,
        model.labels_,
        model.cluster_centers_,
        "Cluster Regions and Centers",
    )


def create_figure(output_path: Path, show: bool) -> None:
    x = build_dataset()
    model = KMeans(n_clusters=4, n_init=20, random_state=12)
    labels = model.fit_predict(x)
    score = silhouette_score(x, labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor("#f3efe7")
    fig.suptitle(
        "Sklearn Tutorial: Finding Structure Without Labels",
        fontsize=17,
        fontweight="bold",
    )

    scatter_plain(axes[0], x)
    scatter_clusters(
        axes[1],
        x,
        labels,
        model.cluster_centers_,
        f"KMeans Labels\nsilhouette = {score:.3f}",
    )
    plot_regions(axes[2], model, x)

    cluster_sizes = np.bincount(labels)
    fig.text(
        0.5,
        0.04,
        "NumPy summary: "
        f"mean = ({np.mean(x[:, 0]):+.2f}, {np.mean(x[:, 1]):+.2f}), "
        f"std = ({np.std(x[:, 0]):.2f}, {np.std(x[:, 1]):.2f}), "
        f"cluster sizes = {cluster_sizes.tolist()}",
        ha="center",
        fontsize=10.5,
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.92))
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
