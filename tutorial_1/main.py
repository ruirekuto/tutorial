from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual tutorial: compare linear and nonlinear decision boundaries.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("moon_tutorial.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def build_dataset(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    return make_moons(n_samples=400, noise=0.22, random_state=random_state)


def plot_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str) -> None:
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.5,
        s=42,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_model(
    ax: plt.Axes,
    model,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    title: str,
) -> None:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    DecisionBoundaryDisplay.from_estimator(
        model,
        np.vstack([x_train, x_test]),
        response_method="predict",
        cmap="coolwarm",
        alpha=0.25,
        ax=ax,
    )

    ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.3,
        s=32,
        alpha=0.7,
        label="train",
    )
    ax.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.8,
        marker="D",
        s=46,
        label="test",
    )
    ax.set_title(f"{title}\naccuracy = {accuracy:.3f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")


def create_figure(output_path: Path, show: bool) -> None:
    x, y = build_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=7,
        stratify=y,
    )

    linear_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(),
    )
    nonlinear_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#f6f4ef")
    fig.suptitle(
        "Sklearn Tutorial: Why Nonlinear Models Matter",
        fontsize=18,
        fontweight="bold",
    )

    plot_points(axes[0, 0], x_train, y_train, "Training Data")
    plot_points(axes[0, 1], x_test, y_test, "Test Data")
    plot_model(
        axes[1, 0],
        linear_model,
        x_train,
        x_test,
        y_train,
        y_test,
        "Logistic Regression\n(linear boundary)",
    )
    plot_model(
        axes[1, 1],
        nonlinear_model,
        x_train,
        x_test,
        y_train,
        y_test,
        "k-NN Classifier\n(nonlinear boundary)",
    )

    train_center = np.mean(x_train, axis=0)
    spread = np.std(x_train, axis=0)
    fig.text(
        0.5,
        0.03,
        "NumPy summary: "
        f"train center = ({train_center[0]:+.2f}, {train_center[1]:+.2f}), "
        f"spread = ({spread[0]:.2f}, {spread[1]:.2f})",
        ha="center",
        fontsize=11,
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
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
