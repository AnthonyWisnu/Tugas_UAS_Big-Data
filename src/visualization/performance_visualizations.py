from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return path


def plot_metric_comparison(comparison_df, metric, output_name):
    data = comparison_df.copy()
    if "dataset" not in data.columns or metric not in data.columns:
        raise ValueError(f"Comparison data must contain dataset and {metric}.")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(data["dataset"], data[metric])
    ax.set_ylim(0, max(1.0, float(data[metric].max()) * 1.15))
    ax.set_ylabel(metric)
    ax.set_title(f"Original vs With Performance: {metric}")
    for index, value in enumerate(data[metric]):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    path = FIGURES_DIR / output_name
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(feature_df):
    if feature_df.empty:
        return None
    data = feature_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(data["feature"], data["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance With Performance")
    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance_final.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion_matrix(confusion_df):
    labels = ["Rendah", "Menengah", "Tinggi"]
    pivot = (
        confusion_df.pivot(index="actual", columns="predicted", values="count")
        .reindex(index=labels, columns=labels)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(pivot.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix With Performance")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    path = FIGURES_DIR / "confusion_matrix_final.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_visualizations():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(require_file(OUTPUT_DIR / "model_metrics.csv"))
    feature = pd.read_csv(require_file(OUTPUT_DIR / "feature_importance_best_model.csv"))
    confusion = pd.read_csv(require_file(OUTPUT_DIR / "confusion_matrix_best_model.csv"))
    test_comparison = comparison[comparison["split"] == "test"].copy()
    test_comparison["dataset"] = "final"

    paths = [
        plot_metric_comparison(test_comparison, "accuracy", "accuracy_final.png"),
        plot_metric_comparison(test_comparison, "macro_f1", "macro_f1_final.png"),
        plot_feature_importance(feature),
        plot_confusion_matrix(confusion),
    ]
    return [path for path in paths if path is not None]


def parse_args():
    return ArgumentParser(description="Generate new performance model visualizations.").parse_args()


def main():
    parse_args()
    paths = generate_visualizations()
    print("Visualisasi performa selesai.")
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
