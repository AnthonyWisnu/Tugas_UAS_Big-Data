import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PERFORMANCE_METRICS_FILE = OUTPUT_DIR / "model_metrics_with_performance.csv"
CONFUSION_FILE = OUTPUT_DIR / "confusion_matrix_best_model_with_performance.csv"
FEATURE_FILE = OUTPUT_DIR / "feature_importance_best_model_with_performance.csv"
LABEL_ORDER = ["Rendah", "Menengah", "Tinggi"]


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def save_figure(fig, paths):
    saved_paths = []
    for path in paths:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_paths.append(path)
    plt.close(fig)
    return saved_paths


def plot_validation_metric(metric):
    df = pd.read_csv(require_file(PERFORMANCE_METRICS_FILE))
    validation_df = df[df["split"] == "validation"].copy()
    if validation_df.empty:
        raise ValueError("No validation rows found in performance metrics output.")

    grouped = (
        validation_df.groupby(["model", "scenario"], as_index=False)[metric]
        .max()
        .sort_values(metric, ascending=False)
        .head(12)
    )
    grouped["label"] = grouped["model"] + " | " + grouped["scenario"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(grouped["label"].iloc[::-1], grouped[metric].iloc[::-1], color="#2e8b57")
    ax.set_title(f"Validation {metric.replace('_', ' ').title()} Transfermarkt + FBref")
    ax.set_xlabel(metric.replace("_", " ").title())
    for bar in bars:
        value = bar.get_width()
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    plt.tight_layout()
    path = FIGURES_DIR / f"transfermarkt_fbref_validation_{metric}.png"
    return save_figure(fig, [path])


def plot_confusion_matrix():
    df = pd.read_csv(require_file(CONFUSION_FILE))
    pivot = df.pivot(index="actual", columns="predicted", values="count").fillna(0)
    pivot = pivot.reindex(index=LABEL_ORDER, columns=LABEL_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(pivot.values, cmap="Blues")
    ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Best Model Confusion Matrix Transfermarkt + FBref")
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    paths = [
        FIGURES_DIR / "confusion_matrix_best_model.png",
        FIGURES_DIR / "confusion_matrix_best_model_with_performance.png",
    ]
    return save_figure(fig, paths)


def plot_feature_importance():
    df = pd.read_csv(require_file(FEATURE_FILE)).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature"], df["importance"], color="#2e8b57")
    ax.set_title("Feature Importance Transfermarkt + FBref")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    paths = [
        FIGURES_DIR / "feature_importance_best_model.png",
        FIGURES_DIR / "feature_importance_best_model_with_performance.png",
    ]
    return save_figure(fig, paths)


def generate_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path_groups = [
        plot_validation_metric("accuracy"),
        plot_validation_metric("macro_f1"),
        plot_confusion_matrix(),
        plot_feature_importance(),
    ]
    paths = [path for group in path_groups for path in group]
    for path in paths:
        print(f"Saved figure: {path}")
    return paths


def parse_args():
    return argparse.ArgumentParser(description="Generate Transfermarkt + FBref model figures.").parse_args()


def main():
    parse_args()
    generate_figures()


if __name__ == "__main__":
    main()
