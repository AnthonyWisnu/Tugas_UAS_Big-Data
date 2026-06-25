from collections import Counter
from itertools import product
from pathlib import Path
from argparse import ArgumentParser

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_XGBOOST = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DATA_FILE = PROJECT_ROOT / "data" / "model" / "players_model.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET_COLUMN = "market_value_category"
LABEL_ORDER = ["Rendah", "Menengah", "Tinggi"]
HYBRID_TARGET_COUNT = 2500

NUMERIC_FEATURES = [
    "age",
    "age_squared",
    "age_peak_distance",
    "is_peak_age",
    "height_m",
    "is_goalkeeper",
    "is_defender",
    "is_midfielder",
    "is_forward",
    "season",
    "club_total_mv_mio",
    "club_total_mv_log",
    "club_total_mv_rank_league_season",
    "club_total_mv_pct_league_season",
    "prev_season_mv",
    "prev_season_mv_log",
    "prev_mv_distance_to_10",
    "prev_mv_distance_to_30",
    "two_seasons_ago_mv",
    "two_seasons_ago_mv_log",
    "has_prev_mv",
    "mv_history_count",
    "prev_growth_rate",
    "prev_growth_rate_clipped",
    "prev_mv_to_club_total_ratio",
    "age_prev_mv_interaction",
]

CATEGORICAL_FEATURES = [
    "age_group",
    "prev_mv_category",
    "two_seasons_ago_mv_category",
    "preferred_foot",
    "pos_category",
    "nationality",
    "club",
    "league",
    "league_rank",
]

PERFORMANCE_NUMERIC_FEATURES = [
    "matches_played",
    "starts",
    "minutes",
    "goals",
    "assists",
    "non_penalty_goals",
    "yellow_cards",
    "red_cards",
    "shots_total",
    "shots_on_target",
    "fouls_committed",
    "fouls_drawn",
    "saves",
    "clean_sheets",
    "goals_against",
    "shots_on_target_against",
    "goals_per_90",
    "assists_per_90",
    "goal_assist_per_90",
    "shots_per_90",
    "shots_on_target_per_90",
    "cards_per_90",
    "starts_rate",
    "save_pct",
    "clean_sheet_pct",
    "has_performance_stats",
]

FORBIDDEN_FEATURES = {
    "market_value_mio",
    "market_value_str",
    "market_value_category",
    "value_category",
    "label",
    "target",
    "mv_growth_rate",
    "position_detail",
    "player_id",
    "player_name",
    "player_url",
    "xg",
    "non_penalty_xg",
    "xg_per_90",
    "non_penalty_xg_per_90",
    "aerial_won",
    "aerial_lost",
    "ball_recoveries",
}


def load_model_data(path=MODEL_DATA_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model data file not found: {path}")
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")
    return df


def select_features(df):
    numeric_candidates = NUMERIC_FEATURES + PERFORMANCE_NUMERIC_FEATURES
    numeric_features = []
    for column in numeric_candidates:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").fillna(0)
        if column in PERFORMANCE_NUMERIC_FEATURES and values.abs().sum() == 0:
            continue
        numeric_features.append(column)
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in df.columns]
    features = numeric_features + categorical_features
    leakage_features = sorted(set(features) & FORBIDDEN_FEATURES)
    if leakage_features:
        raise ValueError(f"Forbidden features selected: {leakage_features}")
    return numeric_features, categorical_features, features


def dataset_config():
    return {
        "data_file": MODEL_DATA_FILE,
        "metrics_file": "model_metrics.csv",
        "comparison_file": "model_comparison_scenarios.csv",
        "classification_file": "classification_report_best_model.csv",
        "confusion_file": "confusion_matrix_best_model.csv",
        "feature_file": "feature_importance_best_model.csv",
        "label_distribution_file": "label_distribution_before_after_sampling.csv",
        "best_model_file": "best_model.pkl",
        "preprocessor_file": "preprocessor.pkl",
        "label_encoder_file": "label_encoder.pkl",
        "figure_confusion_file": "confusion_matrix_best_model.png",
        "figure_feature_file": "feature_importance_best_model.png",
    }


def split_by_season(df):
    train_df = df[df["season"].between(2017, 2021)].copy()
    val_df = df[df["season"] == 2022].copy()
    test_df = df[df["season"].between(2023, 2024)].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train, validation, or test split is empty")
    return train_df, val_df, test_df


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def limit_grid(param_grid, max_configs):
    keys = list(param_grid.keys())
    all_configs = [dict(zip(keys, values)) for values in product(*(param_grid[key] for key in keys))]
    if len(all_configs) <= max_configs:
        return all_configs
    positions = np.linspace(0, len(all_configs) - 1, max_configs, dtype=int)
    return [all_configs[index] for index in positions]


def random_forest_configs(scenario):
    class_weights = ["balanced", "balanced_subsample", None]
    if scenario == "class_weight_balanced":
        class_weights = ["balanced", "balanced_subsample"]
    elif scenario in {"no_sampling", "hybrid_sampling_light"}:
        class_weights = [None]

    grid = {
        "n_estimators": [300, 500],
        "max_depth": [10, 15, 20, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": class_weights,
    }
    return limit_grid(grid, max_configs=24)


def xgboost_configs():
    grid = {
        "n_estimators": [300, 500, 800],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1, 3, 5],
    }
    return limit_grid(grid, max_configs=36)


def logistic_configs(scenario):
    class_weight = "balanced" if scenario == "class_weight_balanced" else None
    return [{"C": value, "class_weight": class_weight} for value in [0.3, 1.0, 3.0]]


def prepare_training_data(X_train, y_train, scenario):
    if scenario in {"no_sampling", "class_weight_balanced"}:
        return X_train, y_train

    counts = Counter(y_train)
    target_count = min(
        counts["Menengah"],
        max(counts["Rendah"], HYBRID_TARGET_COUNT),
    )
    under_strategy = {label: min(count, target_count) for label, count in counts.items()}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_STATE)
    X_under, y_under = rus.fit_resample(X_train, y_train)
    over_strategy = {label: target_count for label in Counter(y_under).keys()}
    ros = RandomOverSampler(sampling_strategy=over_strategy, random_state=RANDOM_STATE)
    return ros.fit_resample(X_under, y_under)


def build_candidate(model_name, params, scenario, label_count):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            C=params["C"],
            class_weight=params["class_weight"],
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if HAS_XGBOOST:
        return XGBClassifier(
            objective="multi:softprob",
            num_class=label_count,
            eval_metric="mlogloss",
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        random_state=RANDOM_STATE,
    )


def model_param_configs(model_name, scenario):
    if model_name == "logistic_regression":
        return logistic_configs(scenario)
    if model_name == "random_forest":
        return random_forest_configs(scenario)
    if HAS_XGBOOST:
        return xgboost_configs()
    return [{}]


def encode_targets(y_train, y_val, y_test):
    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ORDER)
    return (
        label_encoder,
        label_encoder.transform(y_train),
        label_encoder.transform(y_val),
        label_encoder.transform(y_test),
    )


def evaluate_predictions(model_name, scenario, split_name, y_true_encoded, y_pred_encoded, label_encoder):
    labels_encoded = label_encoder.transform(LABEL_ORDER)
    y_true = label_encoder.inverse_transform(y_true_encoded)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )[2]
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_encoded, y_pred_encoded, labels=labels_encoded)
    metrics = {
        "model": model_name,
        "scenario": scenario,
        "split": split_name,
        "accuracy": accuracy_score(y_true_encoded, y_pred_encoded),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1,
        "recall_tinggi": report["Tinggi"]["recall"],
    }
    return metrics, report, matrix


def report_to_rows(model_name, scenario, split_name, report):
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {
                "model": model_name,
                "scenario": scenario,
                "split": split_name,
                "label": label,
            }
            row.update(values)
            rows.append(row)
    return rows


def matrix_to_rows(matrix):
    rows = []
    for i, actual in enumerate(LABEL_ORDER):
        for j, predicted in enumerate(LABEL_ORDER):
            rows.append({"actual": actual, "predicted": predicted, "count": int(matrix[i, j])})
    return rows


def get_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = list(numeric_features)
    cat_transformer = preprocessor.named_transformers_["cat"]
    onehot = cat_transformer.named_steps["onehot"]
    feature_names.extend(onehot.get_feature_names_out(categorical_features).tolist())
    return feature_names


def get_feature_importance(pipeline, feature_names):
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def fit_pipeline(pipeline, X_train, y_train_encoded, y_train_labels, scenario, model_name):
    if scenario == "class_weight_balanced" and model_name == "xgboost" and HAS_XGBOOST:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_labels)
        pipeline.fit(X_train, y_train_encoded, classifier__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train, y_train_encoded)


def save_label_distribution(distributions, file_name="label_distribution_before_after_sampling.csv"):
    rows = []
    for split_name, labels in distributions.items():
        counts = Counter(labels)
        for label in LABEL_ORDER:
            rows.append({"split": split_name, "label": label, "count": int(counts.get(label, 0))})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / file_name, index=False)
    return df


def plot_confusion_matrix(matrix, model_name, scenario, file_name="confusion_matrix_best_model.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Best Model Confusion Matrix: {model_name}, {scenario}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    path = FIGURES_DIR / file_name
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(feature_importance_df, model_name, scenario, file_name="feature_importance_best_model.png"):
    top_features = feature_importance_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title(f"Best Model Feature Importance: {model_name}, {scenario}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = FIGURES_DIR / file_name
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train_and_evaluate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = dataset_config()
    df = load_model_data(config["data_file"])
    numeric_features, categorical_features, features = select_features(df)
    train_df, val_df, test_df = split_by_season(df)

    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[features]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[features]
    y_test = test_df[TARGET_COLUMN]

    scenarios = ["no_sampling", "class_weight_balanced", "hybrid_sampling_light"]
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    validation_rows = []
    best = None
    best_pipeline = None
    best_label_encoder = None
    best_train_labels = None

    for scenario in scenarios:
        X_train_scenario, y_train_scenario = prepare_training_data(X_train, y_train, scenario)
        label_encoder, y_train_encoded, y_val_encoded, y_test_encoded = encode_targets(
            y_train_scenario,
            y_val,
            y_test,
        )

        for model_name in model_names:
            for params in model_param_configs(model_name, scenario):
                preprocessor = build_preprocessor(numeric_features, categorical_features)
                classifier = build_candidate(model_name, params, scenario, len(label_encoder.classes_))
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", classifier),
                    ]
                )
                fit_pipeline(
                    pipeline,
                    X_train_scenario,
                    y_train_encoded,
                    y_train_scenario,
                    scenario,
                    model_name,
                )
                y_val_pred = pipeline.predict(X_val)
                metrics, _, _ = evaluate_predictions(
                    model_name,
                    scenario,
                    "validation",
                    y_val_encoded,
                    y_val_pred,
                    label_encoder,
                )
                metrics["params"] = params
                validation_rows.append(metrics)
                if best is None or metrics["macro_f1"] > best["macro_f1"]:
                    best = metrics
                    best_pipeline = pipeline
                    best_label_encoder = label_encoder
                    best_train_labels = y_train_scenario

    validation_df = pd.DataFrame(validation_rows).sort_values("macro_f1", ascending=False)
    validation_df.to_csv(OUTPUT_DIR / config["comparison_file"], index=False)

    y_test_encoded = best_label_encoder.transform(y_test)
    y_test_pred = best_pipeline.predict(X_test)
    test_metrics, best_report, best_matrix = evaluate_predictions(
        best["model"],
        best["scenario"],
        "test",
        y_test_encoded,
        y_test_pred,
        best_label_encoder,
    )

    metrics_df = pd.concat(
        [
            validation_df.assign(selected_for_test=False),
            pd.DataFrame([{**test_metrics, "params": best["params"], "selected_for_test": True}]),
        ],
        ignore_index=True,
    )
    metrics_df.to_csv(OUTPUT_DIR / config["metrics_file"], index=False)

    pd.DataFrame(report_to_rows(best["model"], best["scenario"], "test", best_report)).to_csv(
        OUTPUT_DIR / config["classification_file"],
        index=False,
    )
    pd.DataFrame(matrix_to_rows(best_matrix)).to_csv(
        OUTPUT_DIR / config["confusion_file"],
        index=False,
    )

    feature_names = get_feature_names(
        best_pipeline.named_steps["preprocessor"],
        numeric_features,
        categorical_features,
    )
    feature_importance_df = get_feature_importance(best_pipeline, feature_names)
    feature_importance_df.to_csv(OUTPUT_DIR / config["feature_file"], index=False)

    joblib.dump(best_pipeline, MODELS_DIR / config["best_model_file"])
    joblib.dump(best_pipeline.named_steps["preprocessor"], MODELS_DIR / config["preprocessor_file"])
    joblib.dump(best_label_encoder, MODELS_DIR / config["label_encoder_file"])
    joblib.dump(best_pipeline, MODELS_DIR / f"{best['model']}.pkl")

    save_label_distribution(
        {
            "train_original": y_train,
            f"train_{best['scenario']}": best_train_labels,
            "validation_original": y_val,
            "test_original": y_test,
        },
        file_name=config["label_distribution_file"],
    )
    plot_confusion_matrix(best_matrix, best["model"], best["scenario"], config["figure_confusion_file"])
    if not feature_importance_df.empty:
        plot_feature_importance(
            feature_importance_df,
            best["model"],
            best["scenario"],
            config["figure_feature_file"],
        )

    summary = {
        "best_model": best["model"],
        "best_scenario": best["scenario"],
        "best_params": best["params"],
        "xgboost_available": HAS_XGBOOST,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "features": features,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "train_label": dict(Counter(y_train)),
        "best_train_label": dict(Counter(best_train_labels)),
        "validation_label": dict(Counter(y_val)),
        "test_label": dict(Counter(y_test)),
        "validation_metrics": validation_df,
        "test_metrics": test_metrics,
        "confusion_matrix": best_matrix,
        "feature_importance": feature_importance_df,
        "dataset_variant": "final",
        "data_file": str(config["data_file"]),
    }
    return summary, metrics_df


def parse_args():
    parser = ArgumentParser(description="Train market value model.")
    return parser.parse_args()


def main():
    parse_args()
    summary, metrics_df = train_and_evaluate()
    print("Training final selesai.")
    print(f"Dataset      : {summary['dataset_variant']}")
    print(f"Data file    : {summary['data_file']}")
    print(f"Best model   : {summary['best_model']}")
    print(f"Best scenario: {summary['best_scenario']}")
    print(f"Best params  : {summary['best_params']}")
    print(f"Train rows   : {summary['train_rows']}")
    print(f"Validation rows: {summary['validation_rows']}")
    print(f"Test rows    : {summary['test_rows']}")
    print("Top validation metrics:")
    print(summary["validation_metrics"].head(10))
    print("Best model test metrics:")
    print(pd.DataFrame([summary["test_metrics"]]))
    return summary, metrics_df


if __name__ == "__main__":
    main()
