import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.modeling.train import (
    CATEGORICAL_FEATURES,
    LABEL_ORDER,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET_COLUMN,
    build_candidate,
    build_preprocessor,
    encode_targets,
    evaluate_predictions,
    fit_pipeline,
    get_feature_importance,
    get_feature_names,
    model_param_configs,
    prepare_training_data,
    report_to_rows,
    split_by_season,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_MODEL_DATA_FILE = PROJECT_ROOT / "data" / "model" / "players_model_with_performance.csv"

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
    "player_id",
    "player_name",
    "player_url",
}


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_dataset(path):
    df = pd.read_csv(require_file(path))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")
    return df


def select_features(df):
    numeric_features = [column for column in NUMERIC_FEATURES if column in df.columns]
    numeric_features.extend(
        [
            column
            for column in PERFORMANCE_NUMERIC_FEATURES
            if column in df.columns and pd.to_numeric(df[column], errors="coerce").fillna(0).abs().sum() > 0
        ]
    )
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in df.columns]
    features = numeric_features + categorical_features
    leakage = sorted((set(features) - {TARGET_COLUMN}) & FORBIDDEN_FEATURES)
    if leakage:
        raise ValueError(f"Forbidden features selected: {leakage}")
    return numeric_features, categorical_features, features


def train_dataset(dataset_path=PERFORMANCE_MODEL_DATA_FILE):
    df = load_dataset(dataset_path)
    numeric_features, categorical_features, features = select_features(df)
    train_df, val_df, test_df = split_by_season(df)
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[features]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[features]
    y_test = test_df[TARGET_COLUMN]

    scenarios = ["no_sampling", "class_weight_balanced", "hybrid_sampling_light"]
    model_names = ["logistic_regression", "xgboost"]
    validation_rows = []
    best = None
    best_pipeline = None
    best_label_encoder = None

    for scenario in scenarios:
        X_train_scenario, y_train_scenario = prepare_training_data(X_train, y_train, scenario)
        label_encoder, y_train_encoded, y_val_encoded, _ = encode_targets(
            y_train_scenario,
            y_val,
            y_test,
        )

        for model_name in model_names:
            for params in model_param_configs(model_name, scenario):
                pipeline = build_pipeline(
                    model_name=model_name,
                    params=params,
                    scenario=scenario,
                    label_count=len(label_encoder.classes_),
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
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
                metrics["dataset"] = "transfermarkt_fbref"
                metrics["params"] = params
                validation_rows.append(metrics)
                if best is None or metrics["macro_f1"] > best["macro_f1"]:
                    best = metrics
                    best_pipeline = pipeline
                    best_label_encoder = label_encoder

    y_test_encoded = best_label_encoder.transform(y_test)
    y_test_pred = best_pipeline.predict(X_test)
    test_metrics, test_report, test_matrix = evaluate_predictions(
        best["model"],
        best["scenario"],
        "test",
        y_test_encoded,
        y_test_pred,
        best_label_encoder,
    )
    test_metrics["dataset"] = "transfermarkt_fbref"
    test_metrics["params"] = best["params"]

    feature_names = get_feature_names(
        best_pipeline.named_steps["preprocessor"],
        numeric_features,
        categorical_features,
    )
    feature_importance = get_feature_importance(best_pipeline, feature_names)
    return {
        "dataset": "transfermarkt_fbref",
        "validation_rows": validation_rows,
        "test_metrics": test_metrics,
        "test_report": test_report,
        "test_matrix": test_matrix,
        "feature_importance": feature_importance,
        "best_pipeline": best_pipeline,
        "label_encoder": best_label_encoder,
        "best": best,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
    }


def build_pipeline(model_name, params, scenario, label_count, numeric_features, categorical_features):
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    classifier = build_candidate(model_name, params, scenario, label_count)
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def matrix_to_rows(matrix, dataset, model, scenario):
    rows = []
    for i, actual in enumerate(LABEL_ORDER):
        for j, predicted in enumerate(LABEL_ORDER):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "scenario": scenario,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[i, j]),
                }
            )
    return rows


def save_results(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    validation_df = pd.DataFrame(result["validation_rows"])
    metrics_df = pd.concat(
        [
            validation_df.assign(selected_for_test=False),
            pd.DataFrame([{**result["test_metrics"], "selected_for_test": True}]),
        ],
        ignore_index=True,
    )
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics_with_performance.csv", index=False)

    report_rows = report_to_rows(
        result["best"]["model"],
        result["best"]["scenario"],
        "test",
        result["test_report"],
    )
    pd.DataFrame(report_rows).to_csv(
        OUTPUT_DIR / "classification_report_best_model_with_performance.csv",
        index=False,
    )
    pd.DataFrame(
        matrix_to_rows(
            result["test_matrix"],
            result["dataset"],
            result["best"]["model"],
            result["best"]["scenario"],
        )
    ).to_csv(OUTPUT_DIR / "confusion_matrix_best_model_with_performance.csv", index=False)
    result["feature_importance"].to_csv(
        OUTPUT_DIR / "feature_importance_best_model_with_performance.csv",
        index=False,
    )

    joblib.dump(result["best_pipeline"], MODELS_DIR / "best_model_with_performance.pkl")
    joblib.dump(result["label_encoder"], MODELS_DIR / "label_encoder_with_performance.pkl")

    print("Saved Transfermarkt + FBref training outputs.")
    print(pd.DataFrame([result["test_metrics"]]))
    return metrics_df


def parse_args():
    parser = argparse.ArgumentParser(description="Train models with Transfermarkt + FBref dataset.")
    parser.add_argument("--dataset-file", default=str(PERFORMANCE_MODEL_DATA_FILE))
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_dataset(args.dataset_file)
    save_results(result)


if __name__ == "__main__":
    main()
