import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st
import xgboost
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_dataset.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model" / "players_model.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "model" / "feature_list.json"
MODEL_METRICS_PATH = PROJECT_ROOT / "data" / "output" / "model_metrics.csv"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "output" / "test_predictions.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "output" / "feature_importance.csv"
BEST_MODEL_SUMMARY_PATH = PROJECT_ROOT / "data" / "output" / "best_model_summary.csv"
FORECAST_2025_PATH = PROJECT_ROOT / "data" / "output" / "forecast_2025.csv"
OVERSAMPLING_SUMMARY_PATH = PROJECT_ROOT / "data" / "output" / "oversampling_summary.csv"
TRAIN_BEFORE_PATH = PROJECT_ROOT / "data" / "model" / "train_before_oversampling.csv"
TRAIN_AFTER_PATH = PROJECT_ROOT / "data" / "model" / "train_after_oversampling.csv"
HIGH_BEFORE_PATH = PROJECT_ROOT / "data" / "model" / "train_high_before_oversampling.csv"
HIGH_AFTER_PATH = PROJECT_ROOT / "data" / "model" / "train_high_after_oversampling.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

LABEL_ORDER = ["Rendah", "Menengah", "Tinggi"]
LABEL_COLORS = {"Rendah": "#3b82f6", "Menengah": "#10b981", "Tinggi": "#f59e0b"}
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


st.set_page_config(page_title="Prediksi Nilai Pasar Pemain", page_icon=None, layout="wide")


def supports_argument(function, argument):
    return argument in inspect.signature(function).parameters


def render_chart(fig):
    if supports_argument(st.plotly_chart, "width"):
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    else:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_table(frame):
    if supports_argument(st.dataframe, "width"):
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def category(value):
    if value < 15:
        return "Rendah"
    if value <= 35:
        return "Menengah"
    return "Tinggi"


def format_money(value):
    return f"EUR {float(value):,.2f} juta"


def format_percent(value):
    return f"{float(value) * 100:.1f}%"


def require_artifacts(paths):
    missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        st.error("Artifact pipeline belum lengkap. Jalankan 02_preprocessing.ipynb lalu 03_training_model.ipynb.")
        st.code("\n".join(missing))
        st.stop()


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_features():
    return json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    return load(BEST_MODEL_PATH)


required = [
    FORECAST_DATA_PATH, MODEL_DATA_PATH, FEATURE_LIST_PATH, MODEL_METRICS_PATH,
    TEST_PREDICTIONS_PATH, FEATURE_IMPORTANCE_PATH, BEST_MODEL_SUMMARY_PATH,
    FORECAST_2025_PATH, OVERSAMPLING_SUMMARY_PATH, TRAIN_BEFORE_PATH,
    TRAIN_AFTER_PATH, HIGH_BEFORE_PATH, HIGH_AFTER_PATH, BEST_MODEL_PATH,
]
require_artifacts(required)

forecast_data = load_csv(FORECAST_DATA_PATH)
model_data = load_csv(MODEL_DATA_PATH)
metrics_data = load_csv(MODEL_METRICS_PATH)
test_predictions = load_csv(TEST_PREDICTIONS_PATH)
feature_importance = load_csv(FEATURE_IMPORTANCE_PATH)
model_summary = load_csv(BEST_MODEL_SUMMARY_PATH).iloc[0]
forecast_2025 = load_csv(FORECAST_2025_PATH)
oversampling_summary = load_csv(OVERSAMPLING_SUMMARY_PATH)
feature_list = load_features()
runtime_mismatch = []
model_python = str(model_summary.get("python_version", ""))
runtime_python = ".".join(map(str, sys.version_info[:3]))
if model_python and ".".join(model_python.split(".")[:2]) != ".".join(runtime_python.split(".")[:2]):
    runtime_mismatch.append(f"Python model={model_python} dashboard={runtime_python}")
if str(model_summary.get("sklearn_version", "")) and str(model_summary.get("sklearn_version")) != sklearn.__version__:
    runtime_mismatch.append(f"scikit-learn model={model_summary.get('sklearn_version')} dashboard={sklearn.__version__}")
if str(model_summary.get("xgboost_version", "")) and str(model_summary.get("xgboost_version")) != xgboost.__version__:
    runtime_mismatch.append(f"xgboost model={model_summary.get('xgboost_version')} dashboard={xgboost.__version__}")
if runtime_mismatch:
    st.error("Versi environment training dan dashboard berbeda. Prediksi dihentikan agar hasil tidak salah.")
    st.code("\n".join(runtime_mismatch))
    st.info("Jalankan Streamlit dari environment/kernel yang sama dengan notebook: python -m streamlit run app/app.py")
    st.stop()
model = load_model()


st.sidebar.header("Filter Data")
season_values = sorted(forecast_data["season"].dropna().astype(int).unique())
selected_seasons = st.sidebar.multiselect("Musim fitur", season_values, default=season_values)
league_values = sorted(forecast_data["league"].dropna().astype(str).unique())
selected_leagues = st.sidebar.multiselect("Liga", league_values, default=league_values)
position_values = sorted(forecast_data["pos_category"].dropna().astype(str).unique())
selected_positions = st.sidebar.multiselect("Posisi", position_values, default=position_values)
filtered = forecast_data[
    forecast_data["season"].isin(selected_seasons)
    & forecast_data["league"].isin(selected_leagues)
    & forecast_data["pos_category"].isin(selected_positions)
].copy()

st.title("Prediksi Nilai Pasar Pemain Sepak Bola")
st.caption(
    "Model menggunakan profil, nilai pasar historis, konteks klub, dan performa musim ini "
    "untuk memprediksi nilai pasar musim berikutnya. Kategori hanya menjadi interpretasi hasil prediksi."
)

overview_tab, evaluation_tab, prediction_tab, importance_tab, explorer_tab = st.tabs([
    "Ringkasan", "Evaluasi Model", "Prediksi 2025", "Faktor Prediksi", "Data Explorer"
])

with overview_tab:
    st.subheader("Ringkasan Dataset Forecast")
    columns = st.columns(5)
    columns[0].metric("Record", f"{len(filtered):,}")
    columns[1].metric("Pemain", f"{filtered['player_id'].nunique():,}")
    columns[2].metric("Liga", f"{filtered['league'].nunique():,}")
    columns[3].metric("Data Berlabel", f"{int(filtered['has_known_next_season_value'].sum()):,}")
    columns[4].metric("Kandidat Forecast 2025", f"{len(forecast_2025):,}")

    left, right = st.columns(2)
    with left:
        season_count = filtered.groupby("season", as_index=False).size()
        fig = px.bar(
            season_count, x="season", y="size", text="size",
            title="Jumlah Record Berdasarkan Musim Fitur",
            labels={"season": "Musim fitur", "size": "Jumlah record"},
        )
        render_chart(fig)
    with right:
        category_count = filtered["current_market_value_category"].value_counts().reindex(LABEL_ORDER).dropna().reset_index()
        category_count.columns = ["category", "records"]
        fig = px.pie(
            category_count, names="category", values="records", hole=0.4,
            category_orders={"category": LABEL_ORDER}, color="category",
            color_discrete_map=LABEL_COLORS, title="Kategori Nilai Pasar Saat Ini",
        )
        render_chart(fig)

    st.info(
        "Contoh alur: statistik dan nilai pasar musim 2023 digunakan untuk memprediksi nilai pasar 2024. "
        "Dengan cara ini model tidak hanya mengubah angka yang sudah diketahui menjadi kategori."
    )

with evaluation_tab:
    st.subheader("Evaluasi Prediksi Musim Berikutnya")
    test_model = metrics_data[(metrics_data["split"] == "test") & (metrics_data["model"] == model_summary["model"])].iloc[0]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Model Terbaik", str(model_summary["model"]).replace("_", " ").title())
    metric_columns[1].metric("MAE Test", format_money(test_model["mae"]))
    metric_columns[2].metric("RMSE Test", format_money(test_model["rmse"]))
    metric_columns[3].metric("R2 Test", f"{test_model['r2']:.3f}")
    metric_columns[4].metric("Akurasi Kategori", format_percent(test_model["category_accuracy"]))
    st.caption(
        "MAE menunjukkan rata-rata selisih absolut antara prediksi dan nilai aktual. "
        "Evaluasi test memakai target musim 2024 yang tidak digunakan saat pemilihan model."
    )

    st.markdown("### Oversampling Data Training")
    sampling_columns = st.columns(4)
    sampling_columns[0].metric("Train Sebelum", f"{int(oversampling_summary['before_records'].sum()):,}")
    sampling_columns[1].metric("Train Sesudah", f"{int(oversampling_summary['after_records'].sum()):,}")
    sampling_columns[2].metric("Tambahan Record", f"{int(oversampling_summary['added_records'].sum()):,}")
    high_added = int(oversampling_summary.loc[oversampling_summary["category"] == "Tinggi", "added_records"].sum())
    sampling_columns[3].metric("Tambahan Kelas Tinggi", f"{high_added:,}")
    sampling_long = oversampling_summary.melt(
        id_vars="category", value_vars=["before_records", "after_records"],
        var_name="stage", value_name="records",
    )
    sampling_long["stage"] = sampling_long["stage"].map({
        "before_records": "Sebelum oversampling",
        "after_records": "Sesudah oversampling",
    })
    sampling_fig = px.bar(
        sampling_long, x="category", y="records", color="stage", barmode="group",
        category_orders={"category": LABEL_ORDER}, title="Distribusi Kategori Target pada Training",
        labels={"category": "Kategori target musim berikutnya", "records": "Jumlah record", "stage": "Tahap"},
    )
    render_chart(sampling_fig)
    st.caption("Oversampling hanya diterapkan pada training. Validation dan test tetap memakai distribusi asli.")

    left, right = st.columns(2)
    with left:
        comparison = metrics_data.copy()
        scenario_label = comparison.get("scenario", pd.Series("baseline", index=comparison.index)).fillna("baseline")
        comparison["model_label"] = comparison["model"].str.replace("_", " ").str.title() + " | " + scenario_label
        fig = px.bar(
            comparison, x="mae", y="model_label", color="split", barmode="group",
            orientation="h", title="Perbandingan MAE Model",
            labels={"mae": "MAE (EUR juta)", "model_label": "Model", "split": "Split"},
        )
        render_chart(fig)
    with right:
        fig = px.scatter(
            test_predictions, x="next_season_market_value_mio", y="predicted_market_value_mio",
            hover_data=["player_name", "club"], opacity=0.6,
            title="Nilai Aktual vs Prediksi pada Test 2024",
            labels={
                "next_season_market_value_mio": "Nilai aktual (EUR juta)",
                "predicted_market_value_mio": "Nilai prediksi (EUR juta)",
            },
        )
        maximum = max(test_predictions["next_season_market_value_mio"].max(), test_predictions["predicted_market_value_mio"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum, line=dict(color="#dc2626", dash="dash"))
        render_chart(fig)

    preview = test_predictions.sort_values("absolute_error_mio").head(20).copy()
    preview = preview.rename(columns={
        "player_name": "Pemain", "club": "Klub",
        "current_market_value_mio": "Nilai 2023",
        "next_season_market_value_mio": "Aktual 2024",
        "predicted_market_value_mio": "Prediksi 2024",
        "absolute_error_mio": "Selisih absolut",
        "actual_category": "Kategori aktual", "predicted_category": "Kategori prediksi",
    })
    render_table(preview[["Pemain", "Klub", "Nilai 2023", "Aktual 2024", "Prediksi 2024", "Selisih absolut", "Kategori aktual", "Kategori prediksi"]])

with prediction_tab:
    st.subheader("Prediksi Nilai Pasar 2025")
    st.caption(
        "Pilih pemain dengan data musim 2024. Anda dapat mengubah beberapa nilai untuk melakukan simulasi. "
        "Model akan memperkirakan nilai pasar musim 2025."
    )
    candidates = forecast_data[forecast_data["season"] == 2024].copy()
    search = st.text_input("Cari pemain, klub, atau liga")
    if search.strip():
        needle = search.strip().lower()
        mask = pd.Series(False, index=candidates.index)
        for column in ["player_name", "club", "league"]:
            mask |= candidates[column].astype(str).str.lower().str.contains(needle, regex=False, na=False)
        candidates = candidates[mask]
    candidates = candidates.sort_values(["current_market_value_mio", "player_name"], ascending=[False, True])
    if candidates.empty:
        st.warning("Tidak ada pemain yang cocok dengan pencarian.")
    else:
        option_map = {
            int(index): f"{row.player_name} | {row.club} | {format_money(row.current_market_value_mio)}"
            for index, row in candidates.iterrows()
        }
        selected_index = st.selectbox("Pilih pemain", list(option_map), format_func=lambda value: option_map[value])
        selected = forecast_data.loc[selected_index].copy()

        with st.form("simulation_form"):
            first, second, third = st.columns(3)
            age = first.number_input("Umur pada musim 2024", min_value=15.0, max_value=50.0, value=float(selected["age"]), step=1.0)
            current_value = second.number_input(
                "Nilai pasar 2024 (EUR juta)", min_value=0.1, max_value=250.0,
                value=float(selected["current_market_value_mio"]), step=0.5,
            )
            minutes = third.number_input("Menit bermain", min_value=0.0, max_value=6000.0, value=float(selected["minutes"]), step=90.0)
            goals = first.number_input("Gol", min_value=0.0, max_value=100.0, value=float(selected["goals"]), step=1.0)
            assists = second.number_input("Assist", min_value=0.0, max_value=100.0, value=float(selected["assists"]), step=1.0)
            shots_on_target = third.number_input(
                "Tembakan tepat sasaran", min_value=0.0, max_value=500.0,
                value=float(selected["shots_on_target"]), step=1.0,
            )
            submitted = st.form_submit_button("Hitung Prediksi 2025", type="primary")

        if submitted:
            simulation = selected[feature_list].to_frame().T.copy()
            simulation["age"] = age
            simulation["current_market_value_mio"] = current_value
            simulation["current_market_value_log"] = np.log1p(current_value)
            simulation["current_market_value_category"] = category(current_value)
            simulation["minutes"] = minutes
            simulation["goals"] = goals
            simulation["assists"] = assists
            simulation["shots_on_target"] = shots_on_target
            previous = float(simulation["previous_market_value_mio"].iloc[0])
            simulation["historical_growth_rate"] = np.clip((current_value - previous) / previous, -5, 5) if previous > 0 else 0
            club_value = float(simulation["club_total_mv_mio"].iloc[0])
            simulation["current_value_to_club_ratio"] = current_value / club_value if club_value > 0 else 0
            predicted_log_growth = float(model.predict(simulation[feature_list])[0])
            predicted_value = current_value * float(np.exp(predicted_log_growth))
            predicted_category = category(predicted_value)
            change = predicted_value - current_value
            change_pct = change / current_value if current_value else 0

            result_columns = st.columns(5)
            result_columns[0].metric("Prediksi Nilai 2025", format_money(predicted_value))
            result_columns[1].metric("Kategori", predicted_category)
            result_columns[2].metric("Perubahan", format_money(change), delta=format_percent(change_pct))
            result_columns[3].metric("Nilai Saat Ini", format_money(current_value))
            result_columns[4].metric("Rata-rata Error Test", format_money(model_summary["test_mae_mio"]))
            st.info(
                f"Interpretasi: prediksi {format_money(predicted_value)} termasuk kategori {predicted_category}. "
                "Rendah berada di bawah EUR 15 juta, Menengah EUR 15 sampai 35 juta, dan Tinggi di atas EUR 35 juta."
            )

    st.divider()
    st.subheader("Daftar Forecast 2025")
    forecast_display = forecast_2025.copy().rename(columns={
        "player_name": "Pemain", "club": "Klub", "league": "Liga", "pos_category": "Posisi",
        "current_market_value_mio": "Nilai 2024", "predicted_market_value_mio": "Prediksi 2025",
        "predicted_change_mio": "Perubahan", "predicted_change_pct": "Perubahan (%)",
        "predicted_category": "Kategori prediksi",
    })
    render_table(forecast_display[["Pemain", "Klub", "Liga", "Posisi", "Nilai 2024", "Prediksi 2025", "Perubahan", "Perubahan (%)", "Kategori prediksi"]].head(100))

with importance_tab:
    st.subheader("Faktor yang Paling Memengaruhi Prediksi")
    st.caption(
        "Nama fitur telah diterjemahkan ke bahasa yang mudah dipahami. Nilai dihitung dengan permutation importance: "
        "semakin besar penurunan performa ketika suatu fitur diacak, semakin penting fitur tersebut secara global."
    )
    top_n = st.slider("Jumlah faktor", 5, min(25, len(feature_importance)), 15)
    importance_view = feature_importance.head(top_n).sort_values("importance")
    fig = px.bar(
        importance_view, x="importance", y="feature_label", orientation="h",
        color="importance", color_continuous_scale="Teal",
        title=f"{top_n} Faktor Utama Prediksi Nilai Pasar",
        labels={"importance": "Dampak terhadap MAE", "feature_label": "Faktor"},
    )
    render_chart(fig)
    table = feature_importance.head(top_n)[["feature_label", "importance", "importance_share"]].copy()
    table.columns = ["Faktor", "Dampak terhadap MAE", "Porsi kepentingan"]
    table["Porsi kepentingan"] = table["Porsi kepentingan"].map(format_percent)
    render_table(table)

with explorer_tab:
    st.subheader("Data Explorer")
    explorer = filtered[[
        "player_name", "club", "league", "season", "pos_category", "age",
        "current_market_value_mio", "next_season_market_value_mio",
        "minutes", "goals", "assists", "has_performance_stats",
    ]].copy()
    explorer = explorer.rename(columns={
        "player_name": "Pemain", "club": "Klub", "league": "Liga", "season": "Musim fitur",
        "pos_category": "Posisi", "age": "Umur", "current_market_value_mio": "Nilai musim ini",
        "next_season_market_value_mio": "Nilai musim berikutnya", "minutes": "Menit",
        "goals": "Gol", "assists": "Assist", "has_performance_stats": "Statistik FBref tersedia",
    })
    render_table(explorer.sort_values(["Musim fitur", "Nilai musim ini"], ascending=[False, False]))

    st.divider()
    st.subheader("Dataset Training Before dan After Oversampling")
    training_options = {
        "Sebelum oversampling - semua kategori": TRAIN_BEFORE_PATH,
        "Sesudah oversampling - semua kategori": TRAIN_AFTER_PATH,
        "Sebelum oversampling - kelas Tinggi": HIGH_BEFORE_PATH,
        "Sesudah oversampling - kelas Tinggi": HIGH_AFTER_PATH,
    }
    selected_training_file = st.selectbox("Pilih dataset training", list(training_options))
    training_table = load_csv(training_options[selected_training_file])
    training_columns = [
        "_oversampling_status", "_is_oversampled_duplicate", "_source_row_id",
        "player_name", "club", "season", "target_season",
        "next_season_market_value_category", "current_market_value_mio",
        "next_season_market_value_mio", "age", "league", "pos_category",
        "minutes", "goals", "assists",
    ]
    available_training_columns = [column for column in training_columns if column in training_table.columns]
    duplicate_count = int(training_table.get("_is_oversampled_duplicate", pd.Series(dtype=bool)).sum())
    training_metrics = st.columns(3)
    training_metrics[0].metric("Rows", f"{len(training_table):,}")
    training_metrics[1].metric("Tambahan Oversampling", f"{duplicate_count:,}")
    training_metrics[2].metric("Kolom", f"{len(training_table.columns):,}")
    render_table(training_table[available_training_columns])
