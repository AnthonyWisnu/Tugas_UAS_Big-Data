import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


LABEL_ORDER = ["Rendah", "Menengah", "Tinggi"]


def label_distribution_chart(df):
    counts = (
        df["market_value_category"]
        .value_counts()
        .reindex(LABEL_ORDER)
        .fillna(0)
        .reset_index()
    )
    counts.columns = ["market_value_category", "count"]
    return px.bar(
        counts,
        x="market_value_category",
        y="count",
        color="market_value_category",
        category_orders={"market_value_category": LABEL_ORDER},
        title="Label Distribution",
    )


def records_by_league_chart(df):
    counts = df["league"].value_counts().reset_index()
    counts.columns = ["league", "count"]
    return px.bar(counts, x="league", y="count", title="Records by League")


def records_by_season_chart(df):
    counts = df.groupby("season", as_index=False).size()
    counts.columns = ["season", "count"]
    return px.line(counts, x="season", y="count", markers=True, title="Records by Season")


def average_market_value_by_season(df):
    data = df.groupby("season", as_index=False)["market_value_mio"].mean()
    return px.line(
        data,
        x="season",
        y="market_value_mio",
        markers=True,
        title="Average Market Value by Season",
        labels={"market_value_mio": "Average Market Value, EUR Mio"},
    )


def average_market_value_by_league(df):
    data = (
        df.groupby("league", as_index=False)["market_value_mio"]
        .mean()
        .sort_values("market_value_mio", ascending=False)
    )
    return px.bar(
        data,
        x="league",
        y="market_value_mio",
        title="Average Market Value by League",
        labels={"market_value_mio": "Average Market Value, EUR Mio"},
    )


def market_value_distribution_by_category(df):
    return px.box(
        df,
        x="market_value_category",
        y="market_value_mio",
        color="market_value_category",
        category_orders={"market_value_category": LABEL_ORDER},
        title="Market Value Distribution by Category",
        labels={"market_value_mio": "Market Value, EUR Mio"},
    )


def top_players_by_market_value(df, n=10):
    required = {"player_name", "club", "league", "season", "market_value_mio"}
    if not required <= set(df.columns):
        return pd.DataFrame()
    return (
        df.sort_values("market_value_mio", ascending=False)
        [["player_name", "club", "league", "season", "market_value_mio", "market_value_category"]]
        .head(n)
    )


def top_clubs_by_total_market_value(df, n=10):
    required = {"club", "league", "season", "club_total_mv_mio"}
    if not required <= set(df.columns):
        return pd.DataFrame()
    return (
        df.sort_values("club_total_mv_mio", ascending=False)
        .drop_duplicates(["club", "league", "season"])
        [["club", "league", "season", "club_total_mv_mio"]]
        .head(n)
    )


def market_value_by_position(df):
    data = (
        df.groupby("pos_category", as_index=False)["market_value_mio"]
        .mean()
        .sort_values("market_value_mio", ascending=False)
    )
    return px.bar(
        data,
        x="pos_category",
        y="market_value_mio",
        title="Average Market Value by Position",
        labels={"market_value_mio": "Average Market Value, EUR Mio"},
    )


def confusion_matrix_chart(confusion_df):
    pivot = confusion_df.pivot(index="actual", columns="predicted", values="count")
    pivot = pivot.reindex(index=LABEL_ORDER, columns=LABEL_ORDER).fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Blues",
            text=pivot.values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Confusion Matrix Best Model", xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def feature_importance_chart(feature_df, n=20):
    data = feature_df.head(n).iloc[::-1]
    return px.bar(
        data,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {n} Feature Importance",
    )
