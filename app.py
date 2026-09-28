from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget

from data_utils import (
    MAIN_INDICES,
    build_summary_metrics,
    drawdown_series,
    format_number,
    format_percent,
    format_summary_table,
    load_index_data,
    make_sample_data,
    monthly_return_table,
)


APP_DIR = Path(__file__).parent
DEFAULT_DATA_PATH = APP_DIR / "data" / "NHIndexMonthly.csv"
NAVY = "#174A63"
ORANGE = "#FF4B00"
GREEN = "#1F8A5B"
RED = "#B42318"
MUTED = "#667085"
GRID = "#E5E9ED"
PLOTLY_COLORS = [
    NAVY,
    ORANGE,
    GREEN,
    "#7A4A9E",
    "#D09A2D",
    "#477FB7",
    "#9C3B3B",
    "#4D8D91",
]


def load_default_data() -> tuple[pd.DataFrame, str]:
    if DEFAULT_DATA_PATH.exists():
        return load_index_data(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH.name
    return make_sample_data(), "Demo data"


INITIAL_DATA, INITIAL_SOURCE = load_default_data()
INITIAL_INDICES = [name for name in MAIN_INDICES if name in INITIAL_DATA["type"].unique()]
if not INITIAL_INDICES:
    INITIAL_INDICES = sorted(INITIAL_DATA["type"].unique())[:5]


def chart_layout(fig: go.Figure, *, height: int = 410, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=30, b=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color="#1F2933", size=12),
        colorway=PLOTLY_COLORS,
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=MUTED, size=14),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return chart_layout(fig, legend=False)


sidebar = ui.sidebar(
    ui.div(
        ui.div("N", class_="brand-mark"),
        ui.div(
            ui.span("NILSSON", class_="brand-dark"),
            ui.span("HEDGE", class_="brand-orange"),
            ui.div("INDEX DASHBOARD", class_="brand-subtitle"),
        ),
        class_="brand-lockup",
    ),
    ui.hr(),
    ui.input_file(
        "data_file",
        "Monthly index CSV",
        accept=[".csv", "text/csv"],
        button_label="Upload CSV",
        placeholder="Using bundled data",
    ),
    ui.output_ui("source_status"),
    ui.input_selectize(
        "indices",
        "Compare indices",
        choices=sorted(INITIAL_DATA["type"].unique()),
        selected=INITIAL_INDICES,
        multiple=True,
        remove_button=True,
    ),
    ui.input_date_range(
        "date_range",
        "Date range",
        start=INITIAL_DATA["date"].min().date(),
        end=INITIAL_DATA["date"].max().date(),
        min=INITIAL_DATA["date"].min().date(),
        max=INITIAL_DATA["date"].max().date(),
    ),
    ui.input_select(
        "detail_index",
        "Index detail",
        choices=sorted(INITIAL_DATA["type"].unique()),
        selected=INITIAL_INDICES[0],
    ),
    ui.hr(),
    ui.download_button("download_filtered", "Download filtered data", class_="btn-download"),
    ui.p(
        "Upload a CSV with date, ror, type, cnt and aumbn columns. Common alternative names are accepted.",
        class_="sidebar-help",
    ),
    width=310,
    bg="#F6F8FA",
)


app_ui = ui.page_sidebar(
    sidebar,
    ui.include_css(APP_DIR / "www" / "styles.css"),
    ui.div(
        ui.div(
            ui.h2("NilssonHedge Index Dashboard"),
            ui.p("Interactive performance, risk and reporting analytics"),
        ),
        ui.div(ui.output_text("as_of_label"), class_="as-of"),
        class_="dashboard-header",
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Overview",
            ui.layout_columns(
                ui.value_box("Best YTD", ui.output_text("best_ytd"), theme="success"),
                ui.value_box("Weakest YTD", ui.output_text("weakest_ytd"), theme="danger"),
                ui.value_box("Selected indices", ui.output_text("selected_count"), theme="primary"),
                ui.value_box("Managers reporting", ui.output_text("manager_total"), theme="primary"),
                col_widths=[3, 3, 3, 3],
                class_="metric-row",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Cumulative Performance"),
                    output_widget("cumulative_chart"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Risk / Return Map"),
                    output_widget("risk_return_chart"),
                    full_screen=True,
                ),
                col_widths=[7, 5],
            ),
            ui.card(
                ui.card_header("Index Summary"),
                ui.output_data_frame("summary_table"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "Index Detail",
            ui.layout_columns(
                ui.value_box("Latest month", ui.output_text("detail_latest"), theme="primary"),
                ui.value_box("YTD", ui.output_text("detail_ytd"), theme="primary"),
                ui.value_box("Annualized return", ui.output_text("detail_ann_return"), theme="primary"),
                ui.value_box("Annualized volatility", ui.output_text("detail_ann_vol"), theme="primary"),
                ui.value_box("Maximum drawdown", ui.output_text("detail_max_dd"), theme="danger"),
                ui.value_box("Managers", ui.output_text("detail_managers"), theme="primary"),
                col_widths=[2, 2, 2, 2, 2, 2],
                class_="metric-row detail-metrics",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Cumulative Performance"),
                    output_widget("detail_performance_chart"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Drawdown"),
                    output_widget("drawdown_chart"),
                    full_screen=True,
                ),
                col_widths=[7, 5],
            ),
            ui.card(
                ui.card_header("Monthly Returns"),
                ui.output_data_frame("monthly_table"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "Correlation",
            ui.layout_columns(
                ui.card(
                    ui.card_header("Return Correlation Matrix"),
                    output_widget("correlation_heatmap"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Closest and Lowest Correlations"),
                    ui.output_data_frame("correlation_pairs"),
                    full_screen=True,
                ),
                col_widths=[7, 5],
            ),
        ),
        ui.nav_panel(
            "Reporting",
            ui.layout_columns(
                ui.card(
                    ui.card_header("Manager Reporting"),
                    output_widget("manager_reporting_chart"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Assets Under Management"),
                    output_widget("aum_chart"),
                    full_screen=True,
                ),
                col_widths=[6, 6],
            ),
            ui.card(
                ui.card_header("Reporting Coverage Snapshot"),
                ui.output_data_frame("reporting_table"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "Data",
            ui.card(
                ui.card_header("Filtered Monthly Index Data"),
                ui.output_data_frame("raw_data_table"),
                full_screen=True,
            ),
        ),
        id="main_nav",
    ),
    title="NilssonHedge Index Dashboard",
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    data_state = reactive.value(INITIAL_DATA)
    source_state = reactive.value(INITIAL_SOURCE)

    @reactive.effect
    @reactive.event(input.data_file)
    def load_uploaded_file():
        upload = input.data_file()
        if not upload:
            return
        try:
            uploaded = upload[0]
            loaded = load_index_data(uploaded["datapath"])
            if loaded.empty:
                raise ValueError("The uploaded CSV contains no valid rows.")
            data_state.set(loaded)
            source_state.set(uploaded["name"])
            ui.notification_show(
                f"Loaded {len(loaded):,} rows from {uploaded['name']}",
                type="message",
                duration=5,
            )
        except Exception as exc:
            ui.notification_show(
                f"Could not load the CSV: {exc}",
                type="error",
                duration=10,
            )

    @reactive.effect
    def refresh_filter_choices():
        df = data_state()
        available = sorted(df["type"].unique())
        preferred = [name for name in MAIN_INDICES if name in available]
        selected = preferred or available[:5]
        ui.update_selectize("indices", choices=available, selected=selected, server=True)
        ui.update_select("detail_index", choices=available, selected=selected[0])
        ui.update_date_range(
            "date_range",
            start=df["date"].min().date(),
            end=df["date"].max().date(),
            min=df["date"].min().date(),
            max=df["date"].max().date(),
        )

    @reactive.calc
    def filtered_data() -> pd.DataFrame:
        df = data_state().copy()
        selected = list(input.indices() or [])
        if selected:
            df = df[df["type"].isin(selected)]
        date_range = input.date_range()
        if date_range and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.sort_values(["type", "date"])

    @reactive.calc
    def summary() -> pd.DataFrame:
        return build_summary_metrics(filtered_data())

    @reactive.calc
    def detail_group() -> pd.DataFrame:
        df = data_state().copy()
        selected = input.detail_index()
        if selected:
            df = df[df["type"] == selected]
        date_range = input.date_range()
        if date_range and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.sort_values("date")

    @reactive.calc
    def detail_metrics() -> pd.Series | None:
        group = detail_group()
        if group.empty:
            return None
        result = build_summary_metrics(group)
        return result.iloc[0] if not result.empty else None

    @render.ui
    def source_status():
        is_demo = source_state() == "Demo data"
        return ui.div(
            ui.span("●", class_="status-dot demo" if is_demo else "status-dot live"),
            ui.span(source_state()),
            class_="source-status",
        )

    @render.text
    def as_of_label():
        df = filtered_data()
        if df.empty:
            return "No data in selected range"
        return f"As of {df['date'].max():%B %Y}"

    @render.text
    def best_ytd():
        data = summary()
        if data.empty:
            return "—"
        row = data.loc[data["YTD"].idxmax()]
        return f"{row['Index']}  {format_percent(row['YTD'])}"

    @render.text
    def weakest_ytd():
        data = summary()
        if data.empty:
            return "—"
        row = data.loc[data["YTD"].idxmin()]
        return f"{row['Index']}  {format_percent(row['YTD'])}"

    @render.text
    def selected_count():
        return format_number(filtered_data()["type"].nunique())

    @render.text
    def manager_total():
        data = summary()
        return format_number(data["Managers"].sum()) if not data.empty else "—"

    @render_widget
    def cumulative_chart():
        df = filtered_data()
        if df.empty:
            return empty_figure("No data in the selected range")
        frames = []
        for index_name, group in df.groupby("type"):
            group = group.sort_values("date").copy()
            group["Cumulative index"] = (1 + group["ror"]).cumprod() * 100
            group["Index"] = index_name
            frames.append(group[["date", "Index", "Cumulative index"]])
        plot_data = pd.concat(frames, ignore_index=True)
        fig = px.line(
            plot_data,
            x="date",
            y="Cumulative index",
            color="Index",
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_traces(line=dict(width=2.2))
        fig.update_yaxes(title="Growth of 100")
        fig.update_xaxes(title=None)
        return chart_layout(fig)

    @render_widget
    def risk_return_chart():
        data = summary().dropna(subset=["Ann. return", "Ann. volatility"])
        if data.empty:
            return empty_figure("Not enough history to calculate risk and return")
        plot_data = data.copy()
        plot_data["AUM size"] = plot_data["AUM ($bn)"].fillna(1).clip(lower=1)
        fig = px.scatter(
            plot_data,
            x="Ann. volatility",
            y="Ann. return",
            text="Index",
            size="AUM size",
            color="YTD",
            color_continuous_scale=[RED, "#F5F5F5", GREEN],
            hover_data={
                "Ann. volatility": ":.1%",
                "Ann. return": ":.1%",
                "YTD": ":.1%",
                "AUM size": ":.1f",
            },
        )
        fig.update_traces(textposition="top center", marker=dict(line=dict(color="white", width=1)))
        fig.update_xaxes(title="Annualized volatility", tickformat=".0%")
        fig.update_yaxes(title="Annualized return", tickformat=".0%")
        fig.update_layout(coloraxis_colorbar=dict(title="YTD", tickformat=".0%"))
        return chart_layout(fig, legend=False)

    @render.data_frame
    def summary_table():
        data = format_summary_table(summary())
        visible_columns = [
            "YTD rank",
            "Index",
            "Latest",
            "YTD",
            "12M",
            "Ann. return",
            "Ann. volatility",
            "Return / risk",
            "Max drawdown",
            "Managers",
            "AUM ($bn)",
        ]
        return render.DataGrid(data[visible_columns], filters=True, height="360px")

    @render.text
    def detail_latest():
        metrics = detail_metrics()
        return format_percent(metrics["Latest"]) if metrics is not None else "—"

    @render.text
    def detail_ytd():
        metrics = detail_metrics()
        return format_percent(metrics["YTD"]) if metrics is not None else "—"

    @render.text
    def detail_ann_return():
        metrics = detail_metrics()
        return format_percent(metrics["Ann. return"]) if metrics is not None else "—"

    @render.text
    def detail_ann_vol():
        metrics = detail_metrics()
        return format_percent(metrics["Ann. volatility"]) if metrics is not None else "—"

    @render.text
    def detail_max_dd():
        metrics = detail_metrics()
        return format_percent(metrics["Max drawdown"]) if metrics is not None else "—"

    @render.text
    def detail_managers():
        metrics = detail_metrics()
        return format_number(metrics["Managers"]) if metrics is not None else "—"

    @render_widget
    def detail_performance_chart():
        group = detail_group().copy()
        if group.empty:
            return empty_figure("No data in the selected range")
        group["Cumulative index"] = (1 + group["ror"]).cumprod() * 100
        fig = px.line(group, x="date", y="Cumulative index")
        fig.update_traces(line=dict(color=NAVY, width=2.6), fill="tozeroy", fillcolor="rgba(23,74,99,0.08)")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Growth of 100")
        return chart_layout(fig, legend=False)

    @render_widget
    def drawdown_chart():
        group = detail_group().copy()
        if group.empty:
            return empty_figure("No data in the selected range")
        group["Drawdown"] = drawdown_series(group["ror"])
        fig = px.area(group, x="date", y="Drawdown")
        fig.update_traces(line=dict(color=RED, width=1.8), fillcolor="rgba(180,35,24,0.18)")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Drawdown", tickformat=".0%")
        return chart_layout(fig, legend=False)

    @render.data_frame
    def monthly_table():
        return render.DataGrid(monthly_return_table(detail_group()), height="330px")

    @reactive.calc
    def correlation_matrix() -> pd.DataFrame:
        df = filtered_data()
        if df.empty:
            return pd.DataFrame()
        returns = df.pivot_table(index="date", columns="type", values="ror").sort_index()
        return returns.corr(min_periods=max(4, min(12, len(returns)) // 2))

    @render_widget
    def correlation_heatmap():
        corr = correlation_matrix()
        if corr.shape[0] < 2:
            return empty_figure("Select at least two indices")
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale=[[0, RED], [0.5, "#FFFFFF"], [1, NAVY]],
                text=np.round(corr.values, 2),
                texttemplate="%{text:.2f}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(hovermode="closest")
        return chart_layout(fig, height=510, legend=False)

    @render.data_frame
    def correlation_pairs():
        corr = correlation_matrix()
        if corr.shape[0] < 2:
            return render.DataGrid(pd.DataFrame(columns=["Index", "Relationship", "Other index", "Correlation"]))
        rows = []
        for index_name in corr.columns:
            values = corr[index_name].drop(index_name, errors="ignore").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "Index": index_name,
                    "Relationship": "Closest",
                    "Other index": values.idxmax(),
                    "Correlation": round(float(values.max()), 2),
                }
            )
            rows.append(
                {
                    "Index": index_name,
                    "Relationship": "Lowest",
                    "Other index": values.idxmin(),
                    "Correlation": round(float(values.min()), 2),
                }
            )
        return render.DataGrid(pd.DataFrame(rows), filters=True, height="430px")

    @render_widget
    def manager_reporting_chart():
        df = filtered_data().dropna(subset=["cnt"])
        if df.empty:
            return empty_figure("Manager-count data is not available")
        fig = px.line(
            df,
            x="date",
            y="cnt",
            color="type",
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_traces(line=dict(width=2))
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Managers reporting")
        return chart_layout(fig)

    @render_widget
    def aum_chart():
        df = filtered_data().dropna(subset=["aumbn"])
        if df.empty:
            return empty_figure("AUM data is not available")
        fig = px.line(
            df,
            x="date",
            y="aumbn",
            color="type",
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_traces(line=dict(width=2))
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="AUM ($bn)")
        return chart_layout(fig)

    @render.data_frame
    def reporting_table():
        data = summary()
        if data.empty:
            return render.DataGrid(pd.DataFrame())
        display = data[["Index", "Managers", "Manager change", "AUM ($bn)", "AUM change"]].copy()
        display["Managers"] = display["Managers"].map(format_number)
        display["Manager change"] = display["Manager change"].map(format_percent)
        display["AUM ($bn)"] = display["AUM ($bn)"].map(lambda value: format_number(value, 1))
        display["AUM change"] = display["AUM change"].map(format_percent)
        return render.DataGrid(display, filters=True, height="360px")

    @render.data_frame
    def raw_data_table():
        data = filtered_data()[["date", "type", "ror", "cnt", "aumbn"]].copy()
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")
        data["ror"] = data["ror"].map(lambda value: round(value, 6))
        data = data.rename(
            columns={
                "date": "Date",
                "type": "Index",
                "ror": "Monthly return",
                "cnt": "Managers",
                "aumbn": "AUM ($bn)",
            }
        )
        return render.DataGrid(data, filters=True, height="620px")

    @render.download(filename=lambda: f"nilssonhedge_filtered_{pd.Timestamp.today():%Y%m%d}.csv")
    def download_filtered():
        export = filtered_data()[["date", "ror", "type", "cnt", "aumbn"]].copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        yield export.to_csv(index=False)


app = App(app_ui, server)


if __name__ == "__main__":
    # Allows app.py to be run directly from PyCharm.
    # For automatic reload while editing, use: shiny run --reload app.py
    app.run(host="127.0.0.1", port=8000)
