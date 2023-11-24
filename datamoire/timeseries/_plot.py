import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_seasonal_decompose(
    data,
    extrapolate_trend="freq",
    period=1,
    title="Time series components",
):
    decomposed = seasonal_decompose(
        x=data,
        extrapolate_trend=extrapolate_trend,
        period=period,
    )
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Series", "Trend", "Seasonal", "Residual"],
        )
        .add_trace(
            go.Scatter(x=data.index, y=decomposed.observed, mode="lines"),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=data.index, y=decomposed.trend, mode="lines"),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=data.index, y=decomposed.seasonal, mode="lines"),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=data.index, y=decomposed.resid, mode="lines"),
            row=4,
            col=1,
        )
        .update_layout(
            height=900,
            title=title,
            margin={"t": 100},
            title_x=0.5,
            showlegend=False,
        )
    )