import numpy as np
from bokeh.io import output_notebook, show
from bokeh.models import Legend, LegendItem, LinearAxis, Range1d
from bokeh.plotting import figure

output_notebook()
palette = ["#404B9F", "#419d78", "#e0a458", "#ffdbb5", "#c04abc"]


def get_figure(**kwargs):
    defaults = {
        "width": 750,
        "height": 400,
        "tools": "",
        "background_fill_color": "#efefef"
    }
    defaults.update(kwargs)
    p = figure(**defaults)
    p.grid.grid_line_color = "#ffffff"
    p.grid.grid_line_width = 1.5
    p.toolbar.autohide = True
    return p


def plot_timeseries(y, inference_sampling_rate: float, kernel_length: float):
    p = get_figure(
        title="Network response vs. time",
        x_axis_label="Time from start [s]",
        y_axis_label="NN Output"
    )
    times = np.arange(len(y)) / inference_sampling_rate
    times += kernel_length  # to indicate where the right edge of the kernel is

    p.line(times, y, line_width=2.0)
    return p


def plot_inference_metrics_vs_time(**dfs):
    time_col = "Time since start (s)"
    latency_col = "Average queue time (us)"
    throughput_col = "Throughput (s' / s)"

    max_latency = max([i[latency_col].max() for i in dfs.values()])
    p = get_figure(
        title="Inference metrics vs. time",
        x_axis_label="Time from start of monitoring [s]",
        y_axis_label="Average queue latency in interval [us]",
        y_range=(-0.01 * max_latency, 1.1 * max_latency),
        width=850 if len(dfs) > 1 else 750
    )

    # add a secondary y axis for throughput
    max_throughput = max([i[throughput_col].max() for i in dfs.values()])
    p.extra_y_ranges = {"throughput": Range1d(-0.01 * max_throughput, 1.1 * max_throughput)}
    throughput_axis = LinearAxis(y_range_name="throughput", axis_label=throughput_col)
    p.add_layout(throughput_axis, "right")

    legend_items = []
    for color, (model_name, df) in zip(palette, dfs.items()):
        if df[time_col].diff().iloc[1] > 1:
            df = df.iloc[1:]

        r1 = p.line(
            df[time_col],
            df[latency_col],
            line_color=color,
            line_width=2.0
        )
        r2 = p.line(
            df[time_col],
            df[throughput_col],
            line_color=color,
            line_width=2.0,
            line_dash="2 2",
            y_range_name="throughput"
        )

        rs = [r1, r2]
        labels = ["Queue time", "Throughput"]
        if len(dfs) > 1:
            model_name = model_name.replace("-", " ").title()
            labels = [f"{model_name} {i.lower()}" for i in labels]

        items = [LegendItem(label=l, renderers=[r]) for l, r in zip(labels, rs)]
        legend_items.extend(items)

    if len(dfs) > 1:
        legend = Legend(items=legend_items)
        p.add_layout(legend, "right")
    else:
        legend = Legend(items=legend_items, location="top_right")
        p.add_layout(legend)
    return p
