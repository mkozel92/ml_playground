import plotly.graph_objs as go
import numpy as np

from visualization.colors import BLUE_SCHEME, PINK_SCHEME


def get_categorical_data_scatter(data: np.array, labels: np.array, name: str, highlight: str = None,
                                 color_scheme: list = BLUE_SCHEME) -> dict:
    np.unique(labels)
    traces = []

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        label_data = data[mask]

        if label == highlight:
            opacity = 1
            size = 12
        else:
            opacity = 0.8
            size = 10

        trace = go.Scatter(
            x=label_data[:, 0],
            y=label_data[:, 1],
            name='%d' % label,
            mode='markers',
            marker=dict(
                size=size,
                color=color_scheme[i],
                opacity=opacity,
                line=dict(
                    width=0,
                )
            )
        )
        traces.append(trace)

    layout = dict(title=name,
                  yaxis=dict(zeroline=False),
                  xaxis=dict(zeroline=False)
                  )
    return {'layout': layout, 'data': traces}


def get_target_histogram(targets: np.array, name: str, color_scheme: list = BLUE_SCHEME) -> dict:
    traces = [go.Histogram(
        x=targets,
        marker=dict(color=color_scheme[1])
    )]
    layout = dict(title=name)
    return {'layout': layout, 'data': traces}

