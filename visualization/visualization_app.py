import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html

from data.data_generation import get_categorical_gaussian_data
from data.data_normalization import feature_rescaling, feature_standardization
from data.dimensionality_reduction import pca, tsne
from visualization.categorical_data_plots import get_categorical_data_scatter, get_target_histogram

app = dash.Dash()

centers = np.array([[1, 1, 1], [5, 5, 6], [10, 10, 7], [20, 20, 30], [5, 6, 7]])
sigmas = np.array([1, 1, 1, 1, 5])
counts = np.array([10, 20, 30, 50, 10])

data, labels = get_categorical_gaussian_data(centers=centers, sigmas=sigmas, counts=counts)

data = feature_standardization(data)
data = tsne(data, 2)

fig_0 = get_categorical_data_scatter(data=data, labels=labels, name='sample data')
fig_1 = get_target_histogram(targets=labels, name='targets histogram')


class LastState(object):

    def __init__(self):
        self.last = None
        self.processed_data = None

    def set_last(self, last: str):
        self.last = last

    def get_last(self) -> str:
        return self.last

    def set_processed_data(self, p_data: np.array):
        self.processed_data = p_data

    def get_processed_data(self) -> np.array:
        return self.processed_data


ls = LastState()
ls.set_processed_data(data)

app.layout = html.Div(
    [html.Div([html.H1("Data dashboard")], style={'textAlign': "center", 'padding': 10}),
     html.Div([dcc.Dropdown(id='drop-down',
                            options=[
                                {'label': 'PCA', 'value': 'pca'},
                                {'label': 't-SNE', 'value': 'tsne'}])]),
     html.Div([html.Div([dcc.Graph(id="scatter-graph",
                                   figure=fig_0)],
                        className="six columns",
                        style={'width': '49%', 'display': 'inline-block'}
                        ),
               html.Div([dcc.Graph(id="hist-graph",
                                   figure=fig_1,
                                   clear_on_unhover=True, )],
                        className="six columns",
                        style={'width': '49%', 'display': 'inline-block'}),
               ]),
     ],
    className="container")


@app.callback(
    dash.dependencies.Output("scatter-graph", "figure"),
    [dash.dependencies.Input("hist-graph", "hoverData"), dash.dependencies.Input("drop-down", 'value')])
def update_scatter(hoverData, dropData):
    if hoverData is not None:
        label = hoverData["points"][0]['x']
        if dropData != ls.get_last():
            if dropData == 'tsne':
                processed_data = tsne(data, 2)
            elif dropData == 'pca':
                processed_data = pca(data, 2)
            else:
                processed_data = data
            ls.set_last(dropData)
            ls.set_processed_data(processed_data)

        return get_categorical_data_scatter(data=ls.get_processed_data(), labels=labels,
                                            name='sample data scatter', highlight=label)
    else:
        return get_categorical_data_scatter(data=ls.get_processed_data(), labels=labels, name='sample data scatter')


if __name__ == '__main__':
    app.run_server()
