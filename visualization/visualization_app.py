import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html

from data.data_generation import get_categorical_gaussian_data
from data.data_normalization import feature_rescaling, feature_standardization
from visualization.categorical_data_plots import get_categorical_data_scatter, get_target_histogram

app = dash.Dash()

centers = np.array([[1, 1], [5, 5], [10, 10], [20, 20], [5,6]])
sigmas = np.array([1, 1, 1, 1, 5])
counts = np.array([10, 20, 30, 50, 10])

data, labels = get_categorical_gaussian_data(centers=centers, sigmas=sigmas, counts=counts)

data = feature_standardization(data)

fig_0 = get_categorical_data_scatter(data=data, labels=labels, name='sample data')
fig_1 = get_target_histogram(targets=labels, name='targets histogram')

app.layout = html.Div(
    [html.Div([html.H1("Data dashboard")], style={'textAlign': "center", 'padding': 10}),
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
    [dash.dependencies.Input("hist-graph", "hoverData")])
def update_scatter(hoverData):
    if hoverData is not None:
        label = hoverData["points"][0]['x']
        return get_categorical_data_scatter(data=data, labels=labels, name='sample data scatter', highlight=label)
    else:
        return get_categorical_data_scatter(data=data, labels=labels, name='sample data scatter')


if __name__ == '__main__':
    app.run_server()
