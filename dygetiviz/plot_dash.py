"""
Plot using [Dash](https://dash.plotly.com/)
"""

import os.path as osp
import random

import dash
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output

import const
from arguments import args
from data.dataloader import load_data
from utils.utils_misc import project_setup

project_setup()

data = load_data()

annotation = data.get("annotation", {})
idx_reference_snapshot = data["idx_reference_snapshot"]
interpolation = data["interpolation"]
node_presence = data["node_presence"]
node2idx = data["node2idx"]
node2label = data["node2label"]
num_nearest_neighbors = data["num_nearest_neighbors"]
perplexity = data["perplexity"]
plot_anomaly_labels = data["plot_anomaly_labels"]
projected_nodes = data["projected_nodes"]
reference_nodes = data["reference_nodes"]
snapshot_names = data["snapshot_names"]
z = data["z"]

idx2node = {idx: node for node, idx in node2idx.items()}

nn = 5

visualization_name = f"{args.dataset_name}_{args.model}_{args.visualization_model}_perplex{perplexity}_nn{nn}_interpolation{interpolation}_snapshot{idx_reference_snapshot}"

fig_cached = pio.read_json(
    osp.join(args.visual_dir, f"Trajectory_{visualization_name}.json"))

node2trace = {
    trace['name']: trace for trace in fig_cached.data
}
nodes = list(node2trace.keys())

# Create a mapping from types to lines
types_to_lines = {node: "" for node in node2label.values()}
# soptions = dict(zip(list(node2idx.keys()), list(node2idx.keys())))

options = []

for node, idx in node2idx.items():
    options.append({
        "label": html.Span(
            [
                "✨",
                html.Span(node, style={
                    'font-size': 15,
                    'padding-left': 10
                }),
            ], style={
                'align-items': 'center',
                'justify-content': 'center'
            }
        ),
        "value": node, # Get a random node as the default value.
    })

for label in set(node2label.values()):
    options.append({
        "label": html.Span(
            [
                "✨",
                html.Span(label, style={
                    'font-size': 15,
                    'padding-left': 10
                }),
            ], style={
                'align-items': 'center',
                'justify-content': 'center'
            }
        ),
        "value": label,
    })

# Start the app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='search-input',
        options=options,
        value='',
        multi=True,
        placeholder="Select a node",
        style={
            'width': '50%'
        },
        clearable=True
    ),
    dcc.Graph(id='dygetviz')
])


@app.callback(
    Output('dygetviz', 'figure'),
    Input('search-input', 'value')
)
def update_graph(search_values):
    fig = go.Figure()
    fig.add_trace(node2trace['background'])

    if search_values:
        for search_value in search_values:
            if search_value in nodes:  # search_value is a line name
                fig.add_trace(node2trace[search_value])

            # if search_value in lines:  # search_value is a line name
            #     fig.add_trace(
            #         go.Scatter(x=x, y=lines[search_value]['data'], mode='lines',
            #                    name=search_value))
            # elif search_value in types_to_lines:  # search_value is a type
            #     for line_name in types_to_lines[search_value]:
            #         fig.add_trace(go.Scatter(x=x, y=lines[line_name]['data'],
            #                                  mode='lines', name=line_name))

    return fig


if __name__ == "__main__":
    print(const.DYGETVIZ)
    app.run_server(debug=True, port=8050)
