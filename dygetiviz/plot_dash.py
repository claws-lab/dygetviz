"""
Plot using [Dash](https://dash.plotly.com/)
"""

import os.path as osp

import dash
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output

import const
import const_viz
from arguments import args
from data.dataloader import load_data
from utils.utils_misc import project_setup
from utils.utils_visual import get_colors

project_setup()


print("Loading data...")
data = load_data()

annotation = data.get("annotation", {})
display_node_type = data["display_node_type"]
idx_reference_snapshot = data["idx_reference_snapshot"]
interpolation = data["interpolation"]
node_presence = data["node_presence"]
node2idx = data["node2idx"]
node2label = data["node2label"]
label2node = data["label2node"]
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

print("Reading visualization cache...")
fig_cached = pio.read_json(
    osp.join(args.visual_dir, f"Trajectory_{visualization_name}.json"))

node2trace = {
    trace['name'].split(' ')[0]: trace for trace in fig_cached.data
}

print("Getting candidate nodes ...")

if args.dataset_name in ["DGraphFin"]:
    nodes = [n for n, l in node2label.items() if l in [0, 1]]
else:
    nodes = list(node2idx.keys())

options = []

# Display a distinct color family for each type of nodes
if display_node_type:
    label2palette = dict(zip(list(label2node.keys()),
                             const_viz.pure_color_palettes[:len(label2node)]))
    label2colors = {label: get_colors(12, palette)[::-1] for label, palette in
                    label2palette.items()}


else:

    label2colors = {
        0: get_colors(50, "Spectral")
    }

print("Adding categories to the dropdown menu ...")

for label, nodes_li in label2node.items():
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


print("Adding nodes to the dropdown menu ...")

for node, idx in node2idx.items():

    # For the DGraphFin dataset, the background nodes (label = 2 or 3) are not meaningful due to insufficient information. So we do not visualize them

    if display_node_type and args.dataset_name in ["DGraphFin"] and node2label.get(node) is None:
        print(f"\tIgnoring node {node} ...")
        continue


    if display_node_type:
        label = node2label[node]


        name = f"{node} ({label})"

    else:
        name = node

    options.append({
        "label": html.Span(
            [
                html.Span(name, style={
                    'font-size': 15,
                    'padding-left': 10
                }),
            ], style={
                'align-items': 'center',
                'justify-content': 'center'
            }
        ),
        "value": node,  # Get a random node as the default value.
    })

# Start the app
app = dash.Dash(__name__)

app.title = f"DyGetViz | {args.dataset_name}"

app.layout = html.Div(
    [
        dcc.Dropdown(
            id='search-input',
            options=options,
            value='',
            multi=True,
            placeholder="Select a node",
            style={
                'width': '100%'
            },
            clearable=True
        ),
        dcc.Graph(id='dygetviz', style={
            'width': '100%',
            # Set the graph width to 100% of its parent container
            'height': '700px'  # Adjust the height as needed
        }),
        html.Div("✨: a category. \n\"(1)\": a node label.", id="note"),

    ])


@app.callback(
    Output('dygetviz', 'figure'),
    Input('search-input', 'value')
)
def update_graph(search_values):
    fig = go.Figure()
    fig.add_trace(node2trace['background'])

    if search_values:
        print(f"Search values:\t{search_values}")
        for idx, value in enumerate(search_values):

            if args.dataset_name == "DGraphFin" and node2label.get(value) is None:
                print(f"Node {value} is a background node, so we ignore it.")
                continue
            # Add a node
            if value in nodes:

                trace = node2trace[value]

                if display_node_type:
                    label = node2label[value]
                    trace.line['color'] = label2colors[label][idx]
                    print(f"\tAdd node:\t{value} ({label})")

                else:
                    trace.line['color'] = label2colors[0][idx]
                    print(f"\tAdd node:\t{value}")

                fig.add_trace(trace)


            # Add a category
            elif value in label2node:
                print(f"\tAdd label:\t{value}")

                for idx_node, node in enumerate(label2node[value]):
                    trace = node2trace[node]
                    trace.line['color'] = label2colors[value][idx_node % 12]
                    fig.add_trace(trace)

            # if search_value in lines:  # search_value is a line name
            #     fig.add_trace(
            #         go.Scatter(x=x, y=lines[search_value]['data'], mode='lines',
            #                    name=search_value))
            # elif search_value in labels_to_lines:  # search_value is a type
            #     for line_name in labels_to_lines[search_value]:
            #         fig.add_trace(go.Scatter(x=x, y=lines[line_name]['data'],
            #                                  mode='lines', name=line_name))

    return fig


if __name__ == "__main__":
    print(const.DYGETVIZ)
    app.run_server(debug=True, port=8050)
