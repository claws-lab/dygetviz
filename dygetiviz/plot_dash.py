"""Run the Dash server to interactively add nodes to the visualization.
Plot using [Dash](https://dash.plotly.com/)
"""

import os.path as osp

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output, State

import const
import const_viz
from arguments import args
from data.dataloader import load_data
from utils.utils_misc import project_setup
from utils.utils_visual import get_colors

project_setup()

print("Loading data...")
data = load_data()

annotation: dict = data.get("annotation", {})
display_node_type: bool = data["display_node_type"]
idx_reference_snapshot: int = data["idx_reference_snapshot"]
interpolation: float = data["interpolation"]
node_presence: np.ndarray = data["node_presence"]
node2idx: dict = data["node2idx"]
node2label: dict = data["node2label"]
label2node: dict = data["label2node"]
num_nearest_neighbors: int = data["num_nearest_neighbors"]
perplexity: int = data["perplexity"]
plot_anomaly_labels: bool = data["plot_anomaly_labels"]
projected_nodes: np.ndarray = data["projected_nodes"]
reference_nodes: np.ndarray = data["reference_nodes"]
snapshot_names: list = data["snapshot_names"]
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
        0: get_colors(10, "Spectral")
    }

print("Adding categories to the dropdown menu ...")
options_categories = []

for label, nodes_li in label2node.items():
    options_categories.append({
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

options_nodes = []

for node, idx in node2idx.items():

    # For the DGraphFin dataset, the background nodes (label = 2 or 3) are not meaningful due to insufficient information. So we do not visualize them
    if display_node_type and args.dataset_name in [
        "DGraphFin"] and node2label.get(node) is None:
        print(f"\tIgnoring node {node} ...")
        continue

    if display_node_type:
        label = node2label[node]

        name = f"{node} ({label})"

    else:
        name = node

    options_nodes.append({
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

options = options_categories + options_nodes

print("Start the app ...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = f"DyGetViz | {args.dataset_name}"

app.layout = html.Div(
    [

        dbc.Row([
            dbc.Col([
                dbc.Label("Add a trajectory:", className="form-label",
                          id="note-trajectory"),
                dcc.Dropdown(
                    id='add-trajectory',
                    options=options,
                    value='',
                    multi=True,
                    placeholder="Select a node",
                    style={
                        'width': '100%'
                    },
                    clearable=True
                )
            ]),
            dbc.Col([
                dbc.Label("Add a background node.", className="form-label",
                          id="note-background"),
                dcc.Dropdown(
                    id='add-background-node-name',
                    options=options,
                    value='',
                    multi=True,
                    placeholder="Select a background node",
                    style={
                        'width': '100%'
                    },
                    clearable=True
                ),
            ])
        ]),

        dcc.Graph(id='dygetviz', style={
            'width': '100%',
            # Set the graph width to 100% of its parent container
            'height': '700px'  # Adjust the height as needed
        }),
        html.Div("✨: a category. \n\"(1)\": a node label.", id="note"),

        dbc.Row([
            dbc.Col([
                html.Label("Change Trajectory Color:"),
                # Dropdown for selecting a node
                dcc.Dropdown(
                    id='node-selector',
                    options=[],
                    value=nodes[0],
                    style={
                        'width': '50%'
                    },
                ),

                # Store the nodes in `trajectory_names`
                dcc.Store(
                    id='trajectory-names-store',
                    data=[]
                ),

                daq.ColorPicker(
                    id='color-picker',
                    label='Color Picker',
                    size=328,
                    value=dict(hex='#119DFF')
                ),
                html.Div(id='color-picker-output-1'),
                html.Button('Update Node Color', id='update-color-button',
                            n_clicks=0, className="me-2"),
            ])
        ]),
    ])


@app.callback(
    Output('dygetviz', 'figure'),
    Output('trajectory-names-store', 'data'),
    Input('add-trajectory', 'value'),
    Input('add-background-node-name', 'value'),
    Input('update-color-button', 'n_clicks'),
    State('node-selector', 'value'),
    State('color-picker', 'value'),
    State('dygetviz', 'figure'),

)
def update_graph(trajectory_names, background_node_names, do_update_color,
                 selected_node, selected_color, current_figure):
    """
    
    :param trajectory_names: Names of the trajectories to be added into the visualization
    :param background_node_names: 
    :param do_update_color: 
    :param selected_node: 
    :param selected_color: 
    :param current_figure: figure from the previous update
    :return: 
    """

    ctx = dash.callback_context
    action_name = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"[Action]\t{action_name}")
    fig = go.Figure()

    if current_figure is None:
        figure_name2trace = {}


    else:
        figure_name2trace = {trace['name']: trace for idx, trace in
                             enumerate(current_figure['data'])}

    def add_background():
        if figure_name2trace.get("background") is None:
            fig.add_trace(node2trace['background'])

    def add_traces():
        for name, trace in figure_name2trace.items():
            if name not in {"background"}:
                fig.add_trace(trace)

    if action_name == '':
        """Launch the app for the first time. 
        
        Only add the background nodes
        """
        add_background()
        return fig, trajectory_names

    elif action_name == 'add-trajectory':

        """
        From Yiqiao: I am not sure if directly modifying `current_figure` is a good practice as it will modify the original object, which can lead to unexpected behavior. In Plotly, figures are mutable objects

        """

        fig = go.Figure()
        add_background()

        new_trajectory_names = list(
            set(trajectory_names) - set(figure_name2trace.keys()))

        print(f"New search values:\t{new_trajectory_names}")
        for idx, value in enumerate(trajectory_names):

            if args.dataset_name == "DGraphFin" and node2label.get(
                    value) is None:
                print(f"Node {value} is a background node, so we ignore it.")
                continue

            # Add a node from previous search

            if value in figure_name2trace:
                trace = figure_name2trace[value]
                fig.add_trace(trace)

            # Add a node
            elif value in nodes:

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

    elif action_name == 'add-background-node-name':
        add_background()
        # Text with <50 characters
        displayed_text = np.array(
            ['' for _ in
             range(len(node2trace['background']['hovertext']))]).astype(
            '<U50')

        hover_text = np.array(list(node2trace['background']['hovertext']))

        mask = np.isin(hover_text, background_node_names)

        displayed_text[mask] = hover_text[mask]

        node2trace['background']['text'] = tuple(displayed_text.tolist())

        add_traces()


    elif action_name == 'update-color-button':

        figure_name2trace[selected_node].line['color'] = selected_color['hex']

        add_traces()

    return fig, trajectory_names


@app.callback(
    Output('node-selector', 'options'),
    Input('trajectory-names-store', 'data')
)
def update_node_selector_options(trajectory_names):
    if not trajectory_names:
        return []  # return an empty list if trajectory_names is empty

    # return a list of options for nodes in trajectory_names
    return [{
        'label': node,
        'value': node
    } for node in trajectory_names]


if __name__ == "__main__":

    print(const.DYGETVIZ)



    """
    `dev_tools_hot_reload`: disable hot-reloading. The code is not reloaded when the file is changed. Setting it to `True` will be very slow.
    """
    app.run_server(debug=True, dev_tools_hot_reload=False, use_reloader=False,
                   port=args.port)
