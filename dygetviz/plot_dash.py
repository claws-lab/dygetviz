"""Run the Dash server to interactively add nodes to the visualization.
Plot using [Dash](https://dash.plotly.com/)
"""

import os.path as osp

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output, State
from tqdm import tqdm

import const
import const_viz
from arguments import parse_args
from data.dataloader import load_data
from utils.utils_data import get_modified_time_of_file, read_markdown_into_html
from utils.utils_misc import project_setup
from utils.utils_visual import get_colors


def main(args):

    print("Loading data...")

    # TODO

    data = load_data(args.dataset_name, False, args)

    annotation: dict = data.get("annotation", {})
    display_node_type: bool = data["display_node_type"]
    idx_reference_snapshot: int = data["idx_reference_snapshot"]
    interpolation: float = data["interpolation"]
    node_presence: np.ndarray = data["node_presence"]
    node2idx: dict = data["node2idx"]
    node2label: dict = data["node2label"]
    label2node: dict = data["label2node"]
    label2name: dict = data["label2name"]
    metadata_df: dict = data["metadata_df"]
    num_nearest_neighbors: int = data["num_nearest_neighbors"]
    perplexity: int = data["perplexity"]
    plot_anomaly_labels: bool = data["plot_anomaly_labels"]
    projected_nodes: np.ndarray = data["projected_nodes"]
    reference_nodes: np.ndarray = data["reference_nodes"]
    snapshot_names: list = data["snapshot_names"]
    z = data["z"]

    args = parse_args()

    idx2node = {idx: node for node, idx in node2idx.items()}

    visualization_name = f"{args.dataset_name}_{args.model}_{args.visualization_model}_perplex{perplexity}_nn{data['num_nearest_neighbors'][0]}_interpolation{interpolation}_snapshot{idx_reference_snapshot}"

    print("Reading visualization cache...")

    path = osp.join(args.visual_dir, f"Trajectory_{visualization_name}.json")

    get_modified_time_of_file(path)

    fig_cached = pio.read_json(path)

    node2trace = {
        trace['name'].split(' ')[0]: trace for trace in fig_cached.data
    }

    print("Getting candidate nodes ...")

    if args.dataset_name in ["DGraphFin"]:
        nodes = [n for n, l in node2label.items() if l in [0, 1]]
    else:
        nodes = list(node2idx.keys())

    options = []

    # If there are multiple node categories, we can display a distinct color family for each type of nodes
    # NOTE: We specifically require that the first color palette is Blue (for normal nodes) and the second one is Red (for anomalous nodes)
    if display_node_type:
        labels = sorted(list(label2node.keys()))
        label2palette = dict(zip(labels,
                                 const_viz.pure_color_palettes[:len(label2node)]))
        label2colors = {label: get_colors(12, label2palette[label])[::-1] for label
                        in labels}


    else:
        # Otherwise, we use a single color family for all nodes. But the colors are very distinct
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
        # Only add trajectories of projected or reference nodes
        if not node in projected_nodes:
            continue

        # For the DGraphFin dataset, the background nodes (label = 2 or 3) are not meaningful due to insufficient information. So we do not visualize them
        if display_node_type and args.dataset_name in [
            "DGraphFin"] and node2label.get(node) is None:
            print(f"\tIgnoring node {node} ...")
            continue

        if display_node_type:
            label = node2label[node]

            name = f"{node} ({label2name[label]})"

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

    print("Reading plotly button explanations ...")
    with open('dygetviz/static/Plotly_Button_Explanations.html', 'r') as file:
        plotly_button_explanations = file.read()

    app.title = f"DyGetViz | {args.dataset_name}"


    print("Reading dataset explanations ...")
    dataset_descriptions = read_markdown_into_html(osp.join(args.data_dir, args.dataset_name, "data_descriptions.md"))


    app.layout = html.Div(
        [
            # Title
            html.H1(f"Dataset: {args.dataset_name}",
                    className="text-center mb-4",
                    style={
                        'color': '#2c3e50',
                        'font-weight': 'bold'
                    }),

            # Dropdown Row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Add a trajectory:",
                                      className="form-label mb-2",
                                      id="note-trajectory",
                                      style={
                                          'font-weight': 'bold',
                                          'color': '#34495e',
                                          'width': '100%'
                                      }),
                        ],
                        className="right",
                        width=3,
                    ),
                    dbc.Col(
                        [
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
                        ],
                        className="center mb-4",
                        # margin-bottom to give some space below
                        width=6

                    )
                ],
                className="text-center mb-4"
                # margin-bottom to give some space below the row
            ),

            # Graph
            dcc.Graph(
                id='dygetviz',
                style={
                    'width': '90%',
                    'height': '700px',
                    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
                    # A subtle shadow for depth
                },
                className="text-center",
            ),
            # html.Div("✨: a category. \n\"(1)\": a node label.", id="note"),

            # Store the nodes in `trajectory_names`
            dcc.Store(
                id='trajectory-names-store',
                data=[]),

            # Yiqiao (2023.8.24): Now we do not consider the color picker since it will give the user too much freedom

            # dbc.Row([
            #     dbc.Col([
            #         html.Label("Change Trajectory Color:"),
            #         # Dropdown for selecting a node
            #         dcc.Dropdown(
            #             id='node-selector',
            #             options=[],
            #             value=nodes[0],
            #             style={
            #                 'width': '50%'
            #             },
            #         ),
            #
            #         # Store the nodes in `trajectory_names`
            #         dcc.Store(
            #             id='trajectory-names-store',
            #             data=[]
            #         ),
            #
            #         daq.ColorPicker(
            #             id='color-picker',
            #             label='Color Picker',
            #             size=328,
            #             value=dict(hex='#119DFF')
            #         ),
            #         html.Div(id='color-picker-output-1'),
            #         html.Button('Update Node Color', id='update-color-button',
            #                     n_clicks=0, className="me-2"),
            #     ])
            # ]),
            dbc.Row([
                dbc.Col([
                    html.H3(f"Dataset Introduction", className="text-center"),
                    html.Iframe(srcDoc=dataset_descriptions,
                                style={"width": "100%", "height": "500px"}),
                    # html.P(explanation, className="text-center"),
                ]),

                dbc.Col([
                    html.H3(f"Panel", className="text-center"),

                    html.Iframe(srcDoc=plotly_button_explanations,
                                style={"width": "100%", "height": "500px"})
                ])
            ]),

        ])


    def generate_node_profile(profile: pd.DataFrame):
        def f(x):
            ret = "<ul>"
            for field in x.index:
                ret += f"<li>{field}: {x[field]}</li>"
            ret += "</ul>"
            return ret

        profile['description'] = profile.apply(f, axis=1)
        return profile

    def convert_scatter_to_scattergl(scatter):
        line = { "color": scatter.line.color, "dash": scatter.line.dash, "shape": scatter.line.shape, "width": scatter.line.width }
        marker = { "size": scatter.marker.size, 'symbol': scatter.marker.symbol}
        return go.Scattergl(x=scatter.x, y=scatter.y, xaxis=scatter.xaxis, yaxis=scatter.yaxis, customdata=scatter.customdata,
                            hovertemplate=scatter.hovertemplate, hovertext=scatter.hovertext, legendgroup=scatter.legendgroup,
                            line= line, marker=marker, mode=scatter.mode, name=scatter.name, showlegend=scatter.showlegend,
                              selectedpoints=scatter.selectedpoints, text=scatter.text, textposition=scatter.textposition)

    # List to keep track of current annotations
    annotations = []


    @app.callback(
        Output('dygetviz', 'figure'),
        Output('trajectory-names-store', 'data'),
        Input('add-trajectory', 'value'),
        Input('dygetviz', 'clickData'),
        State('dygetviz', 'figure'),
        # Input('update-color-button', 'n_clicks'),
        # State('node-selector', 'value'),
        # State('color-picker', 'value'),

    )
    def update_graph(trajectory_names, clickData, current_figure,
                     # do_update_color, selected_node, selected_color,
                     ):
        """

        :param trajectory_names: Names of the trajectories to be added into the visualization
        :param background_node_names:
        :param do_update_color:
        :param selected_node:
        :param selected_color:
        :param current_figure: figure from the previous update
        :return:
        """

        global annotations

        if not trajectory_names:
            trajectory_names = []

        ctx = dash.callback_context
        action_name = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f"[Action]\t{action_name}")

        # This is the template for displaying metadata when hovering over a node

        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False
            )
        )

        if current_figure is None:
            figure_name2trace = {}


        else:
            figure_name2trace = {trace['name']: trace for idx, trace in
                                 enumerate(current_figure['data'])}

        def add_background():

            if figure_name2trace.get("background") is None:
                trace = node2trace['background']
                # trace.hovertemplate = HOVERTEMPLATE
                fig.add_trace(trace)

            if figure_name2trace.get("background") is None and plot_anomaly_labels:
                trace = node2trace['anomaly']
                fig.add_trace(trace)

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
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False
                )
            )

            del figure_name2trace['background']

            add_background()

            new_trajectory_names = list(
                set(trajectory_names) - set(figure_name2trace.keys()))

            print(f"New search values:\t{new_trajectory_names}")
            for idx, value in enumerate(
                    tqdm(trajectory_names, desc="Add Trajectories")):

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
                    trace = convert_scatter_to_scattergl(trace)
                    if display_node_type:
                        label = node2label[value]
                        trace.line['color'] = label2colors[label][idx]
                        print(f"\tAdd node:\t{value} ({label2name[label]})")

                    else:
                        trace.line['color'] = label2colors[0][idx]
                        print(f"\tAdd node:\t{value}")

                    fig.add_trace(trace)


                # Add a category
                elif value in label2node:
                    print(f"\tAdd label:\t{value}")

                    for idx_node, node in enumerate(label2node[value]):
                        trace = node2trace[node]
                        # Haven't tested this since I believe category is broken (after adding only a subset of node trajectories)
                        trace = convert_scatter_to_scattergl(trace)
                        trace.line['color'] = label2colors[value][idx_node % 12]
                        fig.add_trace(trace)


        # elif action_name == 'update-color-button':
        #     # Update the color of the selected trajectory
        #     del figure_name2trace['background']
        #     figure_name2trace[selected_node].line['color'] = selected_color['hex']
        #
        #     add_traces()

        elif action_name == 'dygetviz':
            # Add annotations when user clicks on a node
            """
                    Upon clicking a node, if the node's display is on, we turn the display off. If its display is off, we turn the display on.
                    """

            if clickData:
                del figure_name2trace['background']
                point_data = clickData['points'][0]
                point_idx = point_data['pointIndex']

                displayed_text = np.array(
                    list(node2trace['background']['text'])).astype(
                    '<U50')

                displayed_text[point_idx] = node2trace['background']['hovertext'][
                    point_idx] if not displayed_text[point_idx] else ''

                node2trace['background']['text'] = tuple(displayed_text.tolist())

                add_background()

                add_traces()

            #     point_name = df['name'].iloc[idx]
            #
            #     # Check if the point is already annotated
            #     existing_annotation = next((a for a in annotations if
            #                                 a['x'] == point_data['x'] and a['y'] ==
            #                                 point_data['y']), None)
            #
            #     if existing_annotation:
            #         # Remove existing annotation
            #         annotations.remove(existing_annotation)
            #     else:
            #         # Add new annotation
            #         annotations.append({
            #             'x': point_data['x'],
            #             'y': point_data['y'],
            #             'xref': 'x',
            #             'yref': 'y',
            #             'text': point_name,
            #             'showarrow': False
            #         })
            #
            # fig.update_layout(annotations=annotations)

        return fig, trajectory_names


    # @app.callback(
    #     Output('node-selector', 'options'),
    #     Input('trajectory-names-store', 'data')
    # )
    # def update_node_selector_options(trajectory_names):
    #     """
    #     Archived function
    #     Adjust the colors of existing trajectories
    #
    #     :param trajectory_names:
    #     """
    #     if not trajectory_names:
    #         return []  # return an empty list if trajectory_names is empty
    #
    #     # return a list of options for nodes in trajectory_names
    #     return [{
    #         'label': node,
    #         'value': node
    #     } for node in trajectory_names]


if __name__ == "__main__":
    print(const.DYGETVIZ)
    args = parse_args()
    project_setup()
    print("Start the app ...")
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    """
    `dev_tools_hot_reload`: disable hot-reloading. The code is not reloaded when the file is changed. Setting it to `True` will be very slow.
    """
    app.run_server(debug=True, dev_tools_hot_reload=False, use_reloader=False,
                   port=args.port)
    main(args)
