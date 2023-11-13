"""Run the Dash server to interactively add nodes to the visualization.
Plot using [Dash](https://dash.plotly.com/)
"""

import os.path as osp
import os

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
from data.dataloader import load_data, load_data_description
from utils.utils_misc import project_setup
from utils.utils_visual import get_colors, get_nodes_and_options

args = parse_args()

project_setup()
dataset_names = ['Chickenpox', 'BMCBioinformatics2021', 'Reddit', 'DGraphFin', 'HistWords-EN-GNN']
dataset_data = {}
# Storing all dataset information for quick access (from the load_data() function and the trajectory information)
for dataset_name in dataset_names:
    print(f"Loading data for {dataset_name}...")
    data = load_data(dataset_name)
    visual_dir = osp.join(args.output_dir, "visual", dataset_name)
    nodes, node2trace, label2colors, options, cached_figure = get_nodes_and_options(data, visual_dir)
    try:
        with open(osp.join("data", dataset_name, "data_descriptions.md"), 'r') as file:
            markdown = file.read()
    except:
        markdown = "No description yet"
        

    # Can refactor this into one dict later...

    dataset_description = load_data_description(dataset_name, args)

    # dataset_data[dataset_name] = {"data": data, "nodes": nodes, "node2trace": node2trace, "label2colors":
    #     label2colors,  "options": options, "cached_figure": cached_figure, "dataset_description": dataset_description}
    dataset_data[dataset_name] = {"data": data, "nodes": nodes, "node2trace": node2trace, "label2colors": label2colors,  "options": options, "cached_figure": cached_figure, "markdown": markdown }


print("Start the app ...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
with open('dygetviz/static/Plotly_Button_Explanations.html', 'r') as file:
    plotly_button_explanations = file.read()


app.title = f"DyGetViz | Dynamic Graph Embedding Trajectories Visualization Dashboard"


app.layout = html.Div(
    [   # Title
        html.H1(f"Dataset: {dataset_names[0]}",
                id='dataset-title',
                className="text-center mb-4",
                style={
                    'color': '#2c3e50',
                    'fontWeight': 'bold'
                }),

        # Dropdown Row
        dbc.Row(
            [dbc.Col(
                    [
                        dbc.Label("Select Dataset:",
                                  className="form-label mb-2",
                                  style={
                                      'font-weight': 'bold',
                                      'color': '#34495e',
                                      'width': '100%'
                                  }),
                    ],
                    className="right",
                    width=2,
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id='dataset-selector',
                            options=dataset_names,
                            value=dataset_names[0],
                            searchable=False,
                            clearable=False,
                            style={
                                'width': '100%'
                            }),
                    ],
                    className="right",
                    width=3,
                ),
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
                    width=2,
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
                    width=4
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
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
                # A subtle shadow for depth
            },
            className="text-center",
        ),
        # html.Div("✨: a category. \n\"(1)\": a node label.", id="note"),

        # Store the nodes in `trajectory_names`
        dcc.Store(
            id='trajectory-names-store',
            data=[]),
        dbc.Row([
            dbc.Col([
                html.H3(f"Dataset Introduction", className="text-center"),

                # html.P("", className="text-center", id='data-desc',),
                # dcc.Markdown("", className="text-left", id='dataset-description', dangerously_allow_html=True,style={'text-align': 'left'}),
                dcc.Markdown("", className="text-left", id='data-desc', dangerously_allow_html=True,
                             style={'text-align': 'left'}),
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

def add_background(fig, figure_name2trace, node2trace, plot_anomaly_labels, cached_frames):
    if figure_name2trace.get("background") is None:
        trace = node2trace['background']
        # trace.hovertemplate = HOVERTEMPLATE
        fig.add_trace(trace)
    if figure_name2trace.get("background") is None and plot_anomaly_labels:
        trace = node2trace['anomaly']
        fig.add_trace(trace)
    print("adding cached frames", cached_frames)
    fig.frames = cached_frames

def add_traces(fig, figure_name2trace):
    for name, trace in figure_name2trace.items():
        if name not in {"background"}:
            fig.add_trace(trace)

# List to keep track of current annotations
annotations = []

def convert_scatter_to_scattergl(scatter):
    line = { "color": scatter.line.color, "dash": scatter.line.dash, "shape": scatter.line.shape, "width": scatter.line.width }
    marker = { "size": scatter.marker.size, 'symbol': scatter.marker.symbol}
    return go.Scattergl(x=scatter.x, y=scatter.y, xaxis=scatter.xaxis, yaxis=scatter.yaxis, customdata=scatter.customdata,
                        hovertemplate=scatter.hovertemplate, hovertext=scatter.hovertext, legendgroup=scatter.legendgroup, 
                        line= line, marker=marker, mode=scatter.mode, name=scatter.name, showlegend=scatter.showlegend,
                          selectedpoints=scatter.selectedpoints, text=scatter.text, textposition=scatter.textposition)
@app.callback(
    Output('dygetviz', 'figure'),
    Output('trajectory-names-store', 'data'),
    Output('add-trajectory', 'options'),
    Output('dataset-title', 'children'),
    Output('data-desc', 'children'),
    Input('dataset-selector', 'value'),
    Input('add-trajectory', 'value'),
    Input('dygetviz', 'clickData'),
    State('dygetviz', 'figure'),
    State('add-trajectory', 'options'),
    # Input('update-color-button', 'n_clicks'),
    # State('node-selector', 'value'),
    # State('color-picker', 'value'),

)
def update_graph(dataset_name, trajectory_names, clickData, current_figure, trajectory_options
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
    # visual_dir = osp.join(args.output_dir, "visual", dataset_name)
    # data = load_data(dataset_name)
    # data = dataset_data[dataset_name]['data']
    
    global_store_data = dataset_data[dataset_name]
    nodes, node2trace, label2colors, options, cached_figure = (global_store_data['nodes'], global_store_data['node2trace'], global_store_data['label2colors'], global_store_data['options'], global_store_data['cached_figure'])

    # Update dataset description
    markdown = global_store_data['markdown']

    display_node_type: bool = global_store_data['data']["display_node_type"]
    node2label: dict = global_store_data['data']["node2label"]
    label2node: dict = global_store_data['data']["label2node"]
    plot_anomaly_labels: bool = global_store_data['data']['plot_anomaly_labels']
    title = [f"Dataset: {dataset_name}"]
    global annotations

    if not trajectory_names:
        trajectory_names = []

    ctx = dash.callback_context
    action_name = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"[Action]\t{action_name}")
    # This is the template for displaying metadata when hovering over a node


    # fig = go.Figure(data=cached_figure.data, cached_figure.frames, cached_figure.layout)
    fig = cached_figure
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
        ),
    )

    if current_figure is None or action_name == 'dataset-selector':
        figure_name2trace = {}


    else:
        figure_name2trace = {trace['name']: trace for idx, trace in
                             enumerate(current_figure['data'])}
    if action_name == '' or action_name == 'dataset-selector':
        """Launch the app for the first time. 
        
        Only add the background nodes
        """
        fig = cached_figure
        
        # print(fig)

        return fig, trajectory_names, options, title, markdown



    if action_name == 'add-trajectory':

        """
        From Yiqiao: I am not sure if directly modifying `current_figure` is a good practice as it will modify the original object, which can lead to unexpected behavior. In Plotly, figures are mutable objects

        """

        fig = cached_figure



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


            displayed_text[point_idx] = node2trace['background']['hovertext'][point_idx] if not displayed_text[point_idx] else ''

            node2trace['background']['text'] = tuple(displayed_text.tolist())


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



    # print(fig)
    return fig, trajectory_names, trajectory_options, title, markdown


if __name__ == "__main__":

    print(const.DYGETVIZ)

    """
    `dev_tools_hot_reload`: disable hot-reloading. The code is not reloaded when the file is changed. Setting it to `True` will be very slow.
    """
    app.run_server(debug=True, 
                   dev_tools_hot_reload=False, use_reloader=False,
                   port=args.port)
