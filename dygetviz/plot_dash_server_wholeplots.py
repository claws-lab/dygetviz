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
import re

import dygetviz.const as const
# import const_viz
from dygetviz.components.dygetviz_components import graph_with_loading, dataset_description, interpretation_of_plot, \
    visualization_panel
from dygetviz.arguments import parse_args
from dygetviz.data.dataloader import load_data, load_data_description
from dygetviz.utils.utils_misc import project_setup
from dygetviz.utils.utils_visual import get_colors, get_nodes_and_options

args = parse_args()

project_setup()
# dataset_names = ['Chickenpox', 'BMCBioinformatics2021', 'Reddit', 'DGraphFin', 'HistWords-EN-GNN', 'HistWords-CN-GNN', 'Ant']

dataset_names = ['HistWords-EN-GNN' , 'HistWords-CN-GNN', 'Reddit', 'Ant', 'BMCBioinformatics2021',  'DGraphFin']
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

    # dataset_description_temp = load_data_description(dataset_name, args)

    # dataset_data[dataset_name] = {"data": data, "nodes": nodes, "node2trace": node2trace, "label2colors":
    #     label2colors,  "options": options, "cached_figure": cached_figure, "dataset_description": dataset_description}
    dataset_data[dataset_name] = {"data": data, "nodes": nodes, "node2trace": node2trace, "label2colors": label2colors,  "options": options, "cached_figure": cached_figure, "markdown": markdown }


print("Start the app ...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "../dygetviz/assets/base.css",
                                                "../dygetviz/assets/clinical-analytics.css"])
with open('dygetviz/static/Plotly_Button_Explanations.html', 'r') as file:
    plotly_button_explanations = file.read()


app.title = f"DyGetViz | Dynamic Graph Embedding Trajectories Visualization Dashboard"


app.layout = html.Div(
    id="app-container",
    children=[
        dcc.Location(id='url', refresh=False),
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src="https://brand.gatech.edu/sites/default/files/inline-images/GeorgiaTech_RGB.png"),
                # Title
                # html.H1(f"DyGETViz",
                #       className="text-center mb-4",
                #         ),
                ],
        ),

   # Title
        html.H1(f"{dataset_names[0]}",
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
                        dbc.Label("(Datasets take a while to load)",
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
                            searchable=False,
                            clearable=False,
                            style={
                                'width': '100%'
                            }),
                    ],
                    className="right",
                    width=3,
                ),
            ],
            className="text-center mb-4"
            # margin-bottom to give some space below the row
        ),
        dcc.Loading(
        id="loading-1",  # Use any unique ID
        type="default",  # There are different types of loading spinners to use
        children=dcc.Graph(
            id='dygetviz',
            style={
                'width': '100%',
                'height': '700px',
                # 'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
                # A subtle shadow for depth
            },
            className="text-center",
        ),
        ),
        # html.Div("✨: a category. \n\"(1)\": a node label.", id="note"),

        dbc.Row([
            dbc.Col([
                html.H3(f"Dataset Introduction", className="text-center"),

                # html.P("", className="text-center", id='data-desc',),
                # dcc.Markdown("", className="text-left", id='dataset-description', dangerously_allow_html=True,style={'text-align': 'left'}),
                dcc.Markdown("", className="text-left", id='data-desc', dangerously_allow_html=True,
                             style={'text-align': 'left', 'margin-left': '0.5em'}),
                
                interpretation_of_plot()
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



# Callback to update dropdown based on URL
@app.callback(
    Output('dataset-selector', 'value'),
    Input('url', 'pathname')
)
def update_dropdown_value(pathname):
    # Extract the part of the URL after the last '/'
    url_part = pathname.split('/')[-1]
    
    # Clean the URL part, remove special characters
    cleaned_url_part = re.sub('[^A-Za-z0-9\-]+', ' ', url_part)
    
    # Check if the cleaned URL part is in the dataset names
    if cleaned_url_part in dataset_names:
        return cleaned_url_part
    else:
        # If not found in dataset names, set to the first dataset name
        return dataset_names[0]


# # Callback to update URL based on dropdown value
# @app.callback(
#     Output('url', 'pathname'),
#     [Input('dataset-selector', 'value')]
# )
# def update_url(value):
#     return '/' + value if value else '/'


@app.callback(
    Output('data-desc', 'children'),
    Output('dataset-title', 'children'),
    Input('dataset-selector', 'value'),
)
def update_title(dataset_name):
    global_store_data = dataset_data[dataset_name]
    nodes, node2trace, label2colors, options, cached_figure = (global_store_data['nodes'], global_store_data['node2trace'], global_store_data['label2colors'], global_store_data['options'], global_store_data['cached_figure'])

    # Update dataset description
    markdown = global_store_data['markdown']
    return markdown, dataset_name

@app.callback(
    Output('dygetviz', 'figure'),
    Input('dataset-title', 'children'),
    # Input('dygetviz', 'clickData'),
    State('dygetviz', 'figure'),
    # Input('update-color-button', 'n_clicks'),
    # State('node-selector', 'value'),
    # State('color-picker', 'value'),

)
def update_graph(dataset_name,
                  # clickData, 
                  current_figure
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
    # title = [f"Dataset: {dataset_name}"]
    global annotations


    ctx = dash.callback_context
    action_name = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"[Action]\t{action_name}")
    # This is the template for displaying metadata when hovering over a node


    # fig = go.Figure(data=cached_figure.data, cached_figure.frames, cached_figure.layout)
    fig = cached_figure
    fig.layout.updatemenus = []
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

        return fig

    # elif action_name == 'dygetviz':
    #     # Add annotations when user clicks on a node
    #     """
    #             Upon clicking a node, if the node's display is on, we turn the display off. If its display is off, we turn the display on.
    #             """

    #     if clickData:
    #         del figure_name2trace['background']
    #         point_data = clickData['points'][0]
    #         point_idx = point_data['pointIndex']


    #         displayed_text = np.array(
    #             list(node2trace['background']['text'])).astype(
    #             '<U50')


    #         displayed_text[point_idx] = node2trace['background']['hovertext'][point_idx] if not displayed_text[point_idx] else ''

    #         node2trace['background']['text'] = tuple(displayed_text.tolist())


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
    return fig


server=app.server

if __name__ == "__main__":

    print(const.DYGETVIZ)

    """
    `dev_tools_hot_reload`: disable hot-reloading. The code is not reloaded when the file is changed. Setting it to `True` will be very slow.
    """
    app.run_server( dev_tools_hot_reload=False, use_reloader=False,
                   port=args.port)
    # from gevent.pywsgi import WSGIServer
    
    # Assuming your Dash app instance is named `app`
    # server = WSGIServer(('0.0.0.0', args.port), app.server)
    # print(const.DYGETVIZ)
    # server.serve_forever()


# https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-22-04