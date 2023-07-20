import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np

import plotly.io as pio

# # create some data
# x = np.linspace(0, 10, 100)
# lines = {
#     'sin': np.sin(x),
#     'cos': np.cos(x),
#     'tan': np.tan(x)
# }
#
# # Start the app
# app = dash.Dash(__name__)
#
# app.layout = html.Div([
#     dcc.Dropdown(
#         id='search-input',
#         options=[{'label': name, 'value': name} for name in lines.keys()],
#         value='',
#         multi=True,
#         placeholder="Select a function...",
#         style={'width': '50%'}
#     ),
#     dcc.Graph(id='my-graph')
# ])
#
# @app.callback(
#     Output('my-graph', 'figure'),
#     Input('search-input', 'value')
# )
# def update_graph(search_values):
#     fig = go.Figure()
#
#     if search_values:
#         for search_value in search_values:
#             y = lines.get(search_value)
#             if y is not None:
#                 fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=search_value))
#
#     return fig
#
# if __name__ == '__main__':
#     app.run_server(debug=True)


"""
Search by attribute
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np

# create some data
x = np.linspace(0, 10, 100)
lines = {
    'sin': {'data': np.sin(x), 'type': 'Fraud'},
    'cos': {'data': np.cos(x), 'type': 'Fraud'},
    'tan': {'data': np.tan(x), 'type': 'Normal'},
    'sinc': {'data': np.sinc(x), 'type': 'Normal'},
    'sqrt': {'data': np.sqrt(x), 'type': 'Fraud'}
}

# Create a mapping from types to lines
types_to_lines = {
    type: [name for name, line in lines.items() if line['type'] == type]
    for type in set(line['type'] for line in lines.values())
}

# Start the app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='search-input',
        options=[{'label': name, 'value': name} for name in lines.keys()] + [{'label': f"[Type] {type}", 'value': type} for type in types_to_lines.keys()],
        value='',
        multi=True,
        placeholder="Select a function or type...",
        style={'width': '50%'}
    ),
    dcc.Graph(id='my-graph')
])

@app.callback(
    Output('my-graph', 'figure'),
    Input('search-input', 'value')
)
def update_graph(search_values):
    # fig = go.Figure()

    fig = pio.read_json(osp.join(args.visual_dir, f"Trajectory_{visualization_name}.json"))


    if search_values:
        for search_value in search_values:
            if search_value in lines:  # search_value is a line name
                fig.add_trace(go.Scatter(x=x, y=lines[search_value]['data'], mode='lines', name=search_value))
            elif search_value in types_to_lines:  # search_value is a type
                for line_name in types_to_lines[search_value]:
                    fig.add_trace(go.Scatter(x=x, y=lines[line_name]['data'], mode='lines', name=line_name))

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
