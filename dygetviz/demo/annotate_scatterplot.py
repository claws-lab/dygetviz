# Import libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

import plotly.graph_objects as go

# Sample data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [1, 4, 2, 3, 5],
    'name': ['A', 'B', 'C', 'D', 'E']
})

# Create a Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        config={'staticPlot': False, 'displayModeBar': True},
        figure=px.scatter(df, x='x', y='y')
    )
])

# List to keep track of current annotations
annotations = []


# Callback to annotate point on click
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-plot', 'clickData')
)
def display_click_data(clickData):
    global annotations
    if clickData:
        point_data = clickData['points'][0]
        idx = point_data['pointIndex']
        point_name = df['name'].iloc[idx]

        # Check if the point is already annotated
        existing_annotation = next((a for a in annotations if
                                    a['x'] == point_data['x'] and a['y'] ==
                                    point_data['y']), None)

        if existing_annotation:
            # Remove existing annotation
            annotations.remove(existing_annotation)
        else:
            # Add new annotation
            annotations.append({
                'x': point_data['x'],
                'y': point_data['y'],
                'xref': 'x',
                'yref': 'y',
                'text': point_name,
                'showarrow': False
            })

    fig2 = go.Figure()

    fig = px.scatter(df, x='x', y='y')
    fig.update_layout(annotations=annotations)
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, port=8049)
