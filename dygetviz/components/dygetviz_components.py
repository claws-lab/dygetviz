import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html
from dash import dcc, html


def dataset_description(dataset_name: str):
    component = html.Div([
        html.H4(f"The {'Aging' if dataset_name == 'BMCBioinformatics2021' else dataset_name} "
                f"Dataset",
                className="text-center"),
        html.P([
            dbc.Button("Genetics dataset about aging", id="genetics-dataset-button", color="primary"),
            dbc.Tooltip(
                "Qi Li, Khalique Newaz and Tijana Milenković. Improved supervised prediction of aging‐related genes via weighted dynamic network analysis. BMC Bioinformatics 2021.",
                target="genetics-dataset-button"
            ),
            " originates from a 2021 paper published in BMC Bioinformatics. It provides detailed human gene expression data, capturing the changes that occur over a wide range of human ages -- 37 distinct ages spanning from 20 to 99 years."
        ]),
        html.Span(children=["Dynamic graph embeddings are generated using "]),

        dbc.Button(children=["Graph Convolutional Recurrent Networks (GCRU)"], id="gcru", color="primary"),
        dbc.Tooltip(
            children=["Youngjoo Seo, Michael Defferrard, Pierre Vandergheynst, Xavier Bresson. Structured Sequence "
                      "Modeling with Graph Convolutional Recurrent Networks. CONIP 2018"],
            target="gcru"
            # https://arxiv.org/abs/1612.07659
        ),
        # html.Span("[1].", id="ref1"),
        # dbc.Tooltip(
        #     "[1] Qi Li, Khalique Newaz and Tijana Milenković. Improved supervised prediction of aging‐related genes via weighted dynamic network analysis. BMC Bioinformatics 2021.",
        #     target="ref1"
        # ),

    ])

    return component


def graph_with_loading():
    component = dcc.Loading(
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
    )

    return component


def interpretation_of_plot():
    component = html.Div([
        html.H4("Interpret the plot"),
        html.P(id="anomalous_nodes", children=["🔴 Anomalous nodes, e.g. abnormal genes, fraudsters in a transaction "
                                               "network, or malicious nodes in a social network"]),

        html.P(id="normal_nodes", children=["⚪ Normal nodes"]),
        html.P(children=[
            "🌈 Colorful nodes with identical colors represent trajectory of the same nodes over the given time span."
        ]),
    ])
    return component


def read_button(name: str):
    with open(f'dygetviz/static/{name}.html', 'r') as file:
        plotly_button_explanations = file.read()
        return plotly_button_explanations


def visualization_panel():
    d = {
        "rectangle": "Draw a rectangular region and zoom in",
        "move": "Drag to move the plot",
        "rectangle_select": "Select nodes by drawing a rectangle",
        "freeform_select": "Select nodes by drawing a freeform shape",
        "zoom_in": "Zoom in",
        "zoom_out": "Zoom out",
        "reset": "Reset the plot to its default zoom level.",
        "restore": "Restore view to the default ranges.",
        "plotly_logo": "Plotly official website",
    }

    children_components = []

    for name, desc in d.items():
        # button_icon = html.Iframe(srcDoc=read_button(name),
        #                           style={"width": "60px", "height": "60px", 'display': 'inline-block',
        #                                  'vertical-align': 'middle'})

        svg_html = read_button(name)
        button_icon = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(svg_html)

        button_description = html.Span(desc, style={"margin-left": "5px", 'display': 'inline-block',
                                                    'vertical-align': 'middle',
                                                    'flex-grow': 1
                                                    })

        children_components.append(
            html.Div([button_icon, button_description], style={'display': 'flex', 'align-items': 'center'}))


    # Limit the width of the panel
    component = html.Div(children_components, style={'max-width': '400px'})

    return component
