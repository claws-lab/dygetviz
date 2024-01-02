import base64
import io

import pandas as pd
import dash
from dash import html, dcc
import dash_mantine_components as dmc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag

from dash_iconify import DashIconify



def upload_panel():
    body = dmc.Stack(
        [
            dmc.Stepper(
                id="stepper",
                contentPadding=30,
                active=0,
                size="md",
                breakpoint="sm",
                children=[
                    # dmc.StepperStep(
                    #     label="Add your OpenAI API key",
                    #     icon=DashIconify(
                    #         icon="material-symbols:lock",
                    #     ),
                    #     progressIcon=DashIconify(
                    #         icon="material-symbols:lock",
                    #     ),
                    #     completedIcon=DashIconify(
                    #         icon="material-symbols:lock-open",
                    #     ),
                    #     children=[
                    #         dmc.Stack(
                    #             [
                    #                 dmc.Stack(
                    #                     [
                    #                         dmc.Blockquote(
                    #                             """Welcome to ChartGPT! To get started, fetch your OpenAI API key and paste it below.\
                    #                             Then, upload your CSV file and ask ChartGPT to plot your data. Happy charting 🥳""",
                    #                             icon=DashIconify(
                    #                                 icon="line-md:coffee-half-empty-twotone-loop"
                    #                             ),
                    #                         ),
                    #                         dmc.Center(
                    #                             dmc.Button(
                    #                                 dmc.Anchor(
                    #                                     "Get your API key",
                    #                                     href="https://platform.openai.com/account/api-keys",
                    #                                     target="_blank",
                    #                                     style={
                    #                                         "textDecoration": "none",
                    #                                         "color": "white",
                    #                                     },
                    #                                 ),
                    #                                 fullWidth=False,
                    #                                 rightIcon=DashIconify(
                    #                                     icon="material-symbols:open-in-new"
                    #                                 ),
                    #                             ),
                    #                         ),
                    #                     ],
                    #                 ),
                    #                 dmc.PasswordInput(
                    #                     id="input-api-key",
                    #                     label="API Key",
                    #                     description="Please add your OpenAI API key. It will be used to generate your visualization",
                    #                     placeholder="Your OpenAI API Key",
                    #                     icon=DashIconify(icon="material-symbols:key"),
                    #                     size="sm",
                    #                     required=True,
                    #                 ),
                    #             ]
                    #         )
                    #     ],
                    # ),
                    dmc.StepperStep(
                        label="Upload your CSV file",
                        icon=DashIconify(icon="material-symbols:upload"),
                        progressIcon=DashIconify(icon="material-symbols:upload"),  # style={'border': '1px solid black'}
                        completedIcon=DashIconify(icon="material-symbols:upload"),
                        children=[
                            dmc.Stack(
                                [
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div(
                                            [
                                                "Drag and Drop or",
                                                dmc.Button(
                                                    "Select CSV File",
                                                    ml=10,
                                                    leftIcon=DashIconify(
                                                        icon="material-symbols:upload"
                                                    ),
                                                    style={"textTransform": "capitalize", "background-color": "black"}
                                                ),
                                            ]
                                        ),
                                        max_size=5 * 1024 * 1024,  # 5MB
                                        style={
                                            "borderWidth": "1px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "5px",
                                            "textAlign": "center",
                                            "padding": "10px",
                                            "backgroundColor": "#fafafa",
                                        },
                                        style_reject={
                                            "borderColor": "red",
                                        },
                                        multiple=False,
                                    ),
                                    dmc.Title("Preview", order=3, color="primary"),
                                    html.Div(id="output-data-upload"),
                                ]
                            )
                        ],
                    ),
                    dmc.StepperStep(
                        label="Plot your data 🚀",
                        icon=DashIconify(icon="bi:bar-chart"),
                        progressIcon=DashIconify(icon="bi:bar-chart"),
                        completedIcon=DashIconify(icon="bi:bar-chart-fill"),
                        children=[
                            dmc.Stack(
                                [
                                    dmc.Textarea(
                                        id="input-text",
                                        placeholder="Write here",
                                        autosize=True,
                                        description="""Type in your questions or requests related to your CSV file. GPT will write the code to visualize the data and find the answers you're looking for.""",
                                        maxRows=2,
                                    ),
                                    dmc.Title("Preview", order=3, color="primary"),
                                    html.Div(id="output-data-upload-preview"),
                                ]
                            )
                        ],
                    ),
                    dmc.StepperCompleted(
                        children=[
                            dmc.Stack(
                                [
                                    dmc.Textarea(
                                        id="input-text-retry",
                                        description="""Type in your questions or requests related to your CSV file. GPT will write the code to visualize the data and find the answers you're looking for.""",
                                        placeholder="Write here",
                                        autosize=True,
                                        icon=DashIconify(icon="material-symbols:search"),
                                        maxRows=2,
                                    ),
                                    dmc.LoadingOverlay(
                                        id="output-card",
                                        mih=300,
                                        loaderProps={
                                            "variant": "bars",
                                            "color": "primary",
                                            "size": "xl",
                                        },
                                    ),
                                ]
                            )
                        ]
                    ),
                ],
            ),
            dmc.Group(
                [
                    dmc.Button(
                        "Back",
                        id="stepper-back",
                        display="none",
                        size="md",
                        variant="outline",
                        radius="xl",
                        leftIcon=DashIconify(icon="ic:round-arrow-back"),
                    ),
                    dmc.Button(
                        "Next",
                        id="stepper-next",
                        size="md",
                        radius="xl",
                        rightIcon=DashIconify(
                            icon="ic:round-arrow-forward", id="icon-next"
                        ),
                        style={"textTransform": "capitalize", "background-color": "black"},
                    ),
                ],
                position="center",
                mb=20,
            ),
        ]
    )

    return body
