import os
import os.path as osp
import random
from typing import Union
import time 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dygetviz.const as const
import dygetviz.const_viz as const_viz

import plotly.io as pio
from dygetviz.data.dataloader import load_data
from dygetviz.utils.utils_data import get_modified_time_of_file

from dash import dcc, html

def rgb_to_hex(color):
    # Convert a tuple of RGB values to a hex string
    r, g, b = [int(c * 255) for c in color]
    return f'#{r:02x}{g:02x}{b:02x}'

def random_color(hex=False):
    """Archived as the color is not visually appealing

    hex: bool
    """
    if hex:
        color = "#" + ''.join(
            [random.choice('0123456789ABCDEF') for j in range(6)])
    else:
        color = (
            random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    return color


def get_colors(num_colors, palette_name="Spectral"):
    """Get a list of colors through seaborn

    :param palette_name: Name of the color palette, such as `hls`, `Spectral`
    :param num_colors:
    :return:
    """
    import seaborn as sns

    # Use the hls color space from seaborn128
    colors = sns.color_palette(palette_name, n_colors=num_colors)

    # Convert the color tuples to hex strings
    colors = [rgb_to_hex(color) for color in colors]

    return colors


def get_hovertemplate(fields_in_customdata, is_trajectory=False):

    if is_trajectory:
        hovertemplate = '<b>%{hovertext}</b><br><br>node_color=#B2B2B2<br>x=%{x}<br>y=%{y}<br>display_name=%{text}<br><br>'

    else:
        hovertemplate = '<b>%{hovertext}</b><br><br>node_color=#B2B2B2<br>x=%{x}<br>y=%{y}<br><br>'

    for i, field in enumerate(fields_in_customdata):
        # hovertemplate += f"{field}=" + "%{" + f"hover_data_{i}" + "}<br>"
        hovertemplate += f"{field}=" + "%{" + f"customdata[{i}]" + "}<br>"

    hovertemplate += "<extra></extra>"

    return hovertemplate

def get_nodes_and_options(data, visual_dir, visualization_model=const.TSNE):
    dataset_name: str = data['dataset_name']
    model: str = data['model_name']
    annotation: dict = data.get("annotation", {})
    display_node_type: bool = data["display_node_type"]
    idx_reference_snapshot: int = data["idx_reference_snapshot"]
    interpolation: float = data["interpolation"]
    node_presence: np.ndarray = data["node_presence"]
    node2idx: dict = data["node2idx"]
    node2label: dict = data["node2label"]
    label2node: dict = data["label2node"]
    metadata_df: dict = data["metadata_df"]
    num_nearest_neighbors: int = data["num_nearest_neighbors"]
    perplexity: int = data["perplexity"]
    plot_anomaly_labels: bool = data["plot_anomaly_labels"]
    projected_nodes: np.ndarray = data["projected_nodes"]
    reference_nodes: np.ndarray = data["reference_nodes"]
    snapshot_names: list = data["snapshot_names"]
    z = data["z"]
    nodes: list


    idx2node = {idx: node for node, idx in node2idx.items()}


    visualization_name = f"{dataset_name}_{model}_{visualization_model}_perplex{perplexity}_nn{data['num_nearest_neighbors'][0]}_interpolation{interpolation}_snapshot{idx_reference_snapshot}"
    start_time = time.time()
    print("Reading visualization cache...")

    path = osp.join(visual_dir, f"Trajectory_{visualization_name}.json")

    get_modified_time_of_file(path)
    fig_cached = pio.read_json(path)

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time} seconds")
    start_time = time.time()
    
    print("Generating node2trace...")
    start_time = time.time()
    node2trace = {
        trace['name'].split(' ')[0]: trace for trace in fig_cached.data
    }
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time} seconds")
    start_time = time.time()
    print("Getting candidate nodes ...")

    if dataset_name in ["DGraphFin"]:
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
        label2colors = {label: get_colors(12, label2palette[label])[::-1] for label in labels}
    else:
        # Otherwise, we use a single color family for all nodes. But the colors are very distinct
        label2colors = {
            0: get_colors(10, "Spectral")
        }
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time} seconds")
    start_time = time.time()
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
        if display_node_type and dataset_name in [
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
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time} seconds")
    options = options_categories + options_nodes
    return nodes, node2trace, label2colors, options