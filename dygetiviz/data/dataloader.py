import json
import os.path as osp
import warnings

import numpy as np
import pandas as pd
from numba import NumbaDeprecationWarning

import const
from arguments import args

warnings.simplefilter(action='ignore', category=NumbaDeprecationWarning)


def load_data() -> dict:
    """

    :return: dict that contains the following fields
        z: np.ndarray of shape (num_nodes, num_timesteps, num_dims): node embeddings
        ys: np.ndarray of shape (num_nodes, num_timesteps): node labels
        node2idx: dict that maps node name to node index
        node_presence: np.ndarray of shape (num_nodes, num_timesteps): 1 if node is present at timestep, 0 otherwise
    """

    config = json.load(
        open(osp.join("config", f"{args.dataset_name}.json"), 'r',
             encoding='utf-8'))

    z = np.load(
        osp.join("data", args.dataset_name, f"embeds_{args.dataset_name}.npy"))
    node2idx = json.load(
        open(osp.join("data", args.dataset_name, "node2idx.json"), 'r',
             encoding='utf-8'))
    perplexity = config["perplexity"]

    idx_reference_snapshot = config["idx_reference_snapshot"]


    # Optional arguments
    # Whether to display node type (e.g. anomalous, normal)
    display_node_type = config.get("display_node_type", False)
    interpolation = config.get("interpolation", 0.2)
    num_nearest_neighbors = config.get("num_nearest_neighbors",
                                       [3, 5, 10, 20, 50])

    plot_anomaly_labels = False

    try:
        node2label = json.load(
            open(osp.join("data", args.dataset_name, "node2label.json"), 'r',
                 encoding='utf-8'))



    except FileNotFoundError:
        node2label = {}

    if isinstance(config['reference_nodes'], str) and config[
        'reference_nodes'].endswith("json"):
        reference_nodes = json.load(
            open(osp.join("data", args.dataset_name, config['reference_nodes']),
                 'r', encoding='utf-8'))

    elif isinstance(config['reference_nodes'], list):
        reference_nodes = config['reference_nodes']

    else:
        raise NotImplementedError

    if isinstance(config['projected_nodes'], str) and config[
        'projected_nodes'].endswith("json"):
        projected_nodes = json.load(
            open(osp.join("data", args.dataset_name, config['projected_nodes']),
                 'r', encoding='utf-8'))

    elif isinstance(config['projected_nodes'], list):
        projected_nodes = config['projected_nodes']

    else:
        raise NotImplementedError

    node_presence = None

    if args.dataset_name == "Chickenpox":
        projected_nodes = np.array(
            ["BUDAPEST", "PEST", "BORSOD", "ZALA", "NOGRAD", "TOLNA", "VAS"])

        # All nodes are present since the very beginning
        node_presence = np.ones((z.shape[0], z.shape[1]), dtype=bool)

        reference_nodes = np.array(list(node2idx.keys()))

        weekly_cases = pd.read_csv(
            osp.join("data", args.dataset_name, "hungary_chickenpox.csv"))

        ys = weekly_cases[reference_nodes].values

        snapshot_names = np.arange(0, 522, 1)

    elif args.dataset_name == "DGraphFin":
        plot_anomaly_labels = True
        # Eliminate background nodes
        node2label = {n: l for n, l in node2label.items() if l in [0, 1]}
        snapshot_names = np.arange(17)


    elif args.dataset_name == "Reddit":
        # 2018-01, ..., 2022-12
        snapshot_names = const.dataset_names_60_months


    elif args.dataset_name == "BMCBioinformatics2021":
        snapshot_names = [20, 21, 22, 26, 28, 30, 33, 34, 36, 37, 40, 42, 44,
                          45, 47, 48, 52, 64,
                          66, 69, 70, 74, 75, 80, 82, 83, 84, 85, 86, 87, 90,
                          91, 92, 95, 96, 97,
                          99]


    else:
        # ys = np.load(
        #     osp.join("data", args.dataset_name, f"{args.dataset_name}_ys.npy"))
        pass

    try:
        label2node = {}

        for node, label in node2label.items():
            if label not in label2node:
                label2node[label] = []

            label2node[label].append(node)

    except:
        label2node = {}


    if node_presence is None:
        try:
            node_presence = np.load(
                osp.join("data", args.dataset_name, "node_presence.npy"))

        except FileNotFoundError:
            print(
                "node_presence.npy not found. Assuming all nodes are present at all timesteps.")
            node_presence = np.ones((z.shape[0], z.shape[1]), dtype=bool)

    return {
        "display_node_type": display_node_type,
        "idx_reference_snapshot": idx_reference_snapshot,
        "interpolation": interpolation,
        "label2node": label2node,
        "node2idx": node2idx,
        "node2label": node2label,
        "node_presence": node_presence,
        "num_nearest_neighbors": num_nearest_neighbors,
        "perplexity": perplexity,
        "plot_anomaly_labels": plot_anomaly_labels,
        "projected_nodes": projected_nodes,
        "reference_nodes": reference_nodes,
        "snapshot_names": snapshot_names,
        "z": z,

    }
