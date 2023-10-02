import json
import os.path as osp
import warnings

import numpy as np
import pandas as pd
from numba import NumbaDeprecationWarning

from arguments import args

from dygetviz.data.chickenpox import ChickenpoxDataset

warnings.simplefilter(action='ignore', category=NumbaDeprecationWarning)


def load_data(dataset_name: str, use_tgb: bool=False) -> dict:
    """
    :return: dict that contains the following fields
        z: np.ndarray of shape (num_nodes, num_timesteps, num_dims): node embeddings
        ys: np.ndarray of shape (num_nodes, num_timesteps): node labels
        node2idx: dict that maps node name to node index
        node_presence: np.ndarray of shape (num_nodes, num_timesteps): 1 if node is present at timestep, 0 otherwise
    """

    if use_tgb:
        config_path = osp.join("config", f"TGBL.json")


    else:
        config_path = osp.join("config", f"{dataset_name}.json")

    config = json.load(open(config_path, 'r', encoding='utf-8'))


    try:
        z = np.load(
            osp.join("data", dataset_name, f"{config['model']}_embeds_{dataset_name}_Ep{config['epoch']}_Emb{config['emb_dim']}.npy"))


    except:
        z = np.load(
            osp.join("data", dataset_name, f"embeds_{dataset_name}.npy"))

    assert len(z.shape) == 3

    path_node2idx = osp.join("data", dataset_name, "node2idx.json")

    if osp.exists(path_node2idx):
        print(f"Try loading node2idx.json from {path_node2idx}")
        node2idx = json.load(open(path_node2idx, 'r', encoding='utf-8'))

    else:
        print(f"node2idx.json not found. Using integer node indices as node names.")
        node2idx = {str(i): i for i in range(z.shape[1])}


    perplexity = config["perplexity"]
    model_name = config["model"]


    num_snapshots = config["num_snapshots"]
    assert num_snapshots == z.shape[0]


    idx_reference_snapshot = config.get("idx_reference_snapshot", None)

    if idx_reference_snapshot is None:
        print(f"idx_reference_snapshot not found in config. Assuming the last snapshot ({idx_reference_snapshot}) is the reference snapshot.")
        idx_reference_snapshot = num_snapshots - 1

    # Optional argument
    # Whether to display node type (e.g. anomalous, normal)
    display_node_type = config.get("display_node_type", False)
    interpolation = config.get("interpolation", 0.2)
    num_nearest_neighbors = config.get("num_nearest_neighbors",
                                       [3, 5, 10, 20, 50])

    path_snapshot_names = osp.join("data", dataset_name, "snapshot_names.csv")

    if osp.exists(path_snapshot_names):
        snapshot_names = pd.read_csv(path_snapshot_names)
        snapshot_names = snapshot_names['snapshot'].values

    else:
        snapshot_names = np.arange(num_snapshots).astype(str)

    plot_anomaly_labels = False

    path_node2label = osp.join("data", dataset_name, "node2label.json")

    if osp.exists(path_node2label):
        node2label = json.load(
            open(osp.join("data", dataset_name, "node2label.json"), 'r',
                 encoding='utf-8'))

    else:
        node2label = {}

    if 'reference_nodes' not in config:
        print("reference_nodes not found in config. Assuming all nodes are reference nodes.")
        reference_nodes = np.array(list(node2idx.keys()))



    elif isinstance(config['reference_nodes'], str) and config[
        'reference_nodes'].endswith("json"):
        reference_nodes = json.load(
            open(osp.join("data", dataset_name, config['reference_nodes']),
                 'r', encoding='utf-8'))

    elif isinstance(config['reference_nodes'], list):
        reference_nodes = config['reference_nodes']

    else:
        raise NotImplementedError


    if 'projected_nodes' not in config:
        print("projected_nodes not found in config. Sample 10 nodes to project.")
        projected_nodes = np.random.choice(list(node2idx.keys()), replace=False, size=10)


        if not projected_nodes.dtype == np.int64:
            projected_nodes = projected_nodes.astype(int)
            projected_nodes.sort()
            projected_nodes = projected_nodes.astype(str)

        else:
            projected_nodes.sort()



    elif isinstance(config['projected_nodes'], str) and config[
        'projected_nodes'].endswith("json"):
        projected_nodes = json.load(
            open(osp.join("data", dataset_name, config['projected_nodes']),
                 'r', encoding='utf-8'))

    elif isinstance(config['projected_nodes'], list):
        projected_nodes = config['projected_nodes']

    else:
        raise NotImplementedError

    projected_nodes = np.array(projected_nodes).astype(str)
    reference_nodes = np.array(reference_nodes).astype(str)

    node_presence = None

    # Dataset-specific node profile
    metadata_df = None

    highlighted_nodes = []

    label2name = {}

    if dataset_name == "Chickenpox":
        highlighted_nodes = np.array(
            ["BUDAPEST", "PEST", "BORSOD", "ZALA", "NOGRAD", "TOLNA", "VAS"])

        projected_nodes = np.array(list(node2idx.keys()))

        # All nodes are present since the very beginning
        node_presence = np.ones((z.shape[0], z.shape[1]), dtype=bool)

        reference_nodes = np.array(list(node2idx.keys()))

        # Only plot the first 100 snapshots, otherwise the plot is too crowded
        snapshot_names = snapshot_names[0:100]

        weekly_cases = pd.read_csv(
            osp.join("data", dataset_name, "hungary_chickenpox.csv"))

        ys = weekly_cases[reference_nodes].values

        metadata_df = pd.DataFrame(index=np.arange(len(reference_nodes)))

        metadata_df["node"] = reference_nodes
        metadata_df["Country"] = "Hungary"


    elif dataset_name == "DGraphFin":
        plot_anomaly_labels = True
        # Eliminate background nodes
        node2label = {n: l for n, l in node2label.items() if l in [0, 1]}

        label2name = {
            0: "normal user",
            1: "fraudster"
        }



    # elif dataset_name == "Reddit":
    #     # 2018-01, ..., 2022-12
    #     snapshot_names = const.dataset_names_60_months


    elif dataset_name == "BMCBioinformatics2021":
        label2node = {
            0: "NON-aging-related",
            1: "aging-related"
        }

        plot_anomaly_labels = True

        metadata_df = pd.read_excel(
            osp.join("data", dataset_name, "metadata.xlsx"))
        metadata_df = metadata_df.rename(columns={"entrez": "node"})
        metadata_df = metadata_df.astype({"node": str})
        node2idx = {str(k): v for k, v in node2idx.items()}

        metadata_df = metadata_df.drop(
            columns=["summary", "lineage", "gene_type"])

    elif dataset_name == "HistWords-CN-GNN":

        metadata_df = pd.read_csv(
            osp.join("data", dataset_name, "metadata.csv"))



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

            path_node_presence = osp.join("data", dataset_name, f"{config['model']}_node_presence_{dataset_name}_Ep{config['epoch']}_Emb{config['emb_dim']}.npy")
            if osp.exists(path_node_presence):

                print(f"Try loading node_presence.npy from {path_node_presence}")
                node_presence = np.load(path_node_presence)

            else:
                node_presence = np.load(
                    osp.join("data", dataset_name, "node_presence.npy"))


            assert len(node_presence.shape) == 2

        except FileNotFoundError:
            print(
                "node_presence.npy not found. Assuming all nodes are present at all timesteps.")
            node_presence = np.ones((z.shape[0], z.shape[1]), dtype=bool)

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "display_node_type": display_node_type,
        "highlighted_nodes": highlighted_nodes,
        "idx_reference_snapshot": idx_reference_snapshot,
        "interpolation": interpolation,
        "label2name": label2name,
        "label2node": label2node,
        "metadata_df": metadata_df,
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



def load_data_dtdg(dataset_name: str):
    """
    Load data for embedding training on Discrete-Time Dynamic-Graph (DTDG) models.
    """
    from torch_geometric_temporal.signal import temporal_signal_split

    if dataset_name == "UNComtrade":

        path = osp.join(args.cache_dir, f"full_dataset_{dataset_name}.pt")
        full_dataset = UNComtradeDataset(args)

    elif dataset_name == "Chickenpox":
        full_dataset = ChickenpoxDataset(args)

    else:
        raise NotImplementedError

    train_dataset, test_dataset = temporal_signal_split(full_dataset,
                                                        train_ratio=1.)

    return train_dataset, test_dataset, full_dataset
