import json
import os.path as osp
import warnings

import numpy as np
import pandas as pd
from numba import NumbaDeprecationWarning
from torch_geometric_temporal import ChickenpoxDatasetLoader, \
    EnglandCovidDatasetLoader, METRLADatasetLoader, MontevideoBusDatasetLoader, \
    PedalMeDatasetLoader, WikiMathsDatasetLoader, \
    WindmillOutputLargeDatasetLoader, WindmillOutputMediumDatasetLoader, \
    WindmillOutputSmallDatasetLoader, PemsBayDatasetLoader, \
    TwitterTennisDatasetLoader

from dygetviz.arguments import parse_args
from dygetviz.data.download import download_file_from_google_drive

from dygetviz.data.chickenpox import ChickenpoxDataset
from dygetviz.generate_dtdg_embeds_tgb import train_dynamic_graph_embeds_tgb
from dygetviz.utils.utils_logging import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=NumbaDeprecationWarning)


def load_data(dataset_name=args.dataset_name) -> dict:
    """
    :return: dict that contains the following fields
        z: np.ndarray of shape (num_nodes, num_timesteps, num_dims): node embeddings
        ys: np.ndarray of shape (num_nodes, num_timesteps): node labels
        node2idx: dict that maps node name to node index
        node_presence: np.ndarray of shape (num_nodes, num_timesteps): 1 if node is present at timestep, 0 otherwise
    """

    config = json.load(
        open(osp.join("config", f"{dataset_name}.json"), 'r',
             encoding='utf-8'))

    z = np.load(
        osp.join("data", dataset_name, f"embeds_{dataset_name}.npy"))
    node2idx = json.load(
        open(osp.join("data", dataset_name, "node2idx.json"), 'r',
             encoding='utf-8'))
    perplexity = config["perplexity"]
    model_name = config["model_name"]
    idx_reference_snapshot = config["idx_reference_snapshot"]

    # Optional argument
    # Whether to display node type (e.g. anomalous, normal)
    display_node_type = config.get("display_node_type", False)
    interpolation = config.get("interpolation", 0.2)
    num_nearest_neighbors = config.get("num_nearest_neighbors",
                                       [3, 5, 10, 20, 50])

    snapshot_names = pd.read_csv(osp.join("data", args.dataset_name, f"snapshot_names.csv"))
    snapshot_names = snapshot_names['snapshot'].values

    plot_anomaly_labels = False

    try:
        node2label = json.load(
            open(osp.join("data", dataset_name, "node2label.json"), 'r',
                 encoding='utf-8'))



    except FileNotFoundError:
        node2label = {}

    if isinstance(config['reference_nodes'], str) and config[
        'reference_nodes'].endswith("json"):
        reference_nodes = json.load(
            open(osp.join("data", dataset_name, config['reference_nodes']),
                 'r', encoding='utf-8'))

    elif isinstance(config['reference_nodes'], list):
        reference_nodes = config['reference_nodes']

    else:
        raise NotImplementedError

    if isinstance(config['projected_nodes'], str) and config[
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
    metadata_df = None # pd.DataFrame()

    highlighted_nodes = []

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



    # elif dataset_name == "Reddit":
    #     # 2018-01, ..., 2022-12
    #     snapshot_names = const.dataset_names_60_months


    elif dataset_name == "BMCBioinformatics2021":


        plot_anomaly_labels = True

        metadata_df = pd.read_excel(
            osp.join("data", dataset_name, "metadata.xlsx"))
        metadata_df = metadata_df.rename(columns={"entrez": "node"})
        metadata_df = metadata_df.astype({"node": str})
        node2idx = {str(k): v for k, v in node2idx.items()}

        metadata_df = metadata_df.drop(
            columns=["summary", "lineage", "gene_type"])

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
                osp.join("data", dataset_name, "node_presence.npy"))

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
