import argparse
import json
import logging
import os
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

from arguments import parse_args
from data.download import download_file_from_google_drive

from data.chickenpox import ChickenpoxDataset
from generate_dtdg_embeds_tgb import train_dynamic_graph_embeds_tgb
from utils.utils_logging import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=NumbaDeprecationWarning)


NAME2DATASET_LOADER = {
    "chickenpox": ChickenpoxDatasetLoader,
    "england_covid": EnglandCovidDatasetLoader,
    "metrla": METRLADatasetLoader, #  As of Oct. 2023, this dataset cannot be retrieved
    "montevideo_bus": MontevideoBusDatasetLoader,
    "pedalme_london": PedalMeDatasetLoader,
    "pemsbay": PemsBayDatasetLoader,
    "twitter_tennis_uo17": TwitterTennisDatasetLoader,
    "twitter_tennis_rg17": TwitterTennisDatasetLoader,
    "wikivital_mathematics": WikiMathsDatasetLoader,
    "windmill_large": WindmillOutputLargeDatasetLoader,
    "windmill_medium": WindmillOutputMediumDatasetLoader,
    "windmill_small": WindmillOutputSmallDatasetLoader,

}


def load_data(dataset_name: str, use_tgb: bool=False) -> dict:
    """
    Loads data for dynamic node embedding trajectory visualization. For PyTorch Geometric Temporal (PyG-T) and DGB,
    if the data files do not exist, train the embeddings.

    Args:
        dataset_name (str): Name of the dataset to load.
        use_tgb (bool, optional): Whether to load a [Temporal Graph Benchmark(TGB)](https://tgb.complexdatalab.com/) dataset

    Returns:
        dict: A dictionary containing various fields:
            - 'dataset_name' (str): Name of the loaded dataset.
            - 'model_name' (str): Model name from the configuration.
            - 'display_node_type' (bool): Whether to display the node type.
            - 'highlighted_nodes' (list): List of nodes to be highlighted.
            - 'idx_reference_snapshot' (int): Index of the reference snapshot.
            - 'interpolation' (float): Interpolation parameter.
            - 'label2name' (dict): Mapping of label indices to label names.
            - 'label2node' (dict): Mapping of label indices to nodes.
            - 'metadata_df' (DataFrame or None): Metadata DataFrame if available.
            - 'node2idx' (dict): Mapping of node names to node indices.
            - 'node2label' (dict): Mapping of node names to labels.
            - 'node_presence' (np.ndarray or None): Binary array indicating node presence at each timestep.
            - 'num_nearest_neighbors' (list): List of numbers specifying the nearest neighbors for each node.
            - 'perplexity' (float): Perplexity parameter in t-SNE.
            - 'plot_anomaly_labels' (bool): Whether to plot anomaly labels, e.g. a user is a fake news spreader v.s.
            a normal user, a  gene is aging--related v.s. not.
            - 'projected_nodes' (np.ndarray): Array of nodes to be projected.
            - 'reference_nodes' (np.ndarray): Array of nodes in the reference frame.
            - 'snapshot_names' (np.ndarray or list): Names or indices of snapshots.
            - 'z' (np.ndarray): Node embeddings of shape (num_nodes, num_timesteps, num_dims).

    Raises:
        NotImplementedError: If the 'reference_nodes' or 'projected_nodes' format in the config is not supported.
        AssertionError: If the dimensions of loaded 'z' or 'node_presence' are incorrect.
        FileNotFoundError: If 'node_presence' file is not found and defaults have to be used.

    """

    if use_tgb:
        config_path = osp.join("config", f"TGBL.json")


    else:
        config_path = osp.join("config", f"{dataset_name}.json")

    config = json.load(open(config_path, 'r', encoding='utf-8'))


    try:

        embedding_path = osp.join("data", dataset_name, f"{config['model_name']}_embeds_{dataset_name}_Ep{config['epoch']}_Em"
                                           f"b{config['emb_dim']}.npy")

        if not osp.exists(embedding_path):
            logger.info(f"Embedding file {embedding_path} not found. Training the embeddings.")

            training_config = argparse.Namespace(**json.load(open(osp.join("config", 'TGB_training.json'), 'r',
                                                                  encoding='utf-8')))
            train_dynamic_graph_embeds_tgb(training_config)

        z = np.load(embedding_path)


    except:

        # Try the simplified naming convention
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
    model_name = config["model_name"]

    num_snapshots = z.shape[0]


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
        # snapshot_names = snapshot_names[0:100]

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
        label2name = {
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

            node_presence = None

            if "epoch" in config:

                path_node_presence = osp.join("data", dataset_name, f"{config['model_name']}_node_presence_{dataset_name}_Emb{config['emb_dim']}.npy")
                if osp.exists(path_node_presence):

                    print(f"Try loading node_presence.npy from {path_node_presence}")
                    node_presence = np.load(path_node_presence)

            if node_presence is None:
                node_presence = np.load(
                    osp.join("data", dataset_name, "node_presence.npy"))

                node_presence = node_presence.astype(bool)


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


def load_data_description(dataset_name: str, args) -> str:
    path = osp.join(args.data_dir, dataset_name,  f"data_descriptions.md")

    if osp.exists(path):
        data_description = open(path, 'r', encoding='utf-8').read()
    else:
        data_description = ""

    return data_description



def load_data_dtdg(args, dataset_name: str, use_pyg: bool=False) -> tuple:
    """
    Load data for embedding training on Discrete-Time Dynamic-Graph (DTDG) models.
    """
    from torch_geometric_temporal.signal import temporal_signal_split


    if use_pyg:

        print(f"Using PyG-Temporal {dataset_name} dataset.")

        assert dataset_name in NAME2DATASET_LOADER, f"Invalid dataset name {dataset_name} for pytorch-geometric-temporal."

        os.makedirs("pygt_data", exist_ok=True)

        if dataset_name == "metrla":
            loader = METRLADatasetLoader(raw_data_dir="/tmp/")
            full_dataset = loader.get_dataset(num_timesteps_in=6,
                                         num_timesteps_out=5)
            
        elif dataset_name == "twitter_tennis_rg17":
            # Roland - Garros 2017("rg17")
            # TOOD: Not sure what "mode" means here
            loader = TwitterTennisDatasetLoader("rg17", 1000, None)
            full_dataset = loader.get_dataset()
            
        elif dataset_name == "twitter_tennis_uo17":
            loader = TwitterTennisDatasetLoader("rg17", 1000, None)
            # USOpen 2017("uo17")
            full_dataset = loader.get_dataset()
            
        else:
            loader = NAME2DATASET_LOADER[dataset_name]()
            full_dataset = loader.get_dataset()
            
    else:

        if dataset_name == "UNComtrade":
    
            path = osp.join(args.cache_dir, f"full_dataset_{dataset_name}.pt")
            # TODO
            # full_dataset = UNComtradeDataset(args)
    
        elif dataset_name == "Chickenpox":
            full_dataset = ChickenpoxDataset(args)
    
        else:
            raise NotImplementedError

    train_dataset, test_dataset = temporal_signal_split(full_dataset,
                                                        train_ratio=1.)

    return train_dataset, test_dataset, full_dataset
