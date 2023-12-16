import os.path as osp
import pickle
import zipfile

import torch
import pandas as pd
import numpy as np
from typing import List, Union
from torch_geometric.data import Data
from torch_geometric_temporal import DynamicGraphStaticSignal

from data.dygetviz_dataset import DyGETVizDataset
from data.static_graph_static_signal import StaticGraphStaticSignal

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]

class ChickenpoxDataset(StaticGraphStaticSignal, DyGETVizDataset):
    def __init__(self, args, **kwargs: Additional_Features):
        self.args = args

        self.dataset_name = "Chickenpox"

        DyGETVizDataset.__init__(self, self.dataset_name, **kwargs)


        if osp.exists(self.dataset_path):

            with open(self.dataset_path, "rb") as f:
                d = pickle.load(f)

        else:
            self.download()

            d = self.process()



        node2idx = d["node2idx"]
        targets = d["targets"]
        node_presence = d["node_presence"]
        edge_index = d["edge_index"]
        edge_weight = d["edge_weight"]

        self.num_nodes = len(node2idx)

        limit = np.sqrt(6 / (self.num_nodes + args.embedding_dim))
        features = np.random.uniform(-limit, limit, size=(
            self.num_nodes, args.embedding_dim))

        StaticGraphStaticSignal.__init__(self,
            edge_index=edge_index,
            edge_weight=edge_weight,
            features=features,
            targets=targets,
            node_masks=node_presence,
            **kwargs
        )


    def process(self):
        mapping = pd.read_excel(osp.join(self.cache_dir, "raw_data", "idx2county.xlsx"))

        self.nodes = mapping["county"].values

        node2idx = {row["id"]: row["county"] for idx, row in
                    mapping.iterrows()}

        idx2node = {v: k for k, v in node2idx.items()}

        edges = pd.read_csv(osp.join(self.cache_dir, "raw_data", "hungary_edges.csv"))

        edge_index = [edges[["id_1", "id_2"]].values.T for i in range(522)]

        edge_weight = [np.ones(edges.shape[0]) for i in range(522)]

        # We use the actual #weekly cases as the ground-truth
        weekly_cases = pd.read_csv(
            osp.join(self.cache_dir, "raw_data", "hungary_weekly_chickenpox_cases.csv"))


        # We predict the log2 of the weekly cases
        targets = weekly_cases.loc[:, self.nodes].values
        # There are 522 weeks in total
        targets = [np.log2(targets[i] + 1) for i in range(522)]

        node_presence = np.ones((522, len(self.nodes)))



        d = {
            "targets": targets,
            "node_presence": node_presence,
            "node2idx": node2idx,
            "idx2node": idx2node,
            "edge_index": edge_index,
            "edge_weight": edge_weight
        }

        with open(self.dataset_path, "wb") as f:
            pickle.dump(d, f)

        return d