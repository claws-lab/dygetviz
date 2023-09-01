import torch
from torch_geometric.data import Data

import numpy as np

from typing import List, Union

from torch_geometric_temporal import StaticGraphTemporalSignal, \
    DynamicGraphStaticSignal

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]

class StaticGraphStaticSignal(DynamicGraphStaticSignal):
    def __init__(
            self,
            edge_index: List[Edge_Index],
            edge_weight: List[Edge_Weight],
            features: Node_Features,
            targets: Targets,
            **kwargs: Additional_Features
    ):
        r"""
        Dataset for static graph (constant edges / graph connectivities) and static signal (constant edge features).

        Args:
            edge_index (List[Edge_Index]):
            edge_weight (List[Edge_Weight]:
            features (Node_Features)::
            targets (Targets):
            **kwargs (Additional_Features):
        """
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []

        self.node_masks = kwargs.get("node_masks")
        self.edge_type = kwargs.get("edge_type")

        # if self.ys is not None:
        #
        #     # The node labels remain constant over time
        #     if (isinstance(self.ys, list) and len(self.ys) == 1) or (
        #             isinstance(self.ys, np.ndarray) and len(
        #             self.ys.shape) == 1):
        #         self.ys = [self.ys for _ in range(len(self.edge_index))]

        for key, value in kwargs.items():
            if key in ["node_masks", ]:
                continue

            setattr(self, key, value)
            self.additional_feature_keys.append(key)

        self._set_snapshot_count()

    def __len__(self):
        return self.snapshot_count

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = StaticGraphStaticSignal(
                self.edge_index, # The edge_index for StaticGraphStaticSignal remains constant over time, which is different from DynamicGraphStaticSignal. So we only keep one copy of it.
                self.edge_weight[time_index],
                features=self.features,
                targets=self.targets[time_index],
                node_masks=self.node_masks[time_index],
                **{key: getattr(self, key)[time_index] for key in
                   self.additional_feature_keys}
            )


        else:
            x = torch.tensor(self.features, dtype=torch.float)
            edge_index = torch.tensor(self.edge_index[time_index],
                                      dtype=torch.long)
            edge_weight = torch.tensor(self.edge_weight[time_index],
                                       dtype=torch.float)

            additional_features = self._get_additional_features(time_index)
            if isinstance(self.targets, (np.ndarray, list)):
                y = torch.tensor(self.targets[time_index], dtype=torch.float)
                node_mask = torch.tensor(self.node_masks[time_index],
                                         dtype=torch.bool)
                snapshot = Data(x=x, edge_index=edge_index,
                                edge_attr=edge_weight,
                                y=y, node_mask=node_mask, **additional_features)

            else:
                snapshot = Data(x=x, edge_index=edge_index,
                                edge_attr=edge_weight,
                                **additional_features)

        return snapshot

    def __next__(self):
        if self.t < len(self.edge_index):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration