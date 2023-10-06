import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric_temporal import GConvGRU

MODEL2CLASS = {
    "GConvGRU": GConvGRU,
}


class RecurrentGCN(nn.Module):
    def __init__(self, model, node_features: int, hidden_dim: int = 128,
                 transform_input: bool = True, device: str = "cuda:0",
                 do_node_classification: bool = False, do_node_regression: bool = False,
                 do_edge_classification: bool = False, do_edge_regression: bool = False, do_link_prediction: bool = False,
                 num_classes_nodes: int = 0, num_classes_edges: int = 0):
        r"""
        Initialize Discrete-Time Dynamic Graph (DTDG) model.

        Args:
            model (str): Model type to be used for the GCN.
            node_features (int): Dimension of the input node features.
            hidden_dim (int, optional): Size of the hidden dimension. Defaults to 128.
            transform_input (bool, optional): If True, maps the input features to the same
                                              dimensionality as hidden_dim. Defaults to True.
            device (str, optional): Compute device for the model. Defaults to "cuda:0".
            num_classes_nodes (int, optional): Number of classes for node classification.
                                               Defaults to 0.
            num_classes_edges (int, optional): Number of classes for edge classification.
                                               Defaults to 0.

        """

        super(RecurrentGCN, self).__init__()

        self.transform_input = transform_input
        self.device = device
        self.num_classes_nodes = num_classes_nodes
        self.hidden_dim = hidden_dim
        self.num_classes_edges = num_classes_edges

        self.do_node_classification = do_node_classification
        self.do_node_regression = do_node_regression
        self.do_edge_classification = do_edge_classification
        self.do_edge_regression = do_edge_regression
        self.do_link_prediction = do_link_prediction

        if (do_node_classification and num_classes_nodes == 2) or do_node_regression:
            # For binary classification / regression on nodes, we use a single output neuron.
            self.lin_node = nn.Linear(hidden_dim, 1)

        elif do_node_classification and num_classes_nodes >= 3:
            # For multi-label classification on nodes, we use an MLP with SoftMax.
            self.lin_node = nn.Linear(hidden_dim, num_classes_nodes)

        if (do_edge_classification and self.num_classes_edges == 2) or do_edge_regression:

            self.lin_edge = nn.Linear(hidden_dim, 1)


        elif do_edge_classification and self.num_classes_edges >= 3:
            self.lin_edge = nn.Linear(hidden_dim * 2,
                                                 self.num_classes_edges)


        if self.transform_input:
            self.mlp_input = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
            ).to(device)

        self.recurrent = MODEL2CLASS[model](hidden_dim, hidden_dim, 1).to(device)


        # TODO
        # if args.dataset_name == "Science2013Ant":
        #     # For the Ants dataset, encode the node type as a categorical feature
        #     self.embed_type = nn.Embedding(5, hidden_dim)
        #     self.linear_emb = nn.Linear(hidden_dim * 2, hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, h_0):

        h_1 = F.relu(h_0)
        out = self.node_regressor(h_1)
        return out

    def get_embedding(self, x, edge_index, edge_weight, y):
        if self.transform_input:
            x = self.mlp_input(x)

        h_0 = self.recurrent(x, edge_index, edge_weight)

        # TODO
        if False:  # args.dataset_name in ["Science2013Ant"]:
            h = torch.cat([h_0, self.embed_type(y.int())], dim=1)
            h = F.relu(self.linear_emb(h))
            h = F.dropout(h, p=0.1, training=self.training)
            h = self.linear1(h)

            return h

        else:
            return h_0

    def link_prediction(self, h, src, dst):
        source_node_emb = h[src]
        target_node_emb = h[dst]

        score = torch.sum(source_node_emb * target_node_emb, dim=1)

        return score

    def transform_output(self, h_0):
        out = F.relu(h_0)
        out = F.dropout(out, p=0.1, training=self.training)
        h_1 = self.linear1(out)
        out = F.relu(h_1)
        out = F.dropout(out, p=0.1, training=self.training)

        return out, h_1

    def node_output_layer(self, out):
        score = self.lin_node(out)

        if self.do_node_classification:
            if self.num_classes_nodes == 2:
                score = nn.Sigmoid()(score)
                score = score.squeeze()

            elif self.num_classes_nodes >= 3:
                score = nn.Softmax(dim=1)(score)

            else:
                raise ValueError(
                    "Invalid number of classes for node classification")

        return score

    def edge_output_layer(self, out, edge_index):
        src = out[edge_index[0]]
        dst = out[edge_index[1]]

        score = self.lin_edge(torch.concat([src, dst], dim=1))
        score = nn.LogSoftmax(dim=1)(score)

        return score
