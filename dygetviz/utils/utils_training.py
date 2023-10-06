import warnings

import numpy as np
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n, d = x.size()
    m = y.size(0)

    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def pairwise_cos_sim(A: torch.Tensor, B: torch.Tensor, device="cuda:0"):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)

    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    from torch.nn.functional import normalize
    A_norm = normalize(A, dim=1)  # Normalize the rows of A
    B_norm = normalize(B, dim=1)  # Normalize the rows of B
    cos_sim = torch.matmul(A_norm,
                           B_norm.t())  # Calculate the cosine similarity
    return cos_sim


def pairwise_cos_sim_batch(a: torch.Tensor, b: torch.Tensor, batch_size=16384,
                           device='cuda:0'):
    a = torch.tensor(a, dtype=torch.float, device=device)
    b = torch.tensor(b, dtype=torch.float, device=device)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    res = dot_product_batch(a_norm, b_norm, batch_size, device)

    return res


def dot_product_batch(a: torch.Tensor, b: torch.Tensor, batch_size=16384,
                      device='cuda:0'):
    from tqdm import trange
    res = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float).cpu()

    for i in trange(0, a.shape[0], batch_size, desc="Dot Product"):
        a_batch = a[i:i + batch_size].to(device)
        for j in range(0, b.shape[0], batch_size):
            b_batch = b[j:j + batch_size].to(device)
            res_batch = torch.mm(a_batch, b_batch.transpose(0, 1))
            res[i:i + batch_size, j:j + batch_size] = res_batch.cpu()

    return res


def get_training_args(config: dict):
    training_args = {}


    is_training_objective_set = False
    for field in ["do_node_classification",
                  "do_node_regression",
                  "do_edge_classification",
                  "do_edge_regression",
                  "do_link_prediction"]:
        training_args[field] = config.get(field, False)
        is_training_objective_set = is_training_objective_set or training_args[field]

    assert is_training_objective_set, "At least one training objective must be specified"

    for field in ["num_classes_nodes",
                  "num_classes_edges"]:
        training_args[field] = config.get(field, 0)

    return training_args
