"""Train TGB embeddings using pytorch-geometric-temporal

"""

import json
import logging
import os
import os.path as osp

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

from data.download import download_file_from_google_drive, download_from_GitHub
from dygetviz.data.dataloader import load_data_dtdg
from dygetviz.model.recurrentgcn import RecurrentGCN
from dygetviz.utils.utils_misc import project_setup
from dygetviz.utils.utils_training import get_training_args
from utils.utils_logging import configure_default_logging

# import torch_geometric_temporal as pygt

configure_default_logging()
logger = logging.getLogger(__name__)


def train_dynamic_graph_embeds(args, dataset_name, device, embedding_dim: int,
                               epochs: int, lr: float, model_name: str,
                               save_every: int,
                               step_size: int = 50, use_pyg=True):
    r"""Train dynamic graph embeddings using the torch-geometric-temporal package

    Args:
        dataset_name (str): Name of the dataset
        device (str): Device for embedding training
        embedding_dim (int): Dimension of the embedding
        epochs (int): Number of epochs to train
        lr (float): Learning rate

        save_every (int): How many epochs to perform evaluation and sae the embeddings
        use_pyg (bool): Whether to use the datasets in
        pytorch-geometric-temporal
        package



    Returns:

    """

    # ROOT_PYG = osp.dirname(pygt.__file__)

    DATASET_NAME2GOOGLE_LINK = {
        "chickenpox": "1u0VDqcn6Nn__k57FYdIX-QWZ2gD53H31",
        "england_covid": "10w-rEdZ_FuSLMbRsYttZIbJEZnYz0XNv",
        "montevideo_bus": "1NQlvofuZNT8C3EDTlyxWPrRGOg-xF_jv",
        "mtm_1": "18UvEl3uUCx5L_LuvMgJfnggTrfe-wQdL",
        "pedalme_london": "1RxpPhdJ4xk8SYd2pARiga6TIza8JXlys",
        "twitter_tennis_uo17": "16qKaHv5FpZg_eKQLohEXFCklieOr7QC5",
        "twitter_tennis_rg17": "14GSXJYKNy9Hl14KeOPAY18qhdJQMI_vd",
        "wikivital_mathematics": "Yk5KJRXjNfRQBzJtbvMsovjV2qQQOTj",
    }

    if dataset_name in DATASET_NAME2GOOGLE_LINK:
        os.makedirs(osp.join("data", dataset_name), exist_ok=True)

        url = (f"https://raw.githubusercontent.com/benedekrozemberczki"
               f"/pytorch_geometric_temporal/master/dataset/{dataset_name}.json")

        download_from_GitHub(url, osp.join("data", dataset_name,
                                f"{dataset_name}_pyg_metadata.json"))

        # Downloading from Google Drive is not recommended. Switched to
        # downloading from pytorch-geometric-temporal package

        # download_file_from_google_drive(DATASET_NAME2GOOGLE_LINK[dataset_name],
        #                         osp.join("data", dataset_name,
        #                         f"{dataset_name}_pyg_metadata.json"))


    else:
        logger.warning(f"Metadata of dataset '{dataset_name}' not found in "
                       f"Google Drive.")

    train_dataset, test_dataset, full_dataset = load_data_dtdg(args, dataset_name,
                                                               use_pyg=use_pyg)

    if use_pyg:
        path_config = osp.join("config", f"PYG.json")

    else:
        path_config = osp.join("config", f"{dataset_name}.json")

    config = json.load(open(path_config, 'r', encoding='utf-8'))

    if use_pyg:
        in_channels = train_dataset.features[0].shape[1]

    else:
        in_channels = config["in_channels"]

    training_args = get_training_args(config)

    # if dataset_name in ["DGraphFin"]:
    #     model = RecurrentGCN(node_features=in_channels,
    #                          hidden_dim=embedding_dim,
    #                          transform_input=transform_input,
    #                          num_classes_nodes=2, num_classes_edges=11)
    #
    # elif dataset_name in ["BMCBioinformatics2021"]:
    #     model = RecurrentGCN(node_features=in_channels,
    #                          hidden_dim=embedding_dim,
    #                          transform_input=transform_input,
    #                          num_classes_nodes=2)
    #
    #
    # else:

    transform_input = in_channels != embedding_dim
    model = RecurrentGCN(model=model_name,
                         node_features=in_channels,
                         hidden_dim=embedding_dim,
                         transform_input=transform_input, device=device,
                         **training_args)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.9)

    def train_one_epoch(epoch: int, embeds_li: list):
        model.train()
        scheduler.step()

        if epoch % step_size == 0:
            logger.info(f"Epoch: {epoch}, LR: {scheduler.get_lr()}")

        cost = 0
        node_mask = None

        for idx_snapshot, snapshot in enumerate(train_dataset):
            log_str = f"\tEp.{epoch}\tIter{idx_snapshot}"
            snapshot = snapshot.to(device)

            node_mask = torch.ones(snapshot.num_nodes,
                                   dtype=torch.bool) if not hasattr(snapshot,
                                                                    "node_mask") else snapshot.node_mask

            h_0 = model.get_embedding(snapshot.x, snapshot.edge_index,
                                      snapshot.edge_attr, snapshot.y)

            if training_args["do_link_prediction"]:
                nodes = snapshot.edge_index.unique().cpu()
                nodes = nodes.numpy().reshape(-1)
                neg_target = np.random.choice(nodes,
                                              size=snapshot.edge_index.shape[1])
                neg_target = torch.tensor(neg_target, dtype=torch.long,
                                          device=device)

                pos_score = model.link_prediction(h_0, snapshot.edge_index[0],
                                                  snapshot.edge_index[1])
                neg_score = -model.link_prediction(h_0, snapshot.edge_index[0],
                                                   neg_target)

                loss_link_pred = model.bceloss(pos_score,
                                               torch.ones_like(
                                                   pos_score)) + model.bceloss(
                    neg_score, torch.ones_like(neg_score))

                log_str += f"\tLink: {loss_link_pred.item():.3f}"

                cost += loss_link_pred

            # Transform the embedding from the model's output
            if training_args["do_node_regression"] or training_args[
                "do_edge_regression"] or training_args[
                "do_node_classification"] or \
                    training_args["do_edge_classification"]:
                out, h_1 = model.transform_output(h_0)
                emb = h_1

            if training_args["do_node_regression"]:
                pred_node = model.node_output_layer(out)

                loss_node_regression = torch.mean(
                    (pred_node[node_mask] - snapshot.y[
                        node_mask]) ** 2)

                cost += loss_node_regression * 0.1

                log_str += f"\tNode Reg: {loss_node_regression.item():.3f}"

            if training_args["do_edge_regression"]:
                edge_features = torch.cat(
                    [out[snapshot.edge_index[0]], out[snapshot.edge_index[1]]],
                    dim=1)
                pred_edge = model.edge_output_layer(edge_features)

                loss_edge_regression = torch.mean(
                    (pred_edge.squeeze(1) - snapshot.edge_attr) ** 2)

                cost += loss_edge_regression * 0.1

                log_str += f"\tEdge Reg: {loss_edge_regression.item():.3f}"

            if training_args["do_node_classification"]:

                # if False:
                if dataset_name in ["DGraphFin", "BMCBioinformatics2021"]:

                    if idx_snapshot == len(full_dataset) - 1:

                        node_mask = node_mask if node_mask is not None else (
                                                                                    snapshot.y == 0) | (
                                                                                    snapshot.y == 1)

                        pred_node = model.node_output_layer(out[node_mask])

                        loss_node_classification = nn.BCELoss()(pred_node,
                                                                snapshot.y[
                                                                    node_mask])

                        cost += loss_node_classification * 1.

                        log_str += f"\tNode Cls: {loss_node_classification.item():.3f}"

                    else:

                        loss_node_classification = 0.


                else:
                    node_mask = node_mask if node_mask is not None else (
                                                                                snapshot.y == 0) | (
                                                                                snapshot.y == 1)
                    pred_node = model.node_output_layer(out[node_mask])

                    loss_node_classification = nn.BCELoss()(pred_node,
                                                            snapshot.y[
                                                                node_mask])

                    cost += loss_node_classification * 1.

                    log_str += f"\tNode Cls: {loss_node_classification.item():.3f}"

            if training_args["do_edge_classification"] or training_args[
                "do_edge_regression"]:
                edge_score = model.edge_output_layer(out, snapshot.edge_index)

                # For DGraphFin, edge_type is in 1 - 11
                edge_type = snapshot.edge_type - 1 if dataset_name == "DGraphFin" else snapshot.edge_type
                loss_edge_classification = nn.NLLLoss()(edge_score, edge_type)

                cost += loss_edge_classification * 1.

                log_str += f"\tEdge Cls: {loss_edge_classification.item():.3f}"

            # logger.info(log_str)
            if (epoch + 1) % save_every == 0:
                embeds_li += [emb.detach().cpu().numpy()]

        cost = cost / (idx_snapshot + 1)
        total_loss = cost.item()
        cost.backward()

        optimizer.step()
        optimizer.zero_grad()

        return total_loss

    cache_dir = osp.join("data", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    pbar = trange(1, epochs + 1, desc='Train', leave=True)

    for epoch in pbar:

        # Store the embeddings at each epoch
        embeds_li = []
        loss = train_one_epoch(epoch, embeds_li)
        pbar.set_postfix({
            "epoch": epoch,
            "loss": "{:.3f}".format(loss)

        })

        if (epoch + 1) % save_every == 0:
            embeds: np.ndarray = np.stack([emb for emb in embeds_li])
            del embeds_li
            logger.info(
                f"[Embeds] Saving embeddings for Ep. {epoch + 1} with shape {embeds.shape} ...")
            np.save(osp.join(cache_dir,
                             f"{model_name}_embeds_{dataset_name}_Ep{epoch + 1}_Emb{embedding_dim}.npy"),
                    embeds)


if __name__ == '__main__':
    from arguments import parse_args

    args = parse_args()

    project_setup()

    dataset_name: str = args.dataset_name
    device: str = args.device
    embedding_dim: int = args.embedding_dim
    epochs: int = args.epochs
    save_every = args.eval_every
    lr: float = args.lr
    model_name: str = args.model

    train_dynamic_graph_embeds(args, dataset_name, device, embedding_dim,
                               epochs, lr, model_name, save_every)
