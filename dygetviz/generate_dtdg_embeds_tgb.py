"""
Dynamic Link Prediction with a TGN model with Early Stopping.

References
----------
- https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

Example
-------
To run this script, execute the following command:

.. code-block:: bash

    python examples/linkproppred/tgbl-coin/tgn.py --data "tgbl-coin" --num_run 1 --seed 1

"""

import argparse
import json
import logging
import os
import os.path as osp
import timeit
from pathlib import Path
from typing import List

import numpy as np
import torch
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tgb.utils.utils import set_random_seed, save_results
from torch_geometric.loader import TemporalDataLoader
from tqdm import trange, tqdm

from arguments import parse_args
from model.tgb_modules.decoder import LinkPredictor
from model.tgb_modules.early_stopping import EarlyStopMonitor
from model.tgb_modules.emb_module import GraphAttentionEmbedding
from model.tgb_modules.memory_module import TGNMemory
from model.tgb_modules.msg_agg import LastAggregator
from model.tgb_modules.msg_func import IdentityMessage
from model.tgb_modules.neighbor_loader import LastNeighborLoader
from utils.utils_logging import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)

# Use the CPU as a fallback for GPU-only operations.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def train(args, assoc, criterion, data, device, device_viz, embeds_li: list, epoch: int,
          max_dst_idx: int,
          min_dst_idx: int, model, neighbor_loader, node_presence_li, optimizer,
          snapshot_indices, store_embeds: bool,
          train_data, train_loader, ):
    r"""Trains the Temporal Graph Networks (TGN) model for 1 epoch.

    Save embeddings if necessary.

    Args:
        args (Namespace): Arguments parsed from the command line.
        assoc (Tensor): The association tensor mapping node IDs to their index.
        criterion (Loss Function): The loss function to use.
        data (Data): Graph data.
        device (torch.device): Device to perform computations.
        device_viz (torch.device): Device to store visualization data.
        embeds_li (list): List to store node embeddings.
        epoch (int): Current training epoch.
        max_dst_idx (int): Maximum destination index for negative sampling.
        min_dst_idx (int): Minimum destination index for negative sampling.
        model (dict): Dictionary containing model components ('memory', 'gnn', 'link_pred').
        neighbor_loader (DataLoader): DataLoader for neighbors.
        node_presence_li (list): List to store node presence information.
        optimizer (Optimizer): Optimizer to use.
        snapshot_indices (list): List of snapshot indices for embedding storage.
        store_embeds (bool): Whether to store embeddings for visualization.
        train_data (Data): Training data.
        train_loader (DataLoader): DataLoader for training data.

    Returns:
        float: Average loss over the number of events.
        np.ndarray: Latest node embeddings for visualization, if 'store_embeds' is True.
        np.ndarray: Latest node presence vector, if 'store_embeds' is True.
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    # Index of the current snapshot for embedding storage
    idx_snapshot = 1

    if store_embeds:
        embeddings = torch.zeros((data.num_nodes, args.embedding_dim),
                                 dtype=torch.float32, device=device_viz)

        # Whether the node has appeared in the current snapshot.
        node_presence = torch.zeros((data.num_nodes,),
                                    dtype=torch.bool, device=device_viz)

    else:
        embeddings = node_presence = None

    for idx_batch, batch in enumerate(train_loader):
        batch = batch.to(device)

        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        if store_embeds:
            # Store the up-to-date node embeddings to the rolling cache.
            embeddings[n_id.to(device_viz)] = z.to(device_viz)
            # The number of node present in the snapshot should be non-decreasing.
            node_presence[n_id] = True

            # logger.info(node_presence[idx_snapshot].sum())
            if idx_batch + 1 == snapshot_indices[idx_snapshot]:
                # We are starting a new snapshot.
                embeds_li += [embeddings.detach().cpu().numpy()]
                node_presence_li += [node_presence.detach().cpu().numpy()]

                idx_snapshot += 1

                logger.info(
                    f"[Epoch {epoch}]: {idx_snapshot}-th snapshot (#Nodes={node_presence.sum().item()}) saved at batch {idx_batch}.")

                # All nodes present in the prev snapshot is considered present in the next snapshot.
                # node_presence[idx_snapshot + 1] = node_presence[idx_snapshot]

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events, embeddings, node_presence


@torch.no_grad()
def test(assoc, data, device, device_viz, embeddings: torch.Tensor, embeds_li: list, epoch: int, evaluator, loader,
         metric, model, neg_sampler,
         neighbor_loader, node_presence: torch.Tensor, node_presence_li: List[np.ndarray], snapshot_indices,
         split_mode: str, start_batch_index: int, store_embeds: bool):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        assoc: a vector that maps global node indices to local ones
        data: the dataset object
        device: the device for training
        device_viz: the device for storing all embeddings for visualization
        latest_embeddings: the latest node embeddings for visualization
        latest_node_presence: the latest node presence vector (whether the node has appeared in the latest snapshot)
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: either 'val' or 'test'
        store_embeds: whether to store the embeddings for visualization
    Returns:
        perf_metric: the result of the performance evaluaiton
    """

    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    if store_embeds:
        assert isinstance(node_presence_li, list), "node_presence_li must be a list"
        assert isinstance(node_presence, torch.Tensor), "latest_node_presence must be a torch.Tensor"
        assert isinstance(embeddings, torch.Tensor), "latest_node_presence must be a torch.Tensor"

    else:
        embeddings = node_presence = None

    idx_snapshot = len(embeds_li) + 1

    for idx_batch, pos_batch in enumerate(tqdm(loader, desc=f"{split_mode.capitalize()}", position=0, leave=True)):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t,
                                                 split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]),
                      np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            if store_embeds:
                # Store the up-to-date node embeddings to the rolling cache.
                embeddings[n_id.to(device_viz)] = z.to(device_viz)
                # The number of node present in the snapshot should be non-decreasing.
                node_presence[n_id] = True

                # logger.info(node_presence[idx_snapshot].sum())
                # Pad by the number of batches in the training set
                # For test, we will add #batches in the validation set)
                if idx_snapshot < len(snapshot_indices) and idx_batch + start_batch_index + 1 == snapshot_indices[
                    idx_snapshot]:
                    # We are starting a new snapshot.
                    embeds_li += [embeddings.detach().cpu().numpy()]
                    node_presence_li += [node_presence.detach().cpu().numpy()]
                    idx_snapshot += 1

                    logger.info(
                        f"[Epoch ({epoch})]: {idx_snapshot}-th snapshot (#Nodes={node_presence.sum().item()}) saved at batch {idx_batch + start_batch_index}.")

                    # All nodes present in the prev snapshot is considered present in the next snapshot.
                    # node_presence[idx_snapshot + 1] = node_presence[idx_snapshot]

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    # if store_embeds:
    #     # Save the last snapshot
    #     embeds_li += [embeddings.detach().cpu().numpy()]
    #     node_presence_li += [node_presence.detach().cpu().numpy()]

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics, embeddings, node_presence


def train_dynamic_graph_embeds_tgb(training_config):
    r"""Trains dynamic graph embeddings using the TGN model.

    This function launches the training of dynamic graph embeddings
    using the Temporal Graph Networks (TGN) model.

    Args:
        training_config (Namespace): Training config.

    Note:
        The function uses global variable `args` for some of the parameters.

    """


    training_config.model = "tgn"
    start_overall = timeit.default_timer()

    # ========== set parameters...

    tgb_args = argparse.Namespace(
        **json.load(open(osp.join("config", 'TGBL.json'))))

    logger.info("Arguments")
    logger.info(tgb_args)

    DATA = training_config.dataset_name
    logger.info(f"Dataset: {DATA}")

    LR = tgb_args.lr
    BATCH_SIZE = tgb_args.bs
    SEED = tgb_args.seed
    MEM_DIM = tgb_args.mem_dim
    TIME_DIM = tgb_args.time_dim
    EMB_DIM = training_config.embedding_dim  # tgb_args.emb_dim
    TOLERANCE = tgb_args.tolerance
    PATIENCE = tgb_args.patience
    NUM_RUNS = tgb_args.num_run
    NUM_NEIGHBORS = 10

    MODEL_NAME = 'TGN'
    # ==========

    device = training_config.device
    # Device for storing all embeddings for visualization
    device_viz = training_config.device_viz

    # data loading
    dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")

    mask = torch.zeros_like(dataset.edge_label, dtype=torch.bool)
    train_mask = mask.clone()
    val_mask = mask.clone()
    test_mask = mask.clone()

    N = dataset.edge_label.shape[0]
    train_mask[:int(training_config.train_size * N)] = True
    val_mask[int(training_config.train_size * N): int((training_config.train_size + training_config.val_size) * N)] = True
    test_mask[int((training_config.train_size + training_config.val_size) * N):] = True

    data = dataset.get_TemporalData()

    # Extra step for Mac OS X: Convert dtype to float64
    data.y = data.y.type(torch.float32)
    data.edge_stores[0]['y'] = data.edge_stores[0]['y'].type(torch.float32)
    data.node_stores[0]['y'] = data.node_stores[0]['y'].type(torch.float32)

    data = data.to(device)
    metric = dataset.eval_metric

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # neighhorhood sampler
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS,
                                         device=device)

    # define the model end-to-end
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        MEM_DIM,
        TIME_DIM,
        message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM,
        out_channels=EMB_DIM,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

    model = {'memory': memory,
             'gnn': gnn,
             'link_pred': link_pred}

    optimizer = torch.optim.Adam(
        set(model['memory'].parameters()) | set(
            model['gnn'].parameters()) | set(model['link_pred'].parameters()),
        lr=LR,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    logger.info("==========================================================")
    logger.info(
        f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
    logger.info("==========================================================")

    evaluator = Evaluator(name=DATA)
    neg_sampler = dataset.negative_sampler

    # for saving the results...
    results_path = f'saved_results'
    if not osp.exists(results_path):
        os.mkdir(results_path)
        logger.info('Create directory {}'.format(results_path))
    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

    for run_idx in range(NUM_RUNS):
        logger.info(
            '-------------------------------------------------------------------------------')
        logger.info(f">>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()

        # set the seed for deterministic results...
        torch.manual_seed(run_idx + SEED)
        set_random_seed(run_idx + SEED)

        # define an early stopper
        save_model_dir = f'saved_models/'
        save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir,
                                         save_model_id=save_model_id,
                                         tolerance=TOLERANCE, patience=PATIENCE)

        # ==================================================== Train & Validation
        # loading the validation negative samples
        dataset.load_val_ns()

        val_perf_list = []
        start_train_val = timeit.default_timer()

        # Store the trained embeddings

        # Snapshot indices (the `step` in each epoch) to save the embeddings for visualization.

        num_batches = len(train_loader)
        if training_config.do_val:
            num_batches += len(val_loader)
        if training_config.do_test:
            num_batches += len(test_loader)

        logger.info(f"#batches={num_batches}")
        snapshot_indices = torch.linspace(0, num_batches,
                                          training_config.num_snapshots + 1,
                                          dtype=int)

        for epoch in trange(1, training_config.epochs + 1, desc="Train"):
            # training
            embeds_li, node_presence_li = [], []
            start_epoch_train = timeit.default_timer()

            store_embeds = epoch % training_config.save_embeds_every == 0
            loss, latest_embeddings, latest_node_presence = train(training_config, assoc, criterion,
                                                                  data, device, device_viz,
                                                                  embeds_li, epoch,
                                                                  max_dst_idx, min_dst_idx,
                                                                  model, neighbor_loader,
                                                                  node_presence_li,
                                                                  optimizer, snapshot_indices, store_embeds,
                                                                  train_data, train_loader)
            logger.info(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
            )

            if training_config.do_val:
                # validation
                start_val = timeit.default_timer()
                perf_metric_val, latest_embeddings, latest_node_presence = test(assoc, data, device, device_viz,
                                                                                latest_embeddings, embeds_li, epoch,
                                                                                evaluator,
                                                                                val_loader, metric, model, neg_sampler,
                                                                                neighbor_loader, latest_node_presence,
                                                                                node_presence_li, snapshot_indices,
                                                                                "val",
                                                                                start_batch_index=len(train_loader),
                                                                                store_embeds=store_embeds)
                logger.info(f"\tValidation {metric}: {perf_metric_val: .4f}")
                logger.info(
                    f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
                val_perf_list.append(perf_metric_val)

                # check for early stopping
                if early_stopper.step_check(perf_metric_val, model):
                    break

            # Here we put the test code inside the training loop to save the embeddings, different from the original implementation
            train_val_time = timeit.default_timer() - start_train_val
            logger.info(
                f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

            if training_config.do_test:
                # ==================================================== Test
                # first, load the best model
                early_stopper.load_checkpoint(model)

                # loading the test negative samples
                dataset.load_test_ns()

                # final testing
                start_test = timeit.default_timer()
                perf_metric_test, latest_embeddings, latest_node_presence = test(assoc, data, device, device_viz,
                                                                                 latest_embeddings, embeds_li, epoch,
                                                                                 evaluator,
                                                                                 test_loader, metric, model,
                                                                                 neg_sampler,
                                                                                 neighbor_loader, latest_node_presence,
                                                                                 node_presence_li, snapshot_indices,
                                                                                 "test", start_batch_index=len(
                        train_loader) + len(val_loader), store_embeds=store_embeds)

                logger.info(f"Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
                logger.info(f"\tTest: {metric}: {perf_metric_test: .4f}")
                test_time = timeit.default_timer() - start_test
                logger.info(f"\tTest: Elapsed Time (s): {test_time: .4f}")

                save_results({'model': MODEL_NAME,
                              'data': DATA,
                              'run': run_idx,
                              'seed': SEED,
                              f'val {metric}': val_perf_list,
                              f'test {metric}': perf_metric_test,
                              'test_time': test_time,
                              'tot_train_val_time': train_val_time
                              },
                             results_filename)

            del latest_embeddings, latest_node_presence

            logger.info(
                f">>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
            logger.info('-' * 30)

            if epoch % training_config.save_embeds_every == 0:
                logger.info(f"[Embeds] Saving embeddings for Ep. {epoch} ...")

                embeds_li = np.stack(embeds_li, axis=0)

                os.makedirs(osp.join(training_config.data_dir, training_config.dataset_name),
                            exist_ok=True)
                np.save(osp.join(training_config.data_dir, training_config.dataset_name,
                                 f"{training_config.model}_embeds_{training_config.dataset_name}_Ep{epoch}_Emb{training_config.embedding_dim}.npy"),
                        embeds_li)
                np.save(osp.join(training_config.data_dir, training_config.dataset_name,
                                 f"{training_config.model}_node_presence_{training_config.dataset_name}_Emb{training_config.embedding_dim}.npy"),
                        node_presence_li)

                logger.info("Done!")

    logger.info(
        f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")


if __name__ == "__main__":

    config = argparse.Namespace(**json.load(open(osp.join("config", 'TGB_training.json'))))

    train_dynamic_graph_embeds_tgb(config)
