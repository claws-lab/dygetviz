import json
import os.path as osp

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

from arguments import args
from dygetviz.data.dataloader import load_data_dtdg
from dygetviz.model.recurrentgcn import RecurrentGCN
from dygetviz.utils.utils_misc import project_setup
from dygetviz.utils.utils_training import get_training_args

project_setup()
train_dataset, test_dataset, full_dataset = load_data_dtdg(args.dataset_name)

config = json.load(
    open(osp.join("config", f"{args.dataset_name}.json"), 'r',
         encoding='utf-8'))

training_args = get_training_args(config)

# if args.dataset_name in ["DGraphFin"]:
#     model = RecurrentGCN(node_features=args.in_channels,
#                          hidden_dim=args.embedding_dim,
#                          transform_input=args.transform_input,
#                          num_classes_nodes=2, num_classes_edges=11)
#
# elif args.dataset_name in ["BMCBioinformatics2021"]:
#     model = RecurrentGCN(node_features=args.in_channels,
#                          hidden_dim=args.embedding_dim,
#                          transform_input=args.transform_input,
#                          num_classes_nodes=2)
#
#
# else:
model = RecurrentGCN(model=args.model,
                     node_features=args.in_channels,
                     hidden_dim=args.embedding_dim,
                     transform_input=args.transform_input, **training_args)

model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)


def train(epoch, embeds_li: list):
    model.train()
    scheduler.step()

    if epoch % args.step_size == 0:
        print(f"Epoch: {epoch}, LR: {scheduler.get_lr()}")

    cost = 0
    node_mask = None

    for idx_snapshot, snapshot in enumerate(train_dataset):
        log_str = f"\tEp.{epoch}\tIter{idx_snapshot}"
        snapshot = snapshot.to(args.device)

        h_0 = model.get_embedding(snapshot.x, snapshot.edge_index,
                                  snapshot.edge_attr, snapshot.y)

        # if "node_classification" in args.tasks:
        #     y_hat = model(h_0)
        #
        #     if args.dataset_name == "DGraphFin":
        #
        #         if idx_snapshot == len(full_dataset) - 1:
        #             mask = (snapshot.y == 0) | (snapshot.y == 1)
        #             loss_link_pred = torch.mean((y_hat[mask] - snapshot.y[mask]) ** 2)
        #
        #         else:
        #             loss_link_pred = 0.
        #
        #     else:
        #         loss_link_pred = torch.mean((y_hat - snapshot.y) ** 2)
        #
        #
        #     emb = h_0
        #
        #
        #     cost += loss_link_pred

        if training_args["do_link_prediction"]:
            # if args.dataset_name == "BMCBioinformatics2021" and idx_snapshot != len(
            #         full_dataset) - 1:
            #     # Only do link prediction in the last snapshot, since the edges are static
            #     loss_link_pred = 0.

            # else:

            nodes = snapshot.edge_index.unique().cpu()
            nodes = nodes.numpy().reshape(-1)
            neg_target = np.random.choice(nodes,
                                          size=snapshot.edge_index.shape[1])
            neg_target = torch.tensor(neg_target, dtype=torch.long,
                                      device=args.device)

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
            "do_edge_regression"] or training_args["do_node_classification"] or \
                training_args["do_edge_classification"]:
            out, h_1 = model.transform_output(h_0)
            emb = h_1

        if training_args["do_node_regression"]:
            pred_node = model.node_output_layer(out)

            loss_node_regression = torch.mean(
                (pred_node[snapshot.node_mask] - snapshot.y[
                    snapshot.node_mask]) ** 2)

            cost += loss_node_regression * 0.1

            log_str += f"\tNode Reg: {loss_node_regression.item():.3f}"

        if training_args["do_edge_regression"]:
            # Edge Regression
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
            if args.dataset_name in ["DGraphFin", "BMCBioinformatics2021"]:

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
                                                        snapshot.y[node_mask])

                cost += loss_node_classification * 1.

                log_str += f"\tNode Cls: {loss_node_classification.item():.3f}"

        if training_args["do_edge_classification"] or training_args[
            "do_edge_regression"]:
            edge_score = model.edge_output_layer(out, snapshot.edge_index)

            # For DGraphFin, edge_type is in 1 - 11
            edge_type = snapshot.edge_type - 1 if args.dataset_name == "DGraphFin" else snapshot.edge_type
            loss_edge_classification = nn.NLLLoss()(edge_score, edge_type)

            cost += loss_edge_classification * 1.

            log_str += f"\tEdge Cls: {loss_edge_classification.item():.3f}"



        # print(log_str)
        if (epoch + 1) % args.eval_every == 0:
            embeds_li += [emb.detach().cpu().numpy()]

    cost = cost / (idx_snapshot + 1)
    total_loss = cost.item()
    cost.backward()

    print(f"[Train] Ep.{epoch + 1}\tLoss: {total_loss:.4f}")

    optimizer.step()
    optimizer.zero_grad()


for epoch in trange(args.epochs):

    # Store the embeddings at each epoch
    embeds_li = []
    train(epoch, embeds_li)

    if (epoch + 1) % args.eval_every == 0:
        embeds: np.ndarray = np.stack([emb for emb in embeds_li])
        del embeds_li
        print(embeds.shape)
        print(f"[Embeds] Saving embeddings for Ep. {epoch + 1} ...")
        np.save(osp.join(args.cache_dir,
                         f"{args.model}_embeds_{args.dataset_name}_Ep{epoch + 1}_Emb{args.embedding_dim}.npy"),
                embeds)
