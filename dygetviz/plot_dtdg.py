"""Visualizing Dynamic Graph Embedding Trajectories

Created 2023.7
"""
import os.path as osp
import pickle
import traceback
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm

import const
from arguments import args
from const_viz import *
from data.dataloader import load_data
from utils.utils_misc import project_setup
from utils.utils_training import pairwise_cos_sim
from utils.utils_visual import get_colors, get_hovertemplate
from visualization.anchor_nodes_generator import get_dataframe_for_visualization

warnings.simplefilter(action='ignore', category=FutureWarning)

########## General Parameters. May be overwritten by individual datasets ##########
K = 10

################################

if __name__ == "__main__":

    project_setup()
    data = load_data()

    annotation = data.get("annotation", {})
    highlighted_nodes = data["highlighted_nodes"]
    idx_reference_snapshot = data["idx_reference_snapshot"]
    interpolation = data["interpolation"]
    metadata_df = data["metadata_df"]
    node_presence = data["node_presence"]
    node2idx = data["node2idx"]
    node2label = data["node2label"]
    num_nearest_neighbors = data["num_nearest_neighbors"]
    perplexity = data["perplexity"]
    plot_anomaly_labels = data["plot_anomaly_labels"]
    projected_nodes = data["projected_nodes"]
    reference_nodes = data["reference_nodes"]
    snapshot_names = data["snapshot_names"]
    z = data["z"]

    # Enable this line if we want to include all trajectories when we generate the visualization cache (*.json) files.
    # NOTE: this will make the visualization cache files very large and dataloading very slow.
    # projected_nodes = list(projected_nodes) + list(set(reference_nodes) - set(projected_nodes))

    projected_nodes = np.array(projected_nodes)

    idx2node = {idx: node for node, idx in node2idx.items()}

    """
    ## Select nodes as the anchor nodes (i.e. the reference frame)
    """

    idx_projected_nodes = np.array([node2idx[node] for node in projected_nodes])

    idx_reference_node = np.array([node2idx[n] for n in reference_nodes])

    # The reference nodes (anchor) is only a subset of all nodes, so we need a separate mapping other than `node2idx`
    reference_node2idx = {node: idx for idx, node in enumerate(reference_nodes)}
    reference_idx2node = {idx: node for idx, node in enumerate(reference_nodes)}

    if args.dataset_name in const.dataset_name2months:
        # All nodes with insufficient interactions should be set to 0 already
        assert \
            z[idx_reference_snapshot, idx_reference_node].sum(
                axis=1).nonzero()[
                0].shape[0] == len(reference_nodes)

    ################################

    outputs = get_dataframe_for_visualization(
        z[idx_reference_snapshot, idx_reference_node], args,
        nodes_li=reference_nodes, idx_reference_node=idx_reference_node,
        plot_anomaly_labels=plot_anomaly_labels,
        node2label=node2label,
        perplexity=perplexity,
        metadata_df=metadata_df)

    for nn in num_nearest_neighbors:
        print("-" * 30)
        print(f"> # Nearest Neighbors: {nn}")
        print("-" * 30)

        # Coordinates of the subreddits in the embedding space for each snapshot
        highlighted_idx_node_coords = {
            "x": defaultdict(list),
            "y": defaultdict(list),
            const.IDX_SNAPSHOT: defaultdict(list),
        }

        visualization_name = f"{args.dataset_name}_{args.model}_{args.visualization_model}_perplex{perplexity}_nn{nn}_interpolation{interpolation}_snapshot{idx_reference_snapshot}"

        embedding_train = outputs['embedding']
        df_visual = outputs['df_visual']

        # Plot the anchor nodes in the background

        # df_visual.rename({"Country": 'custom_data_0'}, axis=1, inplace=True)

        if metadata_df is not None:

            fields = metadata_df.columns.tolist()
            fields.remove("node")

            df_visual.rename(
                {field: f'hover_data_{i}' for i, field in enumerate(fields)},
                axis=1, inplace=True)


        else:
            fields = []

        fig_scatter = px.scatter(df_visual, x="x", y="y",
                                 hover_data={
                                     f'hover_data_{i}': True for i, field in
                                     enumerate(fields)
                                 },
                                 size="node_size",
                                 color='node_color', text="display_name",
                                 hover_name="node",
                                 title=f"{args.model}_{args.dataset_name}",
                                 log_x=False,
                                 opacity=0.7)

        # metadata_df contains the node metadata to be displayed when we hover over the nodes

        # Set the background color to white, and remove x/y-axis grid
        fig_scatter.update_layout(plot_bgcolor='white',
                                  xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))

        if metadata_df is not None:
            # assert (fig_scatter.data[0].hovertext == metadata_df.node.values).all()

            fig_scatter.data[0].hovertemplate = get_hovertemplate(
                fields_in_customdata=fields, is_trajectory=False)

        path_coords = osp.join(args.visual_dir, f"{visualization_name}.xlsx")

        # Save the coordinates of the anchor nodes so that we can plot them later using searborn / matplotlib

        with pd.ExcelWriter(path_coords, engine='openpyxl') as writer:
            df_visual.to_excel(writer, sheet_name='background', index=False)


        def adjust_node_color_size(fig):
            """
            Manually adjust the node colors
            :param fig: plotly.graph_objs._figure.Figure
            :return: fig: plotly.graph_objs._figure.Figure
            """
            for i in range(len(fig.data)):
                color = fig.data[i]['name']
                node_type = color_to_node_type[color]
                fig.data[i]['legendgroup'] = fig.data[i][
                    'name'] = node_type
                fig.data[i]['marker']['color'] = color
                fig.data[i]['marker']['size'] = node_type_to_size[
                    node_type]
            return fig


        fig_scatter = adjust_node_color_size(fig_scatter)

        for idx_snapshot, snapshot_name in enumerate(snapshot_names):

            # Original temporal node embeddings to be projected at snapshot `idx_snapshot`
            try:
                z_reference_embeds = z[idx_snapshot,
                                     idx_reference_node, :]

            except:
                raise ValueError(f"Snapshot {idx_snapshot} does not exist")

            z_projected_embeds: np.ndarray = z[idx_snapshot,
                                             idx_projected_nodes,
                                             :]

            cos_sim_mat = pairwise_cos_sim(z_projected_embeds,
                                           z_reference_embeds, args.device)

            z_projected_coords_li = []

            embedding_test: np.ndarray = np.zeros((len(projected_nodes), 2))

            for i, node in enumerate(
                    tqdm(projected_nodes, desc=f"Snapshot {idx_snapshot}")):
                cos_sim_topk_closest_reference_nodes, idx_topk_closest_reference_nodes = \
                    cos_sim_mat[i].topk(nn + 1,
                                        largest=True, )

                possible_idx_reference_node = reference_node2idx.get(node, -1)
                # When calculating the nearest neighbor of `node`, we should exclude `node` itself
                # Algorithm 1 Line 7-9
                cos_sim_topk_closest_reference_nodes: np.ndarray = \
                    cos_sim_topk_closest_reference_nodes[
                        idx_topk_closest_reference_nodes != possible_idx_reference_node][
                    :nn].cpu().numpy()
                idx_topk_closest_reference_nodes: np.ndarray = \
                    idx_topk_closest_reference_nodes[
                        idx_topk_closest_reference_nodes != possible_idx_reference_node][
                    :nn].cpu().numpy()
                # Algorithm 1 Line 10
                topk_closest_reference_nodes = [reference_nodes[idx] for idx in
                                                idx_topk_closest_reference_nodes]

                # Algorithm 1 Line 11
                # (nn, 2)
                z_projected_coords = np.array(embedding_train[
                                                  idx_topk_closest_reference_nodes].tolist()).mean(
                    axis=0)

                # Algorithm 1 Line 12
                embedding_test[i] = embedding_train[
                                        possible_idx_reference_node] * interpolation + z_projected_coords * (
                                            1 - interpolation)

            for i, node in enumerate(projected_nodes):
                idx_node = node2idx[node]

                if node_presence[idx_snapshot, idx_node]:
                    highlighted_idx_node_coords['x'][node] += [
                        embedding_test[i][0]]
                    highlighted_idx_node_coords['y'][node] += [
                        embedding_test[i][1]]
                    highlighted_idx_node_coords[const.IDX_SNAPSHOT][node] += [
                        idx_snapshot]

        colors = get_colors(len(projected_nodes))
        data = []

        # Initialize the figure with the background dots
        fig = go.Figure()

        for trace in fig_scatter.data:
            fig.add_trace(trace)

        dataframes = {}

        for idx_node, node in enumerate(
                tqdm(projected_nodes, desc=f"Adding trajectories")):
            num_total_snapshots = len(
                highlighted_idx_node_coords[const.IDX_SNAPSHOT][node])

            if len(highlighted_idx_node_coords['x'][node]) < 1 or len(
                    highlighted_idx_node_coords['y'][node]) < 1:
                raise ValueError(f"{node} has no coordinates")

            hover_name = [f"Node: {node} | Snapshot: {snap}" for x, snap in
                                 zip(highlighted_idx_node_coords[
                                     const.IDX_SNAPSHOT][
                                     node], snapshot_names)]

            display_name = [f"{node} ({snap})" for i, (x, snap) in
                            enumerate(zip(highlighted_idx_node_coords[
                                              const.IDX_SNAPSHOT][
                                              node], snapshot_names))]

            if len(snapshot_names) <= 10:
                display_name = display_name
            elif len(snapshot_names) <= 20:
                display_name = [name if i % 3 == 0 else "" for i, name in enumerate(display_name)]
            else:
                display_name = [name if i % 10 == 0 else "" for i, name in enumerate(display_name)]



            df = pd.DataFrame({
                'x': highlighted_idx_node_coords['x'][node],
                'y': highlighted_idx_node_coords['y'][node],
                const.IDX_SNAPSHOT:
                    highlighted_idx_node_coords[const.IDX_SNAPSHOT][
                        node],
                'display_name': display_name,
                "hover_name": hover_name,
                'node_color': [colors[idx_node]] * (num_total_snapshots),
            }, index=np.arange(num_total_snapshots))
            df['node'] = node

            if metadata_df is not None:
                df = pd.merge(df, right=metadata_df, on="node")
                df.rename(
                    {field: f'hover_data_{i}' for i, field in
                     enumerate(fields)},
                    axis=1, inplace=True)

            dataframes[str(node)] = df

            if args.dataset_name == "Science2013Ant":
                projected_node_name = f"{node} ({annotation.get(node, '')})"

            elif args.dataset_name == "DGraphFin":
                if node2label[node] == 0:
                    projected_node_name = f"{node} (Normal)"

                elif node2label[node] == 1:
                    projected_node_name = f"{node} (Fraud)"

                elif node2label[node] in [2, 3]:
                    projected_node_name = f"{node} (Background)"

                else:
                    raise ValueError(node2label[node])


            else:
                projected_node_name = str(node)

            fig_line = px.line(df, x='x', y='y', hover_name='hover_name',
                               text="display_name", color='node_color',
                               labels=df["node"],
                               hover_data={
                                   f'hover_data_{i}': True for i, field in
                                   enumerate(fields)
                               })

            if len(fig_line.data) == 0:
                continue


            try:

                fig_line.data[0].line.color = colors[idx_node]
                fig_line.data[0].line.width = 5
                fig_line.data[0].marker.size = 20

                # Change the displayed name in the legend
                fig_line.data[0]['name'] = projected_node_name

                fig_line.data[0].hovertemplate = get_hovertemplate(
                    fields_in_customdata=fields, is_trajectory=True)

            except:
                traceback.print_exc()



            # data += fig_line.data

            # fig_scatter.add_trace(fig_line)

            for trace in fig_line.data:
                fig = fig.add_trace(trace)

        # Write all df's to Excel outside the loop

        mode = 'a' if osp.exists(path_coords) else 'w'

        # Too large to write to Excel
        if len(dataframes) >= 100:
            with open(osp.join(args.visual_dir, f"{visualization_name}.pkl"),
                      'wb') as f:
                pickle.dump(dataframes, f)

        else:
            with pd.ExcelWriter(path_coords, engine='openpyxl', mode=mode,
                                if_sheet_exists='replace') as writer:
                for sheet, df in dataframes.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)

        # fig = go.Figure(data=data + fig_scatter.data)

        fig.update_layout(
            plot_bgcolor='white',  # Set the background color to white
            xaxis=dict(showticklabels=False),  # Hide x-axis tick labels
            yaxis=dict(showticklabels=False),  # Hide y-axis tick labels
            xaxis_showgrid=True,  # Show x-axis grid lines
            yaxis_showgrid=True
        )

        fig.write_html(
            osp.join(args.visual_dir, f"Trajectory_{visualization_name}.html"))

        """
        To load the plot, use:
        fig = pio.read_json(osp.join(args.visual_dir, f"Trajectory_{visualization_name}.json"))
        """
        pio.write_json(fig, osp.join(args.visual_dir,
                                     f"Trajectory_{visualization_name}.json"))
