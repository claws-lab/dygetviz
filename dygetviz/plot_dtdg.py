"""Visualizing Dynamic Graph Embedding Trajectories

Created 2023.7
"""
import logging
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
import torch
from tqdm import tqdm


import const
from arguments import parse_args
from const_viz import *
from data.dataloader import load_data
from utils.utils_logging import configure_default_logging
from utils.utils_misc import project_setup, get_visualization_name
from utils.utils_training import pairwise_cos_sim
from utils.utils_visual import get_colors, get_hovertemplate
from visualization.anchor_nodes_generator import get_dataframe_for_visualization



warnings.simplefilter(action='ignore', category=FutureWarning)


configure_default_logging()
logger = logging.getLogger(__name__)

########## General Parameters. May be overwritten by individual datasets ##########
K = 10


################################

def get_visualization_cache(dataset_name: str, device: str, model_name: str,
                            visualization_dim, visualization_model_name: str):
    r"""

    Args:
        dataset_name (str): Dataset name
        device (str): Device to use for tensor computation
        model_name (str): The DTDG model name (e.g. tgn, DyRep, etc.) that trained the embeddings
        visualization_dim (int): The dimension of the visualization, either 2 or 3 (2D or 3D).
        visualization_model_name (str): The visualization model (e.g. tsne, umap, etc.) that projects the embeddings to 2D/3D

    Returns:

    """

    print("TODO")
    if dataset_name.startswith("tgbl"):

        data = load_data(dataset_name, True)
    else:
        data = load_data(dataset_name)

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
    visual_dir = osp.join("outputs", "visual", dataset_name)

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

    if dataset_name in const.dataset_name2months:
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
        logger.info("-" * 30)
        print(f"> # Nearest Neighbors: {nn}")
        print("-" * 30)

        # Coordinates of the subreddits in the embedding space for each snapshot
        highlighted_idx_node_coords = {
            "x": defaultdict(list),
            "y": defaultdict(list),
            const.IDX_SNAPSHOT: defaultdict(list),
        }

        visualization_name = get_visualization_name(dataset_name, model_name,
                                   visualization_model_name, perplexity, nn,
                                   interpolation, idx_reference_snapshot)

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

        # Specify the number of animation frames (0 to n)
        num_animation_frames = len(snapshot_names) - 1

        # Create the static background DataFrame
        background_df = pd.concat([df_visual] * (num_animation_frames + 1), ignore_index=True)
        background_df[const.IDX_SNAPSHOT] = [frame_value for frame_value in range(num_animation_frames + 1) for _ in range(len(df_visual))]
        fig_scatter = px.scatter(background_df, x="x", y="y",
                                 hover_data={
                                     f'hover_data_{i}': True for i, field in
                                     enumerate(fields)
                                 },
                                 size="node_size",
                                 color='node_color', text="display_name",
                                 hover_name="node",
                                animation_frame=const.IDX_SNAPSHOT,
                                animation_group="node",
                                title = f"{model_name}_{dataset_name}",
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

        path_coords = osp.join(visual_dir, f"{visualization_name}.xlsx")

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
            # Adding static background nodes to each frame
            for i in range(len(fig.frames)):
                fig.frames[i].data = fig.data
                
            return fig

        fig_scatter = adjust_node_color_size(fig_scatter)

        embedding_test_all = []

        for idx_snapshot, snapshot_name in enumerate(snapshot_names):

            # Original temporal node embeddings to be projected at snapshot `idx_snapshot`

            try:
                # (#reference_nodes, embed_dim)
                z_reference_embeds = z[idx_snapshot,
                                     idx_reference_node, :]

            except:
                raise ValueError(f"Snapshot {idx_snapshot} does not exist")

            z_projected_embeds: np.ndarray = z[idx_snapshot,
                                             idx_projected_nodes,
                                             :]

            cos_sim_mat = pairwise_cos_sim(z_projected_embeds,
                                           z_reference_embeds, device)

            z_projected_coords_li = []

            # Vectorize inner loop
            all_possible_idx_reference_node = torch.tensor(
                [reference_node2idx.get(node, -1)
                 for node in projected_nodes], device=device)

            # Compute top-k values for all nodes at once
            cos_sim_topk_values, cos_sim_topk_indices = cos_sim_mat.topk(nn + 1,
                                                                         largest=True)

            mask = (cos_sim_topk_indices != all_possible_idx_reference_node[:,
                                            None])
            # For each row, keep the first nn `True`
            mask = mask & (torch.concat(
                [torch.ones((mask.shape[0], nn), dtype=torch.bool, device=device),
                 (mask.sum(dim=1) <= nn).reshape(-1, 1)], dim=1))

            cos_sim_topk_values = cos_sim_topk_values[mask].reshape(-1,
                                                                    nn).cpu().numpy()
            cos_sim_topk_indices = cos_sim_topk_indices[mask].reshape(-1,
                                                                      nn).cpu().numpy()

            z_projected_coords = np.array(
                embedding_train[cos_sim_topk_indices].tolist()).mean(axis=1)

            # Algorithm 1 Line 12 (Vectorized)
            embedding_test = embedding_train[
                                 all_possible_idx_reference_node.cpu()] * interpolation + z_projected_coords * (
                                         1 - interpolation)


            # if DEBUG:
            for i, node in enumerate(projected_nodes):
                idx_node = node2idx[node]

                if node_presence[idx_snapshot, idx_node]:
                    highlighted_idx_node_coords['x'][node] += [
                        embedding_test[i][0]]
                    highlighted_idx_node_coords['y'][node] += [
                        embedding_test[i][1]]
                    highlighted_idx_node_coords[const.IDX_SNAPSHOT][
                        node] += [
                        idx_snapshot]

            embedding_test_all += [embedding_test]

        embedding_test_all = np.stack(embedding_test_all)

        colors = get_colors(len(projected_nodes))
        data = []

        # Initialize the figure with the background dots
        fig = go.Figure()

        for trace in fig_scatter.data:
            fig.add_trace(trace)
        
        # Copy over frames/layout from original, layout is for the UI
        fig.frames = fig_scatter.frames
        fig.layout = fig_scatter.layout
        
        dataframes = {}

        for idx, node in enumerate(
                tqdm(projected_nodes, desc=f"Adding trajectories")):
            num_total_snapshots = len(
                highlighted_idx_node_coords[const.IDX_SNAPSHOT][node])

            idx_node = node2idx[node]

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
                display_name = [name if i % 3 == 0 else "" for i, name in
                                enumerate(display_name)]
            else:
                display_name = [name if i % 10 == 0 else "" for i, name in
                                enumerate(display_name)]

            df = pd.DataFrame({
                'x': embedding_test_all[:, idx, 0][
                    node_presence[:, idx_node]],
                'y': embedding_test_all[:, idx, 1][
                    node_presence[:, idx_node]],
                const.IDX_SNAPSHOT:
                    node_presence[:, idx_node].nonzero()[0],
                'display_name': display_name,
                "hover_name": hover_name,
                'node_color': [colors[idx]] * (num_total_snapshots),
            }, index=np.arange(num_total_snapshots))
            df['node'] = node

            if metadata_df is not None:
                df = pd.merge(df, right=metadata_df, on="node")
                df.rename(
                    {field: f'hover_data_{i}' for i, field in
                     enumerate(fields)},
                    axis=1, inplace=True)

            dataframes[str(node)] = df

            if dataset_name == "Science2013Ant":
                projected_node_name = f"{node} ({annotation.get(node, '')})"

            elif dataset_name == "DGraphFin":
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

            # Traces in sequence 0-num_animation_frames
            traces_of_line = [px.line(df.loc[0:i], x='x', y='y', hover_name='hover_name',
                               text="display_name", 
                               color='node_color', 
                               labels=df.loc[0:i]['node'],
                               hover_data={
                                   f'hover_data_{i}': True for i, field in enumerate(fields)
                                   }
                                ).data[0] for i in range(num_animation_frames + 1)]

            fig_line = px.line(df, x='x', y='y', hover_name='hover_name',
                               text="display_name",
                                 color='node_color',
                               animation_frame=const.IDX_SNAPSHOT, animation_group='hover_name',
                               labels=df["node"],
                               hover_data={
                                   f'hover_data_{i}': True for i, field in
                                   enumerate(fields)
                               }
                               )

            if len(fig_line.data) == 0:
                continue

            try:
                fig_line.data[0].line.color = colors[idx]

                fig_line.data[0].line.width = 5
                fig_line.data[0].marker.size = 6

                # Change the displayed name in the legend
                fig_line.data[0]['name'] = projected_node_name

                fig_line.data[0].hovertemplate = get_hovertemplate(
                    fields_in_customdata=fields, is_trajectory=True)

                for frame in traces_of_line:
                    frame.line.color = colors[idx]
                    frame.line.width = 5
                    frame.marker.size = 20

                    # Change the displayed name in the legend
                    frame['name'] = projected_node_name

                    frame.hovertemplate = get_hovertemplate(
                        fields_in_customdata=fields, is_trajectory=True)
                    
                # default frames are single points, need to rewrite them as lines up until that point
                frames = [
                    go.Frame(data=traces_of_line[i], name=str(i))
                    for i in range(num_animation_frames + 1)]
                fig_line.frames = frames


            except:
                traceback.print_exc()


            for trace in fig_line.data:
                fig = fig.add_trace(trace)

            frames = [
                go.Frame(data=f.data + fig_line.frames[i].data, name=f.name)
                for i, f in enumerate(fig.frames)]
            fig.frames = frames  

        # Write all pd.Dataframe's to Excel outside the loop
        mode = 'a' if osp.exists(path_coords) else 'w'

        if len(dataframes) >= 100:
            # If #nodes >= 100, loading/saving to excel will take too much time.
            with open(osp.join(visual_dir, f"{visualization_name}.pkl"),
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

        # fig.data = fig.frames[0].data
        # Something goes wrong with the data in the initial set and they can't map properly to the frames, causing viz issues.
        # Recreating the Figure with the first frame as the data
        # for fig.frames[0].data
        # Make everything not default displayed except the background and first two paths, or first one path if there is also anomoly cases
        for data in fig.frames[0].data[3:]:
            data.visible = 'legendonly'
        fig = go.Figure(data=fig.frames[0].data, frames=fig.frames, layout=fig.layout)
        fig.write_html(
            osp.join(visual_dir, f"Trajectory_{visualization_name}.html"))

        """
        To load the plot, use:
        fig = pio.read_json(osp.join(visual_dir, f"Trajectory_{visualization_name}.json"))
        """
        print('writing json to: ', osp.join(args.visual_dir,
                                     f"Trajectory_{visualization_name}.json"))
        
        pio.write_json(fig, osp.join(visual_dir,
                                     f"Trajectory_{visualization_name}.json"))

        return fig


if __name__ == '__main__':
    project_setup()

    args = parse_args()
    get_visualization_cache(dataset_name=args.dataset_name, device=args.device,
                            model_name=args.model, visualization_dim=2,
                            visualization_model_name=args.visualization_model)
