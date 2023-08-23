import numpy as np
import pandas as pd

import const
from const_viz import *


def get_dataframe_for_visualization(z, args, nodes_li, idx_reference_node, **kwargs):
    # Visualization model dictionary.
    from openTSNE import TSNE
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    import umap
    from sklearn.manifold import MDS
    from sklearn.manifold import LocallyLinearEmbedding

    VISUALIZATION_MODELS = {
        "tsne": TSNE,
        "pca": PCA,
        "isomap": Isomap,
        "umap": umap.UMAP,
        "mds": MDS,
        "lle": LocallyLinearEmbedding
    }

    perplexity = kwargs.pop("perplexity")
    metadata_df = kwargs.pop("metadata_df")

    plot_anomaly_labels = kwargs.pop("plot_anomaly_labels")

    if "highlighted_nodes" in kwargs:
        highlighted_nodes = kwargs.pop("highlighted_nodes")
    else:
        highlighted_nodes = np.array([])

    if "node2labels" in kwargs:
        node2labels = kwargs.pop("node2labels")

    else:
        node2labels = {}

    if args.visualization_model == const.TSNE:

        # t-SNE uses several metrics to calculate the nearest neighbors, for example, `cosine`, `euclidean`, `manhattan`, and `chebyshev`. Here we use cosine similarity.

        visualization_model = VISUALIZATION_MODELS[const.TSNE](
            initialization=const.PCA,
            perplexity=perplexity,
            metric="cosine",
            n_jobs=args.num_workers,
            random_state=args.seed,
            verbose=False,
            n_iter=500,
            **kwargs
        )
        embedding_train = visualization_model.fit(z)
    elif args.visualization_model in [const.PCA, const.ISOMAP, const.UMAP, const.MDS, const.LLE]:
        visualization_model = VISUALIZATION_MODELS[args.visualization_model](
            n_components=2,
            n_jobs=args.num_workers, **kwargs
        )
        embedding_train = visualization_model.fit_transform(z)

    position = np.array(embedding_train.tolist()) if args.visualization_model == const.TSNE else embedding_train[:, :args.visualization_dim]

    df_visual = pd.DataFrame(position, columns=['x', 'y'])

    df_visual["node"] = nodes_li
    df_visual[const.IDX_NODE] = idx_reference_node

    df_visual = df_visual.astype({
        'x': 'float32',
        'y': 'float32',
        const.NODE: 'str',
        const.IDX_NODE: 'int32',
    })

    def process_row(row):
        node = row["node"]
        idx_node = row[const.IDX_NODE]
        name = ""
        if node in highlighted_nodes:
            return {
                'node_size': 60,
                'node_color': node_type_to_color['highlighted'],
                'node_type': 'highlighted',
                'display_name': node
            }

        if plot_anomaly_labels and node2labels[node] == 1:
            return {
                'node_size': 5,
                'node_color': node_type_to_color['anomaly'],
                'node_type': 'anomaly',
                'display_name': ""
            }

        return {
            'node_size': 5,
            'node_color': node_type_to_color['background'],
            'node_type': 'background',
            'display_name': name
        }

    df_visual = pd.concat([df_visual, df_visual.apply(process_row, axis=1, result_type="expand")], axis=1)

    df_visual = pd.merge(df_visual, right=metadata_df, on="node")

    return {
        'df_visual': df_visual,
        "embedding": embedding_train,
    }