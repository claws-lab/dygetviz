import argparse
import os
import os.path as osp

import const
from const import *

if platform.system() in ["Windows", "Linux"]:
    import torch

    if torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda:0"
    else:
        DEFAULT_DEVICE = "cpu"


elif platform.system() == "Darwin":
    DEFAULT_DEVICE = "mps:0"


else:
    raise NotImplementedError("Unknown System")

print(f"Your system: {platform.system()}. Default device: {DEFAULT_DEVICE}")

parser = argparse.ArgumentParser(
    description="Dynamic Graph Embedding Trajectory.")
# Parameters for Analysis
parser.add_argument('--do_visual', action='store_true',
                    help="Whether to do visualization")

# Parameters for TGN

parser.add_argument('--background_color', type=str, default="white",
                    help="white")

parser.add_argument('--batch_size', type=int, default=256,
                    help="the batch size for models")

parser.add_argument('--comment', type=str, default="",
                    help="Comment for each run. Useful for identifying each run on Tensorboard")
parser.add_argument('--data_dir', type=str, default="data",
                    help="Location to store all the data.")
parser.add_argument('--dataset_name', type=str, default='Chickenpox',
                    help="Name of dataset.")
parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                    help="Device to use. When using multi-gpu, this is the 'master' device where all operations are performed.")

parser.add_argument('--do_test', action='store_true')
parser.add_argument('--do_val', action='store_true')
parser.add_argument('--do_weighted', action='store_true',
                    help="Construct weighted graph instead of multigraph for each graph snapshot")

parser.add_argument('--dropout', type=float, default=0.1,
                    help="Dropout rate (1 - keep probability).")

parser.add_argument('--embedding_dim', type=int, default=64,
                    help="the embedding size of model")
parser.add_argument('--embedding_dim_user', type=int, default=32,
                    help="The embedding size for the users")
parser.add_argument('--embedding_dim_resource', type=int, default=32,
                    help="The embedding size for the resource (e.g. video)")

parser.add_argument('--epochs', type=int, default=50,
                    help="Number of epochs to train.")
parser.add_argument('--eval_batch_size', type=int, default=256,
                    help="the batch size for models")
parser.add_argument('--eval_every', type=int, default=20,
                    help="How many epochs to perform evaluation?")

parser.add_argument('--eval_embeds_every', type=int, default=-1,
                    help="How many epochs to evaluate embeddings using polarization?")
parser.add_argument('--eval_sample_method', type=str,
                    choices=[RANDOM, PER_INTERACTION, EXCLUDE_POSITIVE],
                    default=EXCLUDE_POSITIVE,
                    help="Negative sampling method for evaluation dataset")

parser.add_argument('--full_dataset_name', type=str, default="60_months",
                    help="Name of the full dataset")

parser.add_argument('--gpus', type=str, default="0",
                    help="GPUs to use. If using 4 GPUs, type 0,1,2,3")

parser.add_argument('--generate_glove_embeds_for_videos', action='store_true',
                    help="Generate GloVe embeddings for video titles and descriptions")

parser.add_argument('--i_end', type=int, default=None,
                    help="Index of the end dataset.")
parser.add_argument('--in_channels', type=int, default=None,
                    help="Index of the end dataset.")

parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--max_seq_length', type=int, default=128,
                    help="Maximum sequence length")

parser.add_argument('--model', type=str, default=None, help="Model name")

parser.add_argument('--node_types', type=str,
                    choices=["v_subreddit", "author_subreddit",
                             "author_resource"], default="v_subreddit",
                    help="What types of node to include in the GCN bipartite graph?")

parser.add_argument('--num_negative_candidates', type=int, default=1000,
                    help="How many negative examples to sample for each video during the initial sampling?")
parser.add_argument('--num_neighbors', type=int, default=10,
                    help="Number of neighboring nodes in GNN")
parser.add_argument('--num_resource_prototypes', type=int, default=-1, help="")

parser.add_argument('--num_workers', type=int, default=1,
                    help="Number of workers for multiprocessing")
parser.add_argument('--perplexity', type=int, default=20,
                    help="Perplexity of the generated t-SNE plot")
parser.add_argument('--pretrained_embeddings_epoch', type=int, default=195,
                    help="Which epoch of the pretrained embeddings (Node2Vec, GCN ...) to use")
parser.add_argument('--output_dir', type=str, default="outputs")

parser.add_argument('--resample_every', type=int, default=1,
                    help="Number of epochs to resample training dataset.")

parser.add_argument('--num_sample_subreddit', type=int, default=-1,
                    help="Number of subreddits to sample in our dataset. Set to -1 if we do not want to sample")

parser.add_argument('--num_nearest_neighbors', type=str, default="[3,5,10,20]",
                    help="Number of Nearest neighbors")

parser.add_argument('--num_sample_resource', type=int, default=-1,
                    help="Number of resource to sample in our dataset. Set to -1 if we do not want to sample")

parser.add_argument('--num_sample_author', type=int, default=-1,
                    help="Number of resource to sample in our dataset. Set to -1 if we do not want to sample")

parser.add_argument('--port', type=int, default=8050)

parser.add_argument('--save_embed_every', type=int, default=10,
                    help="How many epochs to save embeddings for visualization?")

parser.add_argument('--save_model_every', type=int, default=-1,
                    help="How many epochs to save the model weights?")
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument('--step_size', type=int, default=50, help="step size")
parser.add_argument('--task', type=str, default="", help="task_name")
parser.add_argument('--test_size', type=float, default=0.1, help="")
parser.add_argument('--train_neg_sampling_ratio', type=int, default=1,
                    help="How many negative examples to sample for each positive example in training?")
parser.add_argument('--train_sample_method', type=str,
                    choices=[RANDOM, PER_INTERACTION, EXCLUDE_POSITIVE],
                    default=RANDOM,
                    help="Negative sampling method for training dataset")

parser.add_argument('--val_size', type=float, default=0.1, help="")
parser.add_argument('--verbose', action='store_true', help="")

parser.add_argument('--snapshot_interval', type=int, default=1,
                    help="Time interval (in days) between each snapshot. Default: 1 month. Interactions happening within this time interval will be grouped into one snapshot.")

parser.add_argument('--transform_input', action='store_true',
                    help="Whether to transform the input to a new embedding space. This field is automatically set to True if in_channels does not equal to embedding_dim")

parser.add_argument('--suffix', type=str, default="",
                    help="Suffix to append to the end of the log file name")

parser.add_argument('--tasks', type=str,
                    default="['node_classification','link_pred']",
                    help="Tasks to run, passed as a list of strings")

parser.add_argument('--visualization_dim', type=int, choices=[2, 3], default=2,
                    help="Dimension of the generated visualization. Can be 2- or 3-dimensional.")

parser.add_argument('--visualization_model', type=str,
                    choices=[const.TSNE, const.UMAP, const.PCA, const.ISOMAP,
                             const.MDS], default=const.TSNE,
                    help="Visualization model to use")

args = parser.parse_args()

if args.in_channels is None:
    args.in_channels = args.embedding_dim
args.num_nearest_neighbors = eval(args.num_nearest_neighbors)

args.visual_dir = osp.join(args.output_dir, "visual", args.dataset_name)
os.makedirs(args.visual_dir, exist_ok=True)

args.transform_input = args.in_channels != args.embedding_dim
args.tasks = eval(args.tasks)
print(args.tasks)
