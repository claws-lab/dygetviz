import logging

from arguments import parse_args
from generate_dtdg_embeds import train_dynamic_graph_embeds
from plot_dtdg import get_visualization_cache
from utils.utils_logging import configure_default_logging
from utils.utils_misc import project_setup

configure_default_logging()
logger = logging.getLogger(__name__)


def generate_dtdg_embeds_pyg(dataset_name: str, device: str = "cuda:0",
                                                             embedding_dim:
int = 128,
                             epochs: int = 40, lr: float = 1e-3, model_name:
        str =
                             "GConvGRU",
                             save_every: int = 20, visualization_dim=2,
                             visualization_model_name="tsne"):


    train_dynamic_graph_embeds(args, dataset_name, device, embedding_dim,
                               epochs, lr, model_name, save_every)

    get_visualization_cache(dataset_name, device, model_name,
                            visualization_dim, visualization_model_name)





if __name__ == "__main__":

    project_setup()
    args = parse_args()

    generate_dtdg_embeds_pyg("chickenpox", device="cuda:0", embedding_dim=128,
                             epochs=40, lr=1e-3, model_name="GConvGRU",
                             save_every=20, visualization_dim=2,
                             visualization_model_name="tsne")


