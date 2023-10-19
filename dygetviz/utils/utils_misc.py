import logging
import os
import os.path as osp
import random

import numpy as np

from utils.utils_logging import configure_default_logging

configure_default_logging()
logger = logging.getLogger(__name__)


def project_setup():
    from .utils_data import check_cwd
    check_cwd()
    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 20)
    set_seed(42)

    os.makedirs(osp.join("outputs", "visual"), exist_ok=True)


def set_seed(seed, use_torch=True):
    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def get_visualization_name(dataset_name, model_name,
                           visualization_model_name, perplexity, nn,
                           interpolation, idx_reference_snapshot):
    return f"{dataset_name}_{model_name}_{visualization_model_name}_perplex{perplexity}_nn{nn}_interpolation{interpolation}_snapshot{idx_reference_snapshot}"


def get_GPU_memory_allocated_to_tensor(t):
    # calculate the GPU memory occupied by t
    memory_in_MB = t.element_size() * t.nelement() / 1024 / 1024
    logger.info(f"Tensor occupies {memory_in_MB:.2f} MB of GPU memory.")
    return memory_in_MB
