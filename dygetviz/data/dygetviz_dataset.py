import os
import os.path as osp
import zipfile

import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data

from data.download import download_file_from_google_drive
from const import DATASET2FILEID


Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]

class DyGETVizDataset(object):

    def __init__(
            self,
            dataset_name: str,
            **kwargs: Additional_Features
    ):
        self.cache_dir = osp.join("data", dataset_name)
        self.dataset_name = dataset_name
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def dataset_path(self):
        return osp.join(self.cache_dir, "dataset.pkl")

    def download(self):
        raw_data_path = osp.join(self.cache_dir, "raw_data.zip")

        if not osp.exists(osp.join(self.cache_dir, "raw_data")):
            print("Downloading raw data...", end=" ")
            download_file_from_google_drive(DATASET2FILEID[self.dataset_name], raw_data_path)

            with zipfile.ZipFile(raw_data_path, "r") as zip_ref:
                zip_ref.extract("raw_data", self.cache_dir)

                # To extract all files, run:
                # zip_ref.extractall(self.cache_dir)
            print("Done!")

        else:
            print("Raw data already exists.")

