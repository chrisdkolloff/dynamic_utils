import warnings
from typing import Any

import h5py
import torch


class HDFLoader:
    def __init__(self, file_name: str):
        self.file_name = file_name

        with h5py.File(self.file_name, 'r') as hdf:
            self.hdf = hdf
            self.keys = list(hdf.keys())
            self.data = {key: torch.tensor(hdf[key][()]) for key in self.keys}

    def write(self, data_dict: dict):
        """
        Write data to an HDF5 file.

        Args:
            data_dict (dict): A dictionary where the key is the dataset name and
                              the value is the data you want to save.
        """
        with h5py.File(self.file_name, 'w') as hdf:
            for key, value in data_dict.items():
                hdf.create_dataset(key, data=value)

    def read(self, dataset_name: str, default: Any = None):
        """
        Read data from an HDF5 file.

        Args:
            dataset_name (str): The name of the dataset you want to retrieve.
            default (Any): The default return value if the dataset doesn't exist

        Returns:
            Data from the requested dataset.
        """
        if dataset_name in self.keys:
            return self.data[dataset_name][()]
        else:
            return default
            
    def keys(self):
        return self.keys

    def values(self):
        return self.data.values()


class HDFReader(HDFLoader):
    def __init__(self, file_name: str):
        super().__init__(file_name)
        warnings.warn('HDFReader is deprecated. Use HDFLoader instead.')