"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        # self.dataset_list = ['mpi-inf-3dhp', 'coco3D', 'lspet', 'mpii']
        # self.dataset_dict = {'mpi-inf-3dhp': 0, 'coco3D':1, 'lspet':2, 'mpii': 3}

        self.dataset_list = ['mpi-inf-3dhp','coco3D']
        self.dataset_dict = {'mpi-inf-3dhp': 0, 'coco3D':1}

        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]

        self.total_length = sum([len(ds) for ds in self.datasets])
        self.length = max([len(ds) for ds in self.datasets])
        self.data_len = len(self.dataset_list)
        length_itw = sum([len(ds) for ds in self.datasets][1:])
         
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        # self.partition = [.4, .6*len(self.datasets[1])/length_itw,
        #                   .6*len(self.datasets[2])/length_itw,
        #                   .6*len(self.datasets[3])/length_itw]

        self.partition = [0.6, 0.4]
                          
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(self.data_len):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        # return self.length
        return self.length