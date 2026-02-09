######################################################################################
# Copyright (c) muhanzhang, D-VAE, NeurIPS 2019 [GitHub D-VAE]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
import os
import torch
from torch.utils.data import Dataset


NUM_CLASS_DICT = {
    "cifar100": 100,
    "cifar10": 10,
    "aircraft": 30,
    "pets": 37,
}

class MetaTestDataset(Dataset):
    def __init__(self, data_name, num_sample):
        self.num_sample = num_sample
        self.data_name = data_name
        assert data_name in list(NUM_CLASS_DICT.keys()), f"Invalid data_name: {data_name}, supported data_name: {list(NUM_CLASS_DICT.keys())}"
        self.num_class = NUM_CLASS_DICT[data_name]
        dataset_path = f"meta_predictor/dataset/{data_name}bylabel.pt"
        self.x = torch.load(dataset_path)

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        data = []
        classes = list(range(self.num_class))
        for choose_class in classes:  # 对于每一个类，均取num_sample个样本
            cx = self.x[choose_class][0]
            ridx = torch.randperm(len(cx))
            data.append(cx[ridx[: self.num_sample]])
        x = torch.cat(data)
        return x
    
    def get_all_data_per_class(self, device: str):
        data = {}
        classes = list(range(self.num_class))
        for choose_class in classes:
            cx = self.x[choose_class][0]
            data[choose_class] = cx.to(device)
        return data


def load_graph_config(nvt):
    max_n = 20
    graph_config = {}
    graph_config["num_vertex_type"] = nvt + 2  # original types + start/end types
    graph_config["max_n"] = max_n + 2  # maximum number of nodes + start/end nodes
    graph_config["START_TYPE"] = 0  # predefined start vertex type
    graph_config["END_TYPE"] = 1  # predefined end vertex type

    return graph_config


