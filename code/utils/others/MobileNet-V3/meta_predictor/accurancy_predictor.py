######################################################################################
# Copyright (c) muhanzhang, D-VAE, NeurIPS 2019 [GitHub D-VAE]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
import torch
import torch.nn as nn
import warnings
import sys
import os
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append("./")
from meta_predictor.models import PredictorModel
from meta_predictor.meta_dataset import MetaTestDataset, load_graph_config
from search_space.searchspace_utils import (
    arch_str_to_igraph,
    matrix_to_arch_str,
    STR2OPS,
)
from meta_predictor.set_encoder.coreset.coreset_strategy import (
    Coreset_Strategy_Per_Class,
)


class AccRescaler(nn.Module):
    def __init__(self):
        super(AccRescaler, self).__init__()
        data = torch.load("meta_predictor/train_dataset/database_219152_14.0K_train.pt")
        self.std = data["std"]
        self.mean = data["mean"]

    def forward(self, x: torch.Tensor):
        y = 100.0 * x * self.std + self.mean
        # 将小于0的值设为0
        y = torch.clamp(y, min=0.0)
        return y


class MetaPredictor(nn.Module):
    def __init__(self, dataset, hs, nz, nvt, num_sample, device="cuda"):
        super(MetaPredictor, self).__init__()
        self.device = device
        self.dataset = dataset
        self.test_dataset = MetaTestDataset(data_name=dataset, num_sample=num_sample)
        self.model = PredictorModel(
            hs=hs, nz=nz, num_sample=num_sample, graph_config=load_graph_config(nvt=nvt)
        )
        # linear scaler
        # self.rescaler = nn.Sequential(
        #     nn.Linear(1, 1),
        #     nn.LeakyReLU(),
        #     nn.Linear(1, 1),
        #     nn.LeakyReLU(),
        #     nn.Linear(1, 1),
        #     nn.Sigmoid(),
        # )
        # if os.path.exists("./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt"):
        #     self.model.load_state_dict(
        #         torch.load("./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt")
        #     )
        # else:
        #     raise FileNotFoundError(
        #         "Checkpoints not found: './meta_predictor/checkpoints/ednas_ckpt_max_corr.pt', please run 'meta_predictor/meta_train.py' first"
        #     )
        self.model.load_state_dict(
            torch.load("./meta_predictor/checkpoints/ckpt_max_corr.pt")
        )
        # self.rescaler.to(self.device)
        # self.rescaler.eval()
        self.model.to(self.device)
        self.model.eval()
        self.acc_fit = AccRescaler()
        self.acc_fit.to(self.device)
        self.acc_fit.eval()
        self.arch_acc_cache = {}
        self.mseloss = nn.MSELoss(reduction="sum")
        print(">>> Selecting coreset data and encoding it...")
        data_per_class = self.test_dataset.get_all_data_per_class(device=self.device)
        coreset_data = []
        for classes in list(data_per_class.keys()):
            strategy = Coreset_Strategy_Per_Class(
                images_per_class=data_per_class[classes]
            )
            query_idxs = strategy.query(num_sample).to(self.device)
            core_data = data_per_class[classes][query_idxs]
            coreset_data.append(core_data)
        coreset_data = torch.cat(coreset_data, dim=0)
        x_batch = []
        for _ in range(10):  # Collecting data
            x_batch.append(coreset_data)
        coreset_batch = torch.stack(x_batch).to(self.device)
        self.dataset_encodings = self.model.set_encode(coreset_batch.to(self.device))
        print(
            f">>> Coreset data shape: {coreset_batch.shape}, encoding shape: {self.dataset_encodings.shape}"
        )

    def forward(self, x, arch):
        # D_mu = self.model.set_encode(x.to(self.device))
        D_mu = self.dataset_encodings
        G_mu = self.model.graph_encode(arch)
        y_pred = self.model.predict(D_mu, G_mu)
        # y_pred = self.rescaler(y_pred)
        return y_pred

    def collect_data(self, arch_igraph):
        x_batch, g_batch = [], []
        for _ in range(10):
            x_batch.append(self.test_dataset[0])
            g_batch.append(arch_igraph)
        return torch.stack(x_batch).to(self.device), g_batch

    def predictor(self, arch_matrix: torch.Tensor):
        """
        Args:
        arch_matrix: [num_archs, 22, 12], architecture matrix of operations
        """
        arch_str_list, validity_rate = matrix_to_arch_str(x=arch_matrix)
        arch_igraphs = [
            arch_str_to_igraph(arch_str=arch_str) for arch_str in arch_str_list
        ]
        pred_acc_list = []
        with torch.no_grad():
            for i, arch_igraph in enumerate(arch_igraphs):
                if arch_igraph is None:
                    # 如果架构不合法
                    pred_acc_list.append(-1.0)
                else:
                    if arch_str_list[i] in self.arch_acc_cache:
                        # 如果已经计算过该架构的准确率
                        y_pred = self.arch_acc_cache[arch_str_list[i]]
                        pred_acc_list.append(y_pred)
                    else:
                        x, g = self.collect_data(arch_igraph=arch_igraph)
                        y_pred = self.forward(x=x, arch=g)
                        y_pred = torch.mean(y_pred)
                        pred_acc_list.append(y_pred.cpu().detach().item())
                        self.arch_acc_cache[arch_str_list[i]] = y_pred
        pred_acc = torch.tensor(pred_acc_list, device=self.device)
        pred_acc = self.acc_fit(pred_acc)
        return pred_acc, validity_rate


def test():
    def generate_random_valid_arch_matrix():
        # 初始化一个tensor，所有值设为0
        arch_matrix = torch.zeros(22, 12)
        # 第一个操作为input
        arch_matrix[0, STR2OPS["input"]] = 1
        # 最后一个操作为output
        arch_matrix[-1, STR2OPS["output"]] = 1
        # 随机填充中间的操作
        for i in range(1, 21):
            valid_ops = list(STR2OPS.values())[2:-1]  # 排除input和output和none
            rand_op = np.random.choice(valid_ops)
            arch_matrix[i, rand_op] = 1
        # 选取1/5的架构设置为无效架构
        if torch.rand(size=[1]) < 1 / 5:
            arch_matrix[5, STR2OPS["input"]] = 1
        # 将第3, 8, 9, 14层的操作设置为none
        arch_matrix[7, :] = torch.zeros(12)
        arch_matrix[7, STR2OPS["none"]] = 1
        arch_matrix[8, :] = torch.zeros(12)
        arch_matrix[8, STR2OPS["none"]] = 1
        if torch.rand(size=[1]) < 1 / 2:
            arch_matrix[3, :] = torch.zeros(12)
            arch_matrix[3, STR2OPS["none"]] = 1
        if torch.rand(size=[1]) < 1 / 2:
            arch_matrix[14, :] = torch.zeros(12)
            arch_matrix[14, STR2OPS["none"]] = 1
        return arch_matrix

    def generate_random_valid_arch_matrix_list(length):
        x = [generate_random_valid_arch_matrix() for _ in range(length)]
        return torch.stack(x, dim=0)

    matrix = generate_random_valid_arch_matrix_list(length=10)
    p = MetaPredictor(dataset="cifar10", hs=512, nz=56, nvt=27, num_sample=20)
    for _ in range(3):
        # 每次循环的期望输出全部相同
        pred_acc, validity_rate = p.predictor(arch_matrix=matrix)
        print(f"Max acc: {torch.max(pred_acc)}")
        print(f"Min acc: {torch.min(pred_acc)}")
        print(f"Avg acc: {torch.mean(pred_acc)}")
        print(f"Validity rate: {validity_rate}")


# test()
