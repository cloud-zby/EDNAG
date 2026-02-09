import torch
import torch.nn as nn
import copy
import sys

sys.path.append("./")
from meta_predictor.accurancy_predictor import MetaPredictor
from search_space.searchspace_utils import GENO_SHAPE


class FitnessMapping(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x: torch.Tensor):
        x.to(self.device)
        y = torch.zeros_like(x).to(self.device)

        mask1 = x < 30
        mask2 = (x >= 30) & (x < 50)
        mask3 = (x >= 50) & (x < 70)
        mask4 = (x >= 70) & (x < 75)
        mask5 = (x >= 75) & (x < 80)
        mask6 = (x >= 80) & (x < 85)
        mask7 = (x >= 85) & (x < 90)
        mask8 = (x >= 90) & (x < 95)
        mask9 = x >= 95

        y[mask1] = 0.1 * x[mask1]
        y[mask2] = 3 + 1.0 * (x[mask2] - 30)
        y[mask3] = 23 + 2.0 * (x[mask3] - 50)
        y[mask4] = 63 + 3.0 * (x[mask4] - 70)
        y[mask5] = 78 + 5.0 * (x[mask5] - 75)
        y[mask6] = 103 + 10.0 * (x[mask6] - 80)
        y[mask7] = 153 + 30.0 * (x[mask7] - 85)
        y[mask8] = 303 + 40.0 * (x[mask8] - 90)
        y[mask9] = 503 + 50.0 * (x[mask9] - 95)

        return y


class MetaFitness:
    def __init__(self, device, dataset, hs=512, nz=56, nvt=27, num_sample=20):
        self.device = device
        assert dataset in [
            "cifar10",
            "cifar100",
            "aircraft",
            "pets",
        ], "Unsupported dataset: {}".format(dataset)
        self.meta_predictor = MetaPredictor(
            dataset=dataset, hs=hs, nz=nz, nvt=nvt, num_sample=num_sample, device=device
        )
        self.fitness_mapping = FitnessMapping(device=device)

    def meta_arch_fitness(self, operation_matrix: torch.Tensor):
        """
        Args:
        - operation_matrix: Shape [population, 22, 12] or [population, 22 * 12].
        """
        # 如果operation_matrix是二维的，即[population, 22*12]，则将其转换为三维的[population, 22, 12]
        if operation_matrix.dim() == 2:
            operation_matrix = operation_matrix.view(-1, GENO_SHAPE[0], GENO_SHAPE[1])
        # 预测准确率
        pred_acc, valid_rate = self.meta_predictor.predictor(
            arch_matrix=operation_matrix
        )
        # 将预测准确率映射为适应度值
        fitness = self.fitness_mapping(pred_acc)
        return pred_acc, fitness, valid_rate


def test():
    STR2OPS = {
        "input": 0,
        "output": 1,
        "3-3": 2,
        "3-4": 3,
        "3-6": 4,
        "5-3": 5,
        "5-4": 6,
        "5-6": 7,
        "7-3": 8,
        "7-4": 9,
        "7-6": 10,
    }

    def generate_random_valid_arch_matrix():
        # 初始化一个tensor，所有值设为0
        arch_matrix = torch.zeros(22, 12)
        # 第一个操作为input
        arch_matrix[0, STR2OPS["input"]] = 1
        # 最后一个操作为output
        arch_matrix[-1, STR2OPS["output"]] = 1
        # 随机填充中间的操作
        for i in range(1, 21):
            valid_ops = list(STR2OPS.values())[2:]  # 排除input和output
            import numpy as np

            rand_op = np.random.choice(valid_ops)
            arch_matrix[i, rand_op] = 1
        # 选取1/5的架构设置为无效架构
        if torch.rand(size=[1]) < 1 / 5:
            arch_matrix[5, STR2OPS["input"]] = 1
        return arch_matrix

    def generate_random_valid_arch_matrix_list(length):
        x = [generate_random_valid_arch_matrix() for _ in range(length)]
        return torch.stack(x, dim=0)

    matrix = generate_random_valid_arch_matrix_list(length=10)
    p = MetaFitness(device="cuda", dataset="cifar10")
    pred_acc, fitness, valid_rate = p.meta_arch_fitness(operation_matrix=matrix)
    print(f"pred_acc: {pred_acc}")
    print(f"fitness: {fitness}")
    print(f"valid_rate: {valid_rate}")


# test()
