import torch
import numpy as np
import random
import sys

sys.path.append("./")
from search_space.searchspace_utils import matrix_to_arch_str, GENO_SHAPE, STR2OPS


class TimeoutException(Exception):
    pass


def compute_uniqueness(operation_matrix: torch.Tensor):
    """
    Args:
    arch_op_matrices: Shape [population, 22 * 12] or [population, 22, 12].
    """
    # 如果operation_matrix是二维的，即[population, 22*12]，则将其转换为三维的[population, 22, 12]
    if operation_matrix.dim() == 2:
        operation_matrix = operation_matrix.view(-1, GENO_SHAPE[0], GENO_SHAPE[1])
    arch_str_list, _ = matrix_to_arch_str(x=operation_matrix)

    # 去除arch_str_list中所有的""
    valid_arch_str_list = list(filter(lambda x: x != "", arch_str_list))

    total_individuals = len(valid_arch_str_list)
    unique_individuals = len(set(valid_arch_str_list))
    return float(unique_individuals) / float(total_individuals)


def diversity_score(x):
    pop = x.shape[0]
    diversity_scores = torch.zeros(pop)
    for i in range(pop):
        distances = torch.norm(x - x[i], dim=1)
        diversity_scores[i] = torch.sum(distances)
    min_score = torch.min(diversity_scores)
    max_score = torch.max(diversity_scores)
    diversity_scores = (diversity_scores - min_score) / (
        max_score - min_score
    )  # 归一化到[0, 1]
    return diversity_scores


def normalize(x):
    """Normalize the input tensor to [-1, 1].

    Args:
    - x: torch.Tensor, shape (population, geno_dims), the input tensor.
    """
    min_vals, _ = x.min(dim=1, keepdim=True)
    max_vals, _ = x.max(dim=1, keepdim=True)
    normalized_x = (x - min_vals) / (max_vals - min_vals)
    normalized_x = 2 * normalized_x - 1
    return normalized_x


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_population(population_num: int, init_valid_rate: float):
    assert (
        0 < init_valid_rate and init_valid_rate <= 1
    ), f"Incorrect init_valid_rate: {init_valid_rate}, it should be in (0, 1]."

    def generate_random_valid_arch_matrix():
        # 初始化一个tensor，所有值设为0
        arch_matrix = torch.zeros(GENO_SHAPE[0], GENO_SHAPE[1])
        # 第一个操作为input
        arch_matrix[0, STR2OPS["input"]] = 1.0
        # 最后一个操作为output
        arch_matrix[-1, STR2OPS["output"]] = 1.0
        # 随机填充中间的操作
        for i in range(1, GENO_SHAPE[0] - 1):
            valid_ops = list(STR2OPS.values())[2:]  # 排除input和output
            rand_op = np.random.choice(valid_ops)
            arch_matrix[i, rand_op] = 1.0
        return arch_matrix

    x = torch.randn(population_num, GENO_SHAPE[0] * GENO_SHAPE[1])
    # 随机选择一部分种群样本为随机生成的有效架构
    indices = torch.randperm(population_num)[: int(population_num * init_valid_rate)]
    # 注意，此时的有效率可能会略低于init_valid_rate，因为暂未考虑none层过多导致的无效架构
    for idx in indices:
        x[idx] = generate_random_valid_arch_matrix().view(-1)

    return x


def test():
    population_num = 100
    init_valid_rate = 0.5
    x = init_population(population_num=population_num, init_valid_rate=init_valid_rate)
    print(matrix_to_arch_str(x.view(population_num, GENO_SHAPE[0], GENO_SHAPE[1])))


# test()
