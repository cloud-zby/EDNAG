import time
import torch
from evo_diff.fitness import MetaFitness
from evo_diff.corrector import diverse
from evo_diff.utils import diversity_score
from search_space.searchspace_utils import (
    matrix_to_arch_str,
    GENO_SHAPE,
    is_arch_include_none,
)


def get_topk_archs(top_k: int, x: torch.Tensor, fitness_calculator: MetaFitness):
    print(f">>> Selecting top-{top_k} architectures from searched population...")
    arch_str_list, _ = matrix_to_arch_str(x=x.view(-1, GENO_SHAPE[0], GENO_SHAPE[1]))
    pred_acc, fitness, _ = fitness_calculator.meta_arch_fitness(operation_matrix=x)

    # 优先选则包含"none"操作的个体，降低模型的参数量
    is_include_none = is_arch_include_none(x)
    none_indices = torch.argsort(torch.tensor(is_include_none), descending=True)
    n_lower_params = int(0.1 * top_k)
    none_indices = none_indices[:n_lower_params]

    topk_indices = []
    unique_arch_str = set()
    for idx in none_indices:
        arch_str = arch_str_list[idx]
        if arch_str not in unique_arch_str:
            unique_arch_str.add(arch_str)
            topk_indices.append(idx)

    normalized_fitness = (
       ((fitness - fitness.min()) / (fitness.max() - fitness.min()))
    )
    prob = normalized_fitness / normalized_fitness.sum()
    cum_prob = torch.cumsum(prob, dim=0)  # 轮盘赌的累积概率
    iter_start_time = time.time()
    for _ in range(len(topk_indices), top_k):
        succ_select = False
        # 轮盘赌选择，循环直到选择到一个未曾被选择的个体
        while not succ_select:
            if time.time() - iter_start_time > 300:
                print(f">>> Timeout when selecting top-{top_k} architectures.")
                break
            rand_num = torch.rand(size=[1], device="cuda")
            for j in range(len(cum_prob)):
                if j not in topk_indices:
                    if arch_str_list[j] not in unique_arch_str:
                        if rand_num <= cum_prob[j]:
                            topk_indices.append(j)
                            unique_arch_str.add(arch_str_list[j])
                            succ_select = True
                            break

    # 保证topk_indices中的架构arch_str不重复，按照适应度高低直接选择
    # for idx in sorted_indices:
    #     if len(topk_indices) >= top_k:
    #         break
    #     arch_str = arch_str_list[idx]
    #     if arch_str not in unique_arch_str:
    #         unique_arch_str.add(arch_str)
    #         topk_indices.append(idx)

    # 如果架构不足k个，则只返回unique的架构
    if len(topk_indices) < top_k:
        print(
            f">>> Only found {len(topk_indices)} unique architectures in the top-{top_k} architectures."
        )

    # 对topk_indices里面的索引随机打乱
    topk_indices = torch.tensor(topk_indices)
    topk_indices = topk_indices[torch.randperm(topk_indices.size(0))]

    topk_arch_matrix = x[topk_indices]
    topk_arch_str_list = [arch_str_list[idx] for idx in topk_indices]

    for i, arch_str in enumerate(topk_arch_str_list):
        print(
            f"Selected arch {i + 1}/{top_k}: {arch_str}, Pred acc: {pred_acc[topk_indices[i]]:.2f}."
        )

    return topk_arch_matrix, topk_arch_str_list
