import torch
import copy
import time
from evo_diff.fitness import MetaFitness
from evo_diff.utils import TimeoutException, diversity_score, normalize
from search_space.searchspace_utils import is_arch_include_none


def crossover_operator(x: torch.Tensor, y: torch.Tensor, cross_rate: float, eta: float):
    x_new = copy.deepcopy(x)
    y_new = copy.deepcopy(y)
    if torch.rand(size=[1]) <= cross_rate:
        # Simulated Binary Crossover
        rand_num = torch.rand(size=[1])
        if rand_num <= 0.5:
            beta = (rand_num * 2) ** (1 / (1 + eta))
        else:
            beta = (1 / (2 - 2 * rand_num)) ** (1 / (1 + eta))
        x_new = 0.5 * (x + y) - 0.5 * beta * (y - x)
        y_new = 0.5 * (x + y) + 0.5 * beta * (y - x)
    return x_new, y_new


def crossover(population: int, x: torch.Tensor, cross_rate: float, eta: float):
    """Simulated Binary Crossover operation for the population.

    Args:
    - population: int, the number of samples in the population.
    - x: torch.Tensor, shape (population, d).
    - cross_rate: float in [0, 1], the crossover rate.
    - eta: float in [5, 20], the spread factor property. Higher eta means more similar children to parents.
    """
    temp = copy.deepcopy(x)
    for individual in range(0, population - 1, 2):
        if torch.rand(size=[1]) <= cross_rate:
            x[individual], x[individual + 1] = crossover_operator(
                x[individual], x[individual + 1], cross_rate, eta
            )
    return normalize(x)


def mutate(
    population: int,
    x: torch.Tensor,
    mut_rate: float,
    eta: float,
    device: str,
    lower_bound=-1,
    upper_bound=1,
):
    """Polynomial Mutation operation for the population.

    Args:
    - population: int, the number of samples in the population.
    - x: torch.Tensor, shape (population, d), the parent samples.
    - mut_rate: float in [0, 1], the mutation rate.
    - eta: float in [5, 20], the distribution index of mutation. Higher eta means more similar children to parents.
    """
    x = x.to(device=device)
    x_new = copy.deepcopy(x)
    for i in range(population):
        if torch.rand(size=[1]) <= mut_rate:
            # Polynomial Mutation
            delta1 = (x[i] - lower_bound) / (upper_bound - lower_bound)
            delta2 = (upper_bound - x[i]) / (upper_bound - lower_bound)
            mu = torch.rand(size=[1], device=device)
            if mu <= 0.5:
                delta = (2 * mu + (1 - 2 * mu) * ((1 - delta1)) ** (eta + 1)) ** (
                    1 / (1 + eta)
                ) - 1
            else:
                delta = 1 - (
                    2 * (1 - mu) + (2 * mu - 1) * ((1 - delta2) ** (eta + 1))
                ) ** (1 / (1 + eta))
            x_new[i] = x[i] + delta * (upper_bound - lower_bound)
    return normalize(x_new)


def elitism(n_elitism: int, fitness: torch.Tensor):
    # Sort the fitness values indices from large to small
    sorted_indices = torch.argsort(fitness, descending=True)

    elite_indices = []
    last_fitness = -100
    # 从适应度值最大到最小的个体遍历，避免选择了多个相同的个体
    for idx in sorted_indices:
        if fitness[idx] == last_fitness:
            continue
        last_fitness = fitness[idx]
        elite_indices.append(idx.item())
        # 如果精英个体数量达到要求，则停止遍历
        if len(elite_indices) == n_elitism:
            break
    # 如果重复个体实在太多，非重复个体不足n_elitism个，则从适应度值最大的个体中选择
    if len(elite_indices) < n_elitism:
        elite_indices += sorted_indices[: n_elitism - len(elite_indices)].tolist()
    return torch.tensor(elite_indices)


def roulette_wheel(
    normalized_fitness,
    n_elitism,
    n_diver,
    n_lower_params,
    x_parent,
    elite_indices,
    diver_indices,
    none_indices,
    max_iter_time,
    device,
):
    prob = normalized_fitness / normalized_fitness.sum()
    cum_prob = torch.cumsum(prob, dim=0)  # 轮盘赌的累积概率
    selected_indices = []
    iter_start_time = time.time()
    for i in range(n_elitism + n_diver + n_lower_params, x_parent.shape[0]):
        succ_select = False
        # 轮盘赌选择，循环直到选择到一个未曾被选择的个体
        while not succ_select:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            rand_num = torch.rand(size=[1], device=device)
            for j in range(len(cum_prob)):
                if (
                    (j not in elite_indices)
                    and (j not in diver_indices)
                    and (j not in none_indices)
                    and (j not in selected_indices)
                ):
                    if rand_num <= cum_prob[j]:
                        selected_indices.append(j)
                        succ_select = True
                        break
    return selected_indices


def diverse(diver_score: torch.Tensor, fitness: torch.Tensor, n_diver: int):
    sorted_indices = torch.argsort(diver_score, descending=True)
    diver_indices = []
    max_fitness = torch.max(fitness)
    for i in range(len(sorted_indices)):
        if fitness[sorted_indices[i]] >= max_fitness * 0.1:
            diver_indices.append(sorted_indices[i].item())
        if len(diver_indices) == n_diver:
            break
    if len(diver_indices) < n_diver:
        diver_indices += sorted_indices[: n_diver - len(diver_indices)].tolist()
    return torch.tensor(diver_indices)


def select(
    x_prev: torch.Tensor,
    x_next: torch.Tensor,
    elite_rate: float,
    diver_rate: float,
    lower_params_rate: float,
    max_iter_time: float,
    fitness_calculator: MetaFitness,
    device: str,
):
    """Selcet strategy for the population.

    Args:
    - x_parent: torch.Tensor, shape (population, d), the parent samples.
    - x_next: torch.Tensor, shape (population, d), the children samples.
    - fitness: torch.Tensor, shape (population,), the fitness values of the parent samples.
    - elite_rate: float in [0, 1], the elite rate for elitism.
    """
    x_prev = x_prev.to(device=device)
    x_next = x_next.to(device=device)
    x_new = torch.zeros_like(x_prev, device=device)

    # 计算适应度值和多样性分数
    x = torch.cat([x_prev, x_next], dim=0).to(device=device)
    _, fitness, valid_rate = fitness_calculator.meta_arch_fitness(operation_matrix=x)
    fitness = fitness.to(device=device)
    diver_score = diversity_score(x).to(device=device)

    # 选择精英个体
    n_elitism = int(elite_rate * x_prev.shape[0])
    elite_indices = elitism(n_elitism=n_elitism, fitness=fitness).to(device=device)
    x_new[:n_elitism, :] = copy.deepcopy(x[elite_indices])

    # 选择多样性个体
    n_diver = int(diver_rate * x_prev.shape[0])
    diver_indices = diverse(
        diver_score=diver_score, fitness=fitness, n_diver=n_diver
    ).to(device=device)
    x_new[n_elitism : n_elitism + n_diver, :] = copy.deepcopy(x[diver_indices])

    # 优先选则包含"none"操作的个体，降低模型的参数量
    is_include_none = is_arch_include_none(x)
    # 选择参数量最少的个体，即包含"none"操作最多的个体
    none_indices = torch.argsort(torch.tensor(is_include_none, device=device), descending=True)
    n_lower_params = int(lower_params_rate * x_prev.shape[0])
    none_indices = none_indices[:n_lower_params]
    x_new[n_elitism + n_diver : n_elitism + n_diver + n_lower_params, :] = copy.deepcopy(x[none_indices])

    # 在x_parent和x_next中进行roulette wheel selection
    fitness = fitness * (1.0 + diver_score)  # 适应度值和多样性分数的加权
    # 归一化，避免适应度值过大导致的数值不稳定，或者为负数导致相加出错
    normalized_fitness = 0.9 * (
        (fitness - fitness.min()) / (fitness.max() - fitness.min())
    ) + 0.1
    selected_indices = roulette_wheel(
        normalized_fitness,
        n_elitism,
        n_diver,
        n_lower_params,
        x_prev,
        elite_indices,
        diver_indices,
        none_indices,
        max_iter_time,
        device,
    )
    x_new[n_elitism + n_diver + n_lower_params :, :] = copy.deepcopy(
        x[selected_indices]
    )

    return x_new
