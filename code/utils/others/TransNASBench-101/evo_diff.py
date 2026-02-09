import torch
import tqdm
import time
import math
import copy
from utils.mapping import Power, Energy, Identity
from utils.plot import plot_denoise
from utils.corrector import crossover, mutate, select
from utils.predictor import BayesianGenerator
from utils.ddim import DDIMSchedulerCosine
from utils.corrector import TimeoutException
from utils.transnasbench101_fitness import arch_fitness


def evo_diff(
    task,
    search_space,
    api,
    num_step,
    population_num,
    geno_shape,
    temperature,
    diver_rate,
    noise_scale,
    mutate_rate,
    elite_rate,
    mutate_distri_index,
    seed,
    plot_results,
    save_dir,
    max_iter_time,
):
    start_time = time.time()

    # 随机初始化种群样本
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []

    # 在DDIM中使用余弦alpha调度器，选择Energy映射函数
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    # 迭代去噪
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    for t, alpha in bar:
        try:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            # 计算适应度值
            accurancy, fitness = arch_fitness(
                operation_matrix=x, api=api, task=task, search_space=search_space
            )

            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
            if task == "room_layout":
                max_acc = accurancy.min().item()
            avg_acc = accurancy.mean().item()

            # Predictor
            generator = BayesianGenerator(x=x, fitness=fitness, alpha=alpha)
            x = generator.generate(x=x, noise=noise_scale, elite_rate=elite_rate)

            # Corrector
            if t != num_step - 1:
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=mutate_rate,
                    eta=mutate_distri_index,
                )
                x = select(
                    task=task,
                    search_space=search_space,
                    x_prev=x_prev,
                    x_next=x,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    api=api,
                    max_iter_time=max_iter_time,
                )
                x_prev = copy.deepcopy(x)
            else:
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=mutate_rate,
                    eta=math.ceil(mutate_distri_index * 1.5),
                )
                x = select(
                    task=task,
                    search_space=search_space,
                    x_prev=x_prev,
                    x_next=x,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    api=api,
                    max_iter_time=max_iter_time,
                )
                x_prev = copy.deepcopy(x)


            # 保存记录
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            bar.set_postfix(
                {
                    "max_acc": f"{max_acc:.2f}",
                    "avg_acc": f"{avg_acc:.2f}",
                }
            )

        except TimeoutException:
            print(
                f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating..."
            )
            return 0.0, 0.0, x

    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            seed=seed,
            dataset=task,
        )
    end_time = time.time()

    if task == "room_layout":
        accurancy, fitness = arch_fitness(
            operation_matrix=x, api=api, task=task, search_space=search_space
        )
        max_acc = min(accurancy.min().item(), max_acc)
    else:
        accurancy, fitness = arch_fitness(
            operation_matrix=x, api=api, task=task, search_space=search_space
        )
        max_acc = max(accurancy.max().item(), max_acc)

    return max_acc, end_time - start_time, x
