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
from utils.nb201_fitness import arch_fitness
from utils.meta_fitness import meta_arch_fitness
from utils.meta_d2a import FitnessRestorer
from utils.corrector import TimeoutException
from utils.analyse import compute_uniqueness
from utils.meta_d2a import MetaSurrogateUnnoisedModel, load_graph_config, load_model


def evo_diff(
    dataset,
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
    nb201_or_meta,
    max_iter_time,
):
    start_time = time.time()
    assert nb201_or_meta == "nb201", "nb201_or_meta should be nb201, but got {}".format(
        nb201_or_meta
    )

    # 随机初始化种群样本
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []

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
            accurancy, fitness, valid_rate = arch_fitness(
                operation_matrix=x, api=api, dataset=dataset
            )
            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
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
                    x_prev=x_prev,
                    x_next=x,
                    fitness_prev=fitness,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    api=api,
                    dataset=dataset,
                    max_iter_time=max_iter_time,
                    nb201_or_meta=nb201_or_meta,
                    test_dataset=None,
                    meta_surrogate_unnoised_model=None,
                    nasbench201=None,
                    fitness_restorer=None,
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
                    x_prev=x_prev,
                    x_next=x,
                    fitness_prev=fitness,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    api=api,
                    dataset=dataset,
                    max_iter_time=max_iter_time,
                    nb201_or_meta=nb201_or_meta,
                    test_dataset=None,
                    meta_surrogate_unnoised_model=None,
                    nasbench201=None,
                    fitness_restorer=None,
                )
                x_prev = copy.deepcopy(x)

            uniq_rate = compute_uniqueness(arch_op_matrices=x)

            # 保存记录
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            valid_rate_trace.append(valid_rate)
            uniq_rate_trace.append(uniq_rate)
            bar.set_postfix(
                {
                    "max_acc": f"{max_acc:.2f}",
                    "avg_acc": f"{avg_acc:.2f}",
                    "valid_rate": f"{valid_rate:.2f}",
                    "uniq_rate": f"{uniq_rate:.2f}",
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
            valid_rate_trace=valid_rate_trace,
            uniq_rate_trace=uniq_rate_trace,
            seed=seed,
            dataset=dataset,
        )
    end_time = time.time()

    return max_acc_trace[-1], end_time - start_time, uniq_rate, x


def evo_diff_meta(
    dataset,
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
    nb201_or_meta,
    max_iter_time,
):
    assert nb201_or_meta == "meta", "nb201_or_meta should be meta, but got {}".format(
        nb201_or_meta
    )

    # 随机初始化种群样本
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []

    # 在DDIM中使用余弦alpha调度器，选择Energy映射函数
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    nasbench201 = torch.load("meta_acc_predictor/data/nasbench201.pt")
    graph_config = load_graph_config(
        graph_data_name="nasbench201",
        nvt=7,
        data_path="meta_acc_predictor/data/nasbench201.pt",
    )
    meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(
        nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config
    )
    meta_surrogate_unnoised_model = load_model(
        model=meta_surrogate_unnoised_model,
        ckpt_path="meta_acc_predictor/unnoised_checkpoint.pth.tar",
    )
    fitness_restorer = FitnessRestorer(dataset_name=dataset, num_sample=20, seed=seed)
    # 迭代去噪
    start_time = time.time()
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    for t, alpha in bar:
        try:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            # 计算适应度值
            accurancy, fitness, valid_rate = meta_arch_fitness(
                operation_matrix=x,
                api=api,
                dataset=dataset,
                test_dataset=None,
                meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                nasbench201=nasbench201,
                fitness_restorer=fitness_restorer,
            )
            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
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
                    x_prev=x_prev,
                    x_next=x,
                    fitness_prev=fitness,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    api=api,
                    dataset=dataset,
                    max_iter_time=max_iter_time,
                    nb201_or_meta=nb201_or_meta,
                    test_dataset=None,
                    meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                    nasbench201=nasbench201,
                    fitness_restorer=fitness_restorer,
                )
                x_prev = copy.deepcopy(x)
            else:
                x = mutate(
                    population=population_num,
                    x=x,
                    mut_rate=mutate_rate,
                    eta=math.ceil(mutate_distri_index * 1.5),
                )  # 最后一次迭代，减小变异强度
                x = select(
                    x_prev=x_prev,
                    x_next=x,
                    fitness_prev=fitness,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    api=api,
                    dataset=dataset,
                    max_iter_time=max_iter_time,
                    nb201_or_meta=nb201_or_meta,
                    test_dataset=None,
                    meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
                    nasbench201=nasbench201,
                    fitness_restorer=fitness_restorer,
                )
                x_prev = copy.deepcopy(x)

            uniq_rate = compute_uniqueness(arch_op_matrices=x)

            # 保存记录
            avg_acc_trace.append(avg_acc)
            max_acc_trace.append(max_acc)
            valid_rate_trace.append(valid_rate)
            uniq_rate_trace.append(uniq_rate)
            bar.set_postfix(
                {
                    "max_pred_acc": f"{max_acc:.2f}",
                    "avg_pred_acc": f"{avg_acc:.2f}",
                    "valid_rate": f"{valid_rate:.2f}",
                    "uniq_rate": f"{uniq_rate:.2f}",
                }
            )

        except TimeoutException:
            print(
                f"\n>>> Programme exceeded time limit of {max_iter_time} seconds. Terminating..."
            )
            return 0.0, 0.0, x

    end_time = time.time()
    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            valid_rate_trace=valid_rate_trace,
            uniq_rate_trace=uniq_rate_trace,
            seed=seed,
            dataset=dataset,
        )

    return max_acc_trace[-1], end_time - start_time, uniq_rate, x
