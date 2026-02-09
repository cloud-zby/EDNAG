import torch
import tqdm
import time
import copy
import sys

sys.path.append("./")
from evo_diff.mapping import Energy
from evo_diff.plot import plot_denoise
from evo_diff.corrector import mutate, select
from evo_diff.predictor import BayesianGenerator, LatentBayesianGenerator, RandomProjection
from evo_diff.ddim import DDIMSchedulerCosine
from evo_diff.fitness import MetaFitness
from evo_diff.utils import TimeoutException, compute_uniqueness, init_population
from eval_architecture.top_arch import get_topk_archs
from search_space.searchspace_utils import GENO_SHAPE


def evolutionary_diffusion(
    dataset,
    num_step,
    population_num,
    top_k,
    diver_rate,
    noise_scale,
    mutate_rate,
    elite_rate,
    lower_params_rate,
    mutate_distri_index,
    seed,
    plot_results,
    max_iter_time,
    init_valid_rate,
    temperature=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    result_save_dir="./results",
):
    # random_map = RandomProjection(GENO_SHAPE[0] * GENO_SHAPE[1], 2, normalize=True).to(device=device)

    # 随机初始化种群样本
    x = init_population(population_num=population_num, init_valid_rate=init_valid_rate)
    x_prev = copy.deepcopy(x)

    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []

    # 在DDIM中使用余弦alpha调度器，选择Energy映射函数
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    # 适应度计算类
    fitness_calculator = MetaFitness(dataset=dataset, device=device)

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
            accurancy, fitness, valid_rate = fitness_calculator.meta_arch_fitness(
                operation_matrix=x
            )
            fitness = mapping_fn(fitness)
            max_acc = accurancy.max().item()
            avg_acc = accurancy.mean().item()

            # Predictor
            generator = BayesianGenerator(
                x=x.to(device=device),
                fitness=fitness.to(device=device),
                alpha=alpha,
                device=device,
            )
            x = generator.generate(x=x.to(device=device), noise=noise_scale)
            # generator = LatentBayesianGenerator(
            #     x=x.to(device=device),
            #     latent=random_map(x.to(device=device)).detach(),
            #     fitness=fitness.to(device=device),
            #     alpha=alpha,
            #     device=device,
            # )
            # x = generator.generate(noise=noise_scale)

            # Corrector
            x = mutate(
                population=population_num,
                x=x,
                mut_rate=mutate_rate,
                eta=mutate_distri_index, 
                device=device,
            )
            x = select(
                x_prev=x_prev,
                x_next=x,
                elite_rate=elite_rate,
                diver_rate=diver_rate,
                lower_params_rate=lower_params_rate,
                max_iter_time=max_iter_time,
                fitness_calculator=fitness_calculator,
                device=device,
            )
            x_prev = copy.deepcopy(x)

            uniq_rate = compute_uniqueness(operation_matrix=x)

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
            return 0.0, 0.0, 0.0, x

    end_time = time.time()
    if plot_results:
        plot_denoise(
            save_dir=result_save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            valid_rate_trace=valid_rate_trace,
            uniq_rate_trace=uniq_rate_trace,
            seed=seed,
            dataset=dataset,
        )

    topk_arch_matrix, topk_arch_str_list = get_topk_archs(
        top_k=top_k, x=x, fitness_calculator=fitness_calculator
    )

    return (
        max_acc_trace[-1],
        end_time - start_time,
        uniq_rate,
        topk_arch_matrix,
        topk_arch_str_list,
    )
