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
from utils.fitness import arch_fitness


def evo_diff(
    nb_api,
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

    #
    x = torch.randn(population_num, geno_shape[0] * geno_shape[1])
    x_prev = copy.deepcopy(x)

    #
    avg_acc_trace = []
    max_acc_trace = []

    # DDIMalpha，Energy
    scheduler = DDIMSchedulerCosine(num_step=num_step)
    mapping_fn = Energy(temperature=temperature)

    #
    iter_start_time = time.time()
    bar = tqdm.tqdm(scheduler, ncols=120)
    for t, alpha in bar:
        try:
            if time.time() - iter_start_time > max_iter_time:
                raise TimeoutException
            iter_start_time = time.time()

            #
            accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)

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
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    nb_api=nb_api,
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
                    x_prev=x_prev,
                    x_next=x,
                    elite_rate=elite_rate,
                    diver_rate=diver_rate,
                    nb_api=nb_api,
                    max_iter_time=max_iter_time,
                )
                x_prev = copy.deepcopy(x)

            #
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
            dataset="cifar10",
        )
    end_time = time.time()

    accurancy, fitness = arch_fitness(adj_matrix=x, nb_api=nb_api)
    max_acc = max(accurancy.max().item(), max_acc)

    return max_acc, end_time - start_time, x
