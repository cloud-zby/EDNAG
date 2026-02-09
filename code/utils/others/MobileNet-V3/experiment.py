import torch
import time
import os
import sys

sys.path.append("./")
from evo_diff.evo_diffusion import evolutionary_diffusion
from evo_diff.utils import set_random_seed
from eval_architecture.eval_archs import eval_arch


def random_exp(
    experiment_num: int,
    dataset: str,
    num_step: int,
    population_num: int,
    top_k: int,
    diver_rate: float,
    noise_scale: float,
    mutate_rate: float,
    elite_rate: float,
    lower_params_rate: float,
    mutate_distri_index: float,
    plot_results: bool,
    max_iter_time: int,
    init_valid_rate: float,
    train_mode: str,
    result_save_path: str = "./results/search_results.pth",
):
    """Run the evolutionary diffusion algorithm with random seed."""
    for exp in range(experiment_num):
        print(f"\n>>> Experiment {exp + 1}/{experiment_num}.")
        seed = int(time.time())
        set_random_seed(seed=seed)
        print(f">>> Random seed is {seed} in dataset {dataset}.")
        # Check if the experiment already exists
        if os.path.exists(result_save_path):
            result = torch.load(result_save_path)
            if dataset in list(result.keys()):
                if seed in list(result[dataset].keys()):
                    print(f">>> Experiment {seed} in dataset {dataset} already exists.")
                    continue
        # Main experiment
        torch.cuda.empty_cache()
        max_pred_acc, duration, uniq_rate, topk_arch_matrix, topk_arch_str_list = (
            evolutionary_diffusion(
                dataset=dataset,
                num_step=num_step,
                population_num=population_num,
                top_k=top_k,
                diver_rate=diver_rate,
                noise_scale=noise_scale,
                mutate_rate=mutate_rate,
                elite_rate=elite_rate,
                lower_params_rate=lower_params_rate,
                mutate_distri_index=mutate_distri_index,
                seed=seed,
                plot_results=plot_results,
                max_iter_time=max_iter_time,
                init_valid_rate=init_valid_rate,
            )
        )
        print(
            f">>> Search completed in {duration:.2f} seconds with max predicted acc {max_pred_acc:.2f}."
        )
        print(f">>> Uniqueness rate is {uniq_rate:.2f} in searched solutions.")
        arch_acc_list, params_list, flops_list = eval_arch(
            arch_list=topk_arch_str_list,
            dataset=dataset,
            train_mode=train_mode,
            verbose=True,
        )
        print(f">>> Best architecture accuracy on {dataset} is {max(arch_acc_list)}.")
        # Save the results
        if os.path.exists(result_save_path):
            result = torch.load(result_save_path)
            if dataset not in list(result.keys()):
                result[dataset] = {}
            if seed not in list(result[dataset].keys()):
                result[dataset][seed] = {}
            result[dataset][seed] = {
                "dataset": dataset,
                "seed": seed,
                "search_duration": duration,
                "uniq_rate": uniq_rate,
                f"top{top_k}_arch_str_list": topk_arch_str_list,
                f"top{top_k}_arch_acc_list": arch_acc_list,
                f"top{top_k}_arch_params_list": params_list,
                f"top{top_k}_arch_flops_list": flops_list,
            }
        else:
            result = {
                dataset: {
                    seed: {
                        "dataset": dataset,
                        "seed": seed,
                        "search_duration": duration,
                        "uniq_rate": uniq_rate,
                        f"top{top_k}_arch_str_list": topk_arch_str_list,
                        f"top{top_k}_arch_acc_list": arch_acc_list,
                        f"top{top_k}_arch_params_list": params_list,
                        f"top{top_k}_arch_flops_list": flops_list,
                    }
                }
            }
        torch.save(result, result_save_path)
        print(f">>> Experiment {exp + 1}/{experiment_num} results saved to {result_save_path}.")


def reproduce_exp(
    seed_list: list,
    dataset: str,
    num_step: int,
    population_num: int,
    top_k: int,
    diver_rate: float,
    noise_scale: float,
    mutate_rate: float,
    elite_rate: float,
    lower_params_rate: float,
    mutate_distri_index: float,
    plot_results: bool,
    max_iter_time: int,
    init_valid_rate: float,
    train_mode: str,
    result_save_path: str = "./results/search_results.pth",
):
    """Run the evolutionary diffusion algorithm with fixed seed."""
    for seed in seed_list:
        # Check if the experiment already exists
        if os.path.exists(result_save_path):
            result = torch.load(result_save_path)
            if dataset in list(result.keys()):
                if seed in list(result[dataset].keys()):
                    print(f">>> Experiment {seed} in dataset {dataset} already exists.")
                    continue
        # Main experiment
        set_random_seed(seed=seed)
        print(f">>> Experiment seed is {seed} in dataset {dataset}.")
        torch.cuda.empty_cache()
        max_pred_acc, duration, uniq_rate, topk_arch_matrix, topk_arch_str_list = (
            evolutionary_diffusion(
                dataset=dataset,
                num_step=num_step,
                population_num=population_num,
                top_k=top_k,
                diver_rate=diver_rate,
                noise_scale=noise_scale,
                mutate_rate=mutate_rate,
                elite_rate=elite_rate,
                lower_params_rate=lower_params_rate,
                mutate_distri_index=mutate_distri_index,
                seed=seed,
                plot_results=plot_results,
                max_iter_time=max_iter_time,
                init_valid_rate=init_valid_rate,
            )
        )

        print(
            f">>> Search completed in {duration:.2f} seconds with max predicted acc {max_pred_acc:.2f}."
        )
        print(f">>> Uniqueness rate is {uniq_rate:.2f} in searched solutions.")
        arch_acc_list, params_list, flops_list = eval_arch(
            arch_list=topk_arch_str_list,
            dataset=dataset,
            train_mode=train_mode,
            verbose=True,
        )
        print(f">>> Best architecture accuracy on {dataset} is {max(arch_acc_list)}.")
        # Save the results
        if os.path.exists(result_save_path):
            result = torch.load(result_save_path)
            if dataset not in list(result.keys()):
                result[dataset] = {}
            if seed not in list(result[dataset].keys()):
                result[dataset][seed] = {}
            result[dataset][seed] = {
                "dataset": dataset,
                "seed": seed,
                "search_duration": duration,
                "uniq_rate": uniq_rate,
                f"top{top_k}_arch_str_list": topk_arch_str_list,
                f"top{top_k}_arch_acc_list": arch_acc_list,
                f"top{top_k}_arch_params_list": params_list,
                f"top{top_k}_arch_flops_list": flops_list,
            }
        else:
            result = {
                dataset: {
                    seed: {
                        "dataset": dataset,
                        "seed": seed,
                        "search_duration": duration,
                        "uniq_rate": uniq_rate,
                        f"top{top_k}_arch_str_list": topk_arch_str_list,
                        f"top{top_k}_arch_acc_list": arch_acc_list,
                        f"top{top_k}_arch_params_list": params_list,
                        f"top{top_k}_arch_flops_list": flops_list,
                    }
                }
            }
        torch.save(result, result_save_path)
        print(
            f">>> Experiment {seed} results saved to {result_save_path}."
        )
