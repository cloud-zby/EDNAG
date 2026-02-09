import torch
import time
import os
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from config.config import (
    hyper_params_setting,
)
from evo_diff import evo_diff
from TransNASBench101.api import TransNASBenchAPI as API


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main_exp(task: str, search_space: str):
    path2nas_bench_file = "TransNASBench101/transnas-bench_v10141024.pth"
    api = API(path2nas_bench_file)
    assert (
        task in api.task_list
    ), f"ERROR: invalid dataset {task}, expected {api.task_list}"
    assert (
        search_space in api.search_spaces
    ), f"ERROR: invalid search space {search_space}, expected {api.search_spaces}"
    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {task}/{search_space} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )

    # 导入Hyper-parameters设置
    args = hyper_params_setting[search_space][task]
    seed_list = args["seed"]

    avg_duration = torch.tensor(0.0)
    for seed in seed_list:
        # 复现最佳结果
        set_random_seed(seed)
        print(f"\n>>> Running on {task} with seed {seed}...")
        max_acc, duration, _ = evo_diff(
            task=task,
            search_space=search_space,
            api=api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            geno_shape=args["geno_shape"],
            temperature=args["temperature"],  # 温度参数，控制适应度值的数值规模
            diver_rate=args["diver_rate"],  # 多样性程度
            noise_scale=args["noise_scale"],  # 噪声强度，控制扩散的探索性与稳定性
            mutate_rate=args["mutate_rate"],
            elite_rate=args["elite_rate"],
            mutate_distri_index=args["mutate_distri_index"],
            seed=seed,
            plot_results=True,
            save_dir=args["save_dir"],
            max_iter_time=args["max_iter_time"],
        )
        avg_duration += duration
        print(
            f">>> Task {task} with seed {seed} in search space {search_space}, max accuracy: {max_acc:.2f} %"
        )
    print(
        f">>> For {task}, average search duration is {avg_duration / len(seed_list):.2f} seconds.\n"
    )
