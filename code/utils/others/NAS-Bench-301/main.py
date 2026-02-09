import torch
import time
import os
import random
import warnings
import numpy as np
from config.config import (
    hyper_params_setting,
)
from evo_diff import evo_diff
from utils.NB301 import get_api
warnings.filterwarnings("ignore")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main_exp():
    nb_api = get_api()

    # Hyper-parameters
    args = hyper_params_setting
    seed_list = args["seed"]

    for seed in seed_list:
        #
        set_random_seed(seed)
        max_acc, duration, _ = evo_diff(
            nb_api=nb_api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            geno_shape=args["geno_shape"],
            temperature=args["temperature"],  # ，
            diver_rate=args["diver_rate"],  #
            noise_scale=args["noise_scale"],  # ，
            mutate_rate=args["mutate_rate"],
            elite_rate=args["elite_rate"],
            mutate_distri_index=args["mutate_distri_index"],
            seed=seed,
            plot_results=False,
            save_dir=args["save_dir"],
            max_iter_time=args["max_iter_time"],
        )
        print(f">>> Running on cifar10 with seed {seed}, max accuracy: {max_acc:.2f} %")


if __name__ == "__main__":
    main_exp()
