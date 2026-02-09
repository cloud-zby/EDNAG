import torch
import time
import os
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from utils.nb201_fitness import load_nb201_api, get_nb201_arch_str
from config.config import (
    nb201_hyper_params_setting,
    nb201_dataset_list,
    meta_hyper_params_setting,
    meta_dataset_list,
)
from evo_diff import evo_diff, evo_diff_meta
from utils.eval_arch import eval_architectures
from utils.meta_fitness import meta_arch_fitness
from utils.meta_d2a import (
    MetaTestDataset,
    MetaSurrogateUnnoisedModel,
    load_graph_config,
    load_model,
)
from utils.meta_d2a import FitnessRestorer


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_topk_archs(x: torch.Tensor, dataset: str, k: int, api, seed):
    print(f">>> Selecting top-{k} architectures from the population...")
    arch_str_list = []
    population = x.shape[0]
    for i in range(population):
        arch_matrix = x[i].view(8, 7)
        arch_str = get_nb201_arch_str(arch_matrix)
        arch_str_list.append(arch_str)

    # 准备元学习预测器
    test_dataset = MetaTestDataset(
        data_path="./meta_acc_predictor/data/meta_predictor_dataset/",
        data_name=dataset,
        num_sample=20,
    )
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
    nasbench201 = torch.load("meta_acc_predictor/data/nasbench201.pt")
    fitness_restorer = FitnessRestorer(dataset_name=dataset, num_sample=20, seed=seed)

    _, fitness, _ = meta_arch_fitness(
        operation_matrix=x,
        api=api,
        dataset=dataset,
        test_dataset=test_dataset,
        meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
        nasbench201=nasbench201,
        fitness_restorer=fitness_restorer,
    )

    sorted_indices = torch.argsort(fitness, descending=True)
    unique_arch_str = set()
    topk_indices = []

    # 保证topk_indices中的架构arch_str不重复
    for idx in sorted_indices:
        if len(topk_indices) >= k:
            break
        arch_str = arch_str_list[idx]
        if arch_str not in unique_arch_str:
            unique_arch_str.add(arch_str)
            topk_indices.append(idx)

    # 如果架构不足k个，则只返回unique的架构
    if len(topk_indices) < k:
        print(
            f">>> Only found {len(topk_indices)} unique architectures in the top-{k} architectures."
        )

    topk_archs = x[torch.tensor(topk_indices)]
    print(f">>> Top-{k} architectures selected.")
    return topk_archs


def exp_with_rand_seed_in_nb201(dataset: str):
    """
    Experiment for random test using NAS-Bench-201.
    """
    assert dataset in nb201_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api("./nas_201_api/NAS-Bench-201-v1_1-096897.pth", verbose=False)
    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # Show Benchmark API
    total_acc = []
    for i in range(15625):
        acc = api.query_test_acc_by_index(i, dataset)
        total_acc.append(acc)
    print(f"For {dataset}, avg acc: {np.mean(total_acc)}, max acc: {np.max(total_acc)}")

    # 导入Hyper-parameters设置
    args = nb201_hyper_params_setting[dataset]

    # 随机种子实验
    avg_max_acc = torch.tensor(0.0)
    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for exp in range(args["rand_exp_num"]):
        seed = int(time.time())
        set_random_seed(seed)
        print(f"\n>>> Exp {exp}: Running on {dataset} dataset with seed {seed}...")
        max_acc, duration, uniq_rate, _ = evo_diff(
            dataset=dataset,
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
            save_dir=args["save_dir"] + "rand_exp/",
            nb201_or_meta=args["nb201_or_meta"],
            max_iter_time=args["max_iter_time"],
        )
        avg_max_acc += max_acc
        avg_duration += duration
        avg_uniq_rate += uniq_rate
    print(
        f'>>> In {args['rand_exp_num']} random seed experiment on {dataset} dataset, average max accuracy is {avg_max_acc / args["rand_exp_num"]:.2f}, average duration is {avg_duration / args["rand_exp_num"]:.2f} seconds, average uniqueness rate is {avg_uniq_rate / args["rand_exp_num"]:.2f}.\n'
    )


def exp_with_rand_seed_in_meta_predictor(dataset: str):
    """
    Experiment for random test using meta-predictor.
    """
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api("./nas_201_api/NAS-Bench-201-v1_1-096897.pth", verbose=False)

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # 导入Hyper-parameters设置
    args = meta_hyper_params_setting[dataset]

    # 随机种子实验
    avg_max_acc = torch.tensor(0.0)
    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for exp in range(args["rand_exp_num"]):
        seed = int(time.time())
        set_random_seed(seed)
        print(f"\n>>> Exp {exp}: Running on {dataset} dataset with seed {seed}...")
        _, duration, uniq_rate, x = evo_diff_meta(
            dataset=dataset,
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
            save_dir=args["save_dir"] + "rand_exp/",
            nb201_or_meta=args["nb201_or_meta"],
            max_iter_time=args["max_iter_time"],
        )
        x = get_topk_archs(x=x, dataset=dataset, k=args["topk"], api=api, seed=seed)
        max_acc, acc_list = eval_architectures(
            x=x.cpu(),
            api=api,
            dataset_name=dataset,
            image_cutout=args["image_cutout"],
            batch_size=args["batch_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args["LR"],
            momentum=args["momentum"],
            decay=args["decay"],
            nesterov=args["nesterov"],
            train_epochs=args["epochs"],
            warmup_epoch=args["warmup"],
            eta_min=args["eta_min"],
            multi_thread=args["multi_thread"],
            early_stop=args["early_stop"],
        )
        avg_max_acc += max_acc
        avg_duration += duration
        avg_uniq_rate += uniq_rate
    print(
        f'>>> In {args['rand_exp_num']} random seed experiment on {dataset} dataset, average max accuracy is {avg_max_acc / args["rand_exp_num"]:.2f}, average duration is {avg_duration / args["rand_exp_num"]:.2f} seconds, average uniqueness rate is {avg_uniq_rate / args["rand_exp_num"]:.2f}.\n'
    )


def exp_with_fixed_seed_in_nb201(dataset: str):
    """
    Experiment for reproducibility using NAS-Bench-201.
    """
    assert dataset in nb201_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api("./nas_201_api/NAS-Bench-201-v1_1-096897.pth", verbose=False)

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # Show Benchmark API
    total_acc = []
    for i in range(15625):
        acc = api.query_test_acc_by_index(i, dataset)
        total_acc.append(acc)
    print(
        f">>> For {dataset}, avg acc: {np.mean(total_acc)}, max acc: {np.max(total_acc)}"
    )

    # 导入Hyper-parameters设置
    args = nb201_hyper_params_setting[dataset]
    seed_list = args["seed"]

    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for seed in seed_list:
        # 复现最佳结果
        set_random_seed(seed)
        print(f"\n>>> Running on {dataset} dataset with seed {seed}...")
        _, duration, uniq_rate, _ = evo_diff(
            dataset=dataset,
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
            save_dir=args["save_dir"] + "reproduce_exp/",
            nb201_or_meta=args["nb201_or_meta"],
            max_iter_time=args["max_iter_time"],
        )
        avg_duration += duration
        avg_uniq_rate += uniq_rate
    print(
        f">>> For dataset {dataset}, average search duration is {avg_duration / len(seed_list):.2f} seconds, average uniqueness rate is {avg_uniq_rate / len(seed_list):.2f}.\n"
    )


def exp_with_fixed_seed_in_meta_predictor(dataset: str):
    """
    Experiment for reproducibility using meta-predictor.
    """
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api("./nas_201_api/NAS-Bench-201-v1_1-096897.pth", verbose=False)

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # 导入Hyper-parameters设置
    args = meta_hyper_params_setting[dataset]
    seed_list = args["seed"]
    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for seed in seed_list:
        if os.path.exists(f"results/search_log/{dataset}_search.pth"):
            result_log = torch.load(f"results/search_log/{dataset}_search.pth")
        else:
            result_log = {}
        if seed in result_log:
            print(
                ">>> Log already exists, pass this seed. Searching Log: ",
                result_log[seed],
            )
        set_random_seed(seed)
        print(f"\n>>> Running on {dataset} dataset with seed {seed}...")
        _, duration, uniq_rate, x = evo_diff_meta(
            dataset=dataset,
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
            save_dir=args["save_dir"] + "reproduce_exp/",
            nb201_or_meta=args["nb201_or_meta"],
            max_iter_time=args["max_iter_time"],
        )
        x = get_topk_archs(x=x, dataset=dataset, k=args["topk"], api=api, seed=seed)
        max_acc, acc_list = eval_architectures(
            x=x.cpu(),
            api=api,
            dataset_name=dataset,
            image_cutout=args["image_cutout"],
            batch_size=args["batch_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args["LR"],
            momentum=args["momentum"],
            decay=args["decay"],
            nesterov=args["nesterov"],
            train_epochs=args["epochs"],
            warmup_epoch=args["warmup"],
            eta_min=args["eta_min"],
            multi_thread=args["multi_thread"],
            early_stop=args["early_stop"],
        )
        avg_duration += duration
        avg_uniq_rate += uniq_rate
        result_log[seed] = {
            "max_acc": max_acc,
            "duration": duration,
            "uniq_rate": uniq_rate,
            "acc_list": acc_list,
        }
        torch.save(result_log, f"results/search_log/{dataset}_search.pth")
    print(
        f">>> For dataset {dataset}, average search duration is {avg_duration / len(seed_list):.2f} seconds, average uniqueness rate is {avg_uniq_rate / len(seed_list):.2f}.\n"
    )
