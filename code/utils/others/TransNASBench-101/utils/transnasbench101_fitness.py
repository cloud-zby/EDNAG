"""
Search Space Configurations

Macro:
- 64-xxxx-basic or 64-xxxxx-basic or 64-xxxxxx-basic
- x is chosen from {1, 2, 3, 4}, with 'none' denoted as placeholder
- 6 rows, 5 columns

Micro:
- 64-41414-x_xx_xxx
- x is chosen from {0, 1, 2, 3}
- 6 rows, 4 columns
"""

import sys

sys.path.append("TransNASBench101")
sys.path.append("./")
from TransNASBench101.api import TransNASBenchAPI as API
import torch
from utils.mapping import ReScale


def show_api():
    path2nas_bench_file = "TransNASBench101/transnas-bench_v10141024.pth"
    api = API(path2nas_bench_file)

    # show number of architectures and number of tasks
    length = len(api)
    task_list = api.task_list  # list of tasks
    print(
        f"This API contains {length} architectures in total across {len(task_list)} tasks."
    )

    # This API contains 7352 architectures in total across 7 tasks.
    # Check all model encoding
    search_spaces = api.search_spaces  # list of search space names
    all_arch_dict = api.all_arch_dict  # {search_space : list_of_architecture_names}
    for ss in search_spaces:
        print(f"Search space '{ss}' contains {len(all_arch_dict[ss])} architectures.")
    print(f"Names of 7 tasks: {task_list}")
    # Search space 'macro' contains 3256 architectures.
    # Search space 'micro' contains 4096 architectures.
    # Names of 7 tasks: ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']

    metrics_dict = api.metrics_dict  # {task_name : list_of_metrics}
    info_names = api.info_names  # list of model info names
    # check the training information of the example task
    task = "class_object"
    print(f"Task {task} recorded the following metrics: {metrics_dict[task]}")
    print(f"The following model information are also recorded: {info_names}")
    # Task class_object recorded the following metrics: ['train_top1', 'train_top5', 'train_loss', 'valid_top1', 'valid_top5', 'valid_loss', 'test_top1', 'test_top5', 'test_loss', 'time_elapsed']
    # The following model information are also recorded: ['inference_time', 'encoder_params', 'model_params', 'model_FLOPs', 'encoder_FLOPs']

    # for i in range(3300):
    #     print(api.index2arch(i))
    #     import time
    #     time.sleep(0.1)

    # Given arch string
    xarch = api.index2arch(1)  # '64-2311-basic'
    for xtask in api.task_list:
        print(f"----- {xtask} -----")
        print(f"--- info ---")
        for xinfo in api.info_names:
            print(f"{xinfo} : {api.get_model_info(xarch, xtask, xinfo)}")
        print(f"--- metrics ---")
        for xmetric in api.metrics_dict[xtask]:
            print(
                f"{xmetric} : {api.get_single_metric(xarch, xtask, xmetric, mode='best')}"
            )
            print(
                f"best epoch : {api.get_best_epoch_status(xarch, xtask, metric=xmetric)}"
            )
            print(f"final epoch : {api.get_epoch_status(xarch, xtask, epoch=-1)}")
            if ("valid" in xmetric and "loss" not in xmetric) or (
                "valid" in xmetric and "neg_loss" in xmetric
            ):
                print(
                    f"\nbest_arch -- {xmetric}: {api.get_best_archs(xtask, xmetric, 'micro')[0]}"
                )

    print(f"-----------Encoding Information-----------")
    # show encoding information of the example architecture
    print("Macro example network: 64-1234-basic")
    print(
        "- Base channel: 64\n- Macro skeleton: 1234 (4 stacked modules)\n- [m1(normal)-m2(channelx2)-m3(resolution/2)-m4(channelx2 & resolution/2)]\n- Cell structure: basic (ResNet Basic Block)"
    )
    print("*" * 30)
    print("1: normal")
    print("2: channelx2")
    print("3: resolution/2")
    print("4: channelx2 & resolution/2")
    print(">" * 30)
    print("Micro example network: 64-41414-1_02_333")
    print(
        "- Base channel: 64\n- Macro skeleton: 41414 (5 stacked modules)\n  - [m1(channelx2 & resolution/2)-m2(normal)-m3(channelx2 & resolution/2)-m4(normal)-m5(channelx2 & resolution/2)]\n- Cell structure: 1_02_333 (4 nodes, 6 edges)\n  - node0: input tensor\n  - node1: Skip-Connect( node0 )\n  - node2: None( node0 ) + Conv1x1( node1 )\n  - node3: Conv3x3( node0 ) + Conv3x3( node1 ) + Conv3x3( node2 )"
    )
    print(">" * 30)
    print("0: None")
    print("1: Skip-Connect")
    print("2: Conv1x1")
    print("3: Conv3x3")
    print(">" * 30)


def trans_macro_arch_to_str(arch_matrix: torch.Tensor) -> str:
    assert arch_matrix.shape == (
        6,
        5,
    ), f"operation_matrix shape should be (6, 5), but got {arch_matrix.shape}"
    operation_matrix = arch_matrix.argmax(dim=1)
    ops = ["none", "1", "2", "3", "4"]
    arch_str = ""
    for i in range(6):
        if ops[operation_matrix[i].item()] == "none":
            continue
        else:
            arch_str = arch_str + ops[operation_matrix[i].item()]
    return "64-" + arch_str + "-basic"


def trans_micro_arch_to_str(arch_matrix: torch.Tensor) -> str:
    assert arch_matrix.shape == (
        6,
        4,
    ), f"operation_matrix shape should be (6, 4), but got {arch_matrix.shape}"
    operation_matrix = arch_matrix.argmax(dim=1)
    ops = ["0", "1", "2", "3"]
    return (
        "64-41414-"
        + ops[operation_matrix[0].item()]
        + "_"
        + ops[operation_matrix[1].item()]
        + ops[operation_matrix[2].item()]
        + "_"
        + ops[operation_matrix[3].item()]
        + ops[operation_matrix[4].item()]
        + ops[operation_matrix[5].item()]
    )


def get_macro_arch_acc(arch_matrix: torch.Tensor, task: str, api: API) -> float:
    assert arch_matrix.shape == (
        6,
        5,
    ), f"Please only query a macro architecture of (6, 5), not multiple architectures. We got {arch_matrix.shape}."
    assert (
        task in api.task_list
    ), f"task {task} not in task list, expected to be {api.task_list}"
    metric_dict = {
        "class_scene": "valid_top1",
        "class_object": "valid_top1",
        "room_layout": "train_loss",
        "jigsaw": "valid_top1",
        "segmentsemantic": "valid_mIoU",
        "normal": "valid_ssim",
        "autoencoder": "valid_ssim",
    }
    metric = metric_dict[task]
    assert (
        metric in api.metrics_dict[task]
    ), f"metric {metric} not in task {task}, expected to be {api.metrics_dict[task]}"
    arch_str = trans_macro_arch_to_str(arch_matrix)
    metrci_value = api.get_single_metric(arch_str, task, metric, mode="best")
    if task == "normal" or task == "autoencoder" or task == "room_layout":
        metrci_value = metrci_value * 100
    return metrci_value


def get_micro_arch_acc(arch_matrix: torch.Tensor, task: str, api: API) -> float:
    assert arch_matrix.shape == (
        6,
        4,
    ), f"Please only query a micro architecture, not multiple architectures. We got {arch_matrix.shape}."
    assert (
        task in api.task_list
    ), f"task {task} not in task list, expected to be {api.task_list}"
    metric_dict = {
        "class_scene": "valid_top1",
        "class_object": "valid_top1",
        "room_layout": "train_loss",
        "jigsaw": "valid_top1",
        "segmentsemantic": "valid_mIoU",
        "normal": "valid_ssim",
        "autoencoder": "valid_ssim",
    }
    metric = metric_dict[task]
    assert (
        metric in api.metrics_dict[task]
    ), f"metric {metric} not in task {task}, expected to be {api.metrics_dict[task]}"
    arch_str = trans_micro_arch_to_str(arch_matrix)
    metrci_value = api.get_single_metric(arch_str, task, metric, mode="best")
    if task == "normal" or task == "autoencoder" or task == "room_layout":
        metrci_value = metrci_value * 100
    return metrci_value


def diversity_score(x):
    pop = x.shape[0]
    diversity_scores = torch.zeros(pop)
    for i in range(pop):
        distances = torch.norm(x - x[i], dim=1)
        diversity_scores[i] = torch.sum(distances)
    min_score = torch.min(diversity_scores)
    max_score = torch.max(diversity_scores)
    diversity_scores = (diversity_scores - min_score) / (
        max_score - min_score
    )  # 归一化到[0, 1]
    return diversity_scores


def macro_neural_predictor(operation_matrix, api, task):
    population = operation_matrix.shape[0]
    operation_matrix = operation_matrix.view(population, 6, 5)
    acc_list = []
    for i in range(operation_matrix.shape[0]):
        try:
            acc = get_macro_arch_acc(operation_matrix[i], task, api)
        except:
            acc = 1e8 if task == "room_layout" else 0.0
        acc_list.append(acc)
    org_acc_list = torch.tensor(acc_list)
    return org_acc_list


def micro_neural_predictor(operation_matrix, api, task):
    population = operation_matrix.shape[0]
    operation_matrix = operation_matrix.view(population, 6, 4)
    acc_list = []
    for i in range(operation_matrix.shape[0]):
        try:
            acc = get_micro_arch_acc(operation_matrix[i], task, api)
        except:
            acc = 1e8 if task == "room_layout" else 0.0
        acc_list.append(acc)
    org_acc_list = torch.tensor(acc_list)
    return org_acc_list


def arch_fitness(operation_matrix, api, task, search_space):
    assert search_space in [
        "macro",
        "micro",
    ], f"Unsupported search space: {search_space}, expected to be 'macro' or 'micro'"
    if search_space == "macro":
        org_acc = macro_neural_predictor(operation_matrix, api, task)
    elif search_space == "micro":
        org_acc = micro_neural_predictor(operation_matrix, api, task)

    if search_space == "macro":
        rescale_facotr_dict = {
            "class_scene": 1.73,
            "class_object": 2.08,
            "jigsaw": 1.03,
            "segmentsemantic": 3.37,
            "normal": 1.55,
            "autoencoder": 1.30,
        }
    elif search_space == "micro":
        rescale_facotr_dict = {
            "class_scene": 1.82,
            "class_object": 2.15,
            "jigsaw": 1.04,
            "segmentsemantic": 3.80,
            "normal": 1.67,
            "autoencoder": 1.73,
        }

    scaler = ReScale()
    if search_space == "macro" and task == "room_layout":
        fitness = scaler(50.0 + 2814.0 / org_acc, task="room_layout")
    elif search_space == "micro" and task == "room_layout":
        fitness = scaler(50.0 + 2969.0 / org_acc, task="room_layout")
    else:
        fitness = scaler(org_acc * rescale_facotr_dict[task])

    return org_acc, fitness


# if __name__ == "__main__":
#     show_api()

# path2nas_bench_file = "TransNASBench101/transnas-bench_v10141024.pth"
# api = API(path2nas_bench_file)
# min_val = 1e8
# for i in range(4000,7000):
#     xarch = api.index2arch(i)
#     val = api.get_single_metric(xarch, "room_layout", "train_loss", mode="best")
#     if val < min_val:
#         min_val = val

# print(f"min_val: {min_val}")
