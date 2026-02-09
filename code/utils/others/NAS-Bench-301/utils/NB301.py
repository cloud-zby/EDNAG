# NAS-Bench-301 Search Space
import torch
import json
import os
import numpy as np


def show():
    with open("zc_nasbench301.json", "r") as f:
        data = json.load(f)
    acc_list = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    arch_list = []
    for key in data["cifar10"].keys():
        acc = data["cifar10"][key]["val_accuracy"]
        acc_list.append(acc)
        arch_str = separate_parts(key)
        arch_str = split_architecture(arch_str)
        arch_list.append(arch_str)
        tuple_list = extract_tuples(arch_str)
        list1.append(tuple_list[0])
        list2.append(tuple_list[1])
        list3.append(tuple_list[2])
        list4.append(tuple_list[3])
    print("list1:", set(list1))
    print("list2:", set(list2))
    print("list3:", set(list3))
    print("list4:", set(list4))
    print("arch_list:", len(set(arch_list)))
    print(f"In NAS-Bench-301, the global best accuracy is {max(acc_list)}")


def separate_parts(architecture_str):
    architecture = eval(architecture_str)
    part1, _ = architecture
    return str(part1)


def split_architecture(architecture_str):
    architecture = eval(architecture_str)
    group_size = 4
    part1 = architecture[:group_size]
    return str(part1)


def extract_tuples(architecture_str):
    architecture = eval(architecture_str)
    return [str(tup) for tup in architecture]


def get_api(verbose=False):
    with open("zc_nasbench301.json", "r") as f:
        data = json.load(f)

    api = {}
    for key in data["cifar10"].keys():
        acc = data["cifar10"][key]["val_accuracy"]
        arch_str = split_architecture(separate_parts(key))

        if arch_str in api:
            acc = max(api[arch_str], acc)
        api[arch_str] = acc

        if verbose:
            print(f"Architecture: {arch_str}, Accuracy: {acc}")
    if verbose:
        print(f"In NAS-Bench-301, the number of architectures is {len(api)}")
    return api


def convert_architecture(architecture_matrix: torch.Tensor) -> str:
    # architecture_matrix: (4, 14)
    assert architecture_matrix.shape == (
        4,
        14,
    ), f"architecture_matrix must be of shape (4, 14), but got {architecture_matrix.shape}"
    architecture_ops = architecture_matrix.argmax(dim=1)
    ops_idx_1, ops_idx_2, ops_idx_3, ops_idx_4 = (
        architecture_ops[0],
        architecture_ops[1],
        architecture_ops[2],
        architecture_ops[3],
    )
    ops1_dict = {
        0: "(0, 6)",
        1: "(0, 1)",
        2: "(0, 0)",
        3: "(0, 5)",
        4: "(0, 2)",
        5: "(0, 4)",
        6: "(0, 3)",
    }
    ops2_dict = {
        0: "(1, 5)",
        1: "(1, 6)",
        2: "(1, 3)",
        3: "(1, 1)",
        4: "(1, 0)",
        5: "(1, 4)",
        6: "(1, 2)",
    }
    ops3_dict = {
        0: "(0, 6)",
        1: "(0, 1)",
        2: "(1, 5)",
        3: "(1, 6)",
        4: "(1, 3)",
        5: "(0, 0)",
        6: "(0, 5)",
        7: "(1, 1)",
        8: "(0, 2)",
        9: "(0, 4)",
        10: "(1, 0)",
        11: "(1, 4)",
        12: "(0, 3)",
        13: "(1, 2)",
    }
    ops4_dict = {
        0: "(1, 5)",
        1: "(1, 3)",
        2: "(1, 6)",
        3: "(2, 0)",
        4: "(1, 4)",
        5: "(1, 1)",
        6: "(2, 3)",
        7: "(2, 5)",
        8: "(2, 6)",
        9: "(1, 0)",
        10: "(2, 2)",
        11: "(2, 4)",
        12: "(2, 1)",
        13: "(1, 2)",
    }
    ops_idx_1, ops_idx_2, ops_idx_3, ops_idx_4 = (
        int((ops_idx_1 % len(ops1_dict)).item()),
        int((ops_idx_2 % len(ops2_dict)).item()),
        int((ops_idx_3 % len(ops3_dict)).item()),
        int((ops_idx_4 % len(ops4_dict)).item()),
    )
    op1, op2, op3, op4 = (
        ops1_dict[ops_idx_1],
        ops2_dict[ops_idx_2],
        ops3_dict[ops_idx_3],
        ops4_dict[ops_idx_4],
    )
    architecture_str = f"({op1}, {op2}, {op3}, {op4})"
    return architecture_str


def get_acc_by_str(architecture_str, api):
    if architecture_str in api:
        return api[architecture_str]
    else:
        return 0.0


def get_acc_by_matrix(architecture_matrix, api):
    arch_str = convert_architecture(architecture_matrix)
    return get_acc_by_str(arch_str, api)

# show()