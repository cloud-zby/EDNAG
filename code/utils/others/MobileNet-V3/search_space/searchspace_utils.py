import torch
import numpy as np
import igraph

"""MobileNet-V3 Search Space"""
# 网络操作矩阵的形状
GENO_SHAPE = (22, 12)
# STAGE数
NUM_STAGE = 5
# 每个STAGE中最多的LAYER数量
MAX_LAYER_PER_STAGE = 4
# 网络深度列表（每个STAGE中LAYER的数量）
DEPTH_LIST = [2, 3, 4]
# 卷积核大小列表
KS_LIST = [3, 5, 7]
# 网络宽度的EXPAND系数列表，每个卷积层输入通道数C，输出通道数C*EXPAND
EXPAND_LIST = [3, 4, 6]
# 总共12种操作
OPS2STR = {
    0: "input",
    1: "output",
    2: "3-3",
    3: "3-4",
    4: "3-6",
    5: "5-3",
    6: "5-4",
    7: "5-6",
    8: "7-3",
    9: "7-4",
    10: "7-6",
    11: "none",
}
STR2OPS = {
    "input": 0,
    "output": 1,
    "3-3": 2,
    "3-4": 3,
    "3-6": 4,
    "5-3": 5,
    "5-4": 6,
    "5-6": 7,
    "7-3": 8,
    "7-4": 9,
    "7-6": 10,
    "none": 11,
}
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
# 按照上述论文中的设置，将每个操作映射到一个数字，以便用于神经预测器
TYPE_DICT = {
    "2-3-3": 0,
    "2-3-4": 1,
    "2-3-6": 2,
    "2-5-3": 3,
    "2-5-4": 4,
    "2-5-6": 5,
    "2-7-3": 6,
    "2-7-4": 7,
    "2-7-6": 8,
    "3-3-3": 9,
    "3-3-4": 10,
    "3-3-6": 11,
    "3-5-3": 12,
    "3-5-4": 13,
    "3-5-6": 14,
    "3-7-3": 15,
    "3-7-4": 16,
    "3-7-6": 17,
    "4-3-3": 18,
    "4-3-4": 19,
    "4-3-6": 20,
    "4-5-3": 21,
    "4-5-4": 22,
    "4-5-6": 23,
    "4-7-3": 24,
    "4-7-4": 25,
    "4-7-6": 26,
}
EDGE_DICT = {
    2: (2, 3, 3),
    3: (2, 3, 4),
    4: (2, 3, 6),
    5: (2, 5, 3),
    6: (2, 5, 4),
    7: (2, 5, 6),
    8: (2, 7, 3),
    9: (2, 7, 4),
    10: (2, 7, 6),
    11: (3, 3, 3),
    12: (3, 3, 4),
    13: (3, 3, 6),
    14: (3, 5, 3),
    15: (3, 5, 4),
    16: (3, 5, 6),
    17: (3, 7, 3),
    18: (3, 7, 4),
    19: (3, 7, 6),
    20: (4, 3, 3),
    21: (4, 3, 4),
    22: (4, 3, 6),
    23: (4, 5, 3),
    24: (4, 5, 4),
    25: (4, 5, 6),
    26: (4, 7, 3),
    27: (4, 7, 4),
    28: (4, 7, 6),
}

def check_validity_of_matrix(x: torch.Tensor):
    """
    Args:
    x: [22, 12], architecture matrix of operations,
    where 22 = NUM_STAGE * MAX_LAYER_PER_STAGE + INPUT + OUTPUT,
    and 12 = 9 CONV + INPUT + OUTPUT.
    """
    arch_matrix = x.cpu()
    assert (
        tuple(arch_matrix.size()) == GENO_SHAPE
    ), f"Invalid shape: {tuple(arch_matrix.size())}, expected: {GENO_SHAPE}"
    indices = torch.argmax(arch_matrix, dim=1)
    strings = [OPS2STR[idx.item()] for idx in indices]
    # 检测如果第一个和最后一个是input和output，则为有效结构，否则无效
    if strings[0] != "input" or strings[-1] != "output":
        return False
    # 检测中间是否有input或output，如果有则说明架构无效
    if "input" in strings[1:-1] or "output" in strings[1:-1]:
        return False
    # 检测Stage的深度是否合法
    for i in range(NUM_STAGE):
        non_none_layer_count = 0
        for j in range(MAX_LAYER_PER_STAGE):
            if strings[i * MAX_LAYER_PER_STAGE + j + 1] != 'none':    # + 1是因为第一个是input
                non_none_layer_count += 1
        if non_none_layer_count not in DEPTH_LIST:
            return False
    return True


def matrix_to_arch_str(x: torch.Tensor):
    """
    Args:
    x: [num_archs, 22, 12], architecture matrix of operations
    """
    num_archs = x.size(0)
    arch_str_list = []
    num_of_invalid_archs = 0
    for i in range(num_archs):
        arch_matrix = x[i]
        if not check_validity_of_matrix(arch_matrix):
            arch_str_list.append("")
            num_of_invalid_archs += 1
        else:
            indices = torch.argmax(arch_matrix, dim=1)
            strings = [OPS2STR[idx.item()] for idx in indices]
            arch_str = "_".join(strings[1:-1])
            arch_str_list.append(arch_str)
    validity_rate = 1.0 - float(num_of_invalid_archs) / num_archs
    return arch_str_list, validity_rate


def is_arch_include_none(x: torch.Tensor):
    """
    Args:
    x: [num_archs, 22, 12], architecture matrix of operations

    Returns:
    is_include_none: [num_archs], a list of counts of "none" operations in each architecture
    """
    if x.dim() == 2:
        num_archs = x.shape[0]
        x = x.view(num_archs, GENO_SHAPE[0], GENO_SHAPE[1])
    arch_str_list, _ = matrix_to_arch_str(x)
    is_include_none = [float(arch_str.count("none")) for arch_str in arch_str_list]
    return is_include_none


def arch_str_to_matrix(arch_str):
    """
    Args:
    arch_str: str, architecture string
    """
    if arch_str == "":
        return None
    arch_str = arch_str.split("_")
    arch_matrix = torch.zeros(NUM_STAGE * MAX_LAYER_PER_STAGE + 2, len(STR2OPS))
    arch_matrix[0, STR2OPS["input"]] = 1
    arch_matrix[-1, STR2OPS["output"]] = 1
    for i, s in enumerate(arch_str):
        arch_matrix[i + 1, STR2OPS[s]] = 1
    return arch_matrix


def convert_diffusion_arch_to_metad2a_arch(arch_str):
    """
    Args:
    arch_str: str, architecture string
    """
    assert arch_str != "", "Invalid empty architecture string"
    diffusion_arch_str = arch_str.split("_")
    assert len(diffusion_arch_str) == NUM_STAGE * MAX_LAYER_PER_STAGE, f"Invalid architecture string length, expected: {NUM_STAGE * MAX_LAYER_PER_STAGE}, but got: {len(diffusion_arch_str)}"
    # 计算每个STAGE的深度
    depth_list = []
    for i in range(NUM_STAGE):
        non_none_layer_count = 0
        for j in range(MAX_LAYER_PER_STAGE):
            if (
                # 无需+ 1是因为此处的str一定有效，已经去掉了输入输出
                diffusion_arch_str[i * MAX_LAYER_PER_STAGE + j] != "none"
            ):
                non_none_layer_count += 1
        depth_list.append(non_none_layer_count)
    # 转换字符串，为其增加深度属性
    metad2a_arch_str = []
    for i in range(NUM_STAGE):
        for j in range(MAX_LAYER_PER_STAGE):
            if diffusion_arch_str[i * MAX_LAYER_PER_STAGE + j] != "none":
                metad2a_arch_str.append(
                    f"{depth_list[i]}-{diffusion_arch_str[i * MAX_LAYER_PER_STAGE + j]}"
                )
            else:
                metad2a_arch_str.append(f"{depth_list[i]}-3-3")
    return "_".join(metad2a_arch_str)


def arch_str_to_igraph(arch_str):
    """
    Args:
    arch_str: str, architecture string
    """
    if arch_str == "":
        return None
    arch_str = convert_diffusion_arch_to_metad2a_arch(arch_str)
    split_str = arch_str.split("_")
    n = NUM_STAGE * MAX_LAYER_PER_STAGE
    nodes = torch.zeros(n)
    for i, s in enumerate(split_str):
        nodes[i] = TYPE_DICT[s]
    g = igraph.Graph(directed=True)
    g.add_vertices(n + 2)  # + in/out nodes
    g.vs[0]["type"] = 0
    for i, v in enumerate(nodes):
        g.vs[i + 1]["type"] = v + 2  # in node: 0, out node: 1
        g.add_edge(i, i + 1)
    g.vs[n + 1]["type"] = 1
    g.add_edge(n, n + 1)
    return g


def decode_ofa_mbv3_to_igraph(matrix):
    # 5 stages, 4 layers for each stage
    # d: 2, 3, 4
    # e: 3, 4, 6
    # k: 3, 5, 7

    node_types = torch.zeros(NUM_STAGE * MAX_LAYER_PER_STAGE)
    d_list = []
    for i in range(NUM_STAGE):
        for j in range(MAX_LAYER_PER_STAGE):
            d_list.append(matrix["d"][i])
    for i, (ks, e, d) in enumerate(zip(matrix["ks"], matrix["e"], d_list)):
        node_types[i] = TYPE_DICT[f"{d}-{ks}-{e}"]

    # 特殊处理none结点
    for i in range(NUM_STAGE):
        node = EDGE_DICT[
            int(node_types[i * MAX_LAYER_PER_STAGE].item()) + 2 # 输入输出结点为0, 1
        ]  # 第0, 4, 8, 12, 16个结点
        _d, _ks, _e = node
        if _d == 2:   # 仅有前两层为非none层
            node_types[i * MAX_LAYER_PER_STAGE + 2] = TYPE_DICT["2-3-3"]  # none层统一用3-3替代
            node_types[i * MAX_LAYER_PER_STAGE + 3] = TYPE_DICT["2-3-3"]
        elif _d == 3: # 仅有前三层为非none层
            node_types[i * MAX_LAYER_PER_STAGE + 3] = TYPE_DICT["3-3-3"]

    n = NUM_STAGE * MAX_LAYER_PER_STAGE
    g = igraph.Graph(directed=True)
    g.add_vertices(n + 2)  # + in/out nodes
    g.vs[0]["type"] = 0
    for i, v in enumerate(node_types):
        g.vs[i + 1]["type"] = v + 2  # in node: 0, out node: 1
        g.add_edge(i, i + 1)
    g.vs[n + 1]["type"] = 1
    g.add_edge(n, n + 1)
    return g
