import torch


def decode_x_to_NAS_BENCH_201_matrix(x):
    m = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    xys = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
    for i, xy in enumerate(xys):
        m[xy[0]][xy[1]] = int(torch.argmax(torch.tensor(x[i + 1])).item())
    return torch.tensor(m)


def decode_x_to_NAS_BENCH_201_string(x, ops_decoder):
    """_summary_
    Args:
        x (torch.Tensor): x_elem [8, 7]
    Returns:
        arch_str
    """
    m = decode_x_to_NAS_BENCH_201_matrix(x)
    types = ops_decoder
    arch_str = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(
        types[int(m[1][0])],
        types[int(m[2][0])],
        types[int(m[2][1])],
        types[int(m[3][0])],
        types[int(m[3][1])],
        types[int(m[3][2])],
    )
    return arch_str


def compute_uniqueness(arch_op_matrices: torch.Tensor):
    assert arch_op_matrices.shape[1] == 7 * 8, "Invalid shape: {}".format(
        arch_op_matrices.shape
    )
    arch_op_matrices = arch_op_matrices.view(arch_op_matrices.shape[0], 8, 7)
    ops_decoder = [
        "input",
        "output",
        "none",
        "skip_connect",
        "nor_conv_1x1",
        "nor_conv_3x3",
        "avg_pool_3x3",
    ]
    valid_arch_str_list = []
    population = arch_op_matrices.shape[0]
    for i in range(population):
        arch_str = decode_x_to_NAS_BENCH_201_string(arch_op_matrices[i], ops_decoder)
        valid_arch_str_list.append(arch_str)
    return float(len(set(valid_arch_str_list))) / len(valid_arch_str_list)
