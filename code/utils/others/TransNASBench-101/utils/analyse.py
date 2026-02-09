import torch

def decode_x_to_NAS_BENCH_201_matrix(x):
    m = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    xys = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
    for i, xy in enumerate(xys):
        # m[xy[0]][xy[1]] = int(torch.argmax(torch.tensor(x[i+1])).item()) - 2
        m[xy[0]][xy[1]] = int(torch.argmax(torch.tensor(x[i+1])).item())
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
    arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.\
        format(types[int(m[1][0])], types[int(m[2][0])], types[int(m[2][1])],
               types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])])
    return arch_str



