import torch
import sys
sys.path.append("./")
from search_space.searchspace_utils import (
    NUM_STAGE,
    MAX_LAYER_PER_STAGE,
    KS_LIST,
    EXPAND_LIST,
)


def get_blocks_list(expand_coefficient: list, kernel_size: list):
    assert (
        len(expand_coefficient) == NUM_STAGE * MAX_LAYER_PER_STAGE
    ), f"Invalid length of expand_coefficient: {len(expand_coefficient)}, expected: {NUM_STAGE * MAX_LAYER_PER_STAGE}"
    assert (
        len(kernel_size) == NUM_STAGE * MAX_LAYER_PER_STAGE
    ), f"Invalid length of kernel_size: {len(kernel_size)}, expected: {NUM_STAGE * MAX_LAYER_PER_STAGE}"
    blocks_list = [
        {  # 初始卷积层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 16,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 1,
                "mid_channels": None,
                "act_func": "relu",
                "use_se": False,
                "groups": None,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 16,
                "out_channels": 16,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第一层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 16,
                "out_channels": 24,
                "kernel_size": kernel_size[0],
                "stride": 2,
                "expand_ratio": expand_coefficient[0],
                "mid_channels": 96,
                "act_func": "relu",
                "use_se": False,
            },
            "shortcut": None,
        },
        {  # 第二层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 24,
                "out_channels": 24,
                "kernel_size": kernel_size[1],
                "stride": 1,
                "expand_ratio": expand_coefficient[1],
                "mid_channels": 144,
                "act_func": "relu",
                "use_se": False,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 24,
                "out_channels": 24,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第三层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 24,
                "out_channels": 24,
                "kernel_size": kernel_size[2],
                "stride": 1,
                "expand_ratio": expand_coefficient[2],
                "mid_channels": 144,
                "act_func": "relu",
                "use_se": False,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 24,
                "out_channels": 24,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第四层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 24,
                "out_channels": 24,
                "kernel_size": kernel_size[3],
                "stride": 1,
                "expand_ratio": expand_coefficient[3],
                "mid_channels": 144,
                "act_func": "relu",
                "use_se": False,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 24,
                "out_channels": 24,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第五层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 24,
                "out_channels": 40,
                "kernel_size": kernel_size[4],
                "stride": 2,
                "expand_ratio": expand_coefficient[4],
                "mid_channels": 144,
                "act_func": "relu",
                "use_se": True,
            },
            "shortcut": None,
        },
        {  # 第六层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 40,
                "out_channels": 40,
                "kernel_size": kernel_size[5],
                "stride": 1,
                "expand_ratio": expand_coefficient[5],
                "mid_channels": 240,
                "act_func": "relu",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 40,
                "out_channels": 40,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第七层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 40,
                "out_channels": 40,
                "kernel_size": kernel_size[6],
                "stride": 1,
                "expand_ratio": expand_coefficient[6],
                "mid_channels": 240,
                "act_func": "relu",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 40,
                "out_channels": 40,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第八层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 40,
                "out_channels": 40,
                "kernel_size": kernel_size[7],
                "stride": 1,
                "expand_ratio": expand_coefficient[7],
                "mid_channels": 240,
                "act_func": "relu",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 40,
                "out_channels": 40,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第九层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 40,
                "out_channels": 80,
                "kernel_size": kernel_size[8],
                "stride": 2,
                "expand_ratio": expand_coefficient[8],
                "mid_channels": 240,
                "act_func": "h_swish",
                "use_se": False,
            },
            "shortcut": None,
        },
        {  # 第十层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 80,
                "out_channels": 80,
                "kernel_size": kernel_size[9],
                "stride": 1,
                "expand_ratio": expand_coefficient[9],
                "mid_channels": 480,
                "act_func": "h_swish",
                "use_se": False,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 80,
                "out_channels": 80,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第十一层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 80,
                "out_channels": 80,
                "kernel_size": kernel_size[10],
                "stride": 1,
                "expand_ratio": expand_coefficient[10],
                "mid_channels": 480,
                "act_func": "h_swish",
                "use_se": False,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 80,
                "out_channels": 80,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第十二层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 80,
                "out_channels": 80,
                "kernel_size": kernel_size[11],
                "stride": 1,
                "expand_ratio": expand_coefficient[11],
                "mid_channels": 480,
                "act_func": "h_swish",
                "use_se": False,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 80,
                "out_channels": 80,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第十三层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 80,
                "out_channels": 112,
                "kernel_size": kernel_size[12],
                "stride": 1,
                "expand_ratio": expand_coefficient[12],
                "mid_channels": 480,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": None,
        },
        {  # 第十四层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 112,
                "out_channels": 112,
                "kernel_size": kernel_size[13],
                "stride": 1,
                "expand_ratio": expand_coefficient[13],
                "mid_channels": 672,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 112,
                "out_channels": 112,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第十五层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 112,
                "out_channels": 112,
                "kernel_size": kernel_size[14],
                "stride": 1,
                "expand_ratio": expand_coefficient[14],
                "mid_channels": 672,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 112,
                "out_channels": 112,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第十六层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 112,
                "out_channels": 112,
                "kernel_size": kernel_size[15],
                "stride": 1,
                "expand_ratio": expand_coefficient[15],
                "mid_channels": 672,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 112,
                "out_channels": 112,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第十七层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 112,
                "out_channels": 160,
                "kernel_size": kernel_size[16],
                "stride": 2,
                "expand_ratio": expand_coefficient[16],
                "mid_channels": 672,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": None,
        },
        {  # 第十八层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 160,
                "out_channels": 160,
                "kernel_size": kernel_size[17],
                "stride": 1,
                "expand_ratio": expand_coefficient[17],
                "mid_channels": 960,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 160,
                "out_channels": 160,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第十九层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 160,
                "out_channels": 160,
                "kernel_size": kernel_size[18],
                "stride": 1,
                "expand_ratio": expand_coefficient[18],
                "mid_channels": 960,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 160,
                "out_channels": 160,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
        {  # 第二十层
            "name": "ResidualBlock",
            "conv": {
                "name": "MBConvLayer",
                "in_channels": 160,
                "out_channels": 160,
                "kernel_size": kernel_size[19],
                "stride": 1,
                "expand_ratio": expand_coefficient[19],
                "mid_channels": 960,
                "act_func": "h_swish",
                "use_se": True,
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 160,
                "out_channels": 160,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act",
            },
        },
    ]

    return blocks_list


def get_network_configuration(
    expand_coefficient: list, kernel_size: list, num_class: int
):
    """
    Args:
    expand_coefficient: list, expand coefficient of each layer
    """
    blocks_list = get_blocks_list(
        expand_coefficient=expand_coefficient, kernel_size=kernel_size
    )
    network_config = {
        "name": "MobileNetV3",
        "bn": {"momentum": 0.1, "eps": 1e-05, "ws_eps": None},
        "first_conv": {
            "name": "ConvLayer",
            "kernel_size": 3,
            "stride": 2,
            "dilation": 1,
            "groups": 1,
            "bias": False,
            "has_shuffle": False,
            "use_se": False,
            "in_channels": 3,
            "out_channels": 16,
            "use_bn": True,
            "act_func": "h_swish",
            "dropout_rate": 0,
            "ops_order": "weight_bn_act",
        },
        "blocks": blocks_list,
        "final_expand_layer": {
            "name": "ConvLayer",
            "kernel_size": 1,
            "stride": 1,
            "dilation": 1,
            "groups": 1,
            "bias": False,
            "has_shuffle": False,
            "use_se": False,
            "in_channels": 160,
            "out_channels": 960,
            "use_bn": True,
            "act_func": "h_swish",
            "dropout_rate": 0,
            "ops_order": "weight_bn_act",
        },
        "feature_mix_layer": {
            "name": "ConvLayer",
            "kernel_size": 1,
            "stride": 1,
            "dilation": 1,
            "groups": 1,
            "bias": False,
            "has_shuffle": False,
            "use_se": False,
            "in_channels": 960,
            "out_channels": 1280,
            "use_bn": False,
            "act_func": "h_swish",
            "dropout_rate": 0,
            "ops_order": "weight_bn_act",
        },
        "classifier": {
            "name": "LinearLayer",
            "in_features": 1280,
            "out_features": num_class,
            "bias": True,
            "use_bn": False,
            "act_func": None,
            "dropout_rate": 0,
            "ops_order": "weight_bn_act",
        },
    }
    return network_config


def get_network_config(arch_str: str, num_class: int, verbose: bool = False):
    """
    Args:
    arch_str: str, architecture string
    """
    if verbose:
        print(f">>> arch_str: {arch_str}")
    strings = arch_str.split("_")
    kernel_size = []
    expand_coefficient = []
    for s in strings:
        ks, ex = s.split("-")[0], s.split("-")[1]
        ks, ex = int(ks), int(ex)
        assert ks in KS_LIST, f"Invalid kernel size: {ks}, expected in {KS_LIST}"
        assert (
            ex in EXPAND_LIST
        ), f"Invalid expand coefficient: {ex}, expected in {EXPAND_LIST}"
        kernel_size.append(ks)
        expand_coefficient.append(ex)
    if verbose:
        print(
            f">>> kernel_size: {kernel_size}, expand_coefficient: {expand_coefficient}"
        )
    network_config = get_network_configuration(
        expand_coefficient=expand_coefficient,
        kernel_size=kernel_size,
        num_class=num_class,
    )
    if verbose:
        print(f">>> network_config: {network_config}")
    return network_config


def test():
    network_config = get_network_config(
        arch_str="3-3_7-6_3-3_3-3_3-3_3-3_3-3_3-3_3-3_3-3_7-3_3-3_3-4_3-3_3-3_3-6_3-3_3-3_3-3_3-6",
        num_class=10,
        verbose=True,
    )
    print(f">>> network_config: {network_config}")


# test()
