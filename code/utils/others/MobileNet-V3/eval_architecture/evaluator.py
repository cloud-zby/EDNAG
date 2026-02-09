import torch
import os
import json
from eval_architecture.ofa.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3

def parse_string_list(string):
    if isinstance(string, str):
        # convert '[5 5 5 7 7 7 3 3 7 7 7 3 3]' to [5, 5, 5, 7, 7, 7, 3, 3, 7, 7, 7, 3, 3]
        return list(map(int, string[1:-1].split()))
    else:
        return string


def pad_none(x, depth, max_depth):
    new_x, counter = [], 0
    for d in depth:
        for _ in range(d):
            new_x.append(x[counter])
            counter += 1
        if d < max_depth:
            new_x += [None] * (max_depth - d)
    return new_x


def validate_config(config, max_depth=4):
    kernel_size, exp_ratio, depth = config["ks"], config["e"], config["d"]

    if isinstance(kernel_size, str):
        kernel_size = parse_string_list(kernel_size)
    if isinstance(exp_ratio, str):
        exp_ratio = parse_string_list(exp_ratio)
    if isinstance(depth, str):
        depth = parse_string_list(depth)

    assert isinstance(kernel_size, list) or isinstance(kernel_size, int)
    assert isinstance(exp_ratio, list) or isinstance(exp_ratio, int)
    assert isinstance(depth, list)

    if len(kernel_size) < len(depth) * max_depth:
        kernel_size = pad_none(kernel_size, depth, max_depth)
    if len(exp_ratio) < len(depth) * max_depth:
        exp_ratio = pad_none(exp_ratio, depth, max_depth)

    # return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'w': config['w']}
    return {"ks": kernel_size, "e": exp_ratio, "d": depth}


class OFAEvaluator:
    """based on OnceForAll supernet taken from https://github.com/mit-han-lab/once-for-all"""

    def __init__(
        self,
        n_classes=1000,
        model_path=None,
        kernel_size=None,
        exp_ratio=None,
        depth=None,
    ):
        # default configurations
        self.kernel_size = (
            [3, 5, 7] if kernel_size is None else kernel_size
        )  # depth-wise conv kernel size
        self.exp_ratio = [3, 4, 6] if exp_ratio is None else exp_ratio  # expansion rate
        self.depth = (
            [2, 3, 4] if depth is None else depth
        )  # number of MB block repetition

        self.width_mult = 1.2

        self.engine = OFAMobileNetV3(
            n_classes=n_classes,
            dropout_rate=0,
            width_mult=self.width_mult,
            ks_list=self.kernel_size,
            expand_ratio_list=self.exp_ratio,
            depth_list=self.depth,
        )

        init = torch.load(model_path, map_location="cpu")["state_dict"]
        if init['version'] != 'EDNAG_MBV3':
            raise ValueError("Invalid model version")
        self.engine.load_state_dict(init['checkpoint'])

    def sample(self, config=None):
        """randomly sample a sub-network"""
        if config is not None:
            config = validate_config(config)
            self.engine.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])
        else:
            config = self.engine.sample_active_subnet()

        subnet = self.engine.get_active_subnet(preserve_weight=True)
        return subnet, config

    @staticmethod
    def save_net_config(path, net, config_name="net.config"):
        """dump run_config and net_config to the model_folder"""
        net_save_path = os.path.join(path, config_name)
        json.dump(net.config, open(net_save_path, "w"), indent=4)
        print("Network configs dump to %s" % net_save_path)

    @staticmethod
    def save_net(path, net, model_name):
        """dump net weight as checkpoint"""
        if isinstance(net, torch.nn.DataParallel):
            checkpoint = {"state_dict": net.module.state_dict()}
        else:
            checkpoint = {"state_dict": net.state_dict()}
        model_path = os.path.join(path, model_name)
        torch.save(checkpoint, model_path)
        print("Network model dump to %s" % model_path)
