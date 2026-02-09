import torch
import copy
import torch.nn as nn
from eval_architecture.ofa.sub_net.ofa_net import OFASubNet
from torchprofile import profile_macs
from eval_architecture.ofa.sub_net.nsga_net_v2 import NSGANetV2

def set_architecture(
    n_cls, evaluator, drop_path, drop, img_size, device, model_str
):
    
    # g, acc = evaluator.get_architecture(model_str)
    g = OFASubNet(model_str).get_op_dict()
    subnet, config = evaluator.sample(g)
    # print(f"subnet.config", subnet.config)
    net = NSGANetV2.build_from_config(subnet.config, drop_connect_rate=drop_path)
    # net = NSGANetV2.build_from_config(config, drop_connect_rate=drop_path)
    net.load_state_dict(subnet.state_dict())
    print(f">>> Loading parameters from supernet model with {config}.")

    NSGANetV2.reset_classifier(
        net, last_channel=net.classifier.in_features, n_classes=n_cls, dropout_rate=drop
    )
    # calculate #Paramaters and #FLOPS
    inputs = torch.randn(1, 3, img_size, img_size)
    flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
    params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    if torch.cuda.device_count() > 1:
        print(">>> We will use ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
    net = net.to(device)
    return net, params, flops
