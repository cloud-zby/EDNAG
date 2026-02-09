import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import warnings

warnings.filterwarnings("ignore")
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    print(
        f"Mixed Precision Training is not supported in this version of PyTorch {torch.__version__}."
    )
sys.path.append("./")
from eval_architecture.evaluator import OFAEvaluator
from eval_architecture.datasets.get_datasets import get_dataset
from eval_architecture.ofa.sub_net.set_subnet_arch import set_architecture


def train(train_queue, net, criterion, optimizer, grad_clip, device, mode: str):
    if mode == "mixed_prec":
        # 语句from torch.amp import autocast, GradScaler在低版本的torch中不支持
        return train_mixed(train_queue, net, criterion, optimizer, grad_clip, device)
    elif mode == "naive":
        return train_naive(train_queue, net, criterion, optimizer, grad_clip, device)
    elif mode == "accum_grad":
        return train_accum_grad(
            train_queue, net, criterion, optimizer, grad_clip, device
        )
    else:
        raise ValueError(
            f"Invalid training mode {mode}, expected mixed_prec, naive or accum_grad !"
        )


def train_mixed(train_queue, net, criterion, optimizer, grad_clip, device):
    """Mixed Precision Training"""
    net = net.to(device)
    net.train()
    train_loss, correct, total = 0, 0, 0
    bar = tqdm.tqdm(train_queue, ncols=120)

    scaler = GradScaler("cuda")  # 初始化 GradScaler
    for inputs, targets in bar:
        # upsample by bicubic to match imagenet training size
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):  # 自动混合精度
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        bar.set_postfix_str(
            f"train_loss {train_loss / total:.4f}, train_acc {100. * correct / total:.2f}"
        )
    return train_loss / total, 100.0 * correct / total


def train_naive(train_queue, net, criterion, optimizer, grad_clip, device):
    net = net.to(device)
    net.train()
    train_loss, correct, total = 0, 0, 0
    stream = torch.cuda.Stream(device=device)
    bar = tqdm.tqdm(train_queue, ncols=120)
    for inputs, targets in bar:
        # upsample by bicubic to match imagenet training size
        with torch.cuda.stream(stream):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        bar.set_postfix_str(
            f"train_loss {train_loss / total:.4f}, train_acc {100. * correct / total:.2f}"
        )
    return train_loss / total, 100.0 * correct / total


def train_accum_grad(
    train_queue, net, criterion, optimizer, grad_clip, device, accumulation_steps=5
):
    """Accumulated Gradients Training by 5 steps"""
    net = net.to(device)
    net.train()
    train_loss, correct, total = 0, 0, 0
    bar = tqdm.tqdm(train_queue, ncols=120)

    for i, (inputs, targets) in enumerate(bar):
        # upsample by bicubic to match imagenet training size
        inputs, targets = (
            inputs.to(device),
            targets.to(device).long(),
        )  # 确保标签是 LongTensor 类型
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()  # 计算梯度但不更新
        nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清除累计的梯度

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        bar.set_postfix_str(
            f"train_loss {train_loss / total:.4f}, train_acc {100. * correct / total:.2f}"
        )

    # 确保最后剩余的梯度也被更新
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return train_loss / total, 100.0 * correct / total


def infer(valid_queue, net, criterion, device, early_stop=False):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if early_stop and step == 10:
                break
    acc = 100.0 * correct / total

    return test_loss / total, acc


def _eval_arch(
    model_str: str,
    dataset: str,
    train_mode: str,
    lr=0.01,
    momentum=0.9,
    weight_decay=4e-5,
    epochs=20,
    total_param_epochs=180,
    grad_clip=5,
    cutout=True,
    cutout_length=16,
    autoaugment=True,
    drop=0.4,
    drop_path=0.4,
    img_size=224,  # Expand from 32 to 224 for CIFAR10 & CIFAR100 due to pre-trained supernet weights
    batch_size=96,
    verbose=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_perf_path="./eval_architecture/results/model_performance.pth",
):
    # Check if the model has been evaluated before
    if model_str is None or model_str == "":
        return 0, 0, 0
    if os.path.exists(model_perf_path):
        model_performance = torch.load(model_perf_path)
        if (dataset in list(model_performance.keys())) and (
            model_str in list(model_performance[dataset].keys())
        ):
            print(
                f">>> Model has been evaluated on dataset {dataset} and achieved {model_performance[dataset][model_str]} accuracy."
            )
            return model_performance[dataset][model_str], 0, 0
    supernet_model_path = "eval_architecture/checkpoints/ednag_mbv3_supernet"
    evaluator = OFAEvaluator(model_path=supernet_model_path)
    if verbose:
        print(
            f">>> Evaluator loaded super-net checkpoint trained by ImageNet-1k from {supernet_model_path}."
        )
    train_queue, valid_queue, num_class = get_dataset(
        data_name=dataset,
        batch_size=batch_size,
        img_size=img_size,
        autoaugment=autoaugment,
        cutout=cutout,
        cutout_length=cutout_length,
    )
    if verbose:
        print(f">>> Dataset {dataset} loaded successfully with {num_class} classes.")
    net, params, flops = set_architecture(
        n_cls=num_class,
        evaluator=evaluator,
        drop_path=drop_path,
        drop=drop,
        img_size=img_size,
        device=device,
        model_str=model_str,
    )
    if verbose:
        print(">>> Params {:.2f}M, Flops {:.0f}M".format(params, flops))
    net = net.to(device)
    # print(net)

    # parameters = net.parameters()
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    if verbose:
        print(
            f">>> Training model for {epochs} epochs with criterion {criterion} and optimizer {optimizer}."
        )
    max_test_acc = 0

    # 训练最后的一层分类层
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss, train_acc = train(
            train_queue,
            net,
            criterion,
            optimizer,
            grad_clip,
            device,
            mode=train_mode,
        )
        # 测试
        valid_loss, valid_acc = infer(valid_queue, net, criterion, device)
        if max_test_acc < valid_acc:
            max_test_acc = valid_acc
        if verbose:
            print(
                f">>> Epoch {epoch}, lr: {scheduler.get_last_lr()[0]:.4f}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.2f}, test_loss: {valid_loss:.3f}, test_acc: {valid_acc:.2f}."
            )
        scheduler.step()

    # 全参数训练
    for param in net.parameters():
        param.requires_grad = True
    parameters = net.parameters()
    optimizer = optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_param_epochs)
    for epoch in range(total_param_epochs):
        torch.cuda.empty_cache()
        train_loss, train_acc = train(
            train_queue,
            net,
            criterion,
            optimizer,
            grad_clip,
            device,
            mode=train_mode,
        )
        # 测试
        valid_loss, valid_acc = infer(valid_queue, net, criterion, device)
        if max_test_acc < valid_acc:
            max_test_acc = valid_acc
        if verbose:
            print(
                f">>> Epoch {epoch}, lr: {scheduler.get_last_lr()[0]:.4f}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.2f}, test_loss: {valid_loss:.3f}, test_acc: {valid_acc:.2f}."
            )
        scheduler.step()
        if (
            max_test_acc
            >= {"cifar10": 97.52, "cifar100": 86.07, "aircraft": 82.28, "pets": 95.34}[
                dataset
            ]
        ):
            return max_test_acc, params, flops

    # 保存模型准确率
    if os.path.exists(model_perf_path):
        try:  # In order to avoid error "pickle.UnpicklingError: pickle data was truncated"
            model_performance = torch.load(model_perf_path)
        except:
            try:
                model_performance = torch.load(model_perf_path)
            except:
                model_performance = torch.load(model_perf_path)
    else:
        model_performance = {}
    if dataset not in model_performance:
        model_performance[dataset] = {}
    model_performance[dataset][model_str] = max_test_acc
    torch.save(model_performance, model_perf_path)
    if verbose:
        print(
            f">>> Test acc of model on dataset {dataset} is {max_test_acc}, which is saved to {model_perf_path}."
        )
    return max_test_acc, params, flops


def eval_arch(
    arch_list: list,
    dataset: str,
    train_mode: str,
    verbose: bool = True,
):
    assert train_mode in [
        "mixed_prec",
        "naive",
        "accum_grad",
    ], f"Invalid training mode {train_mode}, expected mixed_prec, naive or accum_grad !"
    arch_acc_list = []
    params_list = []
    flops_list = []
    for idx, arch in enumerate(arch_list, start=1):
        if verbose:
            print(f">>> Evaluating architecture {idx}/{len(arch_list)}: {arch}")
        max_test_acc, params, flops = _eval_arch(
            dataset=dataset, model_str=arch, train_mode=train_mode, verbose=verbose
        )
        arch_acc_list.append(max_test_acc)
        params_list.append(params)
        flops_list.append(flops)
    return arch_acc_list, params_list, flops_list


def test():
    arch_acc_list = eval_arch(
        [
            "3-6_5-4_5-3_7-3_7-6_3-3_3-3_5-6_5-6_3-3_7-3_3-3_3-6_5-6_3-6_5-3_3-6_5-6_5-6_5-6"
        ],
        ["cifar10", "cifar100", "aircraft", "pets"][0],
        ["mixed_prec", "naive", "accum_grad"][0],
    )


# if __name__ == "__main__":
#     test()
