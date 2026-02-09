import torch
import time
from utils.mapping import ReScale
from nas_201_api import NASBench201API

def load_nb201_api(path, verbose=False):
    # 加载 NAS-Bench-201 API
    load_api_time = time.time()
    print('>>> Creating the API for NAS-Bench-201')
    api = NASBench201API('./nas_201_api/NAS-Bench-201-v1_0-e61699.pth', verbose=verbose)
    if verbose: print(f'>>> Running time of creating NAS-Bench-201 API: {time.time()-load_api_time:.2f} s')
    return api

def get_nb201_arch_str(operation_matrix):
    assert operation_matrix.shape == (8, 7), f'operation_matrix shape should be (8, 7), but got {operation_matrix.shape}'
    operation_matrix = operation_matrix.argmax(dim=1)
    ops = ['input', 'output', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    arch_str = "|"
    switch = {
            0: lambda: None,  # 输入节点
            7: lambda: None,  # 输出节点
            1: lambda: arch_str + ops[operation_matrix[i].item()] + "~0|",
            2: lambda: arch_str + "+|" + ops[operation_matrix[i].item()] + "~0|",
            3: lambda: arch_str + ops[operation_matrix[i].item()] + "~1|",
            4: lambda: arch_str + "+|" + ops[operation_matrix[i].item()] + "~0|",
            5: lambda: arch_str + ops[operation_matrix[i].item()] + "~1|",
            6: lambda: arch_str + ops[operation_matrix[i].item()] + "~2|"
        }
    for i in range(8):
        if i in switch:
            result = switch[i]()
            if result:
                arch_str = result
    return arch_str

def neural_predictor(operation_matrix, api, dataset):
    population = operation_matrix.shape[0]
    operation_matrix = operation_matrix.view(population, 8, 7)
    acc_list = []
    invaliad_num = 0
    for i in range(operation_matrix.shape[0]):
        try:
            arch_str = get_nb201_arch_str(operation_matrix[i])
            index = api.query_index_by_arch(arch_str)
            acc = api.query_test_acc_by_index(index, dataset)
        except Exception as e:
            acc = 0.0
            invaliad_num += 1
        acc_list.append(acc)
    org_acc = torch.tensor(acc_list)
    valid_rate = 1.0 - float(invaliad_num) / float(operation_matrix.shape[0])
    return org_acc, valid_rate

def diversity_score(x): 
    pop = x.shape[0] 
    diversity_scores = torch.zeros(pop) 
    for i in range(pop): 
        distances = torch.norm(x - x[i], dim=1) 
        diversity_scores[i] = torch.sum(distances) 
    min_score = torch.min(diversity_scores) 
    max_score = torch.max(diversity_scores) 
    diversity_scores = (diversity_scores - min_score) / (max_score - min_score) # 归一化到[0, 1]
    return diversity_scores

def arch_fitness(operation_matrix, api, dataset):
    assert dataset in ['cifar10', 'cifar100', 'ImageNet16-120'], f'Unsupported dataset: {dataset}'
    org_acc, valid_rate = neural_predictor(operation_matrix, api, dataset) # 架构的准确率
    rescale_facotr_dict = {
        'cifar10': 1.0,
        'cifar100': 1.2,
        'ImageNet16-120': 2.0
    }
    fitness = ReScale()(org_acc * rescale_facotr_dict[dataset])
    return org_acc, fitness, valid_rate

