import torch
import time
from utils.mapping import ReScale
from nas_201_api import NASBench201API
from utils.meta_d2a import meta_neural_predictor
from utils.nb201_fitness import get_nb201_arch_str


def neural_predictor(
    operation_matrix,
    api,
    dataset,
    test_dataset,
    meta_surrogate_unnoised_model,
    nasbench201,
    fitness_restorer,
):
    population = operation_matrix.shape[0]
    operation_matrix = operation_matrix.view(population, 8, 7)
    invaliad_num = 0
    arch_str_list = []
    for i in range(operation_matrix.shape[0]):
        arch_str = get_nb201_arch_str(operation_matrix[i])
        if api.query_index_by_arch(arch_str) == -1:
            invaliad_num += 1
            arch_str = ''
        arch_str_list.append(arch_str)
    pred_acc_list = meta_neural_predictor(
        test_dataset=test_dataset,
        meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
        arch_str_list=arch_str_list, 
        dataset_name=dataset,
        nasbench201=nasbench201,
        fitness_restorer=fitness_restorer,
        )
    pred_acc = torch.tensor(pred_acc_list)
    valid_rate = 1.0 - float(invaliad_num) / float(operation_matrix.shape[0])
    return pred_acc, valid_rate


def meta_arch_fitness(
    operation_matrix,
    api,
    dataset,
    test_dataset,
    meta_surrogate_unnoised_model,
    nasbench201,
    fitness_restorer,
):
    assert dataset in ['cifar10', 'cifar100', 'aircraft', 'pets'], 'Unsupported dataset: {}'.format(dataset)
    pred_acc, valid_rate = neural_predictor(
        operation_matrix,
        api,
        dataset,
        test_dataset,
        meta_surrogate_unnoised_model,
        nasbench201,
        fitness_restorer,
    )  # 架构的准确率
    if dataset is 'aircraft':
        pred_acc = pred_acc * 0.92
    elif dataset is 'pets':
        pred_acc = pred_acc * 1.04
    fitness = ReScale()(pred_acc)
    return pred_acc, fitness, valid_rate
