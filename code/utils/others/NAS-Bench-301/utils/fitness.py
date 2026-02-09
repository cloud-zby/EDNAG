import sys

sys.path.append("./")
sys.path.append("./nasbench/")
import torch
from utils.mapping import ReScale
from utils.NB301 import get_acc_by_matrix


def diversity_score(x):
    pop = x.shape[0]
    diversity_scores = torch.zeros(pop)
    for i in range(pop):
        distances = torch.norm(x - x[i], dim=1)
        diversity_scores[i] = torch.sum(distances)
    min_score = torch.min(diversity_scores)
    max_score = torch.max(diversity_scores)
    diversity_scores = (diversity_scores - min_score) / (
        max_score - min_score
    )  # [0, 1]
    return diversity_scores


def neural_predictor(adj_matrices, nb_api):
    batch_size = adj_matrices.shape[0]
    org_acc_list = []
    for i in range(batch_size):
        org_acc = get_acc_by_matrix(adj_matrices[i], nb_api)
        org_acc_list.append(org_acc)
    org_acc_tensor = torch.tensor(org_acc_list)
    return org_acc_tensor


def arch_fitness(adj_matrix, nb_api):
    adj_matrix = adj_matrix.view(-1, 4, 14)
    org_acc = neural_predictor(adj_matrix, nb_api)
    rescale_facotr = 1.05
    scaler = ReScale()
    fitness = scaler(org_acc * rescale_facotr)
    return org_acc, fitness
