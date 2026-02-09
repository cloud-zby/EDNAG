import sys

sys.path.append("./")
sys.path.append("./nasbench/")
import torch
from nasbench import api
from utils.mapping import ReScale

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
    )  # 归一化到[0, 1]
    return diversity_scores


def _neural_predictor(adj_matrix, nb_api):
    # 将adjacency matrix的下三角和主对角线置为0
    adj_matrix = adj_matrix >= 0.5
    adj_matrix = adj_matrix.float().view(7, 7)
    adj_matrix = torch.triu(adj_matrix, diagonal=1)
    adj_list = adj_matrix.int().tolist()

    INPUT = "input"
    OUTPUT = "output"
    CONV1X1 = "conv1x1-bn-relu"
    CONV3X3 = "conv3x3-bn-relu"
    MAXPOOL3X3 = "maxpool3x3"

    model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix=adj_list,
        # Operations at the vertices of the module, matches order of matrix
        ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT],
    )
    try:
        acc = float(nb_api.query(model_spec)["test_accuracy"]) * 100.0
        # acc = float(nb_api.query(model_spec)["validation_accuracy"]) * 100.0
    except Exception as e:
        acc = 0.0
    return acc

def neural_predictor(adj_matrices, nb_api):
    batch_size = adj_matrices.shape[0]
    org_acc_list = []
    for i in range(batch_size):
        org_acc = _neural_predictor(adj_matrices[i], nb_api)
        org_acc_list.append(org_acc)
    org_acc_tensor = torch.tensor(org_acc_list)
    return org_acc_tensor

def arch_fitness(adj_matrix, nb_api):
    adj_matrix = adj_matrix.view(-1, 7, 7)
    org_acc = neural_predictor(adj_matrix, nb_api)

    rescale_facotr = 1.05
    scaler = ReScale()
    fitness = scaler(org_acc * rescale_facotr)

    return org_acc, fitness
