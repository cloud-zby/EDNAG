import torch
import igraph
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from meta_acc_predictor.unnoised_model import MetaSurrogateUnnoisedModel
from torch.utils.data import Dataset
from utils.coreset import Coreset_Strategy_Per_Class


class FitnessRestorer:
    def __init__(self, dataset_name, seed, num_sample=20, save_fig=False):
        self.device = "cuda"
        self.test_dataset = MetaTestDataset(
            data_path="./meta_acc_predictor/data/meta_predictor_dataset/",
            data_name=dataset_name,
            num_sample=num_sample,
        )
        graph_config = load_graph_config(
            graph_data_name="nasbench201",
            nvt=7,
            data_path="meta_acc_predictor/data/nasbench201.pt",
        )
        meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(
            nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config
        )
        self.model = load_model(
            model=meta_surrogate_unnoised_model,
            ckpt_path="meta_acc_predictor/unnoised_checkpoint.pth.tar",
        )
        self.model.to(self.device)
        self.core_dataset = None
        print(">>> Selecting coreset data and encoding it...")
        data_per_class = self.test_dataset.get_all_data_per_class(device=self.device)
        coreset_data = []
        for classes in list(data_per_class.keys()):
            strategy = Coreset_Strategy_Per_Class(
                images_per_class=data_per_class[classes]
            )
            query_idxs = strategy.query(num_sample, save_fig, dataset_name).to(
                self.device
            )
            core_data = data_per_class[classes][query_idxs]
            coreset_data.append(core_data)
        coreset_data = torch.cat(coreset_data, dim=0)
        x_batch = []
        for _ in range(10):  # Collecting data
            x_batch.append(coreset_data)
        coreset_batch = torch.stack(x_batch).to(self.device)
        self.dataset_encodings = self.model.set_encode(coreset_batch.to(self.device))
        self.fitness_dict = {}
        self.fitness_dict_path = (
            f"/results/meta/fitness_cache/fitness_{dataset_name}_{seed}.pth"
        )
        if not os.path.exists(self.fitness_dict_path):
            os.makedirs(name="/results/meta/fitness_cache/", exist_ok=True)
        torch.save(self.fitness_dict, self.fitness_dict_path)
        print(">>> Cache file created at ", self.fitness_dict_path)

    def get_coreset_encode(self):
        return self.dataset_encodings

    def push_fitness(self, arch_str, fitness):
        self.fitness_dict = torch.load(self.fitness_dict_path)
        self.fitness_dict[arch_str] = fitness
        torch.save(self.fitness_dict, self.fitness_dict_path)

    def get_fitness(self, arch_str):
        if arch_str in self.fitness_dict:  # 可能已经读取出来了
            return self.fitness_dict[arch_str]
        else:
            self.fitness_dict = torch.load(self.fitness_dict_path)
            if arch_str in self.fitness_dict:
                return self.fitness_dict[arch_str]
            else:
                return None


class MetaTestDataset(Dataset):
    def __init__(self, data_path, data_name, num_sample):
        self.num_sample = num_sample
        self.data_name = data_name
        num_class_dict = {"cifar100": 100, "cifar10": 10, "aircraft": 30, "pets": 37}
        self.num_class = num_class_dict[data_name]
        self.x = torch.load(os.path.join(data_path, f"{data_name}bylabel.pt"))

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        data = []
        classes = list(range(self.num_class))
        for cls in classes:
            cx = self.x[cls][0]
            ridx = torch.randperm(len(cx))
            data.append(cx[ridx[: self.num_sample]])
        x = torch.cat(data)
        return x

    def get_all_data_per_class(self, device: str):
        data = {}
        classes = list(range(self.num_class))
        for choose_class in classes:
            cx = self.x[choose_class][0]
            data[choose_class] = cx.to(device)
        return data


def decode_NAS_BENCH_201_8_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n)
    for i, node in enumerate(row):
        g.vs[i]["type"] = node[0]
        if i < (n - 2) and i > 0:
            g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i)
    return g, n


def load_graph_config(graph_data_name, nvt, data_path):
    if graph_data_name != "nasbench201":
        raise NotImplementedError(graph_data_name)
    g_list = []
    max_n = 0  # maximum number of nodes
    ms = torch.load(data_path)["arch"]["matrix"]
    for i in range(len(ms)):
        g, n = decode_NAS_BENCH_201_8_to_igraph(ms[i])
        max_n = max(max_n, n)
        g_list.append((g, 0))
    # number of different node types including in/out node
    graph_config = {}
    graph_config["num_vertex_type"] = nvt  # original types + start/end types
    graph_config["max_n"] = max_n  # maximum number of nodes
    graph_config["START_TYPE"] = 0  # predefined start vertex type
    graph_config["END_TYPE"] = 1  # predefined end vertex type
    return graph_config


def get_items(full_target, full_source, source):
    result = []
    for arch_str in source:
        if arch_str == "" or arch_str not in full_source:
            result.append(None)
        else:
            idx = full_source.index(arch_str)
            result.append(full_target[idx])
    return result


def collect_data(arch_igraph, device, test_dataset):
    x_batch, g_batch = [], []
    for _ in range(10):
        x_batch.append(test_dataset[0])
        g_batch.append(arch_igraph)
    return torch.stack(x_batch).to(device), g_batch


def load_model(model, ckpt_path):
    model.cpu()
    model.load_state_dict(torch.load(ckpt_path))
    return model


def get_pred_acc(
    arch_str,
    arch_igraph,
    device,
    test_dataset,
    meta_surrogate_unnoised_model,
    fitness_restorer,
):
    if arch_igraph is None:
        return None, fitness_restorer
    else:
        y_pred = fitness_restorer.get_fitness(arch_str)
        if y_pred is not None:  # Use cached fitness
            return y_pred, fitness_restorer
        else:  # Compute fitness
            x, g = collect_data(arch_igraph, device, test_dataset)
            # D_mu = meta_surrogate_unnoised_model.set_encode(x.to(device))
            D_mu = fitness_restorer.get_coreset_encode().to(device)
            G_mu = meta_surrogate_unnoised_model.graph_encode(g)
            y_pred = meta_surrogate_unnoised_model.predict(D_mu, G_mu)
            y_pred = torch.mean(y_pred)
            fitness_restorer.push_fitness(arch_str, y_pred.item())
            return y_pred.item(), fitness_restorer


def meta_predictor(
    test_dataset,
    meta_surrogate_unnoised_model,
    arch_igraphs,
    arch_str_list,
    device="cuda",
    fitness_restorer=None,
):
    """Internal meta neural architecture performance predictor for NAS-Bench-201.

    Args:
        arch_igraphs: list of igraph.Graph objects, each representing a neural architecture.
        data_path: path to the dataset directory.
        data_name: meta-test dataset name.
        num_sample: number of images as input for set encoder.
        graph_data_name: name of the graph dataset.
        nvt: number of different node types, 7: NAS-Bench-201 including in/out node.
        hs: hidden size of GRUs.
        nz: number of dimensions of latent vectors z.
        device: device to run the neural networks.

    Returns:
        y_pred_list: list of predicted accuracies for the input architectures.
    """
    meta_surrogate_unnoised_model.to(device)
    meta_surrogate_unnoised_model.eval()

    y_pred_list = [None] * len(arch_igraphs)
    with torch.no_grad():
        for i in range(len(arch_igraphs)):
            y_pred_list[i], fitness_restorer = get_pred_acc(
                arch_str_list[i],
                arch_igraphs[i],
                device,
                test_dataset,
                meta_surrogate_unnoised_model,
                fitness_restorer,
            )
    return y_pred_list


def meta_acc_rescale(y_pred_list):
    result = []
    for y_pred in y_pred_list:
        if y_pred is not None:
            rescaled_y_pred = (0.10 + y_pred) * 550 + 40
        else:
            rescaled_y_pred = 0.0
        result.append(rescaled_y_pred)
    return result


def meta_neural_predictor(
    test_dataset,
    meta_surrogate_unnoised_model,
    arch_str_list,
    dataset_name,
    nasbench201,
    fitness_restorer,
):
    """Meta neural architecture performance predictor for NAS-Bench-201.

    Args:
        arch_str_list: list of strings, each representing a neural architecture.
        data_name: meta-test dataset name, must be one of ['cifar10', 'cifar100', 'aircraft', 'pets'].
        num_sample: number of images as input for set encoder, default is 100.

    Returns:
        y_pred_list: list of predicted accuracies for the input architectures.
    """
    assert dataset_name in [
        "cifar10",
        "cifar100",
        "aircraft",
        "pets",
    ], f"Unsupported dataset: {dataset_name}"
    if nasbench201 is None:
        nasbench201 = torch.load("meta_acc_predictor/data/nasbench201.pt")
    arch_igraphs = get_items(
        full_target=nasbench201["arch"]["igraph"],
        full_source=nasbench201["arch"]["str"],
        source=arch_str_list,
    )

    if test_dataset is None:
        test_dataset = MetaTestDataset(
            data_path="./meta_acc_predictor/data/meta_predictor_dataset/",
            data_name=dataset_name,
            num_sample=20,
        )
    if meta_surrogate_unnoised_model is None:
        graph_config = load_graph_config(
            graph_data_name="nasbench201",
            nvt=7,
            data_path="meta_acc_predictor/data/nasbench201.pt",
        )
        meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(
            nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config
        )
        meta_surrogate_unnoised_model = load_model(
            model=meta_surrogate_unnoised_model,
            ckpt_path="meta_acc_predictor/unnoised_checkpoint.pth.tar",
        )

    y_pred_list = meta_predictor(
        test_dataset=test_dataset,
        meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
        arch_igraphs=arch_igraphs,
        arch_str_list=arch_str_list,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fitness_restorer=fitness_restorer,
    )
    y_pred_list = meta_acc_rescale(y_pred_list)
    return y_pred_list


def test_meta_predictor():
    """
    Test the meta_neural_predictor function.

    CIFAR10:
    arch 8  - 93.82%,
    arch 2  - 93.66%,
    arch 10 - 92.04%,
    arch 6  - 91.94%,
    arch 4  - 91.10%,
    arch 9  - 90.99%,
    arch 11 - 90.11%,
    arch 3  - 88.53%,
    arch 1  - 85.86%,
    arch 7  - 75.85%,
    arch 5  - 53.09%,
    arch 12 - 00.00%
    """
    import warnings

    warnings.filterwarnings("ignore")
    arch_str_list = [
        "|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|",
        "|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|",
        "|skip_connect~0|+|nor_conv_3x3~0|skip_connect~1|+|none~0|skip_connect~1|skip_connect~2|",
        "|skip_connect~0|+|skip_connect~0|nor_conv_1x1~1|+|nor_conv_3x3~0|nor_conv_1x1~1|avg_pool_3x3~2|",
        "|avg_pool_3x3~0|+|avg_pool_3x3~0|avg_pool_3x3~1|+|none~0|skip_connect~1|avg_pool_3x3~2|",
        "|skip_connect~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|none~1|none~2|",
        "|none~0|+|skip_connect~0|nor_conv_3x3~1|+|avg_pool_3x3~0|nor_conv_3x3~1|none~2|",
        "|nor_conv_3x3~0|+|none~0|none~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|",
        "|nor_conv_1x1~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|nor_conv_1x1~0|avg_pool_3x3~1|avg_pool_3x3~2|",
        "|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|skip_connect~1|none~2|",
        "|skip_connect~0|+|nor_conv_3x3~0|none~1|+|skip_connect~0|avg_pool_3x3~1|nor_conv_1x1~2|",
        "",
    ]
    for dataset in ["cifar10", "cifar100", "aircraft", "pets"]:
        fitness_restorer = FitnessRestorer(dataset_name=dataset, seed=0, num_sample=20)
        test_dataset = MetaTestDataset(
            data_path="./meta_acc_predictor/data/meta_predictor_dataset/",
            data_name=dataset,
            num_sample=20,
        )
        graph_config = load_graph_config(
            graph_data_name="nasbench201",
            nvt=7,
            data_path="meta_acc_predictor/data/nasbench201.pt",
        )
        meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(
            nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config
        )
        meta_surrogate_unnoised_model = load_model(
            model=meta_surrogate_unnoised_model,
            ckpt_path="meta_acc_predictor/unnoised_checkpoint.pth.tar",
        )
        print(
            f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nDataset: {dataset}"
        )
        y_pred_list = np.array(
            meta_neural_predictor(
                test_dataset,
                meta_surrogate_unnoised_model,
                arch_str_list,
                dataset,
                None,
                fitness_restorer,
            )
        )
        print(f"Predicted accuracies: {y_pred_list}")
        sorted_indices = np.argsort(y_pred_list)[::-1] + 1
        print(f"Sorted indices: {sorted_indices}")
        if dataset == "cifar10":
            print("CIFAR10 acc: [8, 2, 10, 6, 4, 9, 11, 3, 1, 7, 5, 12]")


def draw_cluster_fig(seed):
    # for dataset in ["cifar10", "cifar100", "aircraft", "pets"]:
    for dataset in ["cifar100", "aircraft"]:
        n = 5
        print(f"Dataset: {dataset}, Num_sample: {n}")
        test_dataset = MetaTestDataset(
            data_path="./meta_acc_predictor/data/meta_predictor_dataset/",
            data_name=dataset,
            num_sample=n,
        )
        data_per_class = test_dataset.get_all_data_per_class(device="cuda")
        coreset_data = []
        for classes in list(data_per_class.keys()):
            strategy = Coreset_Strategy_Per_Class(
                images_per_class=data_per_class[classes]
            )
            query_idxs = strategy.query(n, True, dataset, seed).to("cuda")
            core_data = data_per_class[classes][query_idxs]
            coreset_data.append(core_data)
        coreset_data = torch.cat(coreset_data, dim=0)


# if __name__ == '__main__':
#     import warnings
#     warnings.filterwarnings("ignore")
#     test_meta_predictor()

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        draw_cluster_fig(seed=seed)
