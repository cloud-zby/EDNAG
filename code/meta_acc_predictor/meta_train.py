import warnings

warnings.filterwarnings("ignore")
import os
import tqdm
import random
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys

sys.path.append(os.getcwd())
from utils.meta_d2a import load_graph_config
from meta_acc_predictor.unnoised_model import MetaSurrogateUnnoisedModel


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    graph = [item[1] for item in batch]
    acc = torch.stack([item[2] for item in batch])
    return [x, graph, acc]


class MetaTrainDatabaseImgnet(Dataset):
    def __init__(self, data_path, num_sample):
        self.mode = "train"
        self.acc_norm = True
        self.num_sample = num_sample
        self.x = torch.load(os.path.join(data_path, "imgnet32bylabel.pt"))

        mtr_data_path = os.path.join(data_path, "meta_train_tasks_predictor.pt")
        idx_path = os.path.join(data_path, "meta_train_tasks_predictor_idx.pt")

        data = torch.load(mtr_data_path)
        self.acc = data["acc"]
        self.task = data["task"]
        self.graph = data["g"]

        random_idx_lst = torch.load(idx_path)
        self.idx_lst = {}
        self.idx_lst["valid"] = random_idx_lst[:400]
        self.idx_lst["train"] = random_idx_lst[400:]
        self.acc = torch.tensor(self.acc)
        self.mean = torch.mean(self.acc[self.idx_lst["train"]]).item()
        self.std = torch.std(self.acc[self.idx_lst["train"]]).item()
        self.task_lst = torch.load(os.path.join(data_path, "meta_train_task_lst.pt"))

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.idx_lst[self.mode])

    def __getitem__(self, index):
        data = []
        ridx = self.idx_lst[self.mode]
        tidx = self.task[ridx[index]]
        classes = self.task_lst[tidx]
        graph = self.graph[ridx[index]]
        acc = self.acc[ridx[index]]
        for dataset_class in classes:
            cx = self.x[dataset_class - 1][0]
            ridx = torch.randperm(len(cx))
            data.append(cx[ridx[: self.num_sample]])
        x = torch.cat(data)
        if self.acc_norm:
            acc = ((acc - self.mean) / self.std) / 100.0
        else:
            acc = acc / 100.0
        return x, graph, acc


class MetaTrainDatabase(Dataset):
    def __init__(self, dataset_name, data_path, num_sample):
        assert dataset_name in [
            "cifar10",
            "cifar100",
        ], f"Invalid dataset name: {dataset_name}, it should be in [cifar10, cifar100, imagenet16-120]"
        self.mode = "train"
        self.acc_norm = True
        self.num_sample = num_sample
        self.x = torch.load(os.path.join(data_path, f"{dataset_name}bylabel.pt"))
        nasbench201 = torch.load("meta_acc_predictor/data/nasbench201.pt")
        self.acc = nasbench201["test-acc"]["cifar10"]
        self.graph = nasbench201["arch"]["igraph"]
        num_class_dict = {"cifar10": 10, "cifar100": 100}
        self.classes = num_class_dict[dataset_name]
        total_index = list(range(len(nasbench201["arch"]["str"])))
        random.shuffle(total_index)
        self.idx_lst = {}
        self.idx_lst["valid"] = total_index[:1250]
        self.idx_lst["train"] = total_index[1250:]
        self.acc = torch.tensor(self.acc)
        self.mean = torch.mean(self.acc[self.idx_lst["train"]]).item()
        self.std = torch.std(self.acc[self.idx_lst["train"]]).item()

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return len(self.idx_lst[self.mode])

    def __getitem__(self, index):
        data = []
        for dataset_class in range(self.classes):
            cx = self.x[dataset_class][0]
            ridx = torch.randperm(len(cx))
            data.append(cx[ridx[: self.num_sample]])
        x = torch.cat(data)
        idx_lst = self.idx_lst[self.mode]
        graph = self.graph[idx_lst[index]]
        acc = self.acc[idx_lst[index]]
        if self.acc_norm:
            acc = ((acc - self.mean) / self.std) / 100.0
        else:
            acc = acc / 100.0
        return x, graph, acc


def get_meta_train_loader(dataset_name, num_sample, batch_size):
    assert dataset_name in [
        "cifar10",
        "cifar100",
        "imagenet-1k",
    ], f"Invalid dataset name: {dataset_name}, it should be in [cifar10, cifar100, imagenet-1k]"
    if dataset_name in ["cifar10", "cifar100"]:
        dataset = MetaTrainDatabase(
            dataset_name=dataset_name,
            data_path="meta_acc_predictor/data/meta_predictor_dataset",
            num_sample=num_sample,
        )
    else:
        dataset = MetaTrainDatabaseImgnet(
            data_path="meta_acc_predictor/data/meta_predictor_dataset",
            num_sample=num_sample,
        )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )
    return loader


def train_meta_predictor(
    epoches_list,
    num_sample,
    batch_size,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path="meta_acc_predictor/unnoised_checkpoint.pth.tar",
    from_scratch=True,
):
    graph_config = load_graph_config(
        graph_data_name="nasbench201",
        nvt=7,
        data_path="meta_acc_predictor/data/nasbench201.pt",
    )
    model = MetaSurrogateUnnoisedModel(
        nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config
    )
    if not from_scratch and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Load model checkpoints from {checkpoint_path}")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for idx in range(len(epoches_list)):
        dataset_name, epoches = epoches_list[idx]
        print(f"Training meta surrogate predictor model on {dataset_name} dataset")
        loader = get_meta_train_loader(
            dataset_name=dataset_name, num_sample=num_sample, batch_size=batch_size
        )
        model.to(device)
        model.train()
        loader.dataset.set_mode("train")
        train_bar = tqdm.tqdm(range(epoches), ncols=100, desc="Train")
        for epoch in train_bar:
            loss_list = []
            for x, graph, acc in loader:
                optimizer.zero_grad()
                x, graph, acc = x.to(device).float(), graph, acc.to(device).float()
                D_mu = model.set_encode(x.to(device)).float()
                G_mu = model.graph_encode(graph).float()
                y_pred = model.predict(D_mu, G_mu).float()

                y = acc.to(device)
                loss = model.mseloss(y_pred, y.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                loss_list.append(loss.cpu().item())
            train_bar.set_postfix_str(f"loss: {sum(loss_list)/len(loss_list)}")
        model.to(device)
        model.eval()
        loader.dataset.set_mode("valid")
        valid_bar = tqdm.tqdm(loader, ncols=100, desc="Test")
        loss_list = []
        for x, graph, acc in valid_bar:
            x, graph, acc = x.to(device), graph, acc.to(device)
            D_mu = model.set_encode(x.to(device))
            G_mu = model.graph_encode(graph)
            y_pred = model.predict(D_mu, G_mu)
            y = acc.to(device)
            loss = model.mseloss(y_pred, y.unsqueeze(-1))
            valid_bar.set_postfix_str(f"loss: {loss.cpu().item()}")
            loss_list.append(loss.cpu().item())
        print(f"Test finished, average loss is {sum(loss_list) / len(loss_list)}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Train finished, model saved to {checkpoint_path} !")


if __name__ == "__main__":
    # epoches_list = [
    #     ('imagenet-1k', 100),
    #     ('cifar10', 20),
    #     ('cifar100', 40),
    #     ('imagenet-1k', 100),
    #     ('cifar10', 20),
    #     ('cifar100', 40),
    #     ('imagenet-1k', 100),
    # ]
    epoches_list = [("imagenet-1k", 500)]
    train_meta_predictor(
        epoches_list=epoches_list, num_sample=20, batch_size=64, from_scratch=False
    )

"""
cd /d D:/OneDrive/ZHOU_BINGYE/2024-2025学年上/NAS/EvoDiff/DiffEvo-NAS/EvoDiff/NAS-Bench-201_SearchSpace
conda activate base
python meta_acc_predictor/meta_train.py
"""
