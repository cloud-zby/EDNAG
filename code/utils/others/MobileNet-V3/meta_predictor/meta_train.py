"""Modified from MetaD2A"""

import os
import sys
import tqdm
import torch
import warnings
import numpy as np
from torch import optim
from scipy.stats import pearsonr
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append("./")
warnings.filterwarnings("ignore")
from meta_predictor.models import PredictorModel
from meta_predictor.accurancy_predictor import MetaPredictor
from meta_predictor.meta_dataset import load_graph_config
from search_space.searchspace_utils import decode_ofa_mbv3_to_igraph


class MetaTrainDatabase(Dataset):
    def __init__(self, num_sample: int, acc_norm: bool):
        self.mode = "train"
        self.acc_norm = acc_norm
        self.num_sample = num_sample
        self.x = torch.load("meta_predictor/dataset/imgnet32bylabel.pt")

        self.dpath = "meta_predictor/train_dataset/"
        self.dname = f"database_219152_14.0K"
        # self.dpath = "meta_predictor/train_dataset/"
        # self.dname = f"collected_database"

        # if not os.path.exists(self.dpath + f"{self.dname}_train.pt"):
        #     database = torch.load(self.dpath + f"{self.dname}.pt")

        #     rand_idx = torch.randperm(len(database))
        #     test_len = int(len(database) * 0.15)
        #     idxlst = {
        #         "test": rand_idx[:test_len],
        #         "valid": rand_idx[test_len : 2 * test_len],
        #         "train": rand_idx[2 * test_len :],
        #     }

        #     for mode in ["train", "valid", "test"]:
        #         acc, graph, cls, net, flops = [], [], [], [], []
        #         for idx in tqdm.tqdm(
        #             idxlst[mode].tolist(), desc=f"Preprocess data-{mode}"
        #         ):
        #             acc.append(database[idx]["top1"])
        #             net.append(database[idx]["net"])
        #             cls.append(database[idx]["class"])
        #             flops.append(database[idx]["flops"])
        #         if mode == "train":
        #             mean = torch.mean(torch.tensor(acc)).item()
        #             std = torch.std(torch.tensor(acc)).item()
        #         torch.save(
        #             {
        #                 "acc": acc,
        #                 "class": cls,
        #                 "net": net,
        #                 "flops": flops,
        #                 "mean": mean,
        #                 "std": std,
        #             },
        #             self.dpath + f"{self.dname}_{mode}.pt",
        #         )

        self.set_mode(self.mode)

    def set_mode(self, mode):
        self.mode = mode
        data = torch.load(self.dpath + f"{self.dname}_{self.mode}.pt")
        self.acc = data["acc"]
        self.cls = data["class"]
        self.net = data["net"]
        self.flops = data["flops"]
        self.mean = data["mean"]
        self.std = data["std"]

    def __len__(self):
        return len(self.acc)

    def __getitem__(self, index):
        data = []
        classes = self.cls[index]
        acc = self.acc[index]
        graph = self.net[index]

        for i, cls in enumerate(classes):
            cx = self.x[cls.item()][0]
            ridx = torch.randperm(len(cx))
            data.append(cx[ridx[: self.num_sample]])
        x = torch.cat(data)
        if self.acc_norm:
            acc = ((acc - self.mean) / self.std) / 100.0
        else:
            acc = acc / 100.0
        return x, graph, torch.tensor(acc).view(1, 1)


def collate_fn(batch):
    # x = torch.stack([item[0] for item in batch])
    # graph = [item[1] for item in batch]
    # acc = torch.stack([item[2] for item in batch])
    return batch


def forward(model, x, arch):
    """Parameter `model` is expected to  be an object of PredictorModel"""
    D_mu = model.set_encode(x.unsqueeze(0).to("cuda")).unsqueeze(0)
    G_mu = model.graph_encode(arch)
    y_pred = model.predict(D_mu, G_mu)
    return y_pred


def collect_graph(x, arch_igraph):
    x_batch, g_batch = [], []
    for _ in range(10):
        x_batch.append(x)
        g_batch.append(arch_igraph)
    return torch.stack(x_batch).to("cuda"), g_batch


def meta_train(model, num_sample, batch_size, train_epoch, from_scratch):
    """Parameter `model` is expected to  be an object of PredictorModel"""
    dataset = MetaTrainDatabase(num_sample=num_sample, acc_norm=True)
    mtrloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=10, verbose=True
    )
    # MetaD2A预训练模型
    model.load_state_dict(
        torch.load("./meta_predictor/checkpoints/ckpt_max_corr.pt")
    )
    if not from_scratch:
        if os.path.exists("./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt"):
            model.load_state_dict(
                torch.load("./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt")
            )
            print(
                ">>> Load checkpoints from ./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt"
            )
        else:
            print(
                ">>> Checkpoints not found in ./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt"
            )

    # 训练
    print(">>> Start training meta predictor model...")
    for epoch in range(1, train_epoch + 1):
        torch.cuda.empty_cache()
        model.to("cuda")
        model.train()
        mtrloader.dataset.set_mode("train")
        dlen = len(mtrloader.dataset)
        trloss = 0
        y_all, y_pred_all = [], []
        bar = tqdm.tqdm(mtrloader, desc=f"Epoch {epoch}", ncols=120)
        for batch in bar:
            batch_loss = 0
            y_batch, y_pred_batch = [], []
            optimizer.zero_grad()
            for x, g, acc in batch:
                y_pred = forward(model=model, x=x, arch=decode_ofa_mbv3_to_igraph(g))
                y = acc.to("cuda")
                # print(f"y: {y}, y_pred: {y_pred}")
                batch_loss += model.mseloss(y_pred, y)
                y = y.squeeze().tolist()
                y_pred = y_pred.squeeze().tolist()
                y_batch.append(y)
                y_pred_batch.append(y_pred)
                y_all.append(y)
                y_pred_all.append(y_pred)
            batch_loss.backward()
            trloss += float(batch_loss)
            optimizer.step()
            bar.set_postfix_str(
                f"Loss: {batch_loss.item():.4f}"
            )
        loss, corr = trloss / dlen, pearsonr(np.array(y_all), np.array(y_pred_all))[0]
        print(
            f">>> Epoch {epoch}: avg train loss is {loss:.6f}, corr is {corr:.6f}, avg y is {np.mean(y_all):.2f}, avg y_pred is {np.mean(y_pred_all):.2f}"
        )
        scheduler.step(loss)

        # 验证
        if epoch % 10 == 0:
            model.eval()
            valoss = 0
            mtrloader.dataset.set_mode("valid")
            dlen = len(mtrloader.dataset)
            y_all, y_pred_all = [], []
            with torch.no_grad():
                for batch in mtrloader:
                    batch_loss = 0
                    y_batch, y_pred_batch = [], []
                    for x, g, acc in batch:
                        y_pred = forward(
                            model=model, x=x, arch=decode_ofa_mbv3_to_igraph(g)
                        )
                        y = acc.to("cuda")
                        batch_loss += model.mseloss(y_pred, y)
                        y = y.squeeze().tolist()
                        y_pred = y_pred.squeeze().tolist()
                        y_batch.append(y)
                        y_pred_batch.append(y_pred)
                        y_all.append(y)
                        y_pred_all.append(y_pred)
                    valoss += float(batch_loss)
            val_loss, val_corr = (
                valoss / dlen,
                pearsonr(np.array(y_all), np.array(y_pred_all))[0],
            )
            print(
                f"Valid Epoch {epoch}: Valid Loss is {val_loss:.4f}, Valid Corr is {val_corr:.4f}"
            )
            torch.save(
                model.state_dict(), "./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt"
            )
    torch.save(
        model.state_dict(), "./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt"
    )
    print(
        ">>> Training finished! Model saved to ./meta_predictor/checkpoints/ednas_ckpt_max_corr.pt"
    )


def main(
    num_sample=20,
    batch_size=30,
    train_epoch=10,
    hs=512,
    nz=56,
    nvt=27,
    from_scratch=True,
):
    model = PredictorModel(
        hs=hs, nz=nz, num_sample=num_sample, graph_config=load_graph_config(nvt=nvt)
    )
    meta_train(
        model=model,
        num_sample=num_sample,
        batch_size=batch_size,
        train_epoch=train_epoch,
        from_scratch=from_scratch,
    )


# if __name__ == "__main__":
#     main()
