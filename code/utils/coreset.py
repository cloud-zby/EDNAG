"""
Modified from DREAM: Efficient Dataset Distillation by Representative Matching, ICCV 2023
Liu, Yanqing and Gu, Jianyang and Wang, Kai and Zhu, Zheng and Jiang, Wei and You, Yang
"""

import os
import torch
from fast_pytorch_kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Coreset_Strategy_Per_Class:
    def __init__(self, images_per_class):
        self.images = images_per_class

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def query(self, n, save_fig=False, dataset_name=None, seed=None):
        embeddings = self.images
        index = torch.arange(len(embeddings), device="cuda")
        kmeans = KMeans(n_clusters=n, mode="euclidean")
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        dist_matrix = self.euclidean_dist(centers, embeddings)
        q_idxs = index[torch.argmin(dist_matrix, dim=1)]

        if save_fig:
            assert dataset_name is not None, "Please specify the dataset name for saving the figure."
            # 使用PCA将嵌入降到2维以便于可视化
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
            centers_2d = pca.transform(centers.cpu().numpy())

            # 绘制聚类效果图
            plt.figure(figsize=(5, 3.5))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels.cpu().numpy(), cmap='viridis', marker='o', alpha=0.5)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=100, label='Centers')
            plt.title('KMeans Clustering')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend()

            # 保存图像到指定路径
            save_dir = 'results/cluster'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"KMeans_{dataset_name}_{n}_cluster_{seed}.svg"),
                format="svg",
            )
            plt.close()
        return q_idxs
