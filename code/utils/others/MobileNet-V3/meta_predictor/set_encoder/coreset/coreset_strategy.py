"""
Modified from DREAM: Efficient Dataset Distillation by Representative Matching, ICCV 2023
Liu, Yanqing and Gu, Jianyang and Wang, Kai and Zhu, Zheng and Jiang, Wei and You, Yang
"""

import torch
from fast_pytorch_kmeans import KMeans


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

    def query(self, n):
        embeddings = self.images
        index = torch.arange(len(embeddings), device="cuda")
        kmeans = KMeans(n_clusters=n, mode="euclidean")
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        dist_matrix = self.euclidean_dist(centers, embeddings)
        q_idxs = index[torch.argmin(dist_matrix, dim=1)]
        return q_idxs


