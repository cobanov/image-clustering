from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
import numpy as np


def calculate_pca(embeddings):
    pca = PCA(n_components=8)
    pca_embeddings = pca.fit_transform(embeddings.squeeze())
    return pca_embeddings


def calculate_kmeans(embeddings, k):
    centroid, labels = kmeans2(data=embeddings, k=k, minit="points")
    counts = np.bincount(labels)
    return centroid, labels