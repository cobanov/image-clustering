from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
import numpy as np


def calculate_pca(embeddings):
    pca = PCA(n_components=8)
    pca_embeddings = pca.fit_transform(sqz_vec)
    return pca_embeddings


def calculate_kmeans(embeddings):
    centroid, label = kmeans2(data=embeddings, k=3, minit="points")
    counts = np.bincount(label)
    return kmeans_embeddings
