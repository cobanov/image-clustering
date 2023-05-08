from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
import numpy as np


def calculate_pca(embeddings, dim=8):
    print('Calculating PCA')
    pca = PCA(n_components=dim)
    pca_embeddings = pca.fit_transform(embeddings.squeeze())
    print('PCA calculating done!')
    return pca_embeddings


def calculate_kmeans(embeddings, k):
    print('KMeans processing...')
    centroid, labels = kmeans2(data=embeddings, k=k, minit="points")
    counts = np.bincount(labels)
    print('Kmeans done!')
    return centroid, labels