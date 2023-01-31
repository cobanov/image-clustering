from img2vec_pytorch import Img2Vec
import utils
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
import torch
from itertools import compress

import clustering

img2vec = Img2Vec(cuda=True)
CLUSTER_RANGE = 16

# Single Image Test
# img = Image.open("./dataset/tiles_01.jpg").convert('RGB')

# vec = img2vec.get_vec(img, tensor=True)

# Multi Image Test

# list_of_images = utils.read_images_from_directory("dataset")
# pil_images = utils.read_with_pil(list_of_images)


def main():
    # Get image datapaths
    images = utils.read_images_from_directory("./random_2k")

    # Read with PIL
    pil_images = utils.read_with_pil(images)

    # Get embeddings
    vec = img2vec.get_vec(pil_images, tensor=True)
    utils.save_embeddings(vec, "embeddings.pt")

    # Embeddings to PCA
    pca_embeddings = clustering.calculate_pca(vec)

    # PCA to kmeans
    centroid, labels = clustering.calculate_kmeans(pca_embeddings, k=CLUSTER_RANGE)

    # Copy images to different directories
    for label_number in range(CLUSTER_RANGE):
        label_mask = labels == label_number
        label_images = list(compress(pil_images, label_mask))
        utils.create_image_grid(label_images)


if __name__ == "__main__":
    main()
