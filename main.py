import os
import argparse
import shutil
from itertools import compress

import torch
from img2vec_pytorch import Img2Vec
from tqdm import tqdm

import clustering
import utils


def init_parser():
    """
    Initializes the argument parser and adds arguments to it.
    :return: The parser object.
    """
    parser = argparse.ArgumentParser(description="Image clustering CLI")
    parser.add_argument("-i", "--input", help="Input directory path, e.g., ./images")
    parser.add_argument(
        "-c", "--cluster", help="Number of clusters", default=30, type=int
    )
    parser.add_argument("-p", "--pca", help="PCA dimensions", default=16, type=int)
    parser.add_argument("--cpu", help="Run on CPU", action="store_true")
    return parser


def main():
    # Parse command-line arguments
    parser = init_parser()
    args = parser.parse_args()

    cluster_range = args.cluster
    pca_dim = args.pca
    dir_path = args.input

    project_name = os.path.split(dir_path)[-1]
    embedding_path = f"embeddings/{project_name}.pt"
    clusters_directory = f"clusters/{project_name}"

    # Create required directories
    required_dirs = ["embeddings", "clusters"]
    for directory in required_dirs:
        utils.create_dir(directory)

    utils.create_dir(clusters_directory)

    # Get image file paths
    images = utils.read_images_from_directory(dir_path)

    # Read images with PIL
    pil_images = utils.read_with_pil(images)

    # Embeddings
    if os.path.exists(embedding_path):
        print("Embeddings already exist, loading from embeddings folder.")
        embeddings = utils.load_from_embeddings(embedding_path)
    else:
        # Get embeddings
        if args.cpu:
            print("Img2Vec is running on CPU...")
            img2vec = Img2Vec(cuda=False)
        else:
            print("Img2Vec is running on GPU...")
            img2vec = Img2Vec(cuda=True)

        embeddings = img2vec.get_vec(pil_images, tensor=True)
        print("Img2Vec process done.")
        utils.save_embeddings(embeddings, embedding_path)

    # Embeddings to PCA
    pca_embeddings = clustering.calculate_pca(embeddings, dim=pca_dim)

    # PCA to k-means
    centroids, labels = clustering.calculate_kmeans(pca_embeddings, k=cluster_range)

    # Save random sample clusters
    for label_number in tqdm(range(cluster_range)):
        label_mask = labels == label_number
        label_images = list(compress(pil_images, label_mask))
        utils.create_image_grid(label_images, project_name, label_number)

        path_images = list(compress(images, label_mask))
        target_directory = f"./clusters/{project_name}/cluster_{label_number}"
        utils.create_dir(target_directory)

        # Copy images into separate directories
        for img_path in path_images:
            shutil.copy2(img_path, target_directory)


if __name__ == "__main__":
    main()
