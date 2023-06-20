from img2vec_pytorch import Img2Vec
from itertools import compress
import shutil
import os
from tqdm import tqdm
import argparse

import clustering
import utils


def init_parser(**parser_kwargs):
    """
    This function initializes the parser and adds arguments to it
    :return: The parser object is being returned.
    """
    parser = argparse.ArgumentParser(description="Image caption CLI")
    parser.add_argument("-i", "--input", help="Input directoryt path, such as ./images")
    parser.add_argument(
        "-c", "--cluster", help="How many cluster will be", default=30, type=int
    )
    parser.add_argument("-p", "--pca", help="PCA Dimensions", default=16, type=int)
    parser.add_argument("--cpu", help="Run on CPU", action="store_true")

    return parser


def main():
    # CLI Requirements
    parser = init_parser()
    opt = parser.parse_args()

    CLUSTER_RANGE = opt.cluster
    PCA_DIM = opt.pca
    DIR_PATH = opt.input

    project_name = f"{os.path.split(DIR_PATH)[-1]}"
    embedding_path = f"embeddings/{project_name}.pt"
    clusters_directory = f"clusters/{project_name}"

    # Create required directories
    required_dirs = ["embeddings", "clusters"]
    for dir in required_dirs:
        utils.create_dir(dir)

    utils.create_dir(clusters_directory)

    # Get image datapaths
    images = utils.read_images_from_directory(DIR_PATH)

    # Read with PIL
    pil_images = utils.read_with_pil(images)

    # Embeddings
    if os.path.exists(embedding_path):
        print("Embeddings already exists, loading from embeddings folder.")
        vec = utils.load_from_embeddings(embedding_path)

    else:
        # Get embeddings
        if opt.cpu:
            print("Img2Vec is running on CPU...")
            img2vec = Img2Vec(cuda=False)
        else:
            print("Img2Vec is running on CPU...")
            img2vec = Img2Vec(cuda=True)

        vec = img2vec.get_vec(pil_images, tensor=True)
        print("Img2Vec process done.")
        utils.save_embeddings(vec, embedding_path)

    # Embeddings to PCA
    pca_embeddings = clustering.calculate_pca(embeddings=vec, dim=PCA_DIM)

    # PCA to kmeans
    centroid, labels = clustering.calculate_kmeans(pca_embeddings, k=CLUSTER_RANGE)

    # Save random sample clusters
    for label_number in tqdm(range(CLUSTER_RANGE)):
        label_mask = labels == label_number
        label_images = list(compress(pil_images, label_mask))
        utils.create_image_grid(label_images, project_name, label_number)

        path_images = list(compress(images, label_mask))
        target_directory = f"./clusters/{project_name}/cluster_{label_number}"
        utils.create_dir(target_directory)

        # Copy images into seperate directories
        for img_path in path_images:
            shutil.copy2(
                img_path,
                target_directory,
            )


if __name__ == "__main__":
    main()
