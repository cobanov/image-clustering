from img2vec_pytorch import Img2Vec
import utils
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
import torch
from itertools import compress
import os
import clustering
import shutil


img2vec = Img2Vec(cuda=True)
CLUSTER_RANGE = 30
PCA_DIM = 30
DIR_PATH = "./datasets/animals"


project_name = f"{os.path.split(DIR_PATH)[-1]}"
embedding_path = f"embeddings/{project_name}.pt"
clusters_directory = f"clusters/{project_name}"

# Create required directories
required_dirs = ["embeddings", "clusters"]
for dir in required_dirs:
    utils.create_dir(dir)


def main():

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
        print("Img2Vec is running...")

        vec = img2vec.get_vec(pil_images, tensor=True)
        utils.save_embeddings(vec, embedding_path)

    # Embeddings to PCA
    pca_embeddings = clustering.calculate_pca(embeddings=vec, dim=PCA_DIM)

    # PCA to kmeans
    centroid, labels = clustering.calculate_kmeans(pca_embeddings, k=CLUSTER_RANGE)

    # Save random sample clusters
    for label_number in range(CLUSTER_RANGE):
        label_mask = labels == label_number
        label_images = list(compress(pil_images, label_mask))
        utils.create_image_grid(label_images, project_name, label_number)

        path_images = list(compress(images, label_mask))
        target_directory = f"./clusters/{project_name}/cluster_{label_number}"
        utils.create_dir(target_directory)
        for img_path in path_images:
            print(img_path)
            shutil.copy2(img_path, target_directory, )

    ## Copy images to seperate directories
    #! Will be done


if __name__ == "__main__":
    main()
