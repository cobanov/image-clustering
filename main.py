from img2vec_pytorch import Img2Vec
import utils
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2
import torch

# Single Image Test
# img = Image.open("./dataset/tiles_01.jpg")
# vec = img2vec.get_vec(img, tensor=True)

# Multi Image Test

# list_of_images = utils.read_images_from_directory("dataset")
# pil_images = utils.read_with_pil(list_of_images)

# img2vec = Img2Vec(cuda=False)
# vec = img2vec.get_vec(pil_images, tensor=True)
# utils.save_embeddings(vec, "embeddings.pt")


def load_from_embeddings(embedding_path):
    """A brief description."""

    vec = torch.load(embedding_path)
    if vec.dim() > 2:
        vec = torch.squeeze(vec)

def main():
    pass