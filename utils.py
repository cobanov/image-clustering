import glob
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm 

def read_images_from_directory(image_directory: str) -> list:
    """
    > It takes a directory as input and returns a list of all the images in that directory
    :param image_directory: The directory where the images are stored
    :type image_directory: str
    :return: A list of images
    """

    list_of_images = list()
    for ext in ("*.gif", "*.png", "*.jpg", "*.jpeg"):
        list_of_images.extend(
            glob.glob(os.path.join(image_directory, ext))
        )  # ? Remove sorted if it is there
    print(f"Images found: {len(list_of_images)}")

    return list_of_images


def read_with_pil(list_of_images: list, resize=True) -> list:
    """
    > Reads a list of images and returns a list of PIL images
    :param list_of_images: list of image paths
    :type list_of_images: list
    :param resize: If True, resize the image to 512x512, defaults to False (optional)
    :return: A list of PIL images
    """
    print('Images are reading...')
    pil_images = list()
    for img_path in tqdm(list_of_images):
        img = Image.open(img_path).convert("RGB")
        if resize:  #! No hard code
            img.thumbnail((512, 512))
        pil_images.append(img)
    print('Image reading done!')

    return pil_images


def save_embeddings(tensor, path="embeddings.pt"):
    torch.save(tensor, path)
    print(f'Embeddings are saved to {path}')


def load_from_embeddings(embedding_path):
    """A brief description."""

    vec = torch.load(embedding_path)
    print('Embeddings loaded from folder.')
    if vec.dim() > 2:
        vec = torch.squeeze(vec)
    return vec


def create_image_grid(label_images, project_name, label_number):

    for i in range(len(label_images)):
        if i >= 9:
            break
        image = label_images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap="gray", interpolation="none")
        plt.title(f"Class: {label_number}")
        plt.axis("off")
        plt.savefig(f"./clusters/{project_name}/cluster_{label_number}.png")


def create_dir(directory_path):
    if not Path(directory_path).is_dir():
        Path(directory_path).mkdir(exist_ok=True)
    return Path(directory_path).stem
