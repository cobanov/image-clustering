import glob
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def read_images_from_directory(image_directory: str) -> list:
    """
    Reads all images from the given directory and returns a list of image paths.
    :param image_directory: The directory where the images are stored.
    :return: A list of image paths.
    """
    list_of_images = []
    image_extensions = ("*.gif", "*.png", "*.jpg", "*.jpeg")
    for ext in image_extensions:
        list_of_images.extend(glob.glob(os.path.join(image_directory, ext)))
    print(f"Images found: {len(list_of_images)}")
    return list_of_images


def read_with_pil(list_of_images: list, resize=True) -> list:
    """
    Reads a list of images using PIL and returns a list of PIL images.
    :param list_of_images: List of image paths.
    :param resize: If True, resize the image to 512x512. Defaults to True.
    :return: A list of PIL images.
    """
    print("Reading images...")
    pil_images = []
    for img_path in tqdm(list_of_images):
        img = Image.open(img_path).convert("RGB")
        if resize:
            img.thumbnail((512, 512))
        pil_images.append(img)
    print("Image reading done!")
    return pil_images


def save_embeddings(tensor, path="embeddings.pt"):
    """
    Saves the tensor embeddings to a file.
    :param tensor: The tensor to be saved.
    :param path: The path where the tensor should be saved. Defaults to "embeddings.pt".
    """
    torch.save(tensor, path)
    print(f"Embeddings are saved to {path}")


def load_from_embeddings(embedding_path):
    """
    Loads embeddings from the specified path.
    :param embedding_path: The path of the embeddings file.
    :return: The loaded embeddings tensor.
    """
    vec = torch.load(embedding_path)
    print("Embeddings loaded from folder.")
    if vec.dim() > 2:
        vec = torch.squeeze(vec)
    return vec


def create_image_grid(label_images, project_name, label_number):
    """
    Creates a grid of images with labels and saves it to a file.
    :param label_images: List of labeled images.
    :param project_name: The name of the project.
    :param label_number: The label number.
    """

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
    """
    Creates a directory if it doesn't exist.
    :param directory_path: The path of the directory.
    :return: The stem of the directory path.
    """
    directory = Path(directory_path)
    if not directory.is_dir():
        directory.mkdir(exist_ok=True)
    return directory.stem
