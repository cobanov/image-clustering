import glob
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt

def read_images_from_directory(image_directory: str) -> list:
    """
    > It takes a directory as input and returns a list of all the images in that directory
    :param image_directory: The directory where the images are stored
    :type image_directory: str
    :return: A list of images
    """

    list_of_images = list()
    for ext in ("*.gif", "*.png", "*.jpg", '*.jpeg'):
        list_of_images.extend(
            glob.glob(os.path.join(image_directory, ext))
        )  # ? Remove sorted if it is there
    print(f"Images found: {len(list_of_images)}")

    return list_of_images


def read_with_pil(list_of_images: list, resize=False) -> list:
    """
    > Reads a list of images and returns a list of PIL images
    :param list_of_images: list of image paths
    :type list_of_images: list
    :param resize: If True, resize the image to 512x512, defaults to False (optional)
    :return: A list of PIL images
    """

    pil_images = list()
    for img_path in list_of_images:
        img = Image.open(img_path).convert("RGB")
        if resize:  #! No hard code
            img.thumbnail((512, 512))
        pil_images.append(img)

    return pil_images


def save_embeddings(tensor, path="embeddings.pt"):
    torch.save(tensor, path)


def load_from_embeddings(embedding_path):
    """A brief description."""

    vec = torch.load(embedding_path)
    if vec.dim() > 2:
        vec = torch.squeeze(vec)
    return vec


def create_image_grid(label_images, label_number):

    for i in range(len(label_images)):
        if i >= 9:
            break
        image = label_images[i]
        plt.subplot(3, 3, i+1)
        plt.imshow(image, cmap='gray',interpolation='none')
        plt.title(f'Class: {label_number}')
        plt.axis('off')
    plt.show()
