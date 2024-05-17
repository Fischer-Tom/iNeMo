from typing import Union, Sequence, Any
from PIL import Image, ImageDraw
import random

from enum import Enum, auto

import torch
import numpy as np

import torchvision.transforms.functional as tF
from torchvision.utils import make_grid

__all__ = [
    "ImageVariant",
    "ImageType",
    "convert_to",
    "denormalize",
    "visualize_bboxes",
    "visualize_keypoints",
    "visualize_on_grid",
    "visualize_interleaved_on_grid",
]


class ImageVariant(Enum):
    """Enum for image variants."""

    TORCH = auto()
    PIL = auto()
    NUMPY = auto()


ImageType = Union[torch.Tensor, Image.Image, np.ndarray]
""" The type of an image. """


def convert_tensor_to(
    image_tensor: "torch.Tensor", output_type: "ImageVariant"
) -> "ImageType":
    if output_type == ImageVariant.TORCH:
        return image_tensor
    elif output_type == ImageVariant.PIL:
        return tF.to_pil_image(image_tensor)
    elif output_type == ImageVariant.NUMPY:
        return np.array(tF.to_pil_image(image_tensor))
    else:
        raise ValueError("Invalid output type")


def convert_pil_to(image_pil, output_type: "ImageVariant") -> "ImageType":
    if output_type == ImageVariant.TORCH:
        return tF.to_tensor(image_pil)
    elif output_type == ImageVariant.PIL:
        return image_pil
    elif output_type == ImageVariant.NUMPY:
        return np.array(image_pil)
    else:
        raise ValueError("Invalid output type")


def convert_numpy_to(image_numpy, output_type: "ImageVariant") -> "ImageType":
    if output_type == ImageVariant.TORCH:
        return tF.to_tensor(image_numpy)
    elif output_type == ImageVariant.PIL:
        return Image.fromarray(image_numpy)
    elif output_type == ImageVariant.NUMPY:
        return image_numpy
    else:
        raise ValueError("Invalid output type")


def convert_to(image: "ImageType", output_type: "ImageVariant") -> "ImageType":
    """
    Converts an image to a different type.
    Args:
        image: The input image.
        output_type: The output type of the image.
    Returns:
        ImageType: The converted image.
    """
    if isinstance(image, torch.Tensor):
        return convert_tensor_to(denormalize(image), output_type)
    elif isinstance(image, Image.Image):
        return convert_pil_to(image, output_type)
    elif isinstance(image, np.ndarray):
        return convert_numpy_to(image, output_type)
    else:
        raise ValueError("Invalid input type")


def denormalize(
    image_tensor: "torch.Tensor",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    output_type: "ImageVariant" = ImageVariant.TORCH,
) -> "ImageType":
    """
    Denormalizes an image tensor.
    Args:
        image_tensor: The input image tensor.
        mean: The mean used for normalization.
        std: The standard deviation used for normalization.
        output_type: The output type of the image.
    Returns:
        ImageType: The denormalized image.
    """
    mean = torch.as_tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)[
        ..., None, None
    ]
    std = torch.as_tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)[
        ..., None, None
    ]
    image_tensor = image_tensor * std + mean
    return convert_tensor_to(image_tensor, output_type)


def visualize_bboxes(
    image: "ImageType",
    bboxes: Sequence[Sequence[int]],
    output_type: "ImageVariant" = ImageVariant.PIL,
) -> "ImageType":
    """
    Visualizes bounding boxes on a PIL image.
    Args:
        image: The input image.
        bboxes (list): A list of bounding boxes in the format (x_min, y_min, x_max, y_max).
        output_type (ImageVariant): The output type of the image.
    Returns:
        A new image with bounding boxes visualized.
    """
    image = convert_to(image, ImageVariant.PIL)
    colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in bboxes
    ]
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        color = colors[i]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

    return convert_pil_to(result_image, output_type)


def visualize_keypoints(
    image: "ImageType",
    keypoints: Sequence[Sequence[int]],
    iskpvisible: Sequence[bool] = None,
    color=(0, 255, 0),
    output_type: "ImageVariant" = ImageVariant.PIL,
) -> "ImageType":
    """
    Visualizes keypoints on a PIL image.
    Args:
        image: The input image.
        keypoints: A list of keypoints in the format (x, y).
        color: RGB color value for the keypoints.
        output_type: The output type of the image.
    Returns:
        A new image with keypoints visualized.
    """
    image = convert_to(image, ImageVariant.PIL)
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    if iskpvisible is None:
        iskpvisible = np.ones_like(keypoints[:, 0]).astype(bool)
    for (x, y), vis in zip(keypoints, iskpvisible):
        color = (0, 255, 255)
        if vis:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)

    return convert_pil_to(result_image, output_type)


def visualize_on_grid(
    images: "torch.Tensor | Sequence[ImageType]",
    output_type: "ImageVariant" = ImageVariant.PIL,
    **make_grid_kwargs: Any,
) -> "ImageType":
    """
    Visualizes a batch of images on a grid.
    Args:
        images: A batched tensor of images or a sequence of images.
        output_type: The output type of the image.
        **make_grid_kwargs: Keyword arguments for torchvision.utils.make_grid.
    Returns:
        Grid of images.
    """
    if not isinstance(images, torch.Tensor):
        images = torch.stack(
            [convert_to(image, ImageVariant.TORCH) for image in images]
        )
    image_grid = make_grid(images, normalize=True, **make_grid_kwargs)
    return convert_tensor_to(image_grid, output_type)


def visualize_interleaved_on_grid(
    images_a: "torch.Tensor | Sequence[ImageType]",
    images_b: "torch.Tensor | Sequence[ImageType]",
    n_cols: int = 8,
    output_type=ImageVariant.PIL,
) -> "ImageType":
    """
    Visualizes two batches of images on a grid, interleaved, row by row.
    Args:
        images_a: Sequence of images to be visualized on rows with type A.
        images_b: Sequence of images to be visualized on rows with type B.
        n_cols: The number of columns in the grid (or how many images to visualize per row).
        output_type: The output type of the image.
    Notes:
        Let's say images from `images_a` are of type A and images from `images_b` are of type B.
        Then the output grid will have the following structure:
        [
            [A, A, A, ...],
            [B, B, B, ...],
            [A, A, A, ...],
            [B, B, B, ...],
            ...
        ]
    Returns:
        Grid of images, with images from images_a and images_b interleaved row by row.
    """
    if not isinstance(images_a, torch.Tensor):
        # implicitly assume images are of the same shape
        images_a = torch.stack(
            [convert_to(image, ImageVariant.TORCH) for image in images_a]
        )
    if not isinstance(images_b, torch.Tensor):
        images_b = torch.stack(
            [convert_to(image, ImageVariant.TORCH) for image in images_b]
        )
    assert images_a.shape == images_b.shape
    n_images = images_a.shape[0]
    n_rows = n_images // n_cols
    images_a = images_a.reshape(n_cols, n_rows, *images_a.shape[1:])
    images_b = images_b.reshape(n_cols, n_rows, *images_b.shape[1:])
    images = torch.concatenate([images_a, images_b])
    images = images.reshape(-1, *images.shape[2:])
    return visualize_on_grid(images, output_type=output_type)
