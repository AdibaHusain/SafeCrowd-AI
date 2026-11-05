import base64
import io

import cv2
import numpy as np
import torch
from torchvision import transforms


NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes into an RGB NumPy array.

    Output shape:
        H x W x 3
    """
    array = np.frombuffer(image_bytes, dtype=np.uint8)

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Unable to decode the uploaded image."
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def pad_to_multiple(
    image: np.ndarray,
    multiple: int = 8,
) -> np.ndarray:
    """
    Pad an arbitrary image so that height and width
    are divisible by 8.

    CSRNet contains three stride-2 pooling layers,
    so this keeps the spatial dimensions compatible
    with the network.
    """
    height, width = image.shape[:2]

    pad_height = (
        multiple - height % multiple
    ) % multiple

    pad_width = (
        multiple - width % multiple
    ) % multiple

    if pad_height == 0 and pad_width == 0:
        return image

    return np.pad(
        image,
        (
            (0, pad_height),
            (0, pad_width),
            (0, 0),
        ),
        mode="constant",
    )


def to_csrnet_tensor(
    image: np.ndarray,
) -> torch.Tensor:
    """
    Convert an RGB NumPy image into the tensor
    expected by CSRNet.

    Pipeline:
        RGB image
        -> pad to multiple of 8
        -> CHW
        -> float [0, 1]
        -> ImageNet normalization
        -> batch dimension
    """

    image = pad_to_multiple(
        image,
        multiple=8,
    )

    tensor = torch.from_numpy(
        image
    ).permute(2, 0, 1).float() / 255.0

    tensor = NORM(tensor)

    return tensor.unsqueeze(0)


def density_map_to_base64(
    density_map: np.ndarray,
) -> str:
    """
    Convert a density map into a PNG encoded as base64.

    This is for returning a visualization through
    the API. The crowd count itself is calculated
    directly from the raw density map.
    """

    density_map = np.maximum(
        density_map,
        0,
    )

    max_value = float(
        density_map.max()
    )

    if max_value > 0:
        normalized = (
            density_map / max_value * 255
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(
            density_map,
            dtype=np.uint8,
        )

    success, buffer = cv2.imencode(
        ".png",
        normalized,
    )

    if not success:
        raise ValueError(
            "Failed to encode density map."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")