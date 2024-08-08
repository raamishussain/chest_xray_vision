import base64
import pandas as pd
import skimage
import torch
import torchvision
import torchxrayvision as xrv

from fastapi import UploadFile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from chest_xray_vision.data_models import LABEL_MAPPING


class ImageShapeError(Exception):
    """Raised when the image is not a 2D array."""

    pass


def classify_image(image: UploadFile) -> Dict:
    """Classify a chest X-ray image using TorchXRayVision"""

    # Load the model
    model = xrv.models.get_model("densenet121-res224-all")

    # define transformer
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
    )

    # with TemporaryDirectory() as temp_dir:
    #     # read in base64 encoded image from request and save to file
    #     image_path = Path(temp_dir) / "image.png"
    #     with open(image_path, "wb") as f:
    #         f.write(base64.b64decode(image_bytes))

    img = skimage.io.imread(image.file)
    img = xrv.datasets.normalize(img, 255)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        raise ImageShapeError("Image is not a 2D array.")

    # Add color channel
    img = img[None, :, :]

    # transform the image
    img = transform(img)

    # Classify the image
    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        preds = model(img).cpu()
        output["preds"] = dict(
            zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())
        )

    return output


def get_true_label(image_id: str) -> int:
    """Get the true label for the image"""
    image_df = pd.read_csv("chest_xray_vision/chest_xray_data.csv")
    true_label = image_df[image_df["image_id"] == image_id][
        "true_label"
    ].values[0]

    return LABEL_MAPPING[true_label]
