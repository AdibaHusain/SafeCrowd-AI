import time
from pathlib import Path

import torch

from app.models.architectures.csrnet_arch import CSRNet
from app.schemas import InferenceResult
from app.utils.preprocessing import (
    density_map_to_base64,
    to_csrnet_tensor,
)


class CSRNetWrapper:
    """
    Loads the trained CSRNet checkpoint once and
    exposes a simple prediction interface.
    """

    def __init__(self, checkpoint_path: str):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"CSRNet checkpoint not found: "
                f"{checkpoint_path}"
            )

        # No fresh ImageNet weights are needed.
        # We are loading the trained checkpoint.
        self.model = CSRNet(
            load_pretrained_vgg=False
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        # Our training code saves a complete
        # checkpoint dictionary.
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            # This also allows the wrapper to work
            # with a raw state_dict if we ever have one.
            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"CSRNet loaded successfully "
            f"on {self.device}"
        )

        print(
            f"Checkpoint: {checkpoint_path}"
        )

    @torch.no_grad()
    def predict(
        self,
        image,
    ) -> InferenceResult:
        """
        Run CSRNet inference on one RGB image.
        """

        start_time = time.perf_counter()

        tensor = to_csrnet_tensor(
            image
        ).to(self.device)

        density_map = self.model(
            tensor
        )

        density_map_np = (
            density_map
            .squeeze()
            .cpu()
            .numpy()
        )

        # CSRNet estimates crowd count by
        # summing the predicted density map.
        estimated_count = float(
            density_map_np.sum()
        )

        # Density maps should represent
        # non-negative crowd density.
        estimated_count = max(
            0.0,
            estimated_count,
        )

        inference_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        return InferenceResult(
            model_name="csrnet",
            estimated_count=estimated_count,
            density_map_available=True,
            density_map_base64=(
                density_map_to_base64(
                    density_map_np
                )
            ),
            boxes=None,
            inference_time_ms=(
                inference_time_ms
            ),
        )


# The model is loaded once and reused.
# It should NOT be loaded for every request.
_csrnet_model = None


def get_csrnet_model(
    checkpoint_path: str,
) -> CSRNetWrapper:

    global _csrnet_model

    if _csrnet_model is None:
        _csrnet_model = CSRNetWrapper(
            checkpoint_path
        )

    return _csrnet_model