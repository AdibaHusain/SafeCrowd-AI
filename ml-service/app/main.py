from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from app.models.csrnet_wrapper import (
    get_csrnet_model,
)
from app.schemas import InferenceResult
from app.utils.preprocessing import (
    decode_image,
)


app = FastAPI(
    title="SafeCrowdAI ML Service",
    version="1.0.0",
)


# ml-service/
SERVICE_ROOT = Path(
    __file__
).resolve().parent.parent

CHECKPOINT_PATH = (
    SERVICE_ROOT
    / "checkpoints"
    / "csrnet_best.pth"
)


@app.on_event("startup")
def load_models():
    """
    Load CSRNet when FastAPI starts.

    This ensures that a missing/corrupt checkpoint
    is detected during startup instead of when the
    first user sends a request.
    """

    get_csrnet_model(
        str(CHECKPOINT_PATH)
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SafeCrowdAI ML Service",
    }


@app.post(
    "/infer/density",
    response_model=InferenceResult,
)
async def infer_density(
    frame: UploadFile = File(...),
):
    """
    Run CSRNet crowd-density inference
    on an uploaded image.
    """

    try:
        image_bytes = await frame.read()

        if not image_bytes:
            raise ValueError(
                "Uploaded file is empty."
            )

        image = decode_image(
            image_bytes
        )

        model = get_csrnet_model(
            str(CHECKPOINT_PATH)
        )

        return model.predict(
            image
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"CSRNet inference failed: "
                f"{exc}"
            ),
        ) from exc