from chest_xray_vision import __version__
from chest_xray_vision.data_models import (
    ClassifyRequest,
    ClassifyResponse,
    LABEL_MAPPING,
)
from chest_xray_vision.utils import (
    classify_image,
    get_true_label,
    ImageShapeError,
)
from fastapi import FastAPI, HTTPException
from mangum import Mangum


app = FastAPI(
    title="Chest X-ray Vision",
    description="An API for classifying chest X-ray images with lung cancer.",
    version=__version__,
)


@app.post("/classify")
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a chest X-ray image as normal or cancerous."""

    try:
        results = classify_image(request.image_bytes)
    except ImageShapeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if results["preds"]["Nodule"] > 0.505:
        predicted_label = LABEL_MAPPING[1]
    else:
        predicted_label = LABEL_MAPPING[0]

    true_label = get_true_label(request.image_id)

    response = ClassifyResponse(
        image_id=request.image_id,
        predicted_label=predicted_label,
        prediction_confidence=results["preds"]["Nodule"],
        true_label=true_label,
    )

    return response

handler = Mangum(app)