from pydantic import BaseModel

LABEL_MAPPING = {0: "NORMAL", 1: "CANCER"}


class ClassifyResponse(BaseModel):
    image_id: str
    predicted_label: str
    prediction_confidence: float
    true_label: str
