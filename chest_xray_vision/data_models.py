from pydantic import BaseModel

LABEL_MAPPING = {0: "NORMAL", 1: "CANCER"}


class ClassifyRequest(BaseModel):
    image_id: str
    image_bytes: bytes


class ClassifyResponse(BaseModel):
    image_id: str
    predicted_label: str
    prediction_confidence: float
    true_label: str
