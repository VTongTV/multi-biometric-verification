
from typing import Dict, Optional, Any
from pydantic import BaseModel

class BiometricResponse(BaseModel):
    decision: str
    score: float
    details: Dict[str, float]

class IdentificationResponse(BaseModel):
    match_id: str
    score: float
    details: Dict[str, float]

class ErrorResponse(BaseModel):
    error: str
