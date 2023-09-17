from pydantic import BaseModel, validator
from typing import List, Union

class EverythingMode(BaseModel):
    mode: str = "everything"

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "everything"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value

class BoxMode(BaseModel):
    mode: str
    bboxes: List[float]

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "box"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value
    
    @validator('bboxes')
    def validate_box_prompt(cls, value):
        if len(value) != 4:
            raise ValueError("bboxes must have exactly 4 elements")
        if any(not 0 <= v <= 1 for v in value):
            raise ValueError("Each point in 'bboxes' must have values in the range [0, 1]")
        return value

class TextMode(BaseModel):
    mode: str = "text"
    text: str

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "text"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value
    
class PointsMode(BaseModel):
    mode: str = "points"
    points: List[List[float]]
    pointlabel: List[int]

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "points"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value

    @validator('points')
    def validate_point_prompt(cls, value):
        if any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in value):
            raise ValueError("Each point in 'points' must have values in the range [0, 1]")
        if any(len(point) != 2 for point in value):
            raise ValueError("Each point in points must have exactly 2 elements")
        return value
    
class InferenceRequest(BaseModel):
    data: Union[PointsMode, BoxMode, TextMode, EverythingMode]
