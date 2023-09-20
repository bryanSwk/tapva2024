from pydantic import BaseModel, validator, model_validator
from typing import List

class ModeType(BaseModel):

    @model_validator(mode='before')
    @classmethod
    def validate_extra_fields(cls, value):
        allowed_fields = set(cls.__annotations__.keys())
        for key in value.keys():
            if key not in allowed_fields:
                raise ValueError("Extra fields not allowed")
        return value
    
class EverythingMode(ModeType):
    mode: str

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "everything"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value

class BoxMode(ModeType):
    mode: str
    box_prompt: List[List[float]]

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "box"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value
    
    @validator('box_prompt')
    def validate_box_prompt(cls, value):
        for bbox in value:
            if len(bbox) != 4:
                raise ValueError("box_prompt must have exactly 4 elements")
            for v in bbox:
                if not 0 <= v <= 1:
                    raise ValueError("Each point in 'box_prompt' must have values in the range [0, 1]")
        return value

class TextMode(ModeType):
    mode: str
    text_prompt: str

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "text"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value
    
class PointsMode(ModeType):
    mode: str
    point_prompt: List[List[float]]
    point_label: List[int]

    @validator("mode")
    def validate_mode(cls, value):
        expected_mode = "points"
        if value != expected_mode:
            raise ValueError(f"Expected '{expected_mode}' as mode.")
        return value

    @validator('point_prompt')
    def validate_point_prompt(cls, value):
        if any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in value):
            raise ValueError("Each point in 'point_prompt' must have values in the range [0, 1]")
        if any(len(point) != 2 for point in value):
            raise ValueError("Each point in point_prompt must have exactly 2 elements")
        return value

