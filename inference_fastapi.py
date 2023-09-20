import logging
from io import BytesIO
from typing import Dict, Union, List
import cv2
import fastapi
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse
from hydra import compose, initialize
from omegaconf import DictConfig
from schemas import EverythingMode, BoxMode, TextMode, PointsMode
from PIL import Image

from pydantic import Json

from model import InferenceModel

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

def config() -> DictConfig:
    """
    Initialize and return hydra configuration

    Returns:
        cfg (DictConfig): hydra configuration
    """
    with initialize(config_path="./config"):
        cfg = compose(config_name="config.yaml")
    return cfg

@app.on_event("startup")
def startup_event(config: DictConfig = config()) -> None:
    """
    Startup event when fastAPI server starts up
    -------------------------------------------
    Initialization of InferenceModel class
    to reuse the class and reduce the overhead during inference

    Args:
        cfg (DictConfig): hydra configuration
    """
    logging.info("Starting server...")
    app.model = InferenceModel(config)

    logging.info("Start server complete")


@app.get("/ping", status_code=fastapi.status.HTTP_200_OK)
async def ping(
) -> Dict[str, str]:
    """
    Returns the health status of the server.

    Returns:
        Dict[str, str]: JSON payload indicating server health status.
    """
    msg = {"message": "pong"}
    return JSONResponse(content = msg)


@app.post("/infer")
async def infer(image: UploadFile = File(...),
                 mode: str = Form(...),
                 text_prompt: Union[str,None] = Form(None),
                 box_prompt: Union[Json[List[List[float]]], None] = Form(None),
                 point_prompt: Union[Json[List[List[float]]], None] = Form(None),
                 point_label: Union[Json[List[int]], None] = Form(None)
                 ) -> StreamingResponse: 
    """
    Endpoint to perform inference based on the provided mode and input data.

    Args:
        image (UploadFile, required): The input image for the inference.
        mode (str, required): The inference mode. Must be one of ["everything", "text", "box", "points"].
        text_prompt (str, optional): The text prompt for inference (if applicable).
        box_prompt (Json[List[List[float]]], optional): The bounding box prompts in deserialized JSON format (if applicable).
        point_prompt (Json[List[List[float]]], optional): The point prompts in deserialized JSON format (if applicable).
        point_label (Json[List[int]], optional): The point labels in deserialized JSON format (if applicable).

    Returns:
        StreamingResponse: A streaming response containing the inference result image in PNG format.
    
    Raises:
        HTTPException: If there is an error in the inference process or if an invalid mode is provided.
    """
                 
    input_data = {key: value for key, value in locals().items() if key != 'image' and value is not None}

    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data))

    modes = {"everything": EverythingMode,
             "text": TextMode,
             "box": BoxMode,
             "points": PointsMode}
    
    if mode in modes:
        model_class = modes[mode]

        try:
            validated_data = model_class.parse_obj(input_data)
            cv2img = app.model.predict(pil_image, validated_data.model_dump())
            _, im_png = cv2.imencode(".png", cv2img)
            im_png_stream = BytesIO(im_png)
            im_png_stream.seek(0)
        
            return StreamingResponse(im_png_stream, media_type="image/png")            
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
    else:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")


if __name__ == "__main__":

    uvicorn.run(
        "inference_fastapi:app",
        host="127.0.0.1",
        reload=False,
        port=4000,
        log_level=None,
    )
