import logging
from io import BytesIO
from typing import Dict
import fastapi
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.responses import JSONResponse
from hydra import compose, initialize
from omegaconf import DictConfig
from schemas import InferenceRequest, EverythingMode
from pydantic import Json
from PIL import Image

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
def startup_event(config=config()) -> None:
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

@app.post("/infer/")
async def infer(image: UploadFile = File(...), 
                request_data: Json = EverythingMode()
                ):
    """
    Endpoint for image inference.

    Args:
        image (UploadFile): The uploaded image file.
        request_data (Json): The JSON data containing inference request details.

    Returns:
        dict: A dictionary containing the inference result and mode information.
    """
    
    try:
        request = InferenceRequest(data=request_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data))

    allowed_modes = ["everything", "box", "text", "points"]
    print(request.model_dump())
    if request.data.mode not in allowed_modes:
        return {"error": "Invalid mode. Mode must be one of 'everything', 'box', 'text', 'points'"}
    else:
        app.model.predict(pil_image, request.model_dump())
    return {"message": "Image uploaded, processed, and mode selected successfully", "mode": request.data.mode}


if __name__ == "__main__":

    uvicorn.run(
        "inference_fastapi:app",
        host="127.0.0.1",
        reload=False,  # change to false after development
        port=4000,
        log_level=None,
    )
