import io
import os
import logging

import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image, ImageDraw, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIG ---
BUCKET_NAME      = os.environ.get("S3_BUCKET", "runway-model-storage")
MODEL_KEY        = os.environ.get("MODEL_KEY", "checkpoint_epoch_10.pth")
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/checkpoint_epoch_10.pth")
CONF_THRESHOLD   = float(os.environ.get("CONF_THRESHOLD", "0.8"))
MODEL_INPUT_SIZE = 1024
NUM_CLASSES      = 2

BASE_DIR    = os.path.dirname(__file__)
STATIC_DIR  = os.path.join(BASE_DIR, "static")

# --- MODEL SETUP ---
def build_model() -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=NUM_CLASSES,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def load_checkpoint() -> dict:
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Model not found locally — downloading from s3://%s/%s", BUCKET_NAME, MODEL_KEY)
        try:
            boto3.client("s3").download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
            logger.info("Download complete.")
        except (BotoCoreError, ClientError) as e:
            logger.exception("Could not download model from S3")
            raise RuntimeError(f"Failed to fetch model from S3: {e}") from e
    else:
        logger.info("Model found locally (warm start). Skipping download.")

    return torch.load(LOCAL_MODEL_PATH, map_location=torch.device("cpu"))


model = build_model()
checkpoint = load_checkpoint()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
logger.info("Model loaded successfully.")

# Stateless transform — instantiate once
image_transform = transforms.ToTensor()

# --- API ---
app = FastAPI(title="Saransh's Runway Detection API", root_path="/default")


@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/predict_image")
async def predict_image(image_file: UploadFile = File(...)):
    contents = await image_file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    original_width, original_height = image.size

    model_input_image = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    input_tensor = image_transform(model_input_image).unsqueeze(0)

    try:
        with torch.no_grad():
            prediction = model(input_tensor)[0]
    except Exception as e:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from e

    draw = ImageDraw.Draw(image)
    x_scale = original_width / MODEL_INPUT_SIZE
    y_scale = original_height / MODEL_INPUT_SIZE

    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score <= CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.cpu().tolist()
        draw.rectangle(
            [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale],
            outline="red",
            width=5,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
