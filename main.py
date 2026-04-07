import torch
import torchvision
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image, ImageDraw
import io
import numpy as np
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import boto3
import os

# --- S3 CONFIGURATION ---
s3 = boto3.client('s3')
BUCKET_NAME = 'runway-model-storage' 
MODEL_KEY = 'checkpoint_epoch_10.pth'
LOCAL_MODEL_PATH = '/tmp/checkpoint_epoch_10.pth' 

# --- MODEL SETUP ---
num_classes = 2  

model = fasterrcnn_resnet50_fpn_v2(weights=None, 
                                   weights_backbone=ResNet50_Weights.DEFAULT,
                                   num_classes=num_classes)

# Get the number of "in features" for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features 

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

def load_model_from_s3():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Model not found locally. Downloading from S3...")
        try:
            s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not download model from S3: {e}")
            raise e
    else:
        print("Model found locally (Warm Start). Skipping download.")

    return torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))

# Execute the loader
checkpoint = load_model_from_s3()

# Load weights into the model architecture
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")

# --- API SETUP ---
app = FastAPI(title="Saransh's Runway Detection API", root_path="/default")

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.post("/predict_image")
async def predict_image(image_file: UploadFile = File(...)):
    
    contents = await image_file.read()
    
    image = Image.open(io.BytesIO(contents)).convert("RGB")
        
    original_width, original_height = image.size

    model_input_image = image.resize((1024, 1024))
    
    image_transform = transforms.ToTensor()

    input_tensor = image_transform(model_input_image)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)[0]
    
    CONF_THRESHOLD = 0.8
    
    draw = ImageDraw.Draw(image)

    # Calculate the scaling factors
    x_scale = original_width / 1024
    y_scale = original_height / 1024

    boxes = prediction['boxes']
    scores = prediction['scores']

    for i in range(len(scores)):
        score = scores[i]
        
        if score > CONF_THRESHOLD:
            box = boxes[i].cpu().tolist() # [x1, y1, x2, y2]
            
            x1 = box[0] * x_scale
            y1 = box[1] * y_scale
            x2 = box[2] * x_scale
            y2 = box[3] * y_scale
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    buffer = io.BytesIO()
    
    image.save(buffer, format="JPEG")
    
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/jpeg")