import torch
import torchvision
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image, ImageDraw
import io
import numpy as np
from fastapi.responses import StreamingResponse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2  

model = fasterrcnn_resnet50_fpn_v2(weights=None, 
                                     weights_backbone=ResNet50_Weights.DEFAULT,  # <-- THIS IS THE FIX
                                     num_classes=num_classes)

# Get the number of "in features" for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore


model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


checkpoint = torch.load("checkpoint_epoch_10.pth", map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print("Model loaded successfully.")


app = FastAPI(title="Computer Vision API")

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