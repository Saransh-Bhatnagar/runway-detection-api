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

# --- 1. The Artifact (The "Body" + "Brain") ---

print("Loading model architecture...")

num_classes = 2  

model = fasterrcnn_resnet50_fpn_v2(weights=None, 
                                     weights_backbone=ResNet50_Weights.DEFAULT,  # <-- THIS IS THE FIX
                                     num_classes=num_classes)

# Get the number of "in features" for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore


model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# Now we can load our "keychain"
checkpoint = torch.load("checkpoint_epoch_10.pth", map_location=torch.device('cpu'))

# And load our 2-class "brain" into our new 2-class "skull". It will fit perfectly.
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print("Model loaded successfully.")


app = FastAPI(title="My Computer Vision API")

@app.post("/predict_image")
async def predict_image(image_file: UploadFile = File(...)):
    
    # This is the "plumbing" for an image
    # 1. Read the raw bytes of the file
    contents = await image_file.read()
    
    # b. Convert the bytes into a usable image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # --- PRE-PROCESSING ---
    
    # 1. GET ORIGINAL SIZE (for scaling boxes *back*)
    original_width, original_height = image.size

    # 2. RESIZE FOR THE MODEL
    # Create a *new* image object resized for the model
    model_input_image = image.resize((1024, 1024))
    
    image_transform = transforms.ToTensor()
    # 3. Pre-process the *resized* image
    input_tensor = image_transform(model_input_image)
    input_tensor = input_tensor.unsqueeze(0)

    # d. Make the prediction
    with torch.no_grad():
        prediction = model(input_tensor)[0]
    
    # --- POST-PROCESSING & DRAWING ---

    CONF_THRESHOLD = 0.8
    
    # Create a "drawing context" on the *ORIGINAL* image
    draw = ImageDraw.Draw(image)

    # Calculate the scaling factors
    x_scale = original_width / 1024
    y_scale = original_height / 1024

    boxes = prediction['boxes']
    scores = prediction['scores']

    for i in range(len(scores)):
        score = scores[i]
        
        if score > CONF_THRESHOLD:
            # Get the raw box coordinates from the *1024x1024* prediction
            box = boxes[i].cpu().tolist() # [x1, y1, x2, y2]
            
            # 4. SCALE THE BOXES BACK UP
            #    This is the magic.
            x1 = box[0] * x_scale
            y1 = box[1] * y_scale
            x2 = box[2] * x_scale
            y2 = box[3] * y_scale
            
            # 5. Draw the *scaled* box on the *original* image
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    # --- RETURN THE IMAGE ---
    
    # 1. Create a "virtual file" in memory
    buffer = io.BytesIO()
    
    # 2. Save the modified *original* image
    image.save(buffer, format="JPEG")
    
    # 3. "Rewind" the file to the beginning
    buffer.seek(0)
    
    # 4. Stream the virtual file back as a JPEG
    return StreamingResponse(buffer, media_type="image/jpeg")