import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image, ImageDraw
import io
from fastapi.responses import StreamingResponse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 2  
model = fasterrcnn_resnet50_fpn_v2(weights=None, 
                                   weights_backbone=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                   num_classes=num_classes)

in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


checkpoint = torch.load("checkpoint_epoch_10.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")

app = FastAPI(title="Runway Detection API", root_path="/default")

@app.get("/")
def home():
    return {"message": "Runway Detection API is Live. Go to /docs for the UI."}

@app.post("/predict_image")
async def predict_image(image_file: UploadFile = File(...)):
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_width, original_height = image.size

    # Preprocessing
    model_input_image = image.resize((1024, 1024))
    image_transform = transforms.ToTensor()
    input_tensor = image_transform(model_input_image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        prediction = model(input_tensor)[0]

    # Post-processing
    CONF_THRESHOLD = 0.8
    draw = ImageDraw.Draw(image)
    x_scale = original_width / 1024
    y_scale = original_height / 1024

    boxes = prediction['boxes']
    scores = prediction['scores']

    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD:
            box = boxes[i].cpu().tolist()
            x1, y1, x2, y2 = box
            draw.rectangle(
                [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale], 
                outline="red", 
                width=5
            )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
