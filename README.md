# Runway Detection API Service

A containerized FastAPI service for the **"Approach Runway Detection"** model.

This repository contains the production-ready API for the model trained in my Master's research project. While that repo contains the full analysis and training code, this one focuses purely on deployment and serving.

This service accepts an image and returns a new JPEG with the detected runway bounding boxes drawn on it.

---

## 🧠 Tech Stack

- **API:** FastAPI (with Uvicorn)  
- **Containerization:** Docker  
- **ML/CV:** PyTorch, Torchvision, Pillow (PIL)  
- **Language:** Python 3.12  

---

## ⚙️ How to Run It Locally

### 1. Download the Model
This repo does **not** include the `.pth` file. Please download the model (`checkpoint_epoch_10.pth`) from the original project's Google Drive and place it in this folder.

### 2. Build the Docker Image
With Docker Desktop running, build the image:

```bash
docker build -t runway-detection .
```

### 3. Run the Container
Run the newly built image, mapping your local port `8000` to the container's port `80`:

```bash
docker run -p 8000:80 runway-detection
```

The container will start, load the model, and the service will be live.

---

## 🚀 How to Test the API

The API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### ✅ Option 1: Python Test Script (Recommended)

Use the included `test.py` script. You'll need a test image (e.g., `my_test_runway.jpg`) in the same folder.

```bash
# Install the only dependency for testing
pip install requests

# Run the test
python test.py
```

This will send `my_test_runway.jpg` to the API and save the output as `result.jpg`.

### ✅ Option 2: cURL

You can also call the endpoint directly from your terminal:

```bash
curl -X 'POST'   'http://127.0.0.1:8000/predict/image'   -H 'accept: image/jpeg'   -F 'image_file=@my_test_runway.jpg;type=image/jpeg'   --output result.jpg
```

---

