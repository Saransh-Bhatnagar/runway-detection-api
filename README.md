# ✈️ Autonomous Runway Detection System (Serverless AI)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![AWS](https://img.shields.io/badge/AWS-Lambda-orange) ![Docker](https://img.shields.io/badge/Docker-Container-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red)

## 🔍 Overview
A production-grade computer vision pipeline designed to detect airport runways in aerial imagery. This project bridges the "Sim2Real" gap by training on 14,000+ synthetic images and deploying the inference engine as a serverless microservice.

**🔴 Live API Docs:** [[https://YOUR_API_ID.execute-api.eu-west-1.amazonaws.com/default/docs](https://onq18p4b8l.execute-api.eu-west-1.amazonaws.com/default/docs#/default/predict_image_predict_image_post)]([https://YOUR_API_ID.execute-api.eu-west-1.amazonaws.com/default/docs](https://onq18p4b8l.execute-api.eu-west-1.amazonaws.com/default/docs#/default/predict_image_predict_image_post))
*(Note: Please allow 5-10 seconds for Cold Start latency on first request, try refreshing if failed.)*

## 🏗 System Architecture
**Input** (Image) -> **API Gateway** -> **AWS Lambda** (Faster R-CNN) -> **Output** (Bounding Boxes)

## 🚀 Key Features
* **Serverless Deployment:** Deployed on AWS Lambda using Docker, optimizing for zero idle costs.
* **Optimized Inference:** Reduced container size from 12GB to 2.4GB using CPU-only PyTorch builds.
* **Custom Model:** Fine-tuned Faster R-CNN (ResNet-50) with geometric post-processing for high-recall safety.

## 🛠 Tech Stack
* **Framework:** PyTorch, TorchVision, FastAPI
* **Infrastructure:** AWS Lambda, ECR, API Gateway
* **Tools:** Docker, Git

## 💻 Usage
Send a POST request to `/predict_image` with an aerial image file.
