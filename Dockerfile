FROM python:3.11-slim

# 1. Install the AWS Lambda Web Adapter (The "Magic" Link)
# This allows standard FastAPI apps to run on Serverless Lambda
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.7.0 /lambda-adapter /opt/extensions/lambda-adapter

# 2. Set working directory
WORKDIR /app

# 3. Copy dependencies first (for better caching)
COPY requirements.txt .

# 4. Install CPU-only PyTorch and other dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code (main.py, model files, etc.)
COPY . .

# 6. Run the app on port 8080 (Lambda Adapter defaults to 8080)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]