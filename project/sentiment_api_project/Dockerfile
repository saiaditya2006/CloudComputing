# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir fastapi uvicorn transformers torch

# Expose port 8080 (Cloud Run uses this port)
EXPOSE 8080

# Command to start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
