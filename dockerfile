# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for opencv and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model, source code, and other necessary files
COPY src/ ./src/
COPY models/ ./models/
COPY templates/ ./templates/
COPY data/ ./data/

# Create uploads directory
RUN mkdir -p uploads

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "src/app.py"]