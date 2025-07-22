FROM diagonalge/kohya_latest:latest

WORKDIR /workspace

# Install additional Python packages
RUN pip install toml accelerate transformers diffusers

# Copy project files
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/
COPY requirements.txt /workspace/requirements.txt

# Install requirements if exists
RUN pip install -r /workspace/requirements.txt || echo "No requirements.txt found"

# Create required directories
RUN mkdir -p /app/checkpoints \
    /dataset/configs \
    /dataset/outputs \
    /dataset/images \
    /workspace/data \
    /workspace/configs \
    /cache/models \
    /cache/datasets

# Set environment variables sesuai dokumentasi
ENV CONFIG_DIR="/workspace/configs"
ENV OUTPUT_DIR="/workspace/outputs"  
ENV DATASET_DIR="/workspace/data"
ENV CACHE_PATH="/cache"

# Set entrypoint
ENTRYPOINT ["python3", "/workspace/scripts/image_trainer.py"]
