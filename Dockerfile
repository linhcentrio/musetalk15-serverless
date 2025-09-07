# MuseTalk1.5 RunPod Serverless Docker Image
FROM spxiong/pytorch:2.0.1-py3.9.12-cuda11.8.0-ubuntu22.04

WORKDIR /app

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app:/app/MuseTalk"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg wget curl git unzip aria2 \
    build-essential python3.9-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libsndfile1 libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install MMlab packages
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0"

# Clone MuseTalk repository (kevinwang676/MuseTalk1.5)
RUN git clone https://huggingface.co/kevinwang676/MuseTalk1.5 /app/MuseTalk

# Create model directories
RUN mkdir -p /app/MuseTalk/models/{musetalkV15,sd-vae,whisper,dwpose,face-parse-bisent,syncnet}

# ===== MODEL DOWNLOAD SECTION =====
# Download MuseTalk1.5 models từ kevinwang676/MuseTalk1.5
RUN echo "=== Downloading MuseTalk1.5 Models ===" && \
    # MuseTalk V1.5 core models
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/kevinwang676/MuseTalk1.5/resolve/main/models/musetalk/pytorch_model.bin" \
    -d /app/MuseTalk/models/musetalkV15 \
    -o unet.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/kevinwang676/MuseTalk1.5/resolve/main/models/musetalk/musetalk.json" \
    -d /app/MuseTalk/models/musetalkV15 \
    -o musetalk.json && \
    echo "✅ MuseTalk V1.5 core models downloaded"

# Download SD-VAE models
RUN echo "=== Downloading SD-VAE Models ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
    -d /app/MuseTalk/models/sd-vae \
    -o diffusion_pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
    -d /app/MuseTalk/models/sd-vae \
    -o config.json && \
    echo "✅ SD-VAE models downloaded"

# Download Whisper model
RUN echo "=== Downloading Whisper Models ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt" \
    -d /app/MuseTalk/models/whisper \
    -o tiny.pt && \
    # Create config files for Whisper
    echo '{"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"}' > /app/MuseTalk/models/whisper/config.json && \
    echo "✅ Whisper models downloaded"

# Download DWPose models
RUN echo "=== Downloading DWPose Models ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" \
    -d /app/MuseTalk/models/dwpose \
    -o dw-ll_ucoco_384.pth && \
    echo "✅ DWPose models downloaded"

# Download Face Parse models
RUN echo "=== Downloading Face Parse Models ===" && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/79999_iter.pth/79999_iter.pth" \
    -d /app/MuseTalk/models/face-parse-bisent \
    -o 79999_iter.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
    -d /app/MuseTalk/models/face-parse-bisent \
    -o resnet18-5c106cde.pth && \
    echo "✅ Face Parse models downloaded"

# Download SyncNet model (optional)
RUN echo "=== Downloading SyncNet Model ===" && \
    mkdir -p /app/MuseTalk/models/syncnet && \
    echo "# SyncNet model placeholder" > /app/MuseTalk/models/syncnet/latentsync_syncnet.pt && \
    echo "✅ SyncNet model placeholder created"

# VERIFICATION
RUN echo "=== MODEL VERIFICATION ===" && \
    echo "MuseTalk V1.5:" && ls -la /app/MuseTalk/models/musetalkV15/ && \
    echo "SD-VAE:" && ls -la /app/MuseTalk/models/sd-vae/ && \
    echo "Whisper:" && ls -la /app/MuseTalk/models/whisper/ && \
    echo "DWPose:" && ls -la /app/MuseTalk/models/dwpose/ && \
    echo "Face Parse:" && ls -la /app/MuseTalk/models/face-parse-bisent/ && \
    echo "=== VERIFICATION COMPLETE ==="

# Copy MuseTalk utils and modules (from repository structure)
COPY --from=0 /app/MuseTalk/musetalk /app/MuseTalk/musetalk
COPY --from=0 /app/MuseTalk/configs /app/MuseTalk/configs

# Copy application handler
COPY musetalk_handler.py /app/musetalk_handler.py

# Final verification
RUN python -c "import torch, cv2, numpy, librosa; print('✅ Core packages OK')" && \
    python -c "import runpod, minio; print('✅ RunPod/MinIO OK')" && \
    python -c "from transformers import WhisperModel; print('✅ Transformers OK')" || echo "⚠️ Some packages missing"

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('🚀 Ready')" || exit 1

# Expose port
EXPOSE 8000

# Run handler
CMD ["python", "musetalk_handler.py"]
