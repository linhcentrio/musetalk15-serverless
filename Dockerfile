# MuseTalk1.5 RunPod Serverless Docker Image - OPTIMIZED VERSION
FROM spxiong/pytorch:2.0.1-py3.9.12-cuda11.8.0-ubuntu22.04

WORKDIR /app

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/app:/app/MuseTalk" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in single layer for efficiency
RUN apt-get update && apt-get install -y \
    wget curl git unzip aria2 \
    build-essential cmake pkg-config python3-dev python3-pip \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libsndfile1 libasound2-dev \
    libffi-dev libssl-dev libxml2-dev libxslt1-dev libjpeg-dev libpng-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Upgrade pip and install critical Python packages in one step
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir cython setuptools>=60.0.0 wheel>=0.38.0 pip>=21.0 numpy==1.24.3

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MMlab ecosystem with optimized order
RUN pip install --no-cache-dir openmim==0.3.9 mmengine==0.10.7 \
    && pip install --no-cache-dir mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html \
    && pip install --no-cache-dir mmdet==3.3.0 mmpose==1.3.2

# Install chumpy with fallback to source build
RUN pip install --no-cache-dir chumpy==0.70 || \
    (git clone https://github.com/mattloper/chumpy.git /tmp/chumpy && \
     cd /tmp/chumpy && python setup.py install && rm -rf /tmp/chumpy)

# Clone MuseTalk repository (includes all models)
RUN git clone https://huggingface.co/kevinwang676/MuseTalk1.5 /app/MuseTalk

# Create model directories and verify structure
RUN mkdir -p /app/MuseTalk/models/{musetalkV15,musetalk,sd-vae-ft-mse,whisper,dwpose,face-parse-bisent,syncnet} \
    && echo "=== Repository Verification ===" \
    && du -sh /app/MuseTalk \
    && find /app/MuseTalk/models -type f \( -name "*.pth" -o -name "*.bin" -o -name "*.pt" -o -name "*.json" \) 2>/dev/null | head -20

# Fallback download only if absolutely necessary
RUN if [ ! -f "/app/MuseTalk/models/musetalkV15/unet.pth" ] && [ ! -f "/app/MuseTalk/models/musetalk/pytorch_model.bin" ]; then \
    echo "Downloading missing MuseTalk models..." && \
    aria2c --console-log-level=error -c -x 8 -s 8 \
    "https://huggingface.co/kevinwang676/MuseTalk1.5/resolve/main/models/musetalkV15/unet.pth" \
    -d /app/MuseTalk/models/musetalkV15 -o unet.pth; \
else \
    echo "✅ MuseTalk models already present"; \
fi

# Copy application handler
COPY musetalk_handler.py /app/musetalk_handler.py

# Comprehensive verification in single layer
RUN echo "=== PACKAGE VERIFICATION ===" \
    && python -c "import torch; print(f'✅ PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')" \
    && python -c "import cv2, numpy as np; print(f'✅ OpenCV: {cv2.__version__} | NumPy: {np.__version__}')" \
    && python -c "import runpod; print('✅ RunPod OK')" \
    && python -c "from transformers import WhisperModel; print('✅ Transformers OK')" \
    && python -c "import mmengine, mmcv, mmdet, mmpose; print('✅ MMlab Stack OK')" \
    && echo "=== ALL PACKAGES VERIFIED ==="

# Final model verification
RUN echo "=== MODEL FILES VERIFICATION ===" \
    && for dir in musetalkV15 sd-vae whisper dwpose face-parse-bisent; do \
        echo "$dir:" && ls -la /app/MuseTalk/models/$dir/ 2>/dev/null || echo "  Directory not found"; \
    done \
    && echo "=== VERIFICATION COMPLETE ==="

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); import mmengine, mmcv, mmdet, mmpose; print('🚀 Ready')" || exit 1

EXPOSE 8000

CMD ["python", "-u", "musetalk_handler.py"]
