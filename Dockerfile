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

# Install system dependencies + GIT (CRITICAL FIX)
RUN apt-get update && apt-get install -y \
    # Core system tools
    wget \
    curl \
    git \
    unzip \
    aria2 \
    # Build essentials (CRITICAL for wheel building)
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    python3-pip \
    # Media processing
    ffmpeg \
    # Graphics libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Audio libraries
    libsndfile1 \
    libasound2-dev \
    # Additional build dependencies
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools (CRITICAL FIX)
RUN python -m pip install --upgrade pip setuptools wheel

# Copy and install requirements with version locks
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PRE-INSTALL critical build dependencies to avoid chumpy issues
RUN pip install --no-cache-dir \
    cython \
    setuptools>=60.0.0 \
    wheel>=0.38.0 \
    pip>=21.0 \
    numpy==1.24.3

# Install MMlab packages with EXACT VERSIONS from environment.yml
RUN pip install --no-cache-dir openmim==0.3.9

# Install MMengine first (base dependency)
RUN pip install --no-cache-dir mmengine==0.10.7

# Install MMCV with specific CUDA index  
RUN pip install --no-cache-dir mmcv==2.0.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html

# Install MMDetection (CORRECTED VERSION from environment.yml)
RUN pip install --no-cache-dir mmdet==3.3.0

# Install MMPose (CORRECTED VERSION from environment.yml) 
RUN pip install --no-cache-dir mmpose==1.3.2

# Alternative: Install chumpy separately if MMPose still fails
RUN pip install --no-cache-dir chumpy==0.70 || \
    (git clone https://github.com/mattloper/chumpy.git /tmp/chumpy && \
     cd /tmp/chumpy && \
     python setup.py install && \
     rm -rf /tmp/chumpy)

# Clone MuseTalk repository
RUN git clone https://huggingface.co/kevinwang676/MuseTalk1.5 /app/MuseTalk

# Create model directories with proper structure
RUN mkdir -p /app/MuseTalk/models/{musetalkV15,sd-vae,whisper,dwpose,face-parse-bisent,syncnet}

# ===== OPTIMIZED MODEL DOWNLOAD SECTION =====
# Download MuseTalk V1.5 core models
RUN echo "=== Downloading MuseTalk1.5 Models ===" && \
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

# Create SyncNet model placeholder
RUN mkdir -p /app/MuseTalk/models/syncnet && \
    echo "# SyncNet model placeholder" > /app/MuseTalk/models/syncnet/latentsync_syncnet.pt && \
    echo "✅ SyncNet model placeholder created"

# Copy application handler
COPY musetalk_handler.py /app/musetalk_handler.py

# COMPREHENSIVE VERIFICATION
RUN echo "=== COMPREHENSIVE VERIFICATION ===" && \
    python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" && \
    python -c "import numpy as np; print(f'✅ NumPy: {np.__version__}')" && \
    python -c "import runpod; print('✅ RunPod OK')" && \
    python -c "from transformers import WhisperModel; print('✅ Transformers OK')" && \
    python -c "import mmengine; print(f'✅ MMEngine: {mmengine.__version__}')" && \
    python -c "import mmcv; print(f'✅ MMCV: {mmcv.__version__}')" && \
    python -c "import mmdet; print(f'✅ MMDet: {mmdet.__version__}')" && \
    python -c "import mmpose; print(f'✅ MMPose: {mmpose.__version__}')" && \
    echo "=== ALL VERIFICATIONS PASSED ==="

# Final model verification
RUN echo "=== MODEL FILE VERIFICATION ===" && \
    echo "MuseTalk V1.5:" && ls -la /app/MuseTalk/models/musetalkV15/ && \
    echo "SD-VAE:" && ls -la /app/MuseTalk/models/sd-vae/ && \
    echo "Whisper:" && ls -la /app/MuseTalk/models/whisper/ && \
    echo "DWPose:" && ls -la /app/MuseTalk/models/dwpose/ && \
    echo "Face Parse:" && ls -la /app/MuseTalk/models/face-parse-bisent/ && \
    echo "=== MODEL VERIFICATION COMPLETE ==="

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); import mmengine, mmcv, mmdet, mmpose; print('🚀 All systems ready')" || exit 1

# Expose port
EXPOSE 8000

# Run handler with proper error handling
CMD ["python", "-u", "musetalk_handler.py"]
