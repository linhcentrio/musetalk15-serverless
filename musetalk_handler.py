#!/usr/bin/env python3
"""
RunPod Serverless Handler cho MuseTalk1.5 - Real-time Talking Head Generation
Tích hợp MinIO storage và tối ưu hóa performance
Tham khảo từ kevinwang676/MuseTalk1.5
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import torch
import cv2
import numpy as np
import sys
import gc
import json
import traceback
import logging
import subprocess
import imageio
import glob
import pickle
import copy
import re
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
from PIL import Image
from argparse import Namespace
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import WhisperModel
from moviepy.editor import VideoFileClip, AudioFileClip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add MuseTalk paths
sys.path.insert(0, '/app/MuseTalk')

# Import MuseTalk components
try:
    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range
    logger.info("✅ MuseTalk modules imported successfully")
    MUSETALK_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ MuseTalk import error: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    MUSETALK_AVAILABLE = False

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    logger.info("✅ MinIO client initialized")
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# Model configurations (tham khảo từ kevinwang676/MuseTalk1.5)
MODEL_CONFIGS = {
    "models_dir": "/app/MuseTalk/models",
    "unet_model": "/app/MuseTalk/models/musetalkV15/unet.pth",
    "unet_config": "/app/MuseTalk/models/musetalkV15/musetalk.json", 
    "vae_model": "/app/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin",
    "vae_config": "/app/MuseTalk/models/sd-vae/config.json",
    "whisper_model": "/app/MuseTalk/models/whisper",
    "dwpose_model": "/app/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth",
    "face_parse_model": "/app/MuseTalk/models/face-parse-bisent/79999_iter.pth",
    "face_parse_resnet": "/app/MuseTalk/models/face-parse-bisent/resnet18-5c106cde.pth",
    "syncnet_model": "/app/MuseTalk/models/syncnet/latentsync_syncnet.pt"
}

# Global model cache
model_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float32

def verify_models() -> tuple[bool, list]:
    """Verify all required MuseTalk models exist"""
    logger.info("🔍 Verifying MuseTalk models...")
    missing_models = []
    existing_models = []
    total_size = 0
    
    required_files = [
        MODEL_CONFIGS["unet_model"],
        MODEL_CONFIGS["unet_config"],
        MODEL_CONFIGS["vae_model"],
        MODEL_CONFIGS["vae_config"],
        f"{MODEL_CONFIGS['whisper_model']}/config.json",
        MODEL_CONFIGS["dwpose_model"],
        MODEL_CONFIGS["face_parse_model"],
        MODEL_CONFIGS["face_parse_resnet"]
    ]
    
    for path in required_files:
        if os.path.exists(path):
            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                existing_models.append(f"{os.path.basename(path)}: {file_size_mb:.1f}MB")
                total_size += file_size_mb
                logger.info(f"✅ {os.path.basename(path)}: {file_size_mb:.1f}MB")
            except Exception as e:
                logger.error(f"❌ Error checking {path}: {e}")
                missing_models.append(f"{path} (error reading)")
        else:
            missing_models.append(path)
            logger.error(f"❌ Missing: {path}")
    
    if missing_models:
        logger.error(f"❌ Missing {len(missing_models)} models")
        return False, missing_models
    else:
        logger.info(f"✅ All models verified! Total: {total_size:.1f}MB")
        return True, []

def clear_memory():
    """Enhanced memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.synchronize()
        except:
            pass

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL with progress tracking"""
    try:
        logger.info(f"📥 Downloading: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0 and downloaded % (1024 * 1024 * 10) == 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"📥 Progress: {progress:.1f}% ({downloaded/1024/1024:.1f}MB)")
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"✅ Downloaded: {file_size:.1f}MB")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def process_video_frames(video_path: str, output_dir: str) -> tuple[list, float]:
    """Extract frames from video and return frame list + FPS"""
    try:
        logger.info("🎬 Processing video frames...")
        
        if get_file_type(video_path) == "video":
            # Extract frames from video
            frame_dir = os.path.join(output_dir, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            
            reader = imageio.get_reader(video_path)
            fps = get_video_fps(video_path)
            
            for i, frame in enumerate(reader):
                imageio.imwrite(f"{frame_dir}/{i:08d}.png", frame)
            
            reader.close()
            input_img_list = sorted(glob.glob(os.path.join(frame_dir, '*.[jpJP][pnPN]*[gG]')))
        else:
            # Single image
            input_img_list = [video_path]
            fps = 25
            
        logger.info(f"✅ Extracted {len(input_img_list)} frames at {fps} FPS")
        return input_img_list, fps
        
    except Exception as e:
        logger.error(f"❌ Video processing failed: {e}")
        raise e

def process_audio_features(audio_path: str, fps: float = 25) -> tuple[torch.Tensor, list]:
    """Process audio to extract features for MuseTalk"""
    try:
        logger.info("🎵 Processing audio features...")
        
        # Extract audio features using MuseTalk AudioProcessor
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        
        # Get whisper chunks
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        
        logger.info(f"✅ Audio processed: {len(whisper_chunks)} chunks")
        return whisper_input_features, whisper_chunks
        
    except Exception as e:
        logger.error(f"❌ Audio processing failed: {e}")
        raise e

def generate_talking_head(avatar_path: str, audio_path: str, output_path: str, **kwargs) -> str:
    """
    Generate talking head video using MuseTalk1.5
    Tham khảo từ app.py inference function
    """
    try:
        logger.info("🎬 Starting MuseTalk1.5 generation...")
        
        # Extract parameters with defaults from app.py
        bbox_shift = kwargs.get('bbox_shift', 0)
        extra_margin = kwargs.get('extra_margin', 10)
        parsing_mode = kwargs.get('parsing_mode', 'jaw')
        left_cheek_width = kwargs.get('left_cheek_width', 90)
        right_cheek_width = kwargs.get('right_cheek_width', 90)
        batch_size = kwargs.get('batch_size', 8)
        
        logger.info(f"🎯 Generation Parameters:")
        logger.info(f"  bbox_shift: {bbox_shift}")
        logger.info(f"  extra_margin: {extra_margin}")
        logger.info(f"  parsing_mode: {parsing_mode}")
        logger.info(f"  cheek_width: L{left_cheek_width}/R{right_cheek_width}")
        logger.info(f"  batch_size: {batch_size}")
        
        # Verify MuseTalk availability
        if not MUSETALK_AVAILABLE:
            raise RuntimeError("MuseTalk not available")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Extract frames from input video/image
            input_img_list, fps = process_video_frames(avatar_path, temp_dir)
            
            # 2. Process audio features
            whisper_input_features, whisper_chunks = process_audio_features(audio_path, fps)
            
            # 3. Get face landmarks and coordinates
            logger.info("🔍 Extracting face landmarks...")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            
            # Check if face detected
            valid_coords = [coord for coord in coord_list if coord != coord_placeholder]
            if not valid_coords:
                raise ValueError("No face detected in the input image/video")
            
            # 4. Initialize face parser
            fp = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width
            )
            
            # 5. Prepare input latents
            logger.info("🔄 Preparing input latents...")
            input_latent_list = []
            
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                    
                x1, y1, x2, y2 = bbox
                y2 = y2 + extra_margin
                y2 = min(y2, frame.shape[0])
                
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
            
            # Cycle frames for smooth loop
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            
            # 6. Inference batch by batch
            logger.info("🎬 Starting inference...")
            video_num = len(whisper_chunks)
            
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )
            
            res_frame_list = []
            total_batches = int(np.ceil(float(video_num) / batch_size))
            
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total_batches)):
                audio_feature_batch = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=weight_dtype)
                
                # Generate using UNet
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
                
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            
            # 7. Combine results with original frames
            logger.info("🖼️ Combining frames...")
            result_frames_dir = os.path.join(temp_dir, "result_frames")
            os.makedirs(result_frames_dir, exist_ok=True)
            
            combined_frames = []
            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                
                x1, y1, x2, y2 = bbox
                y2 = y2 + extra_margin
                y2 = min(y2, ori_frame.shape[0])
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                    # Use MuseTalk v1.5 blending
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)
                    combined_frames.append(combine_frame)
                except Exception as e:
                    logger.warning(f"⚠️ Frame {i} processing failed: {e}")
                    continue
            
            # 8. Save video without audio first
            logger.info("💾 Saving video...")
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            
            imageio.mimwrite(
                temp_video_path, 
                combined_frames, 
                'FFMPEG', 
                fps=fps, 
                codec='libx264', 
                pixelformat='yuv420p'
            )
            
            # 9. Add audio to video
            logger.info("🎵 Adding audio to video...")
            video_clip = VideoFileClip(temp_video_path)
            audio_clip = AudioFileClip(audio_path)
            
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac', 
                fps=fps
            )
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_clip.close()
        
        # Verify output
        if not os.path.exists(output_path):
            raise FileNotFoundError("Generated video not found")
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ Video generated: {file_size:.1f}MB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Talking head generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading to MinIO: {object_name} ({file_size_mb:.1f}MB)")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Upload completed: {file_url}")
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def validate_input_parameters(job_input: dict) -> tuple[bool, str]:
    """Validate input parameters"""
    try:
        # Required parameters
        required_params = ["avatar_url", "audio_url"]
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"
        
        # Validate URLs
        for param in ["avatar_url", "audio_url"]:
            url = job_input[param]
            try:
                response = requests.head(url, timeout=10)
                if response.status_code != 200:
                    return False, f"{param} not accessible: {response.status_code}"
            except Exception as e:
                return False, f"{param} validation failed: {str(e)}"
        
        # Validate bbox_shift
        bbox_shift = job_input.get("bbox_shift", 0)
        if not (-50 <= bbox_shift <= 50):
            return False, "bbox_shift must be between -50 and 50"
        
        # Validate extra_margin
        extra_margin = job_input.get("extra_margin", 10)
        if not (0 <= extra_margin <= 40):
            return False, "extra_margin must be between 0 and 40"
        
        # Validate parsing_mode
        parsing_mode = job_input.get("parsing_mode", "jaw")
        if parsing_mode not in ["jaw", "raw"]:
            return False, "parsing_mode must be 'jaw' or 'raw'"
        
        # Validate cheek widths
        for param in ["left_cheek_width", "right_cheek_width"]:
            width = job_input.get(param, 90)
            if not (20 <= width <= 160):
                return False, f"{param} must be between 20 and 160"
        
        # Validate batch_size
        batch_size = job_input.get("batch_size", 8)
        if not (1 <= batch_size <= 16):
            return False, "batch_size must be between 1 and 16"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def load_models():
    """Load all MuseTalk models (cached)"""
    global vae, unet, pe, audio_processor, whisper, timesteps
    
    if 'models_loaded' in model_cache:
        return
    
    try:
        logger.info("🔄 Loading MuseTalk models...")
        
        # Load main MuseTalk models
        vae, unet, pe = load_all_model(
            unet_model_path=MODEL_CONFIGS["unet_model"],
            vae_type="sd-vae",
            unet_config=MODEL_CONFIGS["unet_config"],
            device=device
        )
        
        # Move to device and set dtype
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)
        
        timesteps = torch.tensor([0], device=device)
        
        # Initialize audio processor and Whisper
        audio_processor = AudioProcessor(feature_extractor_path=MODEL_CONFIGS["whisper_model"])
        whisper = WhisperModel.from_pretrained(MODEL_CONFIGS["whisper_model"])
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        
        model_cache['models_loaded'] = True
        logger.info("✅ All MuseTalk models loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise e

def handler(job):
    """
    Main RunPod handler for MuseTalk1.5
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Validate input parameters
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            return {
                "error": validation_message,
                "status": "failed",
                "job_id": job_id
            }
        
        # Extract parameters
        avatar_url = job_input["avatar_url"]
        audio_url = job_input["audio_url"]
        
        parameters = {
            "bbox_shift": job_input.get("bbox_shift", 0),
            "extra_margin": job_input.get("extra_margin", 10),
            "parsing_mode": job_input.get("parsing_mode", "jaw"),
            "left_cheek_width": job_input.get("left_cheek_width", 90),
            "right_cheek_width": job_input.get("right_cheek_width", 90),
            "batch_size": job_input.get("batch_size", 8)
        }
        
        logger.info(f"🚀 Job {job_id}: MuseTalk1.5 Generation Started")
        logger.info(f"🖼️ Avatar: {avatar_url}")
        logger.info(f"🎵 Audio: {audio_url}")
        logger.info(f"⚙️ Parameters: {parameters}")
        
        # Verify models
        models_ok, missing_models = verify_models()
        if not models_ok:
            return {
                "error": "Required models are missing",
                "missing_models": missing_models,
                "status": "failed"
            }
        
        # Load models
        load_models()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download inputs
            avatar_path = os.path.join(temp_dir, "avatar.jpg")
            audio_path = os.path.join(temp_dir, "audio.wav")
            
            logger.info("📥 Downloading inputs...")
            if not download_file(avatar_url, avatar_path):
                return {"error": "Failed to download avatar image"}
            
            if not download_file(audio_url, audio_path):
                return {"error": "Failed to download audio file"}
            
            # Generate talking head video
            logger.info("🎬 Generating talking head video...")
            generation_start = time.time()
            
            output_path = os.path.join(temp_dir, "output_video.mp4")
            result_path = generate_talking_head(
                avatar_path, 
                audio_path, 
                output_path,
                **parameters
            )
            
            generation_time = time.time() - generation_start
            
            # Upload result to MinIO
            logger.info("📤 Uploading result...")
            output_filename = f"musetalk_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            try:
                output_url = upload_to_minio(result_path, output_filename)
            except Exception as e:
                return {"error": f"Failed to upload result: {str(e)}"}
            
            # Calculate statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(result_path) / (1024 * 1024)
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.1f}s")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB")
            
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(total_time, 2),
                "generation_time_seconds": round(generation_time, 2),
                "video_info": {
                    "file_size_mb": round(file_size_mb, 2),
                },
                "generation_params": parameters,
                "status": "completed"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": error_msg,
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id
        }
    
    finally:
        clear_memory()

def health_check():
    """Health check function"""
    try:
        # Check CUDA
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Check models
        models_ok, missing = verify_models()
        if not models_ok:
            return False, f"Missing models: {len(missing)}"
        
        # Check MuseTalk
        if not MUSETALK_AVAILABLE:
            return False, "MuseTalk not available"
        
        # Check MinIO
        if not minio_client:
            return False, "MinIO not available"
        
        return True, "All systems operational"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting MuseTalk1.5 Serverless Worker...")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # Health check on startup
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            sys.exit(1)
        
        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎬 Ready to process MuseTalk1.5 requests...")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
