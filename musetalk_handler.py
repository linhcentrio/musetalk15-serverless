#!/usr/bin/env python3
"""
MuseTalk Realtime Handler with Material Caching
Optimized for chat/interactive use cases with avatar reuse
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
import glob
import pickle
import copy
import threading
import queue
import shutil
import zipfile
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add MuseTalk paths
sys.path.insert(0, '/app/MuseTalk')

# Import MuseTalk components
try:
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
    from musetalk.utils.audio_processor import AudioProcessor
    logger.info("✅ MuseTalk modules imported successfully")
    MUSETALK_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ MuseTalk import error: {e}")
    MUSETALK_AVAILABLE = False

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_CACHE_BUCKET = "musetalk-cache"  # Separate bucket for material cache
MINIO_SECURE = False

# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    
    # Ensure cache bucket exists
    if not minio_client.bucket_exists(MINIO_CACHE_BUCKET):
        minio_client.make_bucket(MINIO_CACHE_BUCKET)
    
    logger.info("✅ MinIO client initialized with cache bucket")
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cache = {}
timesteps = torch.tensor([0], device=device)

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

def load_models():
    """Load models with caching"""
    global audio_processor, vae, unet, pe
    
    if 'models_loaded' in model_cache:
        logger.info("✅ Using cached models")
        return
    
    try:
        logger.info("🔄 Loading MuseTalk models...")
        
        # Load all models (following original realtime_inference.py)
        audio_processor, vae, unet, pe = load_all_model()
        
        # Half precision optimization
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
        
        model_cache['models_loaded'] = True
        logger.info("✅ All models loaded and cached")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise e

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    """Extract frames from video"""
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break
    cap.release()

def create_material_cache(avatar_path, cache_dir, avatar_id, bbox_shift, batch_size):
    """
    Create material cache with landmarks, masks, and latents
    Returns: (cache_info, success)
    """
    try:
        logger.info(f"🔄 Creating material cache for avatar: {avatar_id}")
        
        # Setup paths
        full_imgs_path = os.path.join(cache_dir, "full_imgs")
        mask_out_path = os.path.join(cache_dir, "mask")
        os.makedirs(full_imgs_path, exist_ok=True)
        os.makedirs(mask_out_path, exist_ok=True)
        
        # Extract frames from avatar
        if os.path.isfile(avatar_path):
            if get_file_type(avatar_path) == "video":
                video2imgs(avatar_path, full_imgs_path, ext='png')
            else:
                # Single image - copy to frames
                shutil.copy2(avatar_path, os.path.join(full_imgs_path, "00000000.png"))
        else:
            raise ValueError("Invalid avatar path")
        
        # Get image list
        input_img_list = sorted(glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
        if not input_img_list:
            raise ValueError("No images found in avatar")
        
        # Extract landmarks
        logger.info("🔍 Extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        
        # Validate face detection
        valid_coords = [coord for coord in coord_list if coord != coord_placeholder]
        if not valid_coords:
            raise ValueError("No face detected in avatar")
        
        # Prepare latents
        logger.info("🧠 Preparing latents...")
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        
        # Create cycles for smooth animation
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # Prepare masks for advanced blending
        logger.info("🎭 Preparing masks...")
        mask_coords_list_cycle = []
        mask_list_cycle = []
        
        for i, frame in enumerate(tqdm(frame_list_cycle, desc="Creating masks")):
            cv2.imwrite(f"{full_imgs_path}/{str(i).zfill(8)}.png", frame)
            face_box = coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
            mask_coords_list_cycle.append(crop_box)
            mask_list_cycle.append(mask)
        
        # Save all cache data
        coords_path = os.path.join(cache_dir, "coords.pkl")
        mask_coords_path = os.path.join(cache_dir, "mask_coords.pkl")
        latents_path = os.path.join(cache_dir, "latents.pt")
        
        with open(coords_path, 'wb') as f:
            pickle.dump(coord_list_cycle, f)
        
        with open(mask_coords_path, 'wb') as f:
            pickle.dump(mask_coords_list_cycle, f)
        
        torch.save(input_latent_list_cycle, latents_path)
        
        # Create cache info
        cache_info = {
            "avatar_id": avatar_id,
            "bbox_shift": bbox_shift,
            "batch_size": batch_size,
            "created_at": time.time(),
            "num_frames": len(frame_list),
            "num_cycles": len(frame_list_cycle),
            "valid_faces": len(valid_coords),
            "version": "1.0"
        }
        
        # Save cache info
        info_path = os.path.join(cache_dir, "cache_info.json")
        with open(info_path, 'w') as f:
            json.dump(cache_info, f, indent=2)
        
        logger.info(f"✅ Material cache created: {len(frame_list)} frames, {len(valid_coords)} valid faces")
        
        return cache_info, True
        
    except Exception as e:
        logger.error(f"❌ Material cache creation failed: {e}")
        return None, False

def upload_material_cache(cache_dir, avatar_id):
    """
    Upload material cache to MinIO as a ZIP file
    Returns: cache_url
    """
    try:
        logger.info(f"📤 Uploading material cache for {avatar_id}...")
        
        # Create ZIP file
        zip_path = os.path.join(os.path.dirname(cache_dir), f"{avatar_id}_cache.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, cache_dir)
                    zipf.write(file_path, arc_name)
        
        # Upload to MinIO
        cache_object_name = f"avatars/{avatar_id}/{uuid.uuid4().hex[:8]}_cache.zip"
        
        file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        logger.info(f"📦 Uploading cache ZIP: {file_size_mb:.1f}MB")
        
        minio_client.fput_object(MINIO_CACHE_BUCKET, cache_object_name, zip_path)
        
        cache_url = f"https://{MINIO_ENDPOINT}/{MINIO_CACHE_BUCKET}/{quote(cache_object_name)}"
        logger.info(f"✅ Cache uploaded: {cache_url}")
        
        # Cleanup local ZIP
        os.remove(zip_path)
        
        return cache_url
        
    except Exception as e:
        logger.error(f"❌ Cache upload failed: {e}")
        raise e

def download_and_extract_cache(cache_url, extract_dir):
    """
    Download and extract material cache from MinIO
    Returns: (cache_info, success)
    """
    try:
        logger.info(f"📥 Downloading material cache: {cache_url}")
        
        # Download ZIP file
        zip_path = os.path.join(extract_dir, "cache.zip")
        response = requests.get(cache_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Extract ZIP
        cache_data_dir = os.path.join(extract_dir, "cache_data")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(cache_data_dir)
        
        # Load cache info
        info_path = os.path.join(cache_data_dir, "cache_info.json")
        if not os.path.exists(info_path):
            raise ValueError("Invalid cache: missing cache_info.json")
        
        with open(info_path, 'r') as f:
            cache_info = json.load(f)
        
        logger.info(f"✅ Cache downloaded and extracted: {cache_info['num_frames']} frames")
        
        # Cleanup ZIP
        os.remove(zip_path)
        
        return cache_info, cache_data_dir, True
        
    except Exception as e:
        logger.error(f"❌ Cache download failed: {e}")
        return None, None, False

class CachedAvatar:
    """
    Enhanced Avatar class with Material Caching support
    """
    def __init__(self, avatar_id, avatar_path, bbox_shift, batch_size, 
                 material_cache_url=None, force_recreate=False):
        self.avatar_id = avatar_id
        self.avatar_path = avatar_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.material_cache_url = material_cache_url
        self.force_recreate = force_recreate
        self.idx = 0
        
        # Materials
        self.frame_list_cycle = None
        self.coord_list_cycle = None
        self.input_latent_list_cycle = None
        self.mask_coords_list_cycle = None
        self.mask_list_cycle = None
        self.cache_info = None
        
        self.init()
    
    def init(self):
        """Initialize avatar with cached or new materials"""
        try:
            if self.material_cache_url and not self.force_recreate:
                # Try to use cached materials
                success = self.load_cached_materials()
                if success:
                    logger.info("✅ Using cached materials")
                    return
                else:
                    logger.warning("⚠️ Cache loading failed, creating new materials")
            
            # Create new materials
            self.create_new_materials()
            
        except Exception as e:
            logger.error(f"❌ Avatar initialization failed: {e}")
            raise e
    
    def load_cached_materials(self):
        """Load materials from cache"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download and extract cache
                cache_info, cache_data_dir, success = download_and_extract_cache(
                    self.material_cache_url, temp_dir
                )
                
                if not success:
                    return False
                
                # Validate cache compatibility
                if (cache_info['bbox_shift'] != self.bbox_shift):
                    logger.warning(f"⚠️ Cache bbox_shift mismatch: {cache_info['bbox_shift']} vs {self.bbox_shift}")
                    return False
                
                # Load cached data
                coords_path = os.path.join(cache_data_dir, "coords.pkl")
                mask_coords_path = os.path.join(cache_data_dir, "mask_coords.pkl")
                latents_path = os.path.join(cache_data_dir, "latents.pt")
                
                with open(coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                
                with open(mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                
                self.input_latent_list_cycle = torch.load(latents_path)
                
                # Load frame images
                full_imgs_path = os.path.join(cache_data_dir, "full_imgs")
                input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                
                # Load mask images
                mask_out_path = os.path.join(cache_data_dir, "mask")
                input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)
                
                self.cache_info = cache_info
                
                logger.info(f"✅ Cached materials loaded: {len(self.frame_list_cycle)} frames")
                return True
                
        except Exception as e:
            logger.error(f"❌ Cache loading error: {e}")
            return False
    
    def create_new_materials(self):
        """Create new materials and optionally cache them"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create materials
                cache_info, success = create_material_cache(
                    self.avatar_path, temp_dir, self.avatar_id, 
                    self.bbox_shift, self.batch_size
                )
                
                if not success:
                    raise RuntimeError("Failed to create material cache")
                
                # Load created materials
                coords_path = os.path.join(temp_dir, "coords.pkl")
                mask_coords_path = os.path.join(temp_dir, "mask_coords.pkl")
                latents_path = os.path.join(temp_dir, "latents.pt")
                
                with open(coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                
                with open(mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                
                self.input_latent_list_cycle = torch.load(latents_path)
                
                # Load frame images
                full_imgs_path = os.path.join(temp_dir, "full_imgs")
                input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                
                # Load mask images
                mask_out_path = os.path.join(temp_dir, "mask")
                input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)
                
                self.cache_info = cache_info
                
                # Upload cache to MinIO for future use
                try:
                    self.new_cache_url = upload_material_cache(temp_dir, self.avatar_id)
                    logger.info(f"📦 New cache created and uploaded")
                except Exception as e:
                    logger.warning(f"⚠️ Cache upload failed: {e}")
                    self.new_cache_url = None
                
                logger.info(f"✅ New materials created: {len(self.frame_list_cycle)} frames")
                
        except Exception as e:
            logger.error(f"❌ New material creation failed: {e}")
            raise e
    
    def process_frames_realtime(self, res_frame_queue, video_len, result_frames):
        """Real-time frame processing with threading"""
        self.idx = 0
        
        while self.idx < video_len:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
            
            x1, y1, x2, y2 = bbox
            
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
                mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
                
                # Advanced blending with mask
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                result_frames.append(combine_frame)
                
            except Exception as e:
                logger.warning(f"⚠️ Frame {self.idx} processing failed: {e}")
                continue
            
            self.idx += 1
    
    def inference_realtime(self, audio_path, output_path, fps=25):
        """Real-time inference with threading"""
        logger.info("🚀 Starting cached realtime inference...")
        
        # Extract audio features (same as original)
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        result_frames = []
        
        # Start frame processing thread
        process_thread = threading.Thread(
            target=self.process_frames_realtime,
            args=(res_frame_queue, video_num, result_frames)
        )
        process_thread.start()
        
        # Generate frames
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num)/self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        
        # Wait for processing to complete
        process_thread.join()
        
        # Save video
        self.save_video_from_frames(result_frames, output_path, fps, audio_path)
        
        return output_path
    
    def save_video_from_frames(self, frames, output_path, fps, audio_path):
        """Save video from frames with audio"""
        temp_dir = os.path.dirname(output_path)
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save frames
        for i, frame in enumerate(frames):
            cv2.imwrite(f"{frames_dir}/{str(i).zfill(8)}.png", frame)
        
        # Create video
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {frames_dir}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {temp_video}"
        os.system(cmd_img2video)
        
        # Add audio
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_video} {output_path}"
        os.system(cmd_combine_audio)
        
        # Cleanup
        shutil.rmtree(frames_dir)
        if os.path.exists(temp_video):
            os.remove(temp_video)

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL"""
    try:
        logger.info(f"📥 Downloading: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"✅ Downloaded: {file_size:.1f}MB")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
        
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
        
        # Validate optional material_cache_url
        material_cache_url = job_input.get("material_cache_url")
        if material_cache_url:
            try:
                response = requests.head(material_cache_url, timeout=10)
                if response.status_code != 200:
                    return False, f"material_cache_url not accessible: {response.status_code}"
            except Exception as e:
                return False, f"material_cache_url validation failed: {str(e)}"
        
        # Validate numeric parameters
        validations = [
            ("bbox_shift", (-50, 50)),
            ("batch_size", (1, 16)),
            ("fps", (10, 60))
        ]
        
        for param, (min_val, max_val) in validations:
            if param in job_input:
                value = job_input[param]
                if not (min_val <= value <= max_val):
                    return False, f"{param} must be between {min_val} and {max_val}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

@torch.no_grad()
def handler(job):
    """
    Main MuseTalk Realtime Handler with Material Caching
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        
        # Validate input parameters
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            return {"error": validation_message, "status": "failed", "job_id": job_id}
        
        # Extract parameters
        avatar_url = job_input["avatar_url"]
        audio_url = job_input["audio_url"]
        material_cache_url = job_input.get("material_cache_url")  # NEW: Optional cache URL
        force_recreate = job_input.get("force_recreate", False)   # NEW: Force recreate cache
        
        parameters = {
            "bbox_shift": job_input.get("bbox_shift", 0),
            "batch_size": job_input.get("batch_size", 4),
            "fps": job_input.get("fps", 25)
        }
        
        logger.info(f"🚀 Job {job_id}: MuseTalk Realtime Generation with Caching")
        logger.info(f"🖼️ Avatar: {avatar_url}")
        logger.info(f"🎵 Audio: {audio_url}")
        logger.info(f"📦 Cache URL: {material_cache_url or 'None (will create new)'}")
        logger.info(f"🔄 Force Recreate: {force_recreate}")
        logger.info(f"⚙️ Parameters: {parameters}")
        
        # Verify MuseTalk availability
        if not MUSETALK_AVAILABLE:
            return {"error": "MuseTalk modules not available", "status": "failed"}
        
        # Load models
        load_models()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download inputs
            avatar_ext = os.path.splitext(urlparse(avatar_url).path)[1] or '.mp4'
            audio_ext = os.path.splitext(urlparse(audio_url).path)[1] or '.wav'
            
            avatar_path = os.path.join(temp_dir, f"avatar{avatar_ext}")
            audio_path = os.path.join(temp_dir, f"audio{audio_ext}")
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            logger.info("📥 Downloading inputs...")
            if not download_file(avatar_url, avatar_path):
                return {"error": "Failed to download avatar"}
            
            if not download_file(audio_url, audio_path):
                return {"error": "Failed to download audio"}
            
            # Initialize cached avatar
            logger.info("🎭 Initializing cached avatar...")
            generation_start = time.time()
            
            avatar_id = f"avatar_{uuid.uuid4().hex[:8]}"
            cached_avatar = CachedAvatar(
                avatar_id=avatar_id,
                avatar_path=avatar_path,
                bbox_shift=parameters["bbox_shift"],
                batch_size=parameters["batch_size"],
                material_cache_url=material_cache_url,
                force_recreate=force_recreate
            )
            
            # Run realtime inference
            result_path = cached_avatar.inference_realtime(
                audio_path, output_path, fps=parameters["fps"]
            )
            
            generation_time = time.time() - generation_start
            
            if not result_path or not os.path.exists(result_path):
                return {"error": "Video generation failed"}
            
            # Upload result to MinIO
            logger.info("📤 Uploading result...")
            output_filename = f"musetalk_realtime_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            try:
                output_url = upload_to_minio(result_path, output_filename)
            except Exception as e:
                return {"error": f"Upload failed: {str(e)}"}
            
            # Calculate statistics
            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(result_path) / (1024 * 1024)
            
            # Determine cache info for response
            cache_used = material_cache_url is not None and not force_recreate
            new_cache_url = getattr(cached_avatar, 'new_cache_url', None)
            
            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total: {total_time:.1f}s, Generation: {generation_time:.1f}s")
            logger.info(f"📊 Output: {file_size_mb:.1f}MB")
            
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(total_time, 2),
                "generation_time_seconds": round(generation_time, 2),
                "video_info": {
                    "file_size_mb": round(file_size_mb, 2),
                    "fps": parameters["fps"]
                },
                "generation_params": parameters,
                "caching_info": {
                    "cache_used": cache_used,
                    "material_cache_url": material_cache_url,
                    "new_cache_url": new_cache_url,  # NEW: URL for reuse in future requests
                    "cache_stats": cached_avatar.cache_info if cached_avatar.cache_info else None,
                    "performance_boost": f"{100-((generation_time/total_time)*100):.1f}% time saved" if cache_used else "0% (new cache created)"
                },
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

if __name__ == "__main__":
    logger.info("🚀 Starting MuseTalk Realtime Handler with Material Caching...")
    logger.info(f"🔥 PyTorch: {torch.__version__}")
    logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")
    
    try:
        logger.info("🎭 Features: Material Caching, Real-time Inference, Threading")
        logger.info("📦 Cache Strategy: ZIP-based storage on MinIO")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)
