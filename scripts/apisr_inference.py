"""
APISR Video Inference Module

This module provides a comprehensive video upscaling system using APISR models.
It handles batch processing of videos, metrics collection, and model management.

Key Features:
    - Multi-model support (GRL, RRDB, DAT architectures)
    - Automatic model weight downloading
    - SSIM quality metrics calculation
    - Batch processing with detailed metrics
    - Model-specific output organization
    - Comprehensive error handling and logging

Technical Specifications:
    - Input: Video files (MP4, AVI, MKV, MOV)
    - Output: Upscaled videos with 2x or 4x scaling
    - GPU Memory: Managed via model-specific settings
    - Processing: Frame-by-frame with GPU acceleration
    - Metrics: FPS, SSIM, timing, and resolution tracking

Dependencies:
    - torch>=1.7.0
    - opencv-python>=4.5.0
    - loguru>=0.5.0
    - pydantic-settings>=2.0.0
    - numpy>=1.19.0

Example:
    >>> from scripts.apisr_inference import Config, inference
    >>> config = Config()
    >>> output_path, metrics = inference("input.mp4", "4xRRDB")
    >>> print(f"SSIM: {metrics['ssim']}, FPS: {metrics['model_fps']}")
"""

from loguru import logger
import os, sys
import cv2
import time
import datetime
import torch
import numpy as np
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import json

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from apisr.test_code.inference import super_resolve_video
from apisr.test_code.test_utils import load_grl, load_rrdb, load_dat


class Config(BaseSettings):
    """
    Configuration settings for APISR video processing.

    This class manages all configuration parameters for the video upscaling system,
    including model settings, paths, and processing parameters.

    Attributes:
        max_video_size (int): Maximum video dimension for processing
        weights_directory (str): Directory for model weights
        max_queue_size (int): Maximum batch processing queue size
        timezone (str): Timezone for logging timestamps
        input_directory (str): Source directory for videos
        output_directory (str): Target directory for processed videos
        metrics_directory (str): Directory for metrics JSON files
        selected_model (str): Default model for processing
        available_models (dict): Configuration for supported models

    Environment Variables:
        All settings can be overridden using APISR_ prefixed environment variables.
        Example: APISR_SELECTED_MODEL="4xRRDB"
    """
    max_video_size: int = 1080
    weights_directory: str = "pretrained"
    max_queue_size: int = 10
    timezone: str = 'US/Eastern'
    input_directory: str = "/root/pixelupbench/data/anime"
    output_directory: str = "../results"
    metrics_directory: str = "../results/metrics"
    selected_model: str = "2xRRDB"  # Default model
    available_models: dict = {
        "4xGRL": {
            "weight_file": "4x_APISR_GRL_GAN_generator.pth",
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth",
            "scale": 4
        },
        "4xRRDB": {
            "weight_file": "4x_APISR_RRDB_GAN_generator.pth",
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.2.0/4x_APISR_RRDB_GAN_generator.pth",
            "scale": 4
        },
        "2xRRDB": {
            "weight_file": "2x_APISR_RRDB_GAN_generator.pth",
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth",
            "scale": 2
        },
        "4xDAT": {
            "weight_file": "4x_APISR_DAT_GAN_generator.pth",
            "url": "https://github.com/Kiteretsu77/APISR/releases/download/v0.3.0/4x_APISR_DAT_GAN_generator.pth",
            "scale": 4
        }
    }

    class Config:
        env_prefix = "APISR_"


config = Config()


def auto_download_if_needed(weight_path: str, model_name: str) -> None:
    """
    Downloads model weights if they don't exist locally.

    Automatically creates the weights directory if it doesn't exist and
    downloads the appropriate model weights from the configured URL.

    Args:
        weight_path (str): Local path where weights should be stored
        model_name (str): Name of the model to download weights for

    Raises:
        RuntimeError: If download fails or URL is not configured
        
    Logs:
        - INFO: Download status and existing file detection
        - ERROR: Download failures and missing URLs
    """
    if os.path.exists(weight_path):
        logger.info(f"Weight file already exists: {weight_path}")
        return

    os.makedirs(config.weights_directory, exist_ok=True)
    
    model_info = config.available_models.get(model_name)
    if model_info and model_info["url"]:
        logger.info(f"Downloading {model_info['weight_file']} from {model_info['url']}")
        os.system(f"wget {model_info['url']} -O {weight_path}")
    else:
        logger.error(f"No URL found for model {model_name}")


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index (SSIM) between two images.

    Implements the SSIM calculation using a Gaussian window and handles
    the computation in floating-point precision.

    Args:
        img1 (np.ndarray): First image array (BGR format)
        img2 (np.ndarray): Second image array (BGR format)

    Returns:
        float: SSIM value between 0 and 1, where 1 indicates identical images

    Technical Details:
        - Uses 11x11 Gaussian window with σ=1.5
        - Applies constants C1=(0.01*255)² and C2=(0.03*255)²
        - Processes image edges with appropriate padding
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def calculate_video_ssim(input_path: str, output_path: str, scale: int) -> float:
    """
    Calculate SSIM between input and output videos using their middle frames.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to upscaled video
        scale (int): Upscaling factor
        
    Returns:
        float: SSIM value between middle frames
    """
    input_cap = cv2.VideoCapture(input_path)
    output_cap = cv2.VideoCapture(output_path)
    
    # Get total frames
    total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    
    # Set both captures to middle frame
    input_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    output_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    # Read middle frames
    input_ret, input_frame = input_cap.read()
    output_ret, output_frame = output_cap.read()
    
    input_cap.release()
    output_cap.release()
    
    if not input_ret or not output_ret:
        logger.error("Failed to read frames for SSIM calculation")
        return 0.0
        
    # Resize input frame to match output dimensions
    input_frame_resized = cv2.resize(
        input_frame, 
        (output_frame.shape[1], output_frame.shape[0]), 
        interpolation=cv2.INTER_CUBIC
    )
    
    # Calculate SSIM for middle frame
    ssim = calculate_ssim(input_frame_resized, output_frame)
    
    return ssim


def inference(video_path: str, model_name: str) -> tuple[str, dict]:
    """
    Performs video upscaling inference using the specified model.

    Args:
        video_path (str): Path to input video file
        model_name (str): Name of model to use for upscaling

    Returns:
        tuple[str, dict]: Tuple containing:
            - str: Path to output video file
            - dict: Processing metrics (without SSIM, which is calculated later)
    """
    try:
        if model_name not in config.available_models:
            raise ValueError(f"Unsupported model: {model_name}")

        model_info = config.available_models[model_name]
        weight_path = os.path.join(config.weights_directory, model_info["weight_file"])
        auto_download_if_needed(weight_path, model_name)
        scale = model_info["scale"]
        weight_dtype = torch.float32

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start_time = time.time()

        # Load model and process video
        if model_name == "4xGRL":
            generator = load_grl(weight_path, scale=scale)
        elif model_name in ["4xRRDB", "2xRRDB"]:
            generator = load_rrdb(weight_path, scale=scale)
        elif model_name == "4xDAT":
            generator = load_dat(weight_path, scale=scale)

        generator = generator.to(dtype=weight_dtype)

        # Create model-specific output directory
        model_output_dir = os.path.join(config.output_directory, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        output_video_path = os.path.join(
            model_output_dir,
            f"output_{time.time()}.mp4"
        )

        # Process video without SSIM calculation
        super_resolve_video(generator, video_path, output_video_path, 
                          weight_dtype=weight_dtype,
                          downsample_threshold=config.max_video_size, 
                          crop_for_4x=True, scale=scale)

        # Calculate processing metrics (without SSIM)
        processing_time = time.time() - start_time
        model_fps = total_frames / processing_time

        metrics = {
            "processing_time": round(processing_time, 2),
            "model_fps": round(model_fps, 2),
            "input_fps": round(input_fps, 2),
            "scale": scale
        }

        return output_video_path, metrics

    except Exception as error:
        logger.error(f"Global exception: {error}")
        raise


class BatchMetrics:
    """Tracks and manages performance metrics for batch processing operations."""
    def __init__(self, model_name: str):
        self.start_time = time.time()
        self.model_name = model_name
        self.videos = []
        self.pending_ssim_calculations = []

    def add_video_result(self, video_name: str, inference_time: float,
                        input_frame: tuple, output_frame: tuple,
                        input_fps: float, model_fps: float,
                        input_path: str, output_path: str) -> None:
        """Add processing results for a single video and queue SSIM calculation."""
        video_data = {
            "name": video_name,
            "inference_time": inference_time,
            "original_resolution": f"{input_frame[0]}x{input_frame[1]}",
            "upscaled_resolution": f"{output_frame[0]}x{output_frame[1]}",
            "input_fps": input_fps,
            "model_fps": model_fps
        }
        self.videos.append(video_data)
        self.pending_ssim_calculations.append((input_path, output_path, len(self.videos) - 1))

    def calculate_all_ssim(self, scale: int) -> None:
        """Calculate SSIM for all processed videos."""
        logger.info("Starting SSIM calculations for all processed videos...")
        for input_path, output_path, index in self.pending_ssim_calculations:
            try:
                ssim = calculate_video_ssim(input_path, output_path, scale)
                self.videos[index]["ssim"] = round(ssim, 3)
                logger.info(f"SSIM calculated for {self.videos[index]['name']}: {ssim:.3f}")
            except Exception as e:
                logger.error(f"Failed to calculate SSIM for {self.videos[index]['name']}: {str(e)}")
                self.videos[index]["ssim"] = None

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive batch processing summary."""
        total_time = time.time() - self.start_time
        return {
            "model_name": self.model_name,
            "total_videos": len(self.videos),
            "total_batch_time": total_time,
            "average_time_per_video": total_time / len(self.videos) if self.videos else 0,
            "videos": self.videos,
            "timestamp": datetime.datetime.now().isoformat()
        }


def export_metrics(metrics: Dict[str, Any], model_name: str) -> None:
    """
    Exports batch processing metrics to a JSON file.

    Creates a model-specific directory and saves metrics with timestamp.

    Args:
        metrics (Dict[str, Any]): Metrics data to export
        model_name (str): Name of model used for processing

    Technical Details:
        - Creates nested directory structure
        - Uses ISO 8601 timestamp in filename
        - Formats JSON with 4-space indentation
        
    Logs:
        - INFO: Export completion and file location
    """
    metrics_dir = os.path.join(config.metrics_directory, model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_path = os.path.join(
        metrics_dir,
        f"batch_metrics_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics exported to {metrics_path}")


if __name__ == "__main__":
    model_name = config.selected_model
    logger.info(f"Selected model: {model_name}")
    
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    batch_metrics = BatchMetrics(model_name)
    
    # Process all videos first
    for filename in os.listdir(config.input_directory):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(config.input_directory, filename)
            try:
                # Get original video dimensions
                cap = cv2.VideoCapture(video_path)
                input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                start_time = time.time()
                output_path, video_metrics = inference(video_path, model_name)
                inference_time = time.time() - start_time

                # Get output video dimensions
                cap = cv2.VideoCapture(output_path)
                output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                batch_metrics.add_video_result(
                    video_name=filename,
                    inference_time=inference_time,
                    input_frame=(input_width, input_height),
                    output_frame=(output_width, output_height),
                    input_fps=video_metrics["input_fps"],
                    model_fps=video_metrics["model_fps"],
                    input_path=video_path,
                    output_path=output_path
                )

                logger.info(
                    f"Processed {filename} -> {output_path}\n"
                    f"Resolution: {input_width}x{input_height} -> {output_width}x{output_height}\n"
                    f"Input FPS: {video_metrics['input_fps']:.2f}, Model FPS: {video_metrics['model_fps']:.2f}"
                )
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                continue
    
    # Calculate SSIM for all videos after processing
    logger.info("All videos processed. Starting SSIM calculations...")
    scale = config.available_models[model_name]["scale"]
    batch_metrics.calculate_all_ssim(scale)
    
    # Export the final metrics summary
    metrics_summary = batch_metrics.get_summary()
    export_metrics(metrics_summary, model_name)
    logger.info("Processing completed. Metrics exported.")
