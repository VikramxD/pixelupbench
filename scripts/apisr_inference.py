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
    selected_model: str = "4xRRDB"  # Default model
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
    Calculates average SSIM between input and upscaled video frames.

    Processes each frame pair, handling resolution differences by resizing
    the input frames to match the output resolution.

    Args:
        input_path (str): Path to input video file
        output_path (str): Path to upscaled video file
        scale (int): Upscaling factor used

    Returns:
        float: Average SSIM value across all frames

    Technical Details:
        - Resizes input frames using cubic interpolation
        - Processes all frames sequentially
        - Returns 0.0 if video processing fails
    """
    input_cap = cv2.VideoCapture(input_path)
    output_cap = cv2.VideoCapture(output_path)
    
    ssim_values = []
    
    while True:
        input_ret, input_frame = input_cap.read()
        output_ret, output_frame = output_cap.read()
        
        if not input_ret or not output_ret:
            break
            
        # Resize input frame to match output dimensions for comparison
        input_frame_resized = cv2.resize(
            input_frame, 
            (output_frame.shape[1], output_frame.shape[0]), 
            interpolation=cv2.INTER_CUBIC
        )
        
        ssim = calculate_ssim(input_frame_resized, output_frame)
        ssim_values.append(ssim)
    
    input_cap.release()
    output_cap.release()
    
    return sum(ssim_values) / len(ssim_values) if ssim_values else 0.0


def inference(video_path: str, model_name: str) -> tuple[str, dict]:
    """
    Performs video upscaling inference using the specified model.

    Handles the complete inference pipeline including model loading,
    video processing, and metrics calculation.

    Args:
        video_path (str): Path to input video file
        model_name (str): Name of model to use for upscaling

    Returns:
        tuple[str, dict]: Tuple containing:
            - str: Path to output video file
            - dict: Processing metrics including:
                - processing_time (float): Total processing time in seconds
                - model_fps (float): Frames processed per second
                - input_fps (float): Original video frame rate
                - ssim (float): Average SSIM quality metric
                - scale (int): Upscaling factor used

    Raises:
        ValueError: If model_name is not supported
        RuntimeError: If video processing fails

    Technical Details:
        - Automatically downloads model weights if needed
        - Creates model-specific output directories
        - Handles GPU memory management
        - Calculates comprehensive quality metrics
    """
    try:
        if model_name not in config.available_models:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(config.available_models.keys())}")

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

        # Load model based on type
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

        # Process video
        super_resolve_video(generator, video_path, output_video_path, 
                          weight_dtype=weight_dtype,
                          downsample_threshold=config.max_video_size, 
                          crop_for_4x=True, scale=scale)

        # Calculate SSIM after processing
        ssim = calculate_video_ssim(video_path, output_video_path, scale)

        # Calculate processing metrics
        processing_time = time.time() - start_time
        model_fps = total_frames / processing_time

        metrics = {
            "processing_time": round(processing_time, 2),
            "model_fps": round(model_fps, 2),
            "input_fps": round(input_fps, 2),
            "ssim": round(ssim, 3),
            "scale": scale
        }

        return output_video_path, metrics

    except Exception as error:
        logger.error(f"Global exception: {error}")
        raise


class BatchMetrics:
    """
    Manages and tracks metrics for batch video processing operations.

    This class handles the collection and organization of processing metrics
    for multiple videos, providing summary statistics and export capabilities.

    Attributes:
        start_time (float): Batch processing start timestamp
        model_name (str): Name of model being used
        videos (list): List of processed video results

    Technical Details:
        - Tracks per-video metrics including resolution and SSIM
        - Calculates batch-level statistics
        - Provides JSON-compatible data structure
    """
    def __init__(self, model_name: str):
        self.start_time = time.time()
        self.model_name = model_name
        self.videos = []

    def add_video_result(self, video_name: str, inference_time: float,
                        input_frame: tuple, output_frame: tuple,
                        ssim: float) -> None:
        """
        Adds processing results for a single video to the batch metrics.

        Args:
            video_name (str): Name of processed video file
            inference_time (float): Processing time in seconds
            input_frame (tuple): Original resolution (width, height)
            output_frame (tuple): Upscaled resolution (width, height)
            ssim (float): SSIM quality metric

        Technical Details:
            - Rounds numeric values for consistent formatting
            - Stores resolutions in "WxH" string format
        """
        self.videos.append({
            "name": video_name,
            "inference_time": inference_time,
            "original_resolution": f"{input_frame[0]}x{input_frame[1]}",
            "upscaled_resolution": f"{output_frame[0]}x{output_frame[1]}",
            "ssim": round(ssim, 3)
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Generates comprehensive batch processing summary.

        Returns:
            Dict[str, Any]: Summary dictionary containing:
                - model_name (str): Name of model used
                - total_videos (int): Number of videos processed
                - total_batch_time (float): Total processing time
                - average_time_per_video (float): Average processing time
                - videos (list): Detailed metrics for each video
                - timestamp (str): ISO format timestamp

        Technical Details:
            - Calculates aggregate statistics
            - Formats timestamp in ISO 8601 format
            - Includes all individual video metrics
        """
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
                    ssim=video_metrics["ssim"]
                )

                logger.info(
                    f"Processed {filename} -> {output_path}\n"
                    f"Resolution: {input_width}x{input_height} -> {output_width}x{output_height}\n"
                    f"SSIM: {video_metrics['ssim']:.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                continue
    
    # Export the metrics summary
    metrics_summary = batch_metrics.get_summary()
    export_metrics(metrics_summary, model_name)
