"""
Video Upscaling System using Real-ESRGAN and Related Models

This module provides a production-grade video upscaling system that utilizes
Real-ESRGAN models for high-quality video enhancement. It supports multiple
models and configurations with comprehensive error handling and logging.

Typical usage example:
    settings = UpscalerSettings()
    upscaler = VideoUpscaler(settings)
    output_path = upscaler.upscale()

Dependencies:
    - Real-ESRGAN
    - PyTorch
    - OpenCV
    - CUDA Toolkit
"""

from pathlib import Path
from typing import Literal, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
import torch
import cv2
import os
import subprocess
import shutil
import git
from PIL import Image
from tqdm import tqdm
from loguru import logger
from configs.upscaler_settings import UpscalerSettings
import psutil
import time
import GPUtil
import sys


class ResourceMonitor:
    """Monitor system resources during video processing operations.

    This class provides real-time monitoring of GPU, CPU, and memory usage during
    video processing tasks. It tracks resource utilization metrics and processing time
    for performance analysis and optimization.

    Attributes:
        gpu_id (int): The ID of the GPU device to monitor
        start_time (float | None): Timestamp when monitoring started
        end_time (float | None): Timestamp when monitoring ended

    Example:
        monitor = ResourceMonitor(gpu_id=0)
        monitor.start()
        metrics = monitor.get_metrics()
    """

    def __init__(self, gpu_id: int):
        """Initialize the resource monitor.

        Args:
            gpu_id (int): The ID of the GPU device to monitor
        """
        self.gpu_id = gpu_id
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start monitoring resources."""
        self.start_time = time.time()

    def get_metrics(self) -> Dict[str, float]:
        """Get current system resource usage metrics.

        Returns:
            Dict[str, float]: Dictionary containing the following metrics:
                - gpu_memory_used: GPU memory usage in MB
                - gpu_utilization: GPU utilization percentage (0-100)
                - cpu_percent: CPU utilization percentage (0-100)
                - ram_percent: RAM utilization percentage (0-100)
                - time_elapsed: Time elapsed since monitoring started in seconds
        """
        gpu = GPUtil.getGPUs()[self.gpu_id]
        return {
            "gpu_memory_used": gpu.memoryUsed,
            "gpu_utilization": gpu.load * 100,
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "time_elapsed": time.time() - self.start_time if self.start_time else 0,
        }


class VideoUpscaler:
    """Production-grade video upscaling system using Real-ESRGAN.

    This class implements a comprehensive video upscaling pipeline using the Real-ESRGAN
    super-resolution model. It handles the complete workflow including environment setup,
    model management, resource monitoring, and video processing.

    Features:
        - Automatic environment setup and dependency management
        - Multiple model support with automatic weight downloading
        - Resource monitoring and logging
        - Error handling and recovery
        - Progress tracking and metrics reporting

    Attributes:
        settings (UpscalerSettings): Configuration settings for the upscaler
        monitor (ResourceMonitor): System resource monitor instance
        model (Any): Loaded model instance
        device (torch.device): Processing device (GPU)
        realesrgan_path (Path): Path to Real-ESRGAN installation

    Example:
        settings = UpscalerSettings()
        upscaler = VideoUpscaler(settings)
        output_path = upscaler.process_video()
    """

    REALESRGAN_REPO = "https://github.com/xinntao/Real-ESRGAN.git"
    MODEL_URLS = {
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    }

    def __init__(self, settings: UpscalerSettings):
        """Initialize the upscaler with settings."""
        self.settings = settings
        self.monitor = ResourceMonitor(settings.gpu_device)
        self._setup_environment()
        self._setup_logging()
        self.model = None
        self.device = self._setup_device()

    def _setup_logging(self):
        """Configure logging system."""
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.settings.log_dir / f"upscaler_{time.strftime('%Y%m%d_%H%M%S')}.log"

        logger.remove()  # Remove default handler
        logger.add(
            log_file,
            rotation="100 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
            level="INFO",
        )

    def _setup_environment(self):
        """Set up the processing environment for video upscaling.

        This method performs the following setup tasks:
        1. Verifies CUDA GPU availability
        2. Creates necessary directories
        3. Clones and installs Real-ESRGAN if not present
        4. Downloads required model weights

        Raises:
            RuntimeError: If no CUDA-capable GPU is detected
            subprocess.CalledProcessError: If dependency installation fails
        """
        logger.info("Setting up environment...")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not detected. CUDA-capable GPU is required.")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.realesrgan_path = Path("../Real-ESRGAN")
        if not self.realesrgan_path.exists():
            logger.info("Cloning Real-ESRGAN repository...")
            git.Repo.clone_from(self.REALESRGAN_REPO, self.realesrgan_path)
            logger.info("Installing dependencies...")
            subprocess.run(["pip", "install", "-r", str(self.realesrgan_path / "requirements.txt")], check=True)
            subprocess.run(["python", "setup.py", "develop"], cwd=self.realesrgan_path, check=True)

        self._download_model_weights()
        logger.info("Environment setup complete")

    def _download_model_weights(self):
        """Download model weights if not present."""
        weights_dir = self.realesrgan_path / "weights"
        weights_dir.mkdir(exist_ok=True)

        model_path = weights_dir / f"{self.settings.model_name}.pth"
        if not model_path.exists():
            logger.info(f"Downloading {self.settings.model_name} weights...")
            url = self.MODEL_URLS[self.settings.model_name]
            subprocess.run(["wget", url, "-O", str(model_path)], check=True)

    def _setup_device(self) -> torch.device:
        """Set up and return the processing device."""
        device = torch.device(f"cuda:{self.settings.gpu_device}")
        logger.info(f"Using device: {device}")
        return device

    def process_video(self) -> Path:
        """Process and upscale the input video.

        This method handles the complete video processing pipeline:
        1. Initializes resource monitoring
        2. Opens and validates the input video
        3. Configures output paths
        4. Runs the upscaling process
        5. Collects and logs performance metrics

        Returns:
            Path: Path to the processed output video file

        Raises:
            Exception: Any processing errors with detailed error messages
        """
        logger.info(f"Processing video: {self.settings.video_path}")
        self.monitor.start()

        try:
            # Get video info
            cap = cv2.VideoCapture(str(self.settings.video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Set up output
            output_path = self._setup_output_path()

            # Process with Real-ESRGAN
            logger.info("Running Real-ESRGAN upscaling...")
            result_path = self._run_upscaling()

            # Log final metrics
            metrics = self.monitor.get_metrics()
            logger.info("Processing complete. Metrics:")
            for key, value in metrics.items():
                logger.info(f"{key}: {value}")

            return result_path

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
        finally:
            if "cap" in locals():
                cap.release()

    def _setup_output_path(self) -> Path:
        """Set up output path for processed video."""
        output_name = (
            f"{self.settings.video_path.stem}" f"_{self.settings.model_name}" f"_x{self.settings.scale_factor}.mp4"
        )
        return self.settings.output_dir / output_name

    def _run_upscaling(self) -> Path:
        """Execute the Real-ESRGAN upscaling process.

        This method constructs and executes the command-line interface for Real-ESRGAN
        video processing, handling all necessary arguments and options based on the
        current settings.

        The following settings are supported:
        - Model selection
        - Scale factor
        - Tile size
        - Half precision mode
        - Face enhancement
        - Custom output suffix

        Returns:
            Path: Path to the processed output video file

        Raises:
            subprocess.CalledProcessError: If the upscaling process fails
        """
        input_path = str(self.settings.video_path)
        output_dir = str(self.settings.output_dir)

        # Build command with correct arguments
        cmd = [
            "python",
            str(self.realesrgan_path / "inference_realesrgan_video.py"),
            "-i",
            input_path,
            "-o",
            output_dir,
            "-n",
            self.settings.model_name,
            "-s",
            str(self.settings.scale_factor),
            "-t",
            str(self.settings.tile_size),
        ]

        # Optional arguments
        if not self.settings.use_half_precision:
            cmd.append("--fp32")

        if self.settings.face_enhance:
            cmd.append("--face_enhance")

        if self.settings.suffix:
            cmd.extend(["--suffix", self.settings.suffix])

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.debug(f"Process output: {process.stdout}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error: {e.stderr}")
            raise

        # Get output path
        output_filename = f"{self.settings.video_path.stem}_out.mp4"
        output_path = self.settings.output_dir / output_filename

        return output_path


def main():
    """Main entry point."""
    try:
        settings = UpscalerSettings()
        upscaler = VideoUpscaler(settings)
        output_path = upscaler.process_video()
        logger.success(f"Video processed successfully: {output_path}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
