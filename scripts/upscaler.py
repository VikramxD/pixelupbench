import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import cv2
import torch
import subprocess
from tqdm import tqdm
from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import Field
import git
from configs.upscaler_settings import UpscalerSettings

"""
Features:
    - Batch video processing with progress tracking
    - Detailed performance metrics collection
    - Automated environment setup
    - Error handling and recovery
    - JSON-based metrics export

Dependencies:
    - torch with CUDA support
    - Real-ESRGAN
    - OpenCV
    - loguru for logging
    - tqdm for progress tracking

Typical usage:
    settings = UpscalerSettings(input_dir=Path("videos"), model_name="RealESRGAN_x4plus")
    upscaler = VideoUpscaler(settings)
    metrics = upscaler.process_batch()
"""


class BatchMetrics:
    """
    Tracks and manages performance metrics for batch video processing operations.

    This class handles the collection and aggregation of processing metrics across
    multiple videos in a batch, providing summary statistics and individual video results.

    Attributes:
        start_time (float): Unix timestamp when batch processing started
        model_name (str): Name of the upscaling model being used
        num_videos (int): Total number of videos in the batch
        videos_processed (List[Dict[str, Union[str, float]]]): List of processed video results

    Example:
        metrics = BatchMetrics("RealESRGAN_x4plus", num_videos=5)
        metrics.add_video_result("video1.mp4", 10.5)
        summary = metrics.get_summary()
    """

    def __init__(self, model_name: str, num_videos: int) -> None:
        """
        Initialize batch metrics tracker.

        Args:
            model_name: Name of the upscaling model being used
            num_videos: Total number of videos to be processed
        """
        self.start_time: float = time.time()
        self.model_name: str = model_name
        self.num_videos: int = num_videos
        self.videos_processed: List[Dict[str, Union[str, float]]] = []

    def add_video_result(self, video_name: str, inference_time: float) -> None:
        """
        Add processing results for a single video.

        Args:
            video_name: Name of the processed video file
            inference_time: Time taken to process the video in seconds
        """
        self.videos_processed.append({"name": video_name, "inference_time": inference_time})

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive batch processing summary.

        Returns:
            Dictionary containing:
                - model_name: Name of the model used
                - total_videos: Number of videos processed
                - total_batch_time: Total processing time in seconds
                - average_time_per_video: Average processing time per video
                - videos: List of individual video results
                - timestamp: ISO format timestamp of completion
        """
        total_time = time.time() - self.start_time
        return {
            "model_name": self.model_name,
            "total_videos": self.num_videos,
            "total_batch_time": total_time,
            "average_time_per_video": total_time / self.num_videos,
            "videos": self.videos_processed,
            "timestamp": datetime.now().isoformat(),
        }


class VideoUpscaler:
    """
    Production-grade video upscaling system with batch processing capabilities.

    This class implements a comprehensive video upscaling pipeline using Real-ESRGAN,
    supporting batch processing with metrics collection and error handling.

    Attributes:
        settings (UpscalerSettings): Configuration settings for the upscaler
        realesrgan_path (Path): Path to Real-ESRGAN installation

    Class Constants:
        REALESRGAN_REPO (str): URL of the Real-ESRGAN repository

    Example:
        settings = UpscalerSettings(input_dir=Path("videos"))
        upscaler = VideoUpscaler(settings)
        metrics = upscaler.process_batch()
    """

    REALESRGAN_REPO: str = "https://github.com/xinntao/Real-ESRGAN.git"

    def __init__(self, settings: UpscalerSettings) -> None:
        """
        Initialize the upscaler with provided settings.

        Args:
            settings: Configuration settings for video processing

        Raises:
            RuntimeError: If CUDA GPU is not available
        """
        self.settings = settings
        self.realesrgan_path = self._setup_environment()
        logger.info(f"Using model: {settings.model_name}")

    def _setup_environment(self) -> Path:
        """
        Set up the processing environment and dependencies.

        Performs the following setup tasks:
        1. Verifies CUDA GPU availability
        2. Creates output directories
        3. Clones and installs Real-ESRGAN if not present

        Returns:
            Path to the Real-ESRGAN installation

        Raises:
            RuntimeError: If no CUDA-capable GPU is detected
            subprocess.CalledProcessError: If dependency installation fails
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not detected. CUDA-capable GPU is required.")

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        realesrgan_path = Path("../Real-ESRGAN")

        if not realesrgan_path.exists():
            logger.info("Setting up Real-ESRGAN...")
            git.Repo.clone_from(self.REALESRGAN_REPO, realesrgan_path)
            subprocess.run(["pip", "install", "-r", str(realesrgan_path / "requirements.txt")], check=True)

        return realesrgan_path

    def process_video(self, video_path: Path) -> float:
        """
        Process a single video through the upscaling pipeline.

        Handles the complete processing of one video including:
        1. Output path setup
        2. Model configuration
        3. Video processing
        4. Error handling

        Args:
            video_path: Path to the input video file

        Returns:
            Total inference time in seconds

        Raises:
            RuntimeError: If video processing fails
            subprocess.CalledProcessError: If subprocess execution fails
        """
        start_time = time.time()

        output_dir = self.settings.output_dir / self.settings.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_output.mp4"

        cmd = [
            "python",
            str(self.realesrgan_path / "inference_realesrgan_video.py"),
            "-i",
            str(video_path),
            "-o",
            str(output_path),
            "-n",
            self.settings.model_name,
            "-s",
            str(self.settings.scale_factor),
            "-t",
            str(self.settings.tile_size),
        ]

        if not self.settings.use_half_precision:
            cmd.append("--fp32")
        if self.settings.face_enhance:
            cmd.append("--face_enhance")

        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"Processing failed: {process.stderr}")

        inference_time = time.time() - start_time
        return inference_time

    def process_batch(self) -> Dict[str, Any]:
        """
        Process multiple videos in batch mode with metrics collection.

        Processes all MP4 videos in the input directory, collecting metrics
        and handling errors for each video individually. Failed videos are
        logged but don't stop the batch processing.

        Returns:
            Dictionary containing batch processing metrics:
                - Model information
                - Timing statistics
                - Individual video results
                - Completion timestamp

        Raises:
            ValueError: If no MP4 files are found in input directory
            Exception: For other processing errors

        Note:
            Metrics are automatically saved to a JSON file in the output directory
        """
        video_files = list(self.settings.input_dir.glob("*.mp4"))
        if not video_files:
            raise ValueError(f"No MP4 files found in {self.settings.input_dir}")

        metrics = BatchMetrics(self.settings.model_name, len(video_files))
        logger.info(f"Processing {len(video_files)} videos...")

        with tqdm(video_files, desc="Processing videos",ascii=" ▖▘▝▗▚▞█ ") as pbar:
            for video_path in pbar:
                try:
                    inference_time = self.process_video(video_path)
                    metrics.add_video_result(video_path.name, inference_time)
                    pbar.set_postfix({"Last inference": f"{inference_time:.2f}s"})
                except Exception as e:
                    logger.error(f"Failed to process {video_path.name}: {str(e)}")

        # Save batch metrics
        metrics_dir = self.settings.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        metrics_path = metrics_dir / f"batch_metrics_{datetime.now():%Y%m%d_%H%M%S}.json"

        summary = metrics.get_summary()
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=4)

        return summary


def main():
    """Main entry point."""
    try:
        settings = UpscalerSettings()

        upscaler = VideoUpscaler(settings)
        summary = upscaler.process_batch()

        # Print summary
        print("\nBatch Processing Summary:")
        print(f"Model: {summary['model_name']}")
        print(f"Total videos processed: {summary['total_videos']}")
        print(f"Total batch time: {summary['total_batch_time']:.2f}s")
        print(f"Average time per video: {summary['average_time_per_video']:.2f}s")
        print("\nIndividual video times:")
        for video in summary["videos"]:
            print(f"{video['name']}: {video['inference_time']:.2f}s")

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
