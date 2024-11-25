"""
AURA-SR Video Upscaling Pipeline

A Pipeline for benchmarking processing videos using the AURA-SR upscaling model.
Provides comprehensive metrics tracking, error handling, and progress monitoring for
large-scale video processing operations.

Features:
    - Automated batch video processing
    - Real-time progress tracking
    - Performance metrics collection and export
    - Comprehensive error handling
    - GPU acceleration support
    - JSON-based metrics reporting

Dependencies:
    - AURA-SR model
    - PyTorch with CUDA support
    - OpenCV
    - NumPy
    - tqdm for progress tracking
    - loguru for logging

Typical usage:
    settings = AuraSettings(input_dir=Path("videos"))
    upscaler = AuraUpscaler(settings)
    metrics = upscaler.process_batch()
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import cv2
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from aura_sr import AuraSR
from configs.aurasr_settings import AuraSettings

class BatchMetrics:
    """
    Tracks and manages performance metrics for AURA-SR batch processing operations.

    This class handles the collection and aggregation of processing metrics across
    multiple videos, providing summary statistics and individual video results.

    Attributes:
        start_time (float): Unix timestamp when batch processing started
        num_videos (int): Total number of videos in the batch
        videos_processed (List[Dict[str, Union[str, float]]]): List of processed video results

    Example:
        metrics = BatchMetrics(num_videos=5)
        metrics.add_video_result("video1.mp4", 10.5)
        summary = metrics.get_summary()
    """

    def __init__(self, num_videos: int) -> None:
        """
        Initialize batch metrics tracker.

        Args:
            num_videos: Total number of videos to be processed
        """
        self.start_time: float = time.time()
        self.num_videos: int = num_videos
        self.videos_processed: List[Dict[str, Union[str, float]]] = []

    def add_video_result(self, video_name: str, inference_time: float) -> None:
        """
        Add processing results for a single video.

        Args:
            video_name: Name of the processed video file
            inference_time: Time taken to process the video in seconds
        """
        self.videos_processed.append({
            "name": video_name,
            "inference_time": inference_time
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive batch processing summary.

        Returns:
            Dict containing:
                - model_name: Fixed as "AURA-SR"
                - total_videos: Number of videos processed
                - total_batch_time: Total processing time in seconds
                - average_time_per_video: Average processing time per video
                - videos: List of individual video results
                - timestamp: ISO format timestamp of completion
        """
        total_time = time.time() - self.start_time
        return {
            "model_name": "AURA-SR",
            "total_videos": self.num_videos,
            "total_batch_time": total_time,
            "average_time_per_video": total_time / self.num_videos,
            "videos": self.videos_processed,
            "timestamp": datetime.now().isoformat(),
        }


class AuraUpscaler:
    """
    Production-grade video upscaling system using AURA-SR model.

    This class implements a comprehensive video upscaling pipeline using the AURA-SR
    model, supporting batch processing with metrics collection and error handling.

    Attributes:
        settings (AuraSettings): Configuration settings for the upscaler
        device (torch.device): GPU device for processing
        model (AuraSR): Loaded AURA-SR model instance

    Example:
        settings = AuraSettings(input_dir=Path("videos"))
        upscaler = AuraUpscaler(settings)
        metrics = upscaler.process_batch()
    """

    def __init__(self, settings: AuraSettings) -> None:
        """
        Initialize the AURA-SR upscaler with provided settings.

        Args:
            settings: Configuration settings for video processing

        Raises:
            RuntimeError: If GPU initialization fails
            Exception: If model loading fails
        """
        self.settings = settings
        self.device = torch.device(f"cuda:{settings.gpu_device}")
        self.model = self._setup_model()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AURA-SR initialization complete")

    def _setup_model(self) -> AuraSR:
        """
        Initialize and load the AURA-SR model.

        Returns:
            Loaded AURA-SR model instance

        Raises:
            Exception: If model loading fails with detailed error message
        """
        try:
            model = AuraSR.from_pretrained(model_id='fal/AuraSR-v2')
            logger.info("AURA-SR model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load AURA-SR model: {str(e)}")
            raise

    def process_video(self, video_path: Path) -> float:
        """
        Process a single video through the AURA-SR upscaling pipeline.

        Handles the complete processing of one video including:
        1. Video loading and validation
        2. Output path setup
        3. Frame-by-frame processing with proper format conversion
        4. Progress tracking

        Args:
            video_path: Path to the input video file

        Returns:
            float: Total inference time in seconds

        Raises:
            ValueError: If video cannot be read
            Exception: For other processing errors
        """
        start_time = time.time()

        output_dir = self.settings.output_dir / "AURA-SR"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_output.mp4"

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get output dimensions
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read video: {video_path}")

        # Convert test frame and get dimensions
        test_output = self.model.upscale_4x(frame)
        # Convert PIL Image to numpy array if necessary
        if hasattr(test_output, 'convert'):
            test_output = np.array(test_output.convert('RGB'))
        h, w = test_output.shape[:2]

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        # Process video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame and ensure numpy array format
                upscaled_frame = self.model.upscale_4x(frame)
                if hasattr(upscaled_frame, 'convert'):
                    upscaled_frame = np.array(upscaled_frame.convert('RGB'))
                
                # OpenCV expects BGR format
                if upscaled_frame.shape[-1] == 3:  # If RGB
                    upscaled_frame = cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR)
                
                writer.write(upscaled_frame)
                pbar.update(1)

        cap.release()
        writer.release()

        inference_time = time.time() - start_time
        return inference_time

    def process_batch(self) -> Dict[str, Any]:
        """
        Process multiple videos in batch mode with comprehensive metrics collection.

        Processes all MP4 videos in the input directory, collecting metrics
        and handling errors for each video individually. Failed videos are
        logged but don't stop the batch processing.

        Returns:
            Dict containing batch processing metrics:
                - Model information
                - Timing statistics
                - Individual video results
                - Completion timestamp

        Raises:
            ValueError: If no MP4 files are found in input directory
            Exception: For other processing errors

        Note:
            Metrics are automatically saved to a JSON file in the output directory
            with timestamp in the filename.
        """
        video_files = list(self.settings.input_dir.glob("*.mp4"))
        if not video_files:
            raise ValueError(f"No MP4 files found in {self.settings.input_dir}")

        metrics = BatchMetrics(len(video_files))
        logger.info(f"Processing {len(video_files)} videos with AURA-SR...")

        with tqdm(video_files, desc="Processing batch",ascii= " ▖▘▝▗▚▞█ ") as pbar:
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
        metrics_path = metrics_dir / f"aura_batch_metrics_{datetime.now():%Y%m%d_%H%M%S}.json"

        summary = metrics.get_summary()
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=4)

        return summary


def main() -> None:
    """
    Main entry point for the AURA-SR video processing system.

    Handles:
        1. Settings initialization
        2. Upscaler creation
        3. Batch processing execution
        4. Results display
        5. Error handling

    Raises:
        Exception: If batch processing fails, with detailed error logging
    """
    try:
        settings = AuraSettings()
        upscaler = AuraUpscaler(settings)
        summary = upscaler.process_batch()
        print("\nAURA-SR Batch Processing Summary:")
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
