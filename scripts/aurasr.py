"""
AURA-SR Video Upscaling Pipeline

A production-grade pipeline for processing and upscaling videos using the AURA-SR model.
This module provides comprehensive batch processing capabilities with robust error handling,
metrics collection, and progress monitoring.

Key Features:
    - Automated batch video processing with 4x upscaling
    - Real-time progress tracking with ETA
    - Comprehensive performance metrics and JSON reporting
    - GPU acceleration with CUDA support
    - Robust error handling and logging
    - Automatic output organization

Technical Specifications:
    - Input: MP4 video files
    - Output: 4x upscaled MP4 videos
    - GPU Memory: ~4GB for 1080p video
    - Supported Formats: RGB/BGR video frames
    - Processing: Frame-by-frame with batched inference

Dependencies:
    - torch>=1.7.0
    - opencv-python>=4.5.0
    - numpy>=1.19.0
    - tqdm>=4.45.0
    - loguru>=0.5.0
    - aura-sr>=2.0.0

Example:
    >>> from pathlib import Path
    >>> from configs.aurasr_settings import AuraSettings
    >>> 
    >>> settings = AuraSettings(input_dir=Path("videos"))
    >>> upscaler = AuraUpscaler(settings)
    >>> metrics = upscaler.process_batch()
    >>> print(f"Processed {metrics['total_videos']} videos")

Notes:
    - Ensure CUDA toolkit is properly installed for GPU acceleration
    - Input directory should contain only MP4 files for processing
    - Output directory structure is automatically created
    - Metrics are saved in JSON format with timestamps
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
    Comprehensive metrics tracking system for AURA-SR batch processing operations.
    
    Collects, aggregates, and reports performance metrics across multiple video
    processing operations, providing both individual and batch-level statistics.
    
    Attributes:
        start_time (float): Unix timestamp marking batch start
        num_videos (int): Total number of videos in current batch
        videos_processed (List[Dict]): List of processed video results containing:
            - name: Video filename
            - inference_time: Processing duration in seconds
    
    Methods:
        add_video_result: Record metrics for a single video
        get_summary: Generate comprehensive batch processing report
    
    Example:
        >>> metrics = BatchMetrics(num_videos=3)
        >>> metrics.add_video_result("video1.mp4", 45.2)
        >>> summary = metrics.get_summary()
        >>> print(f"Average processing time: {summary['average_time_per_video']:.2f}s")
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

    def add_video_result(self, video_name: str, inference_time: float,
                        original_resolution: tuple, upscaled_resolution: tuple,
                        ssim: float) -> None:
        """
        Add processing results for a single video.

        Args:
            video_name: Name of the processed video file
            inference_time: Time taken to process the video in seconds
            original_resolution: Tuple of (width, height) for input video
            upscaled_resolution: Tuple of (width, height) for output video
            ssim: Average Structural Similarity Index for the video
        """
        self.videos_processed.append({
            "name": video_name,
            "inference_time": inference_time,
            "original_resolution": f"{original_resolution[0]}x{original_resolution[1]}",
            "upscaled_resolution": f"{upscaled_resolution[0]}x{upscaled_resolution[1]}",
            "ssim": round(ssim, 3)
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
    
    Implements a complete video processing pipeline with GPU acceleration,
    batch processing capabilities, and comprehensive error handling.
    
    Attributes:
        settings (AuraSettings): Configuration parameters for processing
        device (torch.device): CUDA device for GPU acceleration
        model (AuraSR): Loaded instance of AURA-SR model
    
    Key Features:
        - Automatic GPU device management
        - Batch video processing with progress tracking
        - Comprehensive metrics collection
        - Organized output directory structure
        - Error resilience with detailed logging
    
    Methods:
        process_video: Process single video with progress tracking
        process_batch: Process multiple videos with metrics collection
    
    Example:
        >>> settings = AuraSettings(
        ...     input_dir=Path("videos"),
        ...     output_dir=Path("results"),
        ...     gpu_device=0
        ... )
        >>> upscaler = AuraUpscaler(settings)
        >>> metrics = upscaler.process_batch()
    
    Raises:
        RuntimeError: If GPU initialization fails
        ValueError: If video processing fails
        Exception: For other processing errors
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

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two frames."""
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

    def process_video(self, video_path: Path) -> tuple:
        """
        Process a single video through the AURA-SR upscaling pipeline.
        
        Args:
            video_path (Path): Path to input MP4 video file
        
        Returns:
            tuple: Contains (inference_time, original_resolution, upscaled_resolution, ssim)
        """
        start_time = time.time()

        output_dir = self.settings.output_dir / "AURA-SR"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_output.mp4"

        # Open video and get original resolution
        cap = cv2.VideoCapture(str(video_path))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get output dimensions
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read video: {video_path}")

        # Convert test frame and get dimensions
        test_output = self.model.upscale_4x(frame)
        if hasattr(test_output, 'convert'):
            test_output = np.array(test_output.convert('RGB'))
        up_height, up_width = test_output.shape[:2]

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (up_width, up_height))

        # Process video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        ssim_values = []
        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB before processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame and ensure numpy array format
                upscaled_frame = self.model.upscale_4x(frame_rgb)
                if hasattr(upscaled_frame, 'convert'):
                    upscaled_frame = np.array(upscaled_frame.convert('RGB'))
                
                # Convert back to BGR for OpenCV
                upscaled_frame = cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR)
                
                # Calculate SSIM
                frame_resized = cv2.resize(frame, (up_width, up_height))
                ssim = self._calculate_ssim(frame_resized, upscaled_frame)
                ssim_values.append(ssim)
                
                writer.write(upscaled_frame)
                pbar.update(1)

        cap.release()
        writer.release()

        inference_time = time.time() - start_time
        average_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0

        return (
            inference_time,
            (orig_width, orig_height),
            (up_width, up_height),
            average_ssim
        )

    def process_batch(self) -> Dict[str, Any]:
        """
        Execute batch processing of multiple videos with comprehensive metrics.
        
        Performs parallel processing of all MP4 videos in the input directory,
        with detailed metrics collection and error handling per video.
        
        Returns:
            Dict[str, Any]: Comprehensive metrics including:
                - model_name: "AURA-SR"
                - total_videos: Number processed
                - total_batch_time: Total processing duration
                - average_time_per_video: Mean processing time
                - videos: List of individual video results
                - timestamp: ISO format completion time
        
        Raises:
            ValueError: If no MP4 files found in input directory
            Exception: For batch processing failures
        
        Notes:
            - Failed videos are logged but don't stop batch processing
            - Metrics are automatically saved to JSON with timestamp
            - Progress bar shows real-time processing status
        """
        video_files = list(self.settings.input_dir.glob("*.mp4"))
        if not video_files:
            raise ValueError(f"No MP4 files found in {self.settings.input_dir}")

        metrics = BatchMetrics(len(video_files))
        logger.info(f"Processing {len(video_files)} videos with AURA-SR...")

        with tqdm(video_files, desc="Processing batch", ascii=" ▖▘▝▗▚▞█ ") as pbar:
            for video_path in pbar:
                try:
                    inference_time, orig_res, up_res, ssim = self.process_video(video_path)
                    metrics.add_video_result(video_path.name, inference_time,
                                          orig_res, up_res, ssim)
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
