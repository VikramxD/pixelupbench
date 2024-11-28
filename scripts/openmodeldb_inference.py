"""
OpenModelDB Video Inference Module

Upscaling videos using OpenModelDB models.
This module provides comprehensive video processing capabilities.

Key Features:
    - Multi-model support with dynamic loading
    - Tile-based processing for memory efficiency
    - Comprehensive metrics collection and JSON export
    - Detailed logging with error tracing
    - Progress tracking with ETA
    - GPU acceleration with CUDA

Technical Specifications:
    - Input: MP4 video files
    - Output: Upscaled MP4 videos with configurable scale factor
    - GPU Memory: Usage based on tile size configuration
    - Processing: Frame-by-frame with tiled inference
    - Metrics: FPS, timing, and resource utilization

Dependencies:
    - torch>=1.7.0
    - opencv-python>=4.5.0
    - loguru>=0.5.0
    - tqdm>=4.45.0
    - numpy>=1.19.0

Example:
    >>> from pathlib import Path
    >>> from configs.openmodeldb_settings import UpscalerSettings, ModelConfig
    >>> 
    >>> settings = UpscalerSettings(input_dir=Path("videos"))
    >>> model_config = ModelConfig(path="Phips/4xNomosWebPhoto_RealPLKSR")
    >>> processor = VideoProcessor(settings, model_config)
    >>> metrics = processor.process_video(Path("input.mp4"))
"""

import cv2
import torch
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from image_gen_aux import UpscaleWithModel

from configs.openmodeldb_settings import ModelConfig, UpscalerSettings


class VideoProcessor:
    """
    video processing class using OpenModelDB upscaling models.
    
    Implements a complete video processing pipeline with GPU acceleration,
    tile-based processing, and comprehensive metrics collection.
    
    Attributes:
        settings (UpscalerSettings): Global configuration parameters
        model_path (str): HuggingFace model path
        tile_size (int): Processing tile dimensions
        model: Loaded upscaling model instance
        output_dir (Path): Directory for processed videos
    
    Technical Details:
        - Uses tiled processing to handle high-resolution videos
        - Implements CUDA acceleration for GPU processing
        - Maintains processing metrics and logs
        - Handles video codec conversion and format management
    
    Example:
        >>> settings = UpscalerSettings(input_dir=Path("videos"))
        >>> model_config = ModelConfig(
        ...     path="Phips/4xNomosWebPhoto_RealPLKSR",
        ...     tile_size=1024
        ... )
        >>> processor = VideoProcessor(settings, model_config)
        >>> metrics = processor.process_video(Path("video.mp4"))
    """

    def __init__(self, settings: UpscalerSettings, model_config: ModelConfig):
        """
        Initialize video processor with settings and model configuration.
        
        Args:
            settings: Global upscaler configuration
            model_config: Model-specific configuration
            
        Raises:
            RuntimeError: If CUDA GPU is not available
            ValueError: If model loading fails
            
        Logs:
            - INFO: Model loading status
            - ERROR: Model loading failures
            - DEBUG: Initialization details
        """
        logger.debug(f"Initializing VideoProcessor with model {model_config.path}")
        self.settings = settings
        self.model_path = model_config.path
        self.tile_size = model_config.tile_size
        
        if not torch.cuda.is_available():
            logger.error("CUDA GPU not detected")
            raise RuntimeError("CUDA-capable GPU is required for video processing")
            
        self.output_dir = settings.output_dir / Path(model_config.path).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {self.output_dir}")
        
        logger.info(f"Loading model: {model_config.path}")
        try:
            self.model = UpscaleWithModel.from_pretrained(
                model_config.path
            ).to(f"cuda:{settings.gpu_device}")
            logger.info(f"Successfully loaded model {model_config.path}")
            logger.debug(f"Model loaded on GPU device {settings.gpu_device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_config.path}: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")

    def process_frame(self, frame: "np.ndarray") -> "np.ndarray":
        """
        Process a single frame using tile-based upscaling.
        
        Implements memory-efficient processing using configurable tile sizes
        and GPU acceleration.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            np.ndarray: Processed frame at higher resolution
            
        Raises:
            RuntimeError: If frame processing fails
            
        Logs:
            - ERROR: Frame processing failures
            - DEBUG: Tile processing details
        """
        try:
            logger.debug(f"Processing frame with tile size {self.tile_size}")
            return self.model(
                frame,
                tiling=True,
                tile_width=self.tile_size,
                tile_height=self.tile_size
            )
        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            raise RuntimeError(f"Frame processing failed: {str(e)}")

    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Process complete video with metrics collection.
        
        Handles the entire video processing pipeline including:
        1. Video loading and validation
        2. Frame-by-frame processing
        3. Output generation
        4. Metrics collection
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dict containing processing metrics:
                - video_name: Name of processed file
                - model_path: Used model path
                - start_time: Processing start timestamp
                - processed_frames: Number of frames processed
                - total_frames: Total video frames
                - fps: Processing speed
                - inference_time: Total processing time
                - total_time: Total execution time
            
        Raises:
            ValueError: If video cannot be read
            RuntimeError: If processing fails
            
        Logs:
            - INFO: Processing progress and results
            - ERROR: Processing failures
            - DEBUG: Detailed processing information
        """
        logger.info(f"Starting processing of video: {video_path}")
        start_time = time.time()
        metrics = {
            "video_name": video_path.name,
            "model_path": self.model_path,
            "start_time": datetime.now().isoformat(),
            "processed_frames": 0,
            "total_frames": 0,
            "fps": 0,
            "inference_time": 0
        }

        try:
            # Setup video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                raise ValueError(f"Failed to open video: {video_path}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            metrics["total_frames"] = total_frames
            logger.debug(f"Video details - Frames: {total_frames}, FPS: {fps}")
            
            # Process first frame to get dimensions
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read first frame from {video_path}")
                raise ValueError(f"Failed to read first frame from {video_path}")
            
            # Get output dimensions
            test_output = self.process_frame(frame)
            h, w = test_output.shape[:2]
            logger.debug(f"Output dimensions: {w}x{h}")
            
            # Setup video writer
            output_path = self.output_dir / f"{video_path.stem}_upscaled.mp4"
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )
            logger.debug(f"Created output video writer: {output_path}")

            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process all frames
            inference_start = time.time()
            with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    processed = self.process_frame(frame)
                    writer.write(processed)
                    metrics["processed_frames"] += 1
                    pbar.update(1)

            inference_time = time.time() - inference_start
            metrics.update({
                "inference_time": inference_time,
                "total_time": time.time() - start_time,
                "fps": metrics["processed_frames"] / inference_time
            })

            logger.info(
                f"Completed processing {video_path.name} - "
                f"FPS: {metrics['fps']:.2f}, "
                f"Time: {metrics['total_time']:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {str(e)}", exc_info=True)
            raise
        finally:
            if 'cap' in locals():
                cap.release()
            if 'writer' in locals():
                writer.release()

        self.save_metrics(metrics, video_path)
        return metrics

    def save_metrics(self, metrics: Dict[str, Any], video_path: Path) -> None:
        """
        Save processing metrics to JSON file.
        
        Creates a timestamped JSON file containing all processing metrics
        and execution statistics.
        
        Args:
            metrics: Collection of processing metrics
            video_path: Path to processed video
            
        Logs:
            - INFO: Metrics saving status
            - DEBUG: Metrics file location
            - ERROR: Saving failures
        """
        try:
            metrics_dir = self.settings.output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            metrics_path = metrics_dir / f"{video_path.stem}_{Path(self.model_path).stem}_metrics.json"
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            logger.info(f"Saved processing metrics to {metrics_path}")
            logger.debug(f"Metrics content: {metrics}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
