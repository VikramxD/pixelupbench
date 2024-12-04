"""
OpenModelDB Video Inference Module

Upscaling videos using OpenModelDB models.
This module provides comprehensive video processing capabilities

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
from typing import Dict, Any, Optional, List
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from image_gen_aux import UpscaleWithModel
import numpy as np
from PIL import Image

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
        ...     path="Phips/4xHFA2kLUDVAESwinIR_light",
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
            self.model = UpscaleWithModel.from_pretrained(model_config.path).to(
                f"cuda:{settings.gpu_device}"
            )
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
            - DEBUG: Frame processing details
        """
        try:
            logger.debug(f"Input frame shape: {frame.shape}")

            # Convert numpy array to PIL Image
            if len(frame.shape) == 2:
                # Convert grayscale to RGB
                frame_pil = Image.fromarray(frame).convert("RGB")
            else:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

            logger.debug(f"PIL Image size: {frame_pil.size} mode: {frame_pil.mode}")

            # Process the frame
            processed = self.model(
                frame_pil,
                tiling=True,
                tile_width=self.tile_size,
                tile_height=self.tile_size,
            )

            # Convert back to numpy array in BGR format for OpenCV
            if isinstance(processed, Image.Image):
                processed = cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)
            elif isinstance(processed, np.ndarray):
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

            logger.debug(f"Output frame shape: {processed.shape}")
            return processed

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Frame processing failed: {str(e)}")

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate SSIM between two frames.
        
        Args:
            img1: First frame
            img2: Second frame
            
        Returns:
            float: SSIM value between 0 and 1
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

    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """Process complete video with metrics collection."""
        logger.info(f"Starting processing of video: {video_path}")
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # Get original video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            input_fps = int(cap.get(cv2.CAP_PROP_FPS))
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Read and process first frame to get dimensions
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Failed to read first frame")

            # Start model inference timing
            model_start_time = time.time()
            test_output = self.process_frame(frame)
            up_height, up_width = test_output.shape[:2]

            # Initialize video writer
            output_path = self.output_dir / f"{video_path.stem}_upscaled.mp4"
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                input_fps,
                (up_width, up_height)
            )

            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Process all frames and calculate SSIM
            ssim_values = []
            with tqdm(total=total_frames, desc=video_path.name) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed = self.process_frame(frame)
                    writer.write(processed)

                    # Calculate SSIM
                    frame_resized = cv2.resize(frame, (up_width, up_height))
                    ssim = self._calculate_ssim(frame_resized, processed)
                    ssim_values.append(ssim)

                    pbar.update(1)

            model_time = time.time() - model_start_time
            inference_time = time.time() - start_time
            average_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
            model_fps = total_frames / model_time  # Raw model inference speed

            metrics = {
                "video_name": video_path.name,
                "model_path": self.model_path,
                "inference_time": inference_time,
                "original_resolution": f"{orig_width}x{orig_height}",
                "upscaled_resolution": f"{up_width}x{up_height}",
                "original_fps": input_fps,
                "model_fps": round(model_fps, 2),
                "ssim": round(average_ssim, 3)
            }

            logger.info(
                f"Completed processing {video_path.name} - "
                f"Time: {inference_time:.2f}s, "
                f"Original FPS: {input_fps}, "
                f"Model FPS: {model_fps:.2f}, "
                f"SSIM: {metrics['ssim']:.3f}"
            )

        finally:
            if "cap" in locals():
                cap.release()
            if "writer" in locals():
                writer.release()

        return metrics


class BatchMetrics:
    """
    Tracks and manages performance metrics for batch processing operations.

    Attributes:
        start_time (float): Batch processing start time
        model_name (str): Name of the model being used
        videos (List[Dict]): List of processed video results
    """

    def __init__(self, model_name: str):
        """Initialize batch metrics tracker."""
        self.start_time = time.time()
        self.model_name = model_name
        self.videos: List[Dict[str, Any]] = []

    def add_video_result(self, video_name: str, inference_time: float,
                        original_resolution: tuple, upscaled_resolution: tuple,
                        ssim: float) -> None:
        """Add processing results for a single video."""
        self.videos.append({
            "name": video_name,
            "inference_time": inference_time,
            "original_resolution": f"{original_resolution[0]}x{original_resolution[1]}",
            "upscaled_resolution": f"{upscaled_resolution[0]}x{upscaled_resolution[1]}",
            "ssim": round(ssim, 3)
        })

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive batch processing summary."""
        total_time = time.time() - self.start_time
        return {
            "model_name": self.model_name,
            "total_videos": len(self.videos),
            "total_batch_time": total_time,
            "average_time_per_video": (total_time / len(self.videos) if self.videos else 0),
            "videos": self.videos,
            "timestamp": datetime.now().isoformat(),
        }


def main() -> None:
    """
    Main entry point for OpenModelDB video processing system.
    
    Environment Variables:
        UPSCALER_INPUT_DIR: Input directory path
        UPSCALER_OUTPUT_DIR: Output directory path
        UPSCALER_GPU_DEVICE: GPU device ID
        UPSCALER_LOG_LEVEL: Logging level
    """
    try:
        # Initialize settings with default values
        settings = UpscalerSettings(
            input_dir=Path("/root/pixelupbench/test/test-real/"),
            models={
                "4xHFA2kLUDVAESwinIR_light": ModelConfig(
                    path= "Phips/4xHFA2kLUDVAESwinIR_light",
                    tile_size=1024
                )
            }
        )
        settings.setup_logging()
        logger.info("Starting OpenModelDB video processing")
        
        # Process videos with each configured model
        for model_name, model_config in settings.models.items():
            logger.info(f"Initializing processor with model: {model_name}")
            processor = VideoProcessor(settings, model_config)
            
            batch_metrics = BatchMetrics(model_name)
            
            # Process all MP4 files in input directory
            video_files = list(settings.input_dir.glob("*.mp4"))
            if not video_files:
                logger.warning(f"No MP4 files found in {settings.input_dir}")
                continue
                
            logger.info(f"Found {len(video_files)} videos to process")
            
            # Process each video
            for video_path in video_files:
                try:
                    logger.info(f"Processing {video_path.name} with {model_name}")
                    start_time = time.time()
                    metrics = processor.process_video(video_path)
                    inference_time = time.time() - start_time
                    batch_metrics.add_video_result(
                        video_path.name,
                        inference_time,
                        tuple(map(int, metrics["original_resolution"].split("x"))),
                        tuple(map(int, metrics["upscaled_resolution"].split("x"))),
                        metrics["ssim"]
                    )
                except Exception as e:
                    logger.error(f"Failed to process {video_path.name}: {str(e)}", exc_info=True)
                    continue
            
            # Save only batch metrics
            metrics_dir = settings.output_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            metrics_path = metrics_dir / f"{model_name}_batch_metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
            
            summary = batch_metrics.get_summary()
            with open(metrics_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Display summary
            print(f"\nProcessing Summary for {model_name}:")
            print(f"Total videos processed: {summary['total_videos']}")
            print(f"Total batch time: {summary['total_batch_time']:.2f}s")
            print(f"Average time per video: {summary['average_time_per_video']:.2f}s")
            print("\nIndividual video results:")
            for video in summary['videos']:
                print(f"{video['name']}:")
                print(f"  Resolution: {video['original_resolution']} → {video['upscaled_resolution']}")
                print(f"  SSIM: {video['ssim']:.3f}")
                print(f"  Time: {video['inference_time']:.2f}s")
            
            logger.info(f"Completed batch processing with {model_name}")
            
    except Exception as e:
        logger.error("Fatal error in main processing loop", exc_info=True)
        raise
    finally:
        logger.info("OpenModelDB processing completed")


if __name__ == "__main__":
    main()
