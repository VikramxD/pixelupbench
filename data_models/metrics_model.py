"""
Unified Metrics Data Model

Provides standardized data structures for collecting and exporting upscaling metrics
across different models and processing pipelines.

Key Features:
    - Consistent metrics format across all upscalers
    - Pydantic-based validation
    - JSON serialization support
    - Comprehensive processing statistics
    - Hierarchical metrics organization
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class VideoMetrics(BaseModel):
    """Individual video processing metrics."""
    
    name: str = Field(..., description="Name of the processed video file")
    inference_time: float = Field(..., description="Processing time in seconds")
    original_resolution: str = Field(..., description="Original video dimensions (WxH)")
    upscaled_resolution: str = Field(..., description="Upscaled video dimensions (WxH)")
    input_fps: float = Field(..., description="Original video frame rate")
    model_fps: float = Field(..., description="Model inference speed (frames/second)")
    effective_fps: Optional[float] = Field(None, description="Actual processing speed including I/O")
    ssim: Optional[float] = Field(None, description="Structural Similarity Index")
    psnr: Optional[float] = Field(None, description="Peak Signal-to-Noise Ratio")


class BatchMetrics(BaseModel):
    """Comprehensive batch processing metrics."""
    
    model_name: str = Field(..., description="Name of the upscaling model")
    total_videos: int = Field(..., description="Number of videos processed")
    total_batch_time: float = Field(..., description="Total processing time in seconds")
    average_time_per_video: float = Field(..., description="Mean processing time per video")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing completion time")
    videos: List[VideoMetrics] = Field(default_factory=list, description="Individual video results")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


def create_video_metrics(
    name: str,
    inference_time: float,
    original_resolution: tuple,
    upscaled_resolution: tuple,
    input_fps: float,
    model_fps: float,
    effective_fps: Optional[float] = None,
    ssim: Optional[float] = None,
    psnr: Optional[float] = None
) -> VideoMetrics:
    """
    Create a standardized video metrics entry.
    
    Args:
        name: Video filename
        inference_time: Processing duration in seconds
        original_resolution: Tuple of (width, height) for input
        upscaled_resolution: Tuple of (width, height) for output
        input_fps: Original video frame rate
        model_fps: Raw model inference speed
        effective_fps: Optional actual processing speed
        ssim: Optional Structural Similarity Index
        psnr: Optional Peak Signal-to-Noise Ratio
    
    Returns:
        VideoMetrics: Standardized metrics object
    """
    return VideoMetrics(
        name=name,
        inference_time=round(inference_time, 2),
        original_resolution=f"{original_resolution[0]}x{original_resolution[1]}",
        upscaled_resolution=f"{upscaled_resolution[0]}x{upscaled_resolution[1]}",
        input_fps=round(input_fps, 2),
        model_fps=round(model_fps, 2),
        effective_fps=round(effective_fps, 2) if effective_fps is not None else None,
        ssim=round(ssim, 3) if ssim is not None else None,
        psnr=round(psnr, 2) if psnr is not None else None
    )


def create_batch_metrics(model_name: str) -> BatchMetrics:
    """
    Create a new batch metrics collector.
    
    Args:
        model_name: Name of the upscaling model
    
    Returns:
        BatchMetrics: New batch metrics instance
    """
    return BatchMetrics(
        model_name=model_name,
        total_videos=0,
        total_batch_time=0.0,
        average_time_per_video=0.0
    )