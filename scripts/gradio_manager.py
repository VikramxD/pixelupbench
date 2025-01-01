"""
Gradio Web Interface for Video Upscaling

Provides a user-friendly web interface for video upscaling using multiple AI models.
Supports real-time progress tracking and comprehensive metrics display.

Features:
    - Multiple model support (RealESRGAN, Swin2SR, AURA-SR)
    - Real-time processing progress
    - Detailed metrics visualization
    - Before/After comparison
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, Dict, Any
from loguru import logger
from upscaler_manager import UpscalerManager
from data_models.config_model import UpscalerConfig
from data_models.metrics_model import BatchMetrics, VideoMetrics, create_video_metrics


def upscale_video(video: gr.File, model_name: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Process and upscale a video using the selected AI model.

    Args:
        video (gr.File): Video file object from Gradio interface
        model_name (str): Name of the upscaling model to use. 
            Options: ["RealESRGAN", "Swin2SR", "AURA-SR"]

    Returns:
        tuple: Contains:
            - str: Path to original video file
            - str: Path to upscaled video file
            - VideoMetrics: Processing metrics including:
                - inference_time: Total processing time
                - original_resolution: Input dimensions
                - upscaled_resolution: Output dimensions
                - input_fps: Original frame rate
                - model_fps: Processing speed
                - ssim: Quality metric (if available)

    Raises:
        ValueError: If video processing fails
        RuntimeError: If model initialization fails
    """
    try:
        logger.info(f"Processing video with {model_name}")
        config = UpscalerConfig()
        manager = UpscalerManager(model_name, config)
        
        # Process video and collect metrics
        output_video, metrics = manager.process_video(video.name)
        
        # Convert raw metrics to standardized format
        video_metrics = create_video_metrics(
            name=Path(video.name).name,
            inference_time=metrics["processing_time"],
            original_resolution=tuple(map(int, metrics["input_resolution"].split("x"))),
            upscaled_resolution=tuple(map(int, metrics["output_resolution"].split("x"))),
            input_fps=metrics["input_fps"],
            model_fps=metrics["model_fps"],
            effective_fps=metrics.get("effective_fps"),
            ssim=metrics.get("ssim")
        )
        
        return video.name, output_video, video_metrics.dict()
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise gr.Error(f"Processing failed: {str(e)}")


# Create the Gradio interface
ui = gr.Interface(
    fn=upscale_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(
            choices=["RealESRGAN", "Swin2SR", "AURA-SR"],
            label="Select Upscaler",
            info="Choose the AI model for upscaling"
        )
    ],
    outputs=[
        gr.Video(label="Original Video"),
        gr.Video(label="Upscaled Video"),
        gr.JSON(
            label="Processing Metrics",
            info="Detailed performance and quality metrics"
        )
    ],
    title="AI Video Upscaler",
    description="""
    Enhance your videos using state-of-the-art AI upscaling models.
    Upload a video and select your preferred upscaling model to begin.
    """,
    article="""
    Supported Models:
    - RealESRGAN: Best for general purpose upscaling
    - Swin2SR: Optimized for detail preservation
    - AURA-SR: Specialized for anime content
    
    Metrics Explanation:
    - inference_time: Total processing duration
    - model_fps: Raw processing speed
    - ssim: Quality metric (higher is better)
    """,
    theme="default"
)


if __name__ == "__main__":
    logger.info("Starting Gradio interface")
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
