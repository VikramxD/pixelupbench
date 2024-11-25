from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional,Literal

class UpscalerSettings(BaseSettings):
    """Configuration settings for video upscaling."""
    
    video_path: Path = Field(
        '/root/pixelupbench/data/realism/low_rel.mp4',  # Making this required
        description="Input video path"
    )
    
    output_dir: Path = Field(
        default=Path("results"),
        description="Output directory"
    )
    
    model_name: Literal[
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "realesr-animevideov3"
    ] = Field(
        default="RealESRGAN_x4plus",
        description="Model name"
    )
    
    scale_factor: int = Field(
        default=4,
        description="The final upsampling scale"
    )
    
    suffix: str = Field(
        default="out",
        description="Suffix of the restored video"
    )
    
    tile_size: int = Field(
        default=0,
        description="Tile size, 0 for no tile during testing"
    )
    
    face_enhance: bool = Field(
        default=False,
        description="Whether to use GFPGAN to enhance face"
    )
    
    use_half_precision: bool = Field(
        default=True,
        description="Use fp16 precision during inference"
    )
    
    gpu_device: int = Field(
        default=0,
        description="GPU device ID"
    )
    
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )

    class Config:
        env_prefix = "UPSCALER_"