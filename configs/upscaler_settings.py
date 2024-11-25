from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class UpscalerSettings(BaseSettings):
    """Configuration settings for batch video upscaling."""

    input_dir: Path = Field(Path('/root/pixelupbench/data/realism'),description="Directory containing input videos")
    output_dir: Path = Field(default=Path("../results"), description="Base directory for outputs")
    model_name: str = Field(default="RealESRGAN_x4plus", description="Model name")
    scale_factor: int = Field(default=4)
    tile_size: int = Field(default=0)
    face_enhance: bool = Field(default=False)
    use_half_precision: bool = Field(default=True)
    gpu_device: int = Field(default=0)

    class Config:
        env_prefix = "UPSCALER_"
