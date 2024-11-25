from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field



class AuraSettings(BaseSettings):
    """Configuration settings for AURA-SR upscaling."""
    
    input_dir: Path = Field(
        '/root/pixelupbench/data/realism',
        description="Directory containing input videos"
    )
    output_dir: Path = Field(
        default=Path("results"),
        description="Base directory for outputs"
    )
    gpu_device: int = Field(
        default=0,
        description="GPU device ID"
    )

    class Config:
        env_prefix = "AURA_"
