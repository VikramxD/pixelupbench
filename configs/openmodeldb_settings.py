"""
Configuration module for OpenModelDB Models

This module provides configuration management and validation for the video upscaling
system using OpenModelDB models. It handles model validation, system settings,
and logging configuration.

The module uses Pydantic for settings management and validation, ensuring type safety
and proper configuration validation at runtime.

Classes:
    ModelConfig: Configuration settings for individual upscaling models
    UpscalerSettings: Global configuration settings for the upscaling system

Functions:
    list_available_models: Returns a sorted list of all available model paths

Constants:
    VALID_MODELS (set): Set of validated and approved model paths

Example:
    >>> settings = UpscalerSettings(
    ...     input_dir="videos/",
    ...     models={
    ...         "4x_upscaler": ModelConfig(
    ...             path="Phips/4xNomosWebPhoto_RealPLKSR",
    ...             tile_size=1024
    ...         )
    ...     }
    ... )
    >>> settings.setup_logging()
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pathlib import Path
from typing import Dict, List
from loguru import logger

# List of validated models
VALID_MODELS = {
    "Phips/1xDeH264_realplksr",
    "Phips/1xDeJPG_HAT",
    "Phips/1xDeJPG_OmniSR",
    "Phips/2xHFA2kCompact",
    "Phips/2xHFA2kOmniSR",
    "Phips/4xNomosWebPhoto_RealPLKSR",
    "Phips/4xNomos8k_atd_jpg",
    "Phips/4xNomosWebPhoto_atd",
    "Phips/4xRealWebPhoto_v4_dat2",
    "Phips/4xLSDIRDAT",
    "Phips/4xLSDIRCompactR3",
    "Phips/4xNomos8kSCHAT-L"
}


class ModelConfig(BaseSettings):
    """
    Configuration settings for individual upscaling models.

    This class validates and stores settings for specific upscaling models,
    ensuring proper configuration before model initialization.

    Attributes:
        path (str): HuggingFace model path in format "owner/model_name".
            Must be one of the pre-validated models in VALID_MODELS.
        tile_size (int): Size of processing tiles for image segmentation.
            Must be a positive multiple of 64 to ensure proper processing.
            Larger values use more memory but may be faster. Default: 1024.

    Raises:
        ValueError: If model path is not in VALID_MODELS or tile_size is invalid.

    Example:
        >>> model_config = ModelConfig(
        ...     path="Phips/4xNomosWebPhoto_RealPLKSR",
        ...     tile_size=1024
        ... )
    """

    path: str = Field(..., description="HuggingFace model path")
    tile_size: int = Field(1024, description="Tile size for processing")

    @field_validator("path")
    def validate_model_path(cls, v: str) -> str:
        """Validate that the model path exists in the approved list."""
        if v not in VALID_MODELS:
            raise ValueError(
                f"Model {v} is not in the list of valid models. "
                f"Please choose from: {sorted(VALID_MODELS)}"
            )
        return v

    @field_validator("tile_size")
    def validate_tile_size(cls, v: int) -> int:
        """Validate tile size is positive and multiple of 64."""
        if v <= 0 or v % 64 != 0:
            raise ValueError("Tile size must be positive and a multiple of 64")
        return v


class UpscalerSettings(BaseSettings):
    """
    Global configuration settings for the video upscaling system.

    This class manages all configuration aspects of the upscaling system,
    including input/output paths, GPU settings, logging configuration,
    and model configurations.

    Attributes:
        input_dir (Path): Directory containing input videos for processing.
            Must exist at runtime.
        output_dir (Path): Base directory for all output files including:
            - Upscaled videos
            - Metrics and analysis
            - Log files
            Defaults to "results" in current directory.
        gpu_device (int): CUDA GPU device ID to use for processing.
            Default: 0 (first GPU).
        log_level (str): Logging level for both file and console output.
            Accepts standard logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
            Default: "INFO".
        models (Dict[str, ModelConfig]): Dictionary mapping model names to their
            configurations. Default includes 4xNomosWebPhoto_RealPLKSR model.

    Environment Variables:
        All settings can be overridden using environment variables with the
        prefix UPSCALER_. For example:
        - UPSCALER_INPUT_DIR
        - UPSCALER_GPU_DEVICE
        - UPSCALER_LOG_LEVEL

    Raises:
        ValueError: If input_dir doesn't exist or other validation fails.

    Example:
        >>> settings = UpscalerSettings(
        ...     input_dir=Path("videos"),
        ...     output_dir=Path("results"),
        ...     gpu_device=0,
        ...     log_level="INFO"
        ... )
        >>> settings.setup_logging()
    """

    input_dir: Path = Field(..., description="Directory containing input videos")
    output_dir: Path = Field(
        default=Path("../results"), description="Base directory for outputs"
    )
    gpu_device: int = Field(default=0, description="GPU device ID")
    log_level: str = Field(default="INFO", description="Logging level")
    models: Dict[str, ModelConfig] = Field(
        default={
            "4xLSDIRDAT": ModelConfig(
                path="Phips/4xLSDIRDAT"
            )
        }
    )

    @field_validator("input_dir")
    def validate_input_dir(cls, v: Path) -> Path:
        """Validate input directory exists."""
        if not v.exists():
            raise ValueError(f"Input directory {v} does not exist")
        return v

    def setup_logging(self) -> None:
        """
        Configure logging system with file and console outputs.

        Creates log directory if it doesn't exist and sets up loguru logger
        with appropriate formats and retention policies.
        """
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "upscaler.log"

        # Remove default logger and set up new configuration
        logger.remove()

        # File logging
        logger.add(
            log_file,
            rotation="100 MB",
            retention="30 days",
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
        )

        # Console logging
        logger.add(
            lambda msg: print(msg, flush=True),
            level=self.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
            colorize=True,
        )

    class Config:
        env_prefix = "UPSCALER_"


def list_available_models() -> List[str]:
    """
    Get a sorted list of all available upscaling models.

    Returns a list of validated model paths that can be used in the ModelConfig.
    These models have been tested and verified to work with the system.

    Returns:
        List[str]: Sorted list of available model paths in format "owner/model_name"

    Example:
        >>> models = list_available_models()
        >>> print(models[0])
        'Phips/1xDeH264_realplksr'
    """
    return sorted(VALID_MODELS)
