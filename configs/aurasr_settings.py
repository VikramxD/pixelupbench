"""
Configuration module for AURA-SR Video Upscaling Script

This module provides configuration management and validation for the AURA-SR
video upscaling system, including path management, GPU settings, and logging
configuration.

The module uses Pydantic for settings management and validation, ensuring type safety
and proper configuration validation at runtime.

Classes:
    AuraSettings: Global configuration settings for AURA-SR upscaling system

Environment Variables:
    All settings can be overridden using environment variables with AURA_ prefix:
    - AURA_INPUT_DIR: Path to input videos directory
    - AURA_OUTPUT_DIR: Path to output directory
    - AURA_GPU_DEVICE: GPU device ID
    - AURA_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from loguru import logger


class AuraSettings(BaseSettings):
    """
    Configuration settings for AURA-SR video upscaling system.
    
    This class manages all configuration aspects of the AURA-SR system,
    including input/output paths, GPU settings, and logging configuration.
    
    Attributes:
        input_dir (Path): Directory containing input videos for processing.
            Must exist at runtime.
        output_dir (Path): Base directory for all output files including:
            - Upscaled videos
            - Performance metrics
            - Log files
            Defaults to "../results".
        gpu_device (int): CUDA GPU device ID to use for processing.
            Default: 0 (first GPU).
        log_level (str): Logging level for both file and console output.
            Accepts standard logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
            Default: "INFO".
    
    Raises:
        ValueError: If input_dir doesn't exist or other validation fails.
    
    Example:
        >>> settings = AuraSettings(
        ...     input_dir=Path("videos"),
        ...     output_dir=Path("results"),
        ...     gpu_device=0
        ... )
        >>> settings.setup_logging()
    """
    
    input_dir: Path = Field(
        '/root/pixelupbench/data/realism',
        description="Directory containing input videos"
    )
    output_dir: Path = Field(
        default=Path("../results"),
        description="Base directory for outputs"
    )
    gpu_device: int = Field(
        default=0,
        description="GPU device ID"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    @field_validator('input_dir')
    def validate_input_dir(cls, v: Path) -> Path:
        """
        Validate that input directory exists.
        
        Args:
            v (Path): Input directory path to validate
            
        Returns:
            Path: Validated input directory path
            
        Raises:
            ValueError: If directory doesn't exist
        """
        if not v.exists():
            raise ValueError(f"Input directory {v} does not exist")
        return v

    def setup_logging(self) -> None:
        """
        Configure logging system with file and console outputs.
        
        Sets up loguru logger with both file and console handlers:
        - File logging with rotation and retention policies
        - Console logging with color formatting
        - Custom format including timestamps and log levels
        
        Creates log directory if it doesn't exist.
        
        Example:
            >>> settings = AuraSettings()
            >>> settings.setup_logging()
            >>> logger.info("AURA-SR processing started")
        """
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "aura_upscaler.log"

        # Remove default logger and set up new configuration
        logger.remove()
        
        # File logging with rotation
        logger.add(
            log_file,
            rotation="100 MB",
            retention="30 days",
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True
        )
        
        # Console logging with colors
        logger.add(
            lambda msg: print(msg, flush=True),
            level=self.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
            colorize=True
        )

        logger.info("AURA-SR logging system initialized")

    class Config:
        """Pydantic configuration class."""
        env_prefix = "AURA_"
