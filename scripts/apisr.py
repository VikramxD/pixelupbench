from loguru import logger
import os, sys
import cv2
import time
import datetime, pytz
import torch
import numpy as np
from torchvision.utils import save_image
from pydantic import BaseSettings
from typing import List
import json

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from APISR.test_code.inference import super_resolve_video
from APISR.test_code.test_utils import load_grl, load_rrdb, load_dat


class Config(BaseSettings):
    """
    Pydantic configuration class for demo settings.

    Attributes:
        max_video_size (int): Maximum size (in pixels) for the shorter side of the video. Larger videos will be resized.
        weights_directory (str): Directory to store the pre-trained model weights.
        max_queue_size (int): Maximum size of the Gradio queue.
        timezone (str): Timezone to use for logging.
        input_directory (str): Directory for input videos.
        output_directory (str): Directory for processed videos.
        metrics_directory (str): Directory for metrics output.
        model_weights (dict): Mapping of model names to their weight filenames.
    """
    max_video_size: int = 1080
    weights_directory: str = "pretrained"
    max_queue_size: int = 10
    timezone: str = 'US/Eastern'
    input_directory: str = "inputs"
    output_directory: str = "outputs"
    metrics_directory: str = "metrics"
    model_weights: dict = {
        "4xGRL": "4x_APISR_GRL_GAN_generator.pth",
        "4xRRDB": "4x_APISR_RRDB_GAN_generator.pth",
        "2xRRDB": "2x_APISR_RRDB_GAN_generator.pth",
        "4xDAT": "4x_APISR_DAT_GAN_generator.pth"
    }

    class Config:
        env_prefix = "APISR_"


config = Config()


def auto_download_if_needed(weight_path: str) -> None:
    """
    Download model weights if they are not already present locally.

    Args:
        weight_path (str): Path to the weight file.
    """
    if os.path.exists(weight_path):
        logger.info(f"Weight file already exists: {weight_path}")
        return

    os.makedirs(config.weights_directory, exist_ok=True)

    weight_urls = {
        "4x_APISR_RRDB_GAN_generator.pth": "https://github.com/Kiteretsu77/APISR/releases/download/v0.2.0/4x_APISR_RRDB_GAN_generator.pth",
        "4x_APISR_GRL_GAN_generator.pth": "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth",
        "2x_APISR_RRDB_GAN_generator.pth": "https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth",
        "4x_APISR_DAT_GAN_generator.pth": "https://github.com/Kiteretsu77/APISR/releases/download/v0.3.0/4x_APISR_DAT_GAN_generator.pth"
    }

    filename = os.path.basename(weight_path)
    if filename in weight_urls:
        url = weight_urls[filename]
        logger.info(f"Downloading {filename} from {url}")
        os.system(f"wget {url} -O {weight_path}")
    else:
        logger.error(f"No URL found for {filename}")


def inference(video_path: str, model_name: str) -> str:
    """
    Perform inference on a video using the specified model.
    """
    try:
        weight_dtype = torch.float32

        # Validate model name
        if model_name not in config.model_weights:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(config.model_weights.keys())}")

        # Get weight path from config
        weight_filename = config.model_weights[model_name]
        weight_path = os.path.join(config.weights_directory, weight_filename)
        auto_download_if_needed(weight_path)

        # Load the model based on type
        if model_name == "4xGRL":
            generator = load_grl(weight_path, scale=4)
        elif model_name in ["4xRRDB", "2xRRDB"]:
            scale = 2 if model_name == "2xRRDB" else 4
            generator = load_rrdb(weight_path, scale=scale)
        elif model_name == "4xDAT":
            generator = load_dat(weight_path, scale=4)

        generator = generator.to(dtype=weight_dtype)

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Current time: {datetime.datetime.now(pytz.timezone(config.timezone))}")

        # Create output directory if it doesn't exist
        os.makedirs(config.output_directory, exist_ok=True)
        
        # Generate output path in the configured directory
        output_video_path = os.path.join(
            config.output_directory,
            f"output_{time.time()}.mp4"
        )

        super_resolve_video(generator, video_path, output_video_path, weight_dtype=weight_dtype,
                          downsample_threshold=config.max_video_size, crop_for_4x=True)

        return output_video_path

    except Exception as error:
        logger.error(f"Global exception: {error}")
        raise


def export_metrics(metrics: List[dict], output_dir: str = None) -> None:
    """
    Export batch processing metrics to a JSON file.
    """
    output_dir = output_dir or config.metrics_directory
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(
        output_dir,
        f"batch_metrics_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics exported to {metrics_path}")
