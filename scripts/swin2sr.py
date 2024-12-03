"""
Swin2SR Video Upscaling Pipeline

A production-grade pipeline for processing and upscaling videos using the Swin2SR model.
This module provides comprehensive batch processing capabilities with robust error handling,
metrics collection, and progress monitoring.

Key Features:
    - Automated batch video processing with 2x upscaling
    - Real-time progress tracking with ETA
    - Comprehensive performance metrics and JSON reporting
    - GPU acceleration with CUDA support
    - Robust error handling and logging

Dependencies:
    - torch>=1.7.0
    - transformers>=4.0.0
    - PIL>=8.0.0
    - numpy>=1.19.0
    - tqdm>=4.45.0
    - loguru>=0.5.0
    - opencv-python>=4.5.0
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Union, Any
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from loguru import logger
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

class BatchMetrics:
    def __init__(self, num_videos: int) -> None:
        self.start_time: float = time.time()
        self.num_videos: int = num_videos
        self.videos_processed: List[Dict[str, Union[str, float]]] = []

    def add_video_result(self, video_name: str, inference_time: float,
                         original_resolution: tuple, upscaled_resolution: tuple,
                         ssim: float) -> None:
        self.videos_processed.append({
            "name": video_name,
            "inference_time": inference_time,
            "original_resolution": f"{original_resolution[0]}x{original_resolution[1]}",
            "upscaled_resolution": f"{upscaled_resolution[0]}x{upscaled_resolution[1]}",
            "ssim": round(ssim, 3)
        })

    def get_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        return {
            "model_name": "Swin2SR",
            "total_videos": self.num_videos,
            "total_batch_time": total_time,
            "average_time_per_video": total_time / self.num_videos,
            "videos": self.videos_processed,
            "timestamp": datetime.now().isoformat(),
        }

class Swin2SRUpscaler:
    def __init__(self, input_dir: Path, output_dir: Path, gpu_device: int = 0) -> None:
        """Initialize the Swin2SR upscaler with provided settings."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
        self.processor, self.model = self._setup_model()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Swin2SR initialization complete")

    def _setup_model(self):
        """Initialize and load the Swin2SR model."""
        try:
            processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
            model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
            model.to(self.device)
            logger.info("Swin2SR model loaded successfully")
            return processor, model
        except Exception as e:
            logger.error(f"Failed to load Swin2SR model: {str(e)}")
            raise

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two frames."""
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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through Swin2SR."""
        frame_pil = Image.fromarray(frame)
        inputs = self.processor(frame_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        
        return output

    def process_video(self, video_path: Path) -> tuple:
        """Process a single video through the Swin2SR upscaling pipeline."""
        start_time = time.time()

        output_dir = self.output_dir / "Swin2SR"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_output.mp4"

        cap = cv2.VideoCapture(str(video_path))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read video: {video_path}")

        test_output = self.process_frame(frame)
        up_height, up_width = test_output.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (up_width, up_height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ssim_values = []
        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                upscaled_frame = self.process_frame(frame_rgb)
                upscaled_frame = cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR)

                frame_resized = cv2.resize(frame, (up_width, up_height))
                ssim = self._calculate_ssim(frame_resized, upscaled_frame)
                ssim_values.append(ssim)

                writer.write(upscaled_frame)
                pbar.update(1)

        cap.release()
        writer.release()

        inference_time = time.time() - start_time
        average_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0

        return (
            inference_time,
            (orig_width, orig_height),
            (up_width, up_height),
            average_ssim
        )

    def process_batch(self) -> Dict[str, Any]:
        """Execute batch processing of multiple videos with comprehensive metrics."""
        video_files = list(self.input_dir.glob("*.mp4"))
        if not video_files:
            raise ValueError(f"No MP4 files found in {self.input_dir}")

        metrics = BatchMetrics(len(video_files))
        logger.info(f"Processing {len(video_files)} videos with Swin2SR...")

        with tqdm(video_files, desc="Processing batch", ascii=" ▖▘▝▗▚▞█ ") as pbar:
            for video_path in pbar:
                try:
                    inference_time, orig_res, up_res, ssim = self.process_video(video_path)
                    metrics.add_video_result(video_path.name, inference_time,
                                             orig_res, up_res, ssim)
                    pbar.set_postfix({"Last inference": f"{inference_time:.2f}s"})
                except Exception as e:
                    logger.error(f"Failed to process {video_path.name}: {str(e)}")

        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        metrics_path = metrics_dir / f"swin2sr_batch_metrics_{datetime.now():%Y%m%d_%H%M%S}.json"

        summary = metrics.get_summary()
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=4)

        return summary

def main() -> None:
    """Main entry point for the Swin2SR video processing system."""
    try:
        input_dir = Path("/root/pixelupbench/data/realism")
        output_dir = Path("../results")
        upscaler = Swin2SRUpscaler(input_dir, output_dir)
        summary = upscaler.process_batch()
        print("\nSwin2SR Batch Processing Summary:")
        print(f"Total videos processed: {summary['total_videos']}")
        print(f"Total batch time: {summary['total_batch_time']:.2f}s")
        print(f"Average time per video: {summary['average_time_per_video']:.2f}s")

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()