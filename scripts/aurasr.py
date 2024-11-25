"""
Video Upscaling Benchmark System for AURA-SR
Processes videos using AURA-SR model with inference time tracking.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import cv2
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from aura_sr import AuraSR
from configs.aurasr_settings import AuraSettings

class BatchMetrics:
    """Simple batch metrics tracker."""

    def __init__(self, num_videos: int):
        self.start_time = time.time()
        self.num_videos = num_videos
        self.videos_processed = []

    def add_video_result(self, video_name: str, inference_time: float):
        """Add single video result."""
        self.videos_processed.append({"name": video_name, "inference_time": inference_time})

    def get_summary(self) -> Dict:
        """Get batch processing summary."""
        total_time = time.time() - self.start_time
        return {
            "model_name": "AURA-SR",
            "total_videos": self.num_videos,
            "total_batch_time": total_time,
            "average_time_per_video": total_time / self.num_videos,
            "videos": self.videos_processed,
            "timestamp": datetime.now().isoformat(),
        }


class AuraUpscaler:
    """Video upscaling system using AURA-SR."""

    def __init__(self, settings: AuraSettings):
        """Initialize AURA-SR upscaler."""
        self.settings = settings
        self.device = torch.device(f"cuda:{settings.gpu_device}")
        self.model = self._setup_model()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AURA-SR initialization complete")

    def _setup_model(self) -> AuraSR:
        """Setup AURA-SR model."""
        try:
            model = AuraSR.from_pretrained(model_id='fal/AuraSR-v2')
            logger.info("AURA-SR model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load AURA-SR model: {str(e)}")
            raise

    def process_video(self, video_path: Path) -> float:
        """Process single video and return inference time."""
        start_time = time.time()

        output_dir = self.settings.output_dir / "AURA-SR"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_output.mp4"

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get output dimensions
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read video: {video_path}")

        test_output = self.model.upscale_4x(frame)
        h, w = test_output.shape[:2]

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        # Process video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                upscaled_frame = self.model.upscale_4x(frame)
                writer.write(upscaled_frame)
                pbar.update(1)

        cap.release()
        writer.release()

        inference_time = time.time() - start_time
        return inference_time

    def process_batch(self) -> Dict:
        """Process all videos and return batch metrics."""
        video_files = list(self.settings.input_dir.glob("*.mp4"))
        if not video_files:
            raise ValueError(f"No MP4 files found in {self.settings.input_dir}")

        metrics = BatchMetrics(len(video_files))
        logger.info(f"Processing {len(video_files)} videos with AURA-SR...")

        with tqdm(video_files, desc="Processing batch",ascii= " ▖▘▝▗▚▞█ ") as pbar:
            for video_path in pbar:
                try:
                    inference_time = self.process_video(video_path)
                    metrics.add_video_result(video_path.name, inference_time)
                    pbar.set_postfix({"Last inference": f"{inference_time:.2f}s"})
                except Exception as e:
                    logger.error(f"Failed to process {video_path.name}: {str(e)}")

        # Save batch metrics
        metrics_dir = self.settings.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        metrics_path = metrics_dir / f"aura_batch_metrics_{datetime.now():%Y%m%d_%H%M%S}.json"

        summary = metrics.get_summary()
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=4)

        return summary


def main():
    """Main entry point."""
    try:
        settings = AuraSettings(input_dir=Path("input_videos"))

        upscaler = AuraUpscaler(settings)
        summary = upscaler.process_batch()
        print("\nAURA-SR Batch Processing Summary:")
        print(f"Total videos processed: {summary['total_videos']}")
        print(f"Total batch time: {summary['total_batch_time']:.2f}s")
        print(f"Average time per video: {summary['average_time_per_video']:.2f}s")
        print("\nIndividual video times:")
        for video in summary["videos"]:
            print(f"{video['name']}: {video['inference_time']:.2f}s")

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
