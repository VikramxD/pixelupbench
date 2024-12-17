# PixelUpBench: Video Upscaling Models Benchmark

A comprehensive benchmark suite for evaluating various video upscaling models, focusing on both performance and quality metrics.

## Overview

PixelUpBench provides detailed benchmarking results for state-of-the-art video upscaling models, comparing their performance across different video types and resolutions. This benchmark helps developers and researchers choose the most suitable upscaling model for their specific use case.

## Test Environment

```
Hardware Specifications:
- GPU: NVIDIA A40 (48GB VRAM)
- CUDA Version: 11.8
- Python Version: 3.10

Test Configuration:
- Scale Factors: 4x (primary), 2x (selected models)
- Video Types: Realistic and Anime content
- Resolution Range: 360p to 1080p input
```

## Test Dataset

### Video Specifications

| Video | Duration (s) | FPS | Resolution | Content Type |
|-------|-------------|-----|------------|--------------|
| generated.mp4 | 4 | 25.0 | 704x480 | Synthetic |
| input.mp4 | 12 | 23.98 | 1280x720 | Real-world |
| low_rel.mp4 | 10 | 25.0 | 640x360 | Real-world |
| low_res.mp4 | 8 | 25.0 | 360x640 | Real-world |
| restore.mp4 | 14 | 18.0 | 480x360 | Real-world |
| test_real.mp4 | 10 | 30.0 | 1280x720 | Real-world |
| test_anime.mp4 | 10 | 30.0 | 1280x720 | Anime |

## Benchmark Results

### 4x Upscaling Models

| Model | Avg Time(s)↓ | Avg SSIM↑ | Avg model_fps↓ | VRAM Usage (GB) | Best Use Case |
|-------|-------------|------------|----------------|-----------------|---------------|
| 4xLSDIRCompactR3 | 277.17 | 0.867 | 0.715 | 8.2 | Fast processing, general content |
| 4xNomosRealPLKSR | 405.10 | 0.837 | 0.486 | 10.4 | Balanced quality-speed |
| RealESRGAN_x4plus | 464.26 | - | 0.447 | 6.8 | General purpose |
| 4xNomos2_otf_esrgan | 704.06 | 0.869 | 0.292 | 12.6 | High quality, no time constraint |
| AURA-SR | 791.87 | 0.910 | 0.25 | 14.2 | Maximum quality |
| 4xHFA2kLUDVAESwinIR_light | 911.84 | 0.841 | 0.2155 | 9.8 | Memory-constrained systems |
| 4xNomos2_hq_atd | 2332.86 | 0.907 | 0.09 | 16.4 | Highest quality, offline processing |

### 2x Upscaling Models

| Model | Avg Time(s)↓ | Avg SSIM↑ | Avg model_fps↓ | VRAM Usage (GB) | Best Use Case |
|-------|-------------|------------|----------------|-----------------|---------------|
| 2xHFA2kCompact | 77.20 | 0.903 | 2.8154 | 4.2 | Fast processing |
| 2xNomosUni_span_multijpg | 81.41 | 0.947 | 2.6364 | 4.8 | General purpose |
| 2xHFA2k_LUDVAE_compact | 80.06 | 0.901 | 2.6856 | 4.4 | Memory-efficient |
| 2xNomosUni_esrgan_multijpg | 109.87 | 0.941 | 1.9656 | 5.2 | High quality |
| 2xRRDB APISR | 333.032 | 0.697 | 2.77 | 6.4 | Fast inference |

## Model Analysis

### Performance Categories

1. **Ultra-Fast Processing** (>2 FPS)
   - 2x Models: 2xHFA2kCompact, 2xNomosUni_span_multijpg
   - Best for: Real-time processing, streaming applications

2. **Balanced Performance** (0.4-2 FPS)
   - Models: 4xLSDIRCompactR3, 4xNomosRealPLKSR, RealESRGAN_x4plus
   - Best for: General purpose upscaling with good quality-speed trade-off

3. **Quality Focused** (<0.4 FPS)
   - Models: AURA-SR, 4xNomos2_hq_atd
   - Best for: Professional video enhancement, offline processing

### Quality Metrics

1. **SSIM Performance**
   - Highest: 0.977 (4xNomos2_hq_atd on restore.mp4)
   - Most Consistent: 4xHFA2kLUDVAESwinIR_light (0.766-0.889 range)
   - Best Overall: AURA-SR (0.910 average)

2. **Visual Quality Characteristics**
   - Detail Preservation: 4xNomos2_hq_atd, AURA-SR
   - Artifact Handling: 4xNomosRealPLKSR, RealESRGAN_x4plus
   - Edge Sharpness: 4xNomos2_otf_esrgan

## Optimization Guidelines

### Resource Requirements

1. **VRAM Considerations**
   - Light Models (<8GB): RealESRGAN_x4plus, 2x models
   - Medium Models (8-12GB): 4xLSDIRCompactR3, 4xNomosRealPLKSR
   - Heavy Models (>12GB): AURA-SR, 4xNomos2_hq_atd

2. **Processing Time Factors**
   - Resolution scaling is non-linear
   - Portrait videos require more processing time
   - Batch processing can improve throughput

### Best Practices

1. **Model Selection**
   - For real-time: Use 2x models or 4xLSDIRCompactR3
   - For quality: AURA-SR or 4xNomos2_hq_atd
   - For balanced: 4xNomosRealPLKSR or RealESRGAN_x4plus

2. **Optimization Techniques**
   - Use appropriate batch sizes for your VRAM
   - Consider resolution preprocessing
   - Implement proper memory management

## Notes

- ↑ Higher is better
- ↓ Lower is better
- SSIM: Structural Similarity Index (0-1)
- model_fps: Frames processed per second
- All tests conducted on identical hardware
- VRAM usage may vary with input resolution

## Contributing

Contributions to PixelUpBench are welcome! Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
