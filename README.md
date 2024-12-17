# Video Upscaling Models Benchmark

## Test Environment
```
Hardware: NVIDIA A40 (48GB)
CUDA: 11.8
Python: 3.10
Test Videos: 7 clips with varying resolutions
Scale Factor: 4x (2x for Swin2SR)
```

## Input Video Information

| Video | Duration (s) | FPS |
| --- | --- | --- |
| generated.mp4 | 4 | 25.0 |
| input.mp4 | 12 | 23.98 |
| low_rel.mp4 | 10 | 25.0 |
| low_res.mp4 | 8 | 25.0 |
| restore.mp4 | 14 | 18.0 |
| test_real.mp4 | 10 | 30.0 |
| test_anime.mp4 | 10 | 30.0 |

## Models Benchmarked

| Model | Scale | Avg Time(s)↓ | Avg SSIM↑ | Avg model_fps↓ | Remarks/Comments |
|-------|-------|-------------|------------|----------------|------------------|
| 4xLSDIRCompactR3 | 4x | 277.17 | 0.867 | 0.715 | Extremely fast, but oversmoothens (water-painted effect) |
| 4xNomosRealPLKSR | 4x | 405.10 | 0.837 | 0.486 | Consistent, solves aliasing, slight oversharpening/line artifacts |
| RealESRGAN_x4plus | 4x | 464.26 | - | 0.447 | Consistent, solves oversharpened edges |
| 4xNomos2_otf_esrgan | 4x | 704.06 | 0.869 | 0.292 | Good quality, solves edge oversharpening, very long inference time |
| AURA-SR | 4x | 791.87 | 0.910 | 0.25 | High SSIM, but over-sharpens with line artifacts |
| 4xHFA2kLUDVAESwinIR_light | 4x | 911.84 | 0.841 | 0.2155 | Decent, but long inference time |
| Swin2SR | 2x | 970.82 | 0.711 | 0.207 | 2x upscale only, mediocre results for cost |
| 4xNomos2_hq_atd | 4x | 2332.86 | 0.907 | 0.09 | Very high quality, extremely slow |

### Additional Models (2x)

| Model | Scale | Avg Time(s)↓ | Avg SSIM↑ | Avg model_fps↓ | Remarks/Comments |
|-------|-------|-------------|------------|----------------|------------------|
| 2xHFA2kCompact | 2x | 77.20 | 0.903 | 2.8154 | Leaves line artifacts, poor portrait video handling |
| 2xNomosUni_span_multijpg | 2x | 81.41 | 0.947 | 2.6364 | Consistent, handles portrait videos well |
| 2xHFA2k_LUDVAE_compact | 2x | 80.06 | 0.901 | 2.6856 | Leaves line artifacts in outputs |
| 2xHFA2kReal-CUGAN | 2x | 83.11 | 0.907 | 2.604 | Noisy outputs, poor portrait video handling |
| 2xNomosUni_esrgan_multijpg | 2x | 109.87 | 0.941 | 1.9656 | Good quality, handles portrait videos well |
| 2xHFA2kOmniSR | 2x | 185.42 | 0.899 | 1.62 | Not recommended, produces blurry video |
| 2xRRDB APISR | 2x | 333.032 | 0.697 | 2.77 | Good results, fast inference time |

## Video-Specific Performance

### generated.mp4 (704x480 → 2816x1920, 100 frames)
| Model | Time (s)↓ | SSIM↑ | model_fps |
|-------|-----------|--------|-----------|
| 4xLSDIRCompactR3 | 231.69 | 0.810 | 0.4315 |
| 4xNomosRealPLKSR | 298.41 | 0.722 | 0.335 |
| RealESRGAN_x4plus | 380.33 | - | 0.263 |
| 4xNomos2_otf_esrgan | 627.62 | 0.779 | 0.159 |
| AURA-SR | 655.99 | 0.790 | 0.152 |
| 4xHFA2kLUDVAESwinIR_light | 735.64 | 0.766 | 0.136 |
| Swin2SR | 812.05 | 0.672 | 0.123 |
| 4xNomos2_hq_atd | 1574.59 | 0.817 | 0.0635 |

### low_rel.mp4 (640x360 → 2560x1440)
| Model | Time (s)↓ | SSIM↑ | model_fps |
|-------|-----------|--------|-----------|
| 4xLSDIRCompactR3 | 332.57 | 0.771 | 0.301 |
| 4xNomosRealPLKSR | 387.52 | 0.804 | 0.258 |
| RealESRGAN_x4plus | 623.31 | - | 0.161 |
| 4xNomos2_otf_esrgan | 874.52 | 0.818 | 0.114 |
| AURA-SR | 946.81 | 0.920 | 0.106 |
| 4xHFA2kLUDVAESwinIR_light | 1047.35 | 0.841 | 0.095 |
| Swin2SR | 1198.10 | 0.487 | 0.083 |
| 4xNomos2_hq_atd | 2091.30 | 0.915 | 0.048 |

### low_res.mp4 (360x640 → 1440x2560)
| Model | Time (s)↓ | SSIM↑ | model_fps |
|-------|-----------|--------|-----------|
| 4xLSDIRCompactR3 | 286.88 | 0.927 | 0.349 |
| 4xNomosRealPLKSR | 430.79 | 0.894 | 0.232 |
| RealESRGAN_x4plus | 509.75 | - | 0.196 |
| 4xNomos2_otf_esrgan | 768.43 | 0.927 | 0.13 |
| AURA-SR | 816.58 | 0.967 | 0.122 |
| 4xHFA2kLUDVAESwinIR_light | 960.56 | 0.866 | 0.104 |
| Swin2SR | 1047.09 | 0.786 | 0.096 |
| 4xNomos2_hq_atd | 3575.46 | 0.918 | 0.028 |

### restore.mp4 (480x360 → 1920x1440)
| Model | Time (s)↓ | SSIM↑ | model_fps |
|-------|-----------|--------|-----------|
| 4xLSDIRCompactR3 | 257.55 | 0.960 | 0.388 |
| RealESRGAN_x4plus | 343.47 | - | 0.291 |
| 4xNomosRealPLKSR | 503.67 | 0.926 | 0.198 |
| 4xNomos2_otf_esrgan | 545.68 | 0.953 | 0.183 |
| AURA-SR | 748.09 | 0.962 | 0.134 |
| 4xHFA2kLUDVAESwinIR_light | 903.80 | 0.889 | 0.111 |
| Swin2SR | 826.03 | 0.900 | 0.121 |
| 4xNomos2_hq_atd | 2090.10 | 0.977 | 0.048 |

### Additional RealESRGAN_x4plus Results

| Model | Video | Inference Time (s) | Original FPS | model_fps |
|-------|-------|-------------------|--------------|-----------|
| RealESRGAN_x4plus | test_real.mp4 | 197.87 | 30.0 | 1.516 |
| RealESRGAN_x4plus | test_anime.mp4 | 277.62 | 30.0 | 1.081 |

Average Time per Video: 237.78s

## Performance Analysis

### Processing Time
- Range: 77.20s - 3575.46s (46.3x difference)
- Portrait videos take longer to process
- Processing time scales non-linearly with resolution

### SSIM Quality
- Range: 0.697 - 0.977
- Best overall: 4xNomos2_hq_atd (0.977)
- Best single score: 4xNomos2_hq_atd on restore.mp4 (0.977)
- Most consistent: 4xHFA2kLUDVAESwinIR_light (0.766-0.889 range)

### Resolution Impact
- Portrait orientation shows increased processing time
- Higher resolutions generally require longer processing
- Quality patterns consistent across similar resolutions

## Technical Details
- Single-pass processing
- CUDA acceleration enabled
- Full pipeline overhead included
- Direct frame-by-frame processing
- Memory usage varies with resolution
- GPU utilization varies by model

Note:
```
↑ Higher is better
↓ Lower is better
SSIM: Structural Similarity Index (0-1)
Times in seconds
All tests run on identical hardware
