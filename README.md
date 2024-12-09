# Video Upscaling Models Benchmark

## Test Environment
```
Hardware: NVIDIA A40 (48GB)
CUDA: 11.8
Python: 3.10
Test Videos: 4 clips with varying resolutions
Scale Factor: 4x (2x for Swin2SR)
```

## Models Benchmarked

| Model | Scale | Avg Time(s)↓ | Avg SSIM↑ |
|-------|-------|-------------|------------|
| 4xLSDIRCompactR3 | 4x | 277.17 | 0.867 |
| 4xNomosRealPLKSR | 4x | 405.10 | 0.837 |
| RealESRGAN_x4plus | 4x | 464.26 | - |
| 4xNomos2_otf_esrgan | 4x | 704.06 | 0.869 |
| AURA-SR | 4x | 791.87 | 0.910 |
| 4xHFA2kLUDVAESwinIR_light | 4x | 911.84 | 0.841 |
| Swin2SR | 2x | 970.82 | 0.711 |
| 4xNomos2_hq_atd | 4x | 2332.86 | 0.907 |

## Video-Specific Performance

### generated.mp4 (704x480 → 2816x1920)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 231.69 | 0.810 |
| 4xNomosRealPLKSR | 298.41 | 0.722 |
| RealESRGAN_x4plus | 380.33 | - |
| 4xNomos2_otf_esrgan | 627.62 | 0.779 |
| AURA-SR | 655.99 | 0.790 |
| 4xHFA2kLUDVAESwinIR_light | 735.64 | 0.766 |
| Swin2SR | 812.05 | 0.672 |
| 4xNomos2_hq_atd | 1574.59 | 0.817 |

### low_rel.mp4 (640x360 → 2560x1440)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 332.57 | 0.771 |
| 4xNomosRealPLKSR | 387.52 | 0.804 |
| RealESRGAN_x4plus | 623.31 | - |
| 4xNomos2_otf_esrgan | 874.52 | 0.818 |
| AURA-SR | 946.81 | 0.920 |
| 4xHFA2kLUDVAESwinIR_light | 1047.35 | 0.841 |
| Swin2SR | 1198.10 | 0.487 |
| 4xNomos2_hq_atd | 2091.30 | 0.915 |

### low_res.mp4 (360x640 → 1440x2560)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 286.88 | 0.927 |
| 4xNomosRealPLKSR | 430.79 | 0.894 |
| RealESRGAN_x4plus | 509.75 | - |
| 4xNomos2_otf_esrgan | 768.43 | 0.927 |
| AURA-SR | 816.58 | 0.967 |
| 4xHFA2kLUDVAESwinIR_light | 960.56 | 0.866 |
| Swin2SR | 1047.09 | 0.786 |
| 4xNomos2_hq_atd | 3575.46 | 0.918 |

### restore.mp4 (480x360 → 1920x1440)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 257.55 | 0.960 |
| RealESRGAN_x4plus | 343.47 | - |
| 4xNomosRealPLKSR | 503.67 | 0.926 |
| 4xNomos2_otf_esrgan | 545.68 | 0.953 |
| AURA-SR | 748.09 | 0.962 |
| 4xHFA2kLUDVAESwinIR_light | 903.80 | 0.889 |
| Swin2SR | 826.03 | 0.900 |
| 4xNomos2_hq_atd | 2090.10 | 0.977 |

## Performance Analysis

### Processing Time
- Range: 277.17s - 2332.86s (8.4x difference)
- Portrait videos take longer to process
- Processing time scales non-linearly with resolution

### SSIM Quality
- Range: 0.711 - 0.910
- Best overall: AURA-SR (0.910)
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
```
