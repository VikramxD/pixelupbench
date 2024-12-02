# 🚀 Benchmarking Video Upscaling Models

## 🎯 Test Environment and Setup
```
Hardware: NVIDIA A40 (48GB)
CUDA: 11.8
Python: 3.10
Input Videos: 4 test clips with varying resolutions
```

## 🔍 Key Observations

1. **Model Performance Characteristics**
   - 4xLSDIRCompactR3: Best speed-to-quality ratio (277.17s avg, 0.867 SSIM)
   - AURA-SR: Highest quality but slowest (791.87s avg, 0.910 SSIM)
   - Processing time varies up to 2.86x between models

2. **Resolution Impact**
   - Portrait videos (360x640) take longer to process
   - Landscape format shows better processing efficiency
   - Processing time scales non-linearly with resolution

3. **Quality-Speed Trade-offs**
   - Faster models show slight quality degradation
   - Higher SSIM scores correlate with longer processing times
   - Quality improvements come with significant speed penalties

## 📊 Model Performance Summary

| Model | Avg Time(s)↓ | Avg SSIM↑ | Speed Rank | Quality Rank |
|-------|-------------|------------|------------|--------------|
| 4xLSDIRCompactR3 | 277.17 | 0.867 | 1 | 2 |
| 4xNomosRealPLKSR | 405.10 | 0.837 | 2 | 3 |
| RealESRGAN_x4plus | 464.26 | - | 3 | - |
| AURA-SR | 791.87 | 0.910 | 4 | 1 |

## 📈 Detailed Results by Video

### generated.mp4 (704x480 → 2816x1920)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 231.69 | 0.810 |
| 4xNomosRealPLKSR | 298.41 | 0.722 |
| RealESRGAN_x4plus | 380.33 | - |
| AURA-SR | 655.99 | 0.790 |

### low_rel.mp4 (640x360 → 2560x1440)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 332.57 | 0.771 |
| 4xNomosRealPLKSR | 387.52 | 0.804 |
| RealESRGAN_x4plus | 623.31 | - |
| AURA-SR | 946.81 | 0.920 |

### low_res.mp4 (360x640 → 1440x2560)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 286.88 | 0.927 |
| 4xNomosRealPLKSR | 430.79 | 0.894 |
| RealESRGAN_x4plus | 509.75 | - |
| AURA-SR | 816.58 | 0.967 |

### restore.mp4 (480x360 → 1920x1440)
| Model | Time (s)↓ | SSIM↑ |
|-------|-----------|--------|
| 4xLSDIRCompactR3 | 257.55 | 0.960 |
| 4xNomosRealPLKSR | 503.67 | 0.926 |
| RealESRGAN_x4plus | 343.47 | - |
| AURA-SR | 748.09 | 0.962 |

## 📊 Batch Processing Statistics

| Model | Total Time (s) | Avg Time/Video (s) |
|-------|----------------|-------------------|
| 4xLSDIRCompactR3 | 1108.70 | 277.17 |
| 4xNomosRealPLKSR | 1620.39 | 405.10 |
| RealESRGAN_x4plus | 1857.05 | 464.26 |
| AURA-SR | 3167.46 | 791.87 |

## 💡 Model Selection Guide

1. **For Speed Priority**
   - Choose: 4xLSDIRCompactR3
   - Best for: Batch processing, time-sensitive applications
   - Trade-off: Slightly lower quality than AURA-SR

2. **For Quality Priority**
   - Choose: AURA-SR
   - Best for: High-quality upscaling requirements
   - Trade-off: Significantly longer processing times

3. **For Balanced Performance**
   - Choose: 4xNomosRealPLKSR
   - Best for: General-purpose upscaling
   - Trade-off: Middle-ground on both speed and quality

## 📝 Notes
```
↑ Higher is better
↓ Lower is better
SSIM: Structural Similarity Index (0-1, higher is better)
All tests conducted under identical conditions
```

## 🔧 Technical Details
- All models perform 4x upscaling
- Processing done with CUDA acceleration
- Results based on single-pass processing
- SSIM measured against reference frames

