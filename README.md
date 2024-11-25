# 🚀 Benchmarking Pixel Based Upscaling Models 

## 🎯 Test Setup
```
Input  → 480x840p @ 30FPS
Output → 1920x3360p @ 30FPS (4x Upscale)
Clips  → 5-10 seconds each
Dataset→ 10 clips (5 real-life + 5 anime)
```

## 🔥 Models Under Test

| Model | Focus | Scale | Status |
|-------|-------------|-------|---------|
| RealESRGAN_x4plus | Real-life Enhancement | 4x | ⏳ |
| RealESRGAN_x4plus_anime_6B | Anime Optimization | 4x | ⏳ |
| realesr-animevideov3 | Anime Video Focus | 4x | ⏳ |
| AURA-SR | Universal Video | 4x | ⏳ |

## 📊 Performance Tracking

### 🎥 Real-Life Videos

| Video | Model | Time (s)↓ | VRAM (GB)↓ | FPS↑ |
|:------|:------:|:-----------:|:------------:|:------:|
| clip_01 | RealESRGAN_x4plus | - | - | - |
| clip_01 | AURA-SR | - | - | - |
| clip_02 | RealESRGAN_x4plus | - | - | - |
| clip_02 | AURA-SR | - | - | - |
| clip_03 | RealESRGAN_x4plus | - | - | - |
| clip_03 | AURA-SR | - | - | - |
| clip_04 | RealESRGAN_x4plus | - | - | - |
| clip_04 | AURA-SR | - | - | - |
| clip_05 | RealESRGAN_x4plus | - | - | - |
| clip_05 | AURA-SR | - | - | - |

### 🎨 Anime Videos

| Video | Model | Time (s)↓ | VRAM (GB)↓ | FPS↑ |
|:------|:------:|:-----------:|:------------:|:------:|
| anime_01 | RealESRGAN_x4plus_anime_6B | - | - | - |
| anime_01 | realesr-animevideov3 | - | - | - |
| anime_01 | AURA-SR | - | - | - |
| anime_02 | RealESRGAN_x4plus_anime_6B | - | - | - |
| anime_02 | realesr-animevideov3 | - | - | - |
| anime_02 | AURA-SR | - | - | - |
| anime_03 | RealESRGAN_x4plus_anime_6B | - | - | - |
| anime_03 | realesr-animevideov3 | - | - | - |
| anime_03 | AURA-SR | - | - | - |
| anime_04 | RealESRGAN_x4plus_anime_6B | - | - | - |
| anime_04 | realesr-animevideov3 | - | - | - |
| anime_04 | AURA-SR | - | - | - |
| anime_05 | RealESRGAN_x4plus_anime_6B | - | - | - |
| anime_05 | realesr-animevideov3 | - | - | - |
| anime_05 | AURA-SR | - | - | - |

## 💻 Test Environment
```
🖥️ GPU: NVIDIA A100 (80GB)
⚡ CUDA: 11.8
🐍 Python: 3.10
```

## 📝 Notes
```
↑ Higher is better
↓ Lower is better
🔄 Each test runs 3 times (averaged)
```


