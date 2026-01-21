# 🚗 Adaptive Smart Parking Detection System

A research-grade parking occupancy detection system using YOLOv8, featuring adaptive geometric refinement and temporal confidence aggregation. This system is designed for high-resolution CCTV applications where stability, occlusion robustness, and camera-drift compensation are critical.

## 🌟 Key Features

### 🧠 Research-Grade Algorithms
- **TCOD (Temporal Confidence-Based Occupancy)**: Uses Exponential Smoothing ($S_t = \lambda S_{t-1} + (1-\lambda)s_t$) to eliminate state flickering.
- **Double-Threshold Hysteresis**: Prevents rapid state switching using an "Uncertainty Zone" [0.4 - 0.6].
- **Dynamic Polygon Refinement (DPR)**: Automatically adapts parking slot boundaries to account for camera shake or drift based on vehicle centroid history.
- **Occlusion-Robust Decay**: Maintains "memory" of parked cars even when briefly obscured by pedestrians or passing traffic.

### 🛠️ Production-Ready Features
- **High-Resolution Processing**: 720p/1080p inference mode optimized for small/distant object detection.
- **Video-Specific Persistence**: Automatically saves and loads parking polygons based on the video source filename.
- **Slot-Level Analytics**: Detailed second-by-second tracking of occupancy durations and utilization percentages.
- **Stability Metrics**: Real-time computation of the **Stability Index (SI)** for academic evaluation.

## 🚀 Quick Start

### Installation
```bash
pip install ultralytics opencv-python numpy torch
```

### Running the System
```bash
python Polyline_parking.py
```

## ⌨️ Controls & Interaction

| Key / Mouse | Action |
|-------------|--------|
| **Left Click Drag** | Draw new parking polygons |
| **Right Click** | Delete a specific polygon |
| **'s'** | **Save**: Save polygons to video-specific JSON |
| **'u'** | **Undo**: Remove the last drawn polygon |
| **'c'** | **Clear**: Wipe all polygons for the current video |
| **'a'** | **Analytics**: Export session data to JSON |
| **'m'** | **Metrics**: Print research stability metrics to console |
| **'d'** | **Debug**: Toggle raw YOLO detection boxes |
| **SPACE** | **Quit**: Exits the program |

## 📊 Evaluation & Research
The system includes specialized metrics for research papers:
1.  **Stability Index (SI)**: State changes per second (lower is better).
2.  **False Occupancy Rate**: Measures temporal jitter/flickering.
3.  **Utilization Rate**: Percentage of time slots are occupied vs free.

Detailed documentation on the mathematical models can be found in the `brain/` directory artifacts.

## 📂 Project Structure
- `Polyline_parking.py`: Main system implementation.
- `parking_polygons_[video].json`: Persistence files for parking slots.
- `slot_analytics.json`: Exported session data for dashboarding.
- `.gitignore`: Configured to exclude large model files (>100MB).

---
*Developed for research and production-grade smart city applications.*
