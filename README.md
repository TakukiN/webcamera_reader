# Webcam Reader Application

## Overview
This application captures video from a webcam, processes each frame to remove reflections using ERRNet, and performs OCR (Optical Character Recognition) to detect Japanese and English text. It also supports barcode detection, high-resolution image reconstruction, and various interactive features.

## Functional Specifications

### 1. Core Features
- **Webcam Capture**: Captures video from the default webcam at 1280x720 resolution.
- **Reflection Removal**: Uses ERRNet to remove reflections from each frame before further processing.
- **OCR**: Detects and displays Japanese and English text from the processed frames.
- **Barcode Detection**: Identifies and displays barcode information (type and data) on the frame.
- **High-Resolution Image Reconstruction**: Supports generating high-resolution images from multiple frames with different blending modes (min, median, max).

### 2. Interactive Features
- **Key Controls**:
  - `o`: Toggle OCR on/off.
  - `f`: Toggle focus lock on/off.
  - `h`: Toggle high-resolution image generation on/off.
  - `r`: Reconstruct/update the high-resolution image.
  - `b`: Switch blending modes (min/median/max).
  - `a`: Toggle label detection mode (automatic/manual).
  - `q`: Quit the application.

### 3. Label Detection
- **Automatic Mode**: Attempts to automatically detect label corners. Falls back to manual mode if detection fails.
- **Manual Mode**: Allows the user to manually select four points for label detection.

### 4. Error Handling
- Logs errors and warnings for various operations, including OCR, barcode processing, and frame processing.

### 5. Dependencies
- OpenCV
- PyTorch
- Tesseract OCR
- PyZBar
- PIL (Pillow)
- Ultralytics (YOLOv8)

## Installation
1. Ensure all dependencies are installed:
   ```bash
   pip install opencv-python pytesseract pyzbar pillow ultralytics
   ```
2. Download the ERRNet model and place it in the project directory as `ERRNet.pth`.

## Usage
Run the application using:
```bash
python webcam_reader.py
```

## Notes
- The application requires a webcam to function.
- Ensure that the Tesseract OCR executable is correctly configured in the script.

## Image Synthesis Process

### Steps to Create a High-Resolution Image

1. **Enable High-Resolution Image Generation**:
   - Press the `h` key to enable high-resolution image generation.
   - The status "高精細生成中" (High-Resolution Generation Active) will be displayed at the bottom of the screen.

2. **Select Blending Mode**:
   - Press the `b` key to switch between blending modes:
     - `min`: Uses the darkest parts of the images.
     - `median`: Uses the median values of the images.
     - `max`: Uses the brightest parts of the images.
   - The current blending mode will be displayed at the bottom of the screen.

3. **Set Label Detection Mode**:
   - Press the `a` key to toggle between automatic and manual label detection:
     - **Automatic Mode**: Attempts to automatically detect the four corners of the label.
     - **Manual Mode**: Allows you to manually select four points for label detection.

4. **Execute Image Synthesis**:
   - Press the `r` key to generate a high-resolution image from the captured frames.
   - The generated image will be automatically saved and displayed in a separate window.
   - The file will be saved with the name format `high_res_YYYYMMDD_HHMMSS.png`.

### Recommended Operation Steps

1. Press `h` to start high-resolution generation.
2. Press `b` to select the blending mode (e.g., `median`).
3. Press `a` to choose the label detection mode.
4. Move the camera slowly to capture multiple frames.
5. Press `r` to generate the high-resolution image.

### Notes
- During high-resolution generation, moving the camera slowly will yield better results.
- Ensure that the label is fully visible in the frame.
- Choose the blending mode based on your needs:
  - `min`: Use to remove reflections.
  - `median`: Suitable for general purposes.
  - `max`: Use to brighten dark areas.

## ライセンス

MIT License 