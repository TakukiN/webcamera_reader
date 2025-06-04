# Webcam Reader with High-Resolution Image Generation

A Python application that uses a webcam to read text and barcodes from labels, with high-resolution image generation capabilities.

## Features

- Real-time text recognition (OCR) for Japanese and English text
- Barcode reading
- High-resolution image generation from multiple frames
- Automatic and manual label detection
- Focus control
- Multiple blending modes for high-resolution image generation

## Prerequisites

### External Applications

1. **Tesseract-OCR**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install the Windows installer (tesseract-ocr-w64-setup-5.5.0.20241111.exe)
   - Default installation path: `C:\Program Files\Tesseract-OCR`
   - Make sure to install Japanese language data during installation

2. **Python 3.8 or later**
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python>=4.5.0
- numpy>=1.19.0
- pytesseract>=0.3.8
- pyzbar>=0.1.8
- Pillow>=8.0.0
- torch>=2.0.0
- torchvision>=0.15.0
- scipy>=1.7.0
- matplotlib>=3.4.0
- tqdm>=4.65.0
- ultralytics>=8.0.0

### Model Files

Download the following model files and place them in the project root directory:

1. YOLOv8-seg model:
   - File: `yolov8n-seg.pt`
   - The model will be automatically downloaded on first run

2. SAM model:
   - File: `sam_vit_b_01ec64.pth`
   - Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

## Usage

1. Run the application:
```bash
python webcam_reader.py
```

2. When the application starts:
   - The first frame will be used to detect or select the label area
   - In automatic mode, it will try to detect the label corners
   - If automatic detection fails, you'll be prompted to manually select the corners
   - Click on the four corners of the label in clockwise order

3. Key Controls:
   - `o`: Toggle OCR on/off
   - `f`: Toggle focus lock on/off
   - `h`: Toggle high-resolution image generation
   - `r`: Generate/update high-resolution image
   - `b`: Switch blending modes (min/median/max)
   - `a`: Toggle between automatic and manual label detection
   - `q`: Quit the application

## High-Resolution Image Generation

The application can generate high-resolution images by combining multiple frames:

1. Press the `h` key to enable high-resolution image generation.
2. The application will start capturing frames and tracking the label.
3. Press the `b` key to switch between blending modes:
   - `min`: Takes the minimum value for each pixel
   - `median`: Takes the median value for each pixel
   - `max`: Takes the maximum value for each pixel
4. Press the `r` key to generate a high-resolution image from the captured frames.
5. The generated image will be saved as `high_res_YYYYMMDD_HHMMSS.png`.

## Label Detection

The application supports both automatic and manual label detection:

1. Press the `a` key to toggle between automatic and manual label detection.
2. In automatic mode, the application will try to detect the label corners using computer vision techniques.
3. In manual mode, you'll be prompted to click on the four corners of the label.
4. The corners should be selected in clockwise order, starting from the top-left.

## Troubleshooting

1. If Tesseract-OCR is not found:
   - Make sure Tesseract-OCR is installed in the default location
   - Or update the path in `webcam_reader.py`:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```

2. If camera access fails:
   - Make sure no other application is using the camera
   - Check if the camera is properly connected
   - Try running the application with administrator privileges

3. If OCR accuracy is low:
   - Ensure good lighting conditions
   - Keep the label steady and in focus
   - Try adjusting the camera distance

## License

This project is licensed under the MIT License - see the LICENSE file for details. 