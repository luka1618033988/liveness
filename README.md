# Real-Time Face and Liveness Detection

This project implements a real-time face detection and liveness detection system using OpenCV in Python. It captures video from a webcam, processes frames to detect faces, calculates optical flow in different grid sections of the frame, and determines the liveness of the detected face.

## Features
- **Face Detection**: Detects faces in real-time using OpenCV's Haar Cascades.
- **Liveness Detection**: Determines if the detected face is live or a photograph based on optical flow calculations in different grid sections. Note that I stabilize the frame before processing to eliminate the effect of camera shake from the flow.

All of the checks should be successful in order to pass liveness detection.

## Installation

To run this project, you need Python 3.8 and OpenCV installed. You can install OpenCV using pip:

```bash
pip install opencv-python
```

## Configuration

The configuration parameters are set in the script as follows:

- `DETECTION_TIME`: Duration for which the detection runs after the spacebar is pressed.
- `FLOW_THRESHOLD`: Threshold for the optical flow difference between regions of interest.
- `WIDTHS`, `HEIGHTS`: Dimensions for dividing the video frame into grids for analysis.

## Usage

To run the script, execute the following command in the terminal:

```bash
python main.py
```

Press the spacebar to start the liveness detection process. The status of the detection (Passed/Failed) will be displayed on the screen in corresponding color.

## Development Mode

The script includes a development mode that shows additional gridlines and flow values for debugging purposes. Activate it by setting the `--dev` flag to `True`:

```bash
python main.py --dev True
```

## Failed Experiments
1. Calculate 2d Fourier Transform of the image and check for high frequency components. This method failed because the Fourier Transform of a photograph and a live face are very similar. In case we had datasets of photographs and live faces, we could have trained a classifier to distinguish between the two, however project requirements did not allow for this.
2. Spectral Analysis of the image, rationale being that radiation due to temperature of human body was somehow captured in webcam. This method failed because the spectral analysis of a photograph and a live face are very similar as well (also, mapping RGB back to light wavelengths is not straightforward). Current webcams come with additional filters that block IR light, which would have been useful for this method.
3. Focus check. In this method I tried to change focus of the webcam from min to max and compare the sharpness of the image in different regions. This method failed because quality of my webcam was not good enough to detect the difference in sharpness.