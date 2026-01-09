# Hand Gesture Recognition Model

This project implements a real-time hand gesture recognition system using OpenCV, CVZone, and TensorFlow/Keras. It detects hand landmarks and classifies gestures to control a drone (or other applications).

## Project Structure

- **test2_final.py**: The main script for running the real-time gesture recognition.
- **Model/**: Contains the trained Keras model (`mymodel_new.h5`).
- **demo.mp4**: A video demonstration of the project.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install opencv-python cvzone tensorflow numpy matplotlib
```

## Usage

Run the final script to start the hand gesture recognition:

```bash
python test2_final.py
```

The application will open your webcam window. It detects your hand and predicts the gesture.
The model recognizes the following gestures:
- Flip
- Backward
- Down
- Up
- Forward
- Right
- Left
- Land

To quit the application, press `q`.

## Demo
checkout demo.mp4
<video src="./demo.mp4" controls="controls" style="max-width: 100%;">
</video>
