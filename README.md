# Emoji Gesture Recognition App

This desktop application uses computer vision to recognize hand gestures and facial expressions, automatically converting them into corresponding emojis.

## Features

- Real-time hand gesture recognition
- Real-time facial expression recognition
- Automatic emoji copying to clipboard
- Easy switching between hand and face recognition modes
- Adjustable recognition sensitivity
- Visual feedback for successful detections

## Requirements

- Python 3.8+
- Webcam (built-in or USB)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```
2. Select your preferred mode (Hand Gestures or Facial Expressions)
3. Perform gestures or expressions in front of the camera
4. When a gesture/expression is recognized, the corresponding emoji will be automatically copied to your clipboard

## Supported Gestures/Expressions

### Hand Gestures
- OK sign → 👌
- Thumbs up → 👍
- Peace sign → ✌️
- Fist → ✊

### Facial Expressions
- Smile → 😊
- Laugh → 😄
- Surprise → 😮
- Neutral → 😐

## Customization

You can adjust the recognition sensitivity using the slider in the application interface.
