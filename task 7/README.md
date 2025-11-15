# CABAC Coder for Video

This project implements a simplified Context-Adaptive Binary Arithmetic Coding (CABAC) encoder in Python and compares its behaviour on two types of video:
- A generic AI-generated walking video (YouTube)
- A specialized screen recording of Visual Studio Code

We test four CABAC context setups: `generic`, `screen_recording`, `surveillance` and `animation`, and compare compression ratio and context adaptation on both clips.

## Requirements

- Python 3.x  
- opencv-python  
- numpy  
- matplotlib  

Install with:

    pip install opencv-python numpy matplotlib

## How to run

    python app7.py

Then follow the menu:
1. Choose "Compare ALL CABAC algorithms"
2. Enter the path to the video file
3. Enter the number of frames (e.g. 40 for the generic clip, 50 for the screen recording)
4. Answer "yes" to persist context between frames

Plots and results are saved as PNG images and printed in the console.


