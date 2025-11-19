# Gazenet Implementation Plan

## 1. Current State
We have successfully extracted the core "Gaze Estimation Engine" from the original repository and set it up as a standalone project.

### Architecture Overview
The system operates in a linear pipeline:
1.  **Input**: Captures video frame from Webcam (using `opencv`).
2.  **Face Detection**: Detects the user's face using `uniface` (RetinaFace model).
3.  **Preprocessing**: Crops the face region, resizes it to 448x448, and normalizes pixel values.
4.  **Inference**: Feeds the processed face image into our **ResNet-34** model (`models/resnet.py`).
5.  **Output**: The model predicts two continuous values: **Pitch** (vertical angle) and **Yaw** (horizontal angle).

### Directory Structure
*   `models/`: Contains the PyTorch definition of the ResNet architecture.
*   `utils/`: Helper functions for data loading and visualization (drawing the gaze arrow).
*   `weights/`: Stores the pre-trained model files (`resnet34.pt`, etc.).
*   `main.py`: The entry point that runs the webcam loop and prints/visualizes gaze angles.
*   `config.py`: Configuration for model parameters (bins, angle range).

## 2. Roadmap

### Phase 1: Calibration & Mapping (Next Step)
The raw Pitch/Yaw angles need to be converted into Screen Coordinates (X, Y).
*   [ ] Create `calibration.py`.
*   [ ] Implement a UI that asks the user to look at specific screen points (Corners + Center).
*   [ ] Record "Gaze Vector" vs "Screen Pixel" pairs.
*   [ ] Train a simple Linear Regression or Polynomial regressor to map Gaze -> Screen.

### Phase 2: Mouse Control
*   [ ] Create `control.py`.
*   [ ] Use the calibration model to predict screen coordinates in real-time.
*   [ ] Use a library like `pyautogui` to move the mouse cursor.
*   [ ] Implement "Dwell Click" (click if gaze stays still for X seconds) or "Blink Click".

### Phase 3: Browser Integration
*   [ ] **Option A (Extension)**: Build a local WebSocket server in Python that sends coordinates to a Chrome Extension.
*   [ ] **Option B (Wrapper)**: Build a custom browser UI (e.g., using PyQt5 WebEngine) that responds directly to gaze events.

## 3. Immediate Action Items
1.  Verify weights are downloaded.
2.  Start Phase 1 (Calibration).
