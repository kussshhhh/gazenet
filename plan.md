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
## 4. Technical Deep Dive

### How do we get Coordinates from a Vector?
The "Vector" is just two angles: **Pitch** (up/down) and **Yaw** (left/right).
To get Screen Coordinates $(X, Y)$, we use **Polynomial Regression**.

We assume there is a mathematical relationship:
$$ X_{screen} = A \cdot Yaw + B \cdot Pitch + C $$
$$ Y_{screen} = D \cdot Yaw + E \cdot Pitch + F $$

During **Calibration**, we collect known pairs of $(Yaw, Pitch)$ and $(X, Y)$. We then solve for the weights $A, B, C, D, E, F$ that minimize the error.

### The "Same Vector" Problem
*   **Question**: "What if two points fall on the same vector?"
*   **Answer**: This happens if you move your head. If you shift your head 10cm to the right but keep looking at the same angle, you will be looking at a different spot on the wall (or screen).
*   **Solution**: Our current simple model assumes your head position is relatively **stationary** (like sitting in front of a laptop). If you move your head significantly, you must **Recalibrate**.
*   **Advanced Solution (Future)**: We would need to estimate the 3D Head Position $(X, Y, Z)$ relative to the camera and intersect the gaze ray with the screen plane. This requires camera calibration and is much more complex.

### Handling Error Rates (Jitter)
The raw output from the AI model is noisy. It "jitters" even when your eyes are still.
*   **Solution**: We implement **Smoothing** (Exponential Moving Average).
    $$ P_{current} = \alpha \cdot P_{raw} + (1 - \alpha) \cdot P_{previous} $$
    *   If $\alpha$ is low (e.g., 0.1), the cursor is very smooth but laggy.
    *   If $\alpha$ is high (e.g., 0.9), the cursor is fast but jittery.
    *   We currently use $\alpha=0.5$ in `control.py`.

### What happens after we get coordinates?
Once we have $(X, Y)$, we have two paths:
1.  **OS Level Control (`control.py`)**: We move the actual system mouse. This works for *everything* (Browser, Games, Desktop).
2.  **Browser Integration**:
    *   We can inject JavaScript to show a "Gaze Cursor" inside the web page.
    *   We can implement **"Smart Snapping"**: If your gaze is near a button, we "snap" the cursor to the button center to make clicking easier.

