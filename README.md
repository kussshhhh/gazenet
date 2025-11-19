# Gazenet

**Gazenet** is a gaze-controlled interface for your computer. It uses deep learning (ResNet) to estimate where you are looking on the screen using just your webcam, allowing you to control the mouse cursor with your eyes.

## 🚀 Setup

We use **uv** for fast and reliable dependency management.

### 1. Prerequisites
*   **Python 3.10+**
*   **uv** (Fast Python package installer)
    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 2. Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <your-repo-url>
    cd gazenet
    ```

2.  **Create a virtual environment**:
    ```bash
    uv venv
    ```

3.  **Activate the environment**:
    ```bash
    # macOS/Linux
    source .venv/bin/activate

    # Windows
    .venv\Scripts\activate
    ```

4.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

### 3. Download Weights

You need the pre-trained ResNet models to run the gaze estimation. We have a script to download them for you.

```bash
sh download_weights.sh
```

*This will download `resnet18.pt`, `resnet34.pt`, and `resnet50.pt` into the `weights/` directory.*

---

## 🎮 Usage

### Step 1: Calibration (Required)
Before you can control the mouse, the system needs to learn how your gaze maps to your screen coordinates.

```bash
python calibration.py
```
*   **Instructions**: A series of yellow dots will appear on the screen. Look at each dot until it turns green and the system moves to the next one.
*   **Output**: This generates a `calibration.json` file.

### Step 2: Mouse Control
Once calibrated, you can start the mouse control script.

```bash
python control.py
```
*   **Controls**:
    *   **Move**: Look around to move the mouse.
    *   **Exit**: Press `q` to quit.

### Debugging / Visualization
If you just want to see the gaze vector (arrow) without moving the mouse:

```bash
python main.py
```

## 📁 Project Structure
*   `models/`: ResNet architecture definitions.
*   `utils/`: Helper functions for face detection and visualization.
*   `weights/`: Pre-trained model weights.
*   `calibration.py`: Script to map gaze angles to screen pixels.
*   `control.py`: Main script for mouse control.
*   `main.py`: Debug script for visualizing gaze vectors.