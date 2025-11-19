import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
import argparse
import json
import time
import uniface
from config import data_config
from utils.helpers import get_model, pre_process

logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Calibration")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name")
    parser.add_argument("--weight", type=str, default="weights/resnet34.pt", help="Path to weights")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name")
    parser.add_argument("--output", type=str, default="calibration.json", help="Output calibration file")
    args = parser.parse_args()

    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    return args

def get_gaze_vector(frame, face_detector, gaze_detector, device, idx_tensor, params):
    bboxes, keypoints = face_detector.detect(frame)
    if len(bboxes) == 0:
        return None

    # Take the largest face
    bbox = max(bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    h, w, _ = frame.shape
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)

    image = frame[y_min:y_max, x_min:x_max]
    if image.size == 0:
        return None

    image = pre_process(image).to(device)
    pitch, yaw = gaze_detector(image)

    pitch_predicted = F.softmax(pitch, dim=1)
    yaw_predicted = F.softmax(yaw, dim=1)

    pitch_deg = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
    yaw_deg = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

    return pitch_deg.item(), yaw_deg.item()

def main(args):
    # Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    logging.info(f"Using device: {device}")

    face_detector = uniface.RetinaFace()
    gaze_detector = get_model(args.model, args.bins, inference_mode=True)
    gaze_detector.load_state_dict(torch.load(args.weight, map_location=device))
    gaze_detector.to(device)
    gaze_detector.eval()

    idx_tensor = torch.arange(args.bins, device=device, dtype=torch.float32)

    # Setup Window
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open webcam")
        return

    # Calibration Points (Normalized 0-1)
    points = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
    ]
    
    # Get screen resolution (approximate via window size, or assume standard)
    # For accurate mapping, we need actual screen resolution. 
    # OpenCV doesn't give screen size easily without a window.
    # We will assume the window is full screen and use its size.
    # Wait for window to initialize
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Calibration", dummy)
    cv2.waitKey(100)
    
    # Try to get window size
    # Note: cv2.getWindowImageRect might return (x, y, w, h)
    try:
        _, _, screen_w, screen_h = cv2.getWindowImageRect("Calibration")
    except:
        screen_w, screen_h = 1920, 1080 # Fallback
        logging.warning(f"Could not get window size, using fallback: {screen_w}x{screen_h}")

    logging.info(f"Screen Resolution: {screen_w}x{screen_h}")

    calibration_data = []

    for pt_idx, (px, py) in enumerate(points):
        target_x = int(px * screen_w)
        target_y = int(py * screen_h)
        
        logging.info(f"Look at point {pt_idx+1}/{len(points)}: ({target_x}, {target_y})")

        # Show point and wait
        start_time = time.time()
        samples = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Create display image
            display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            
            # Draw target point
            cv2.circle(display, (target_x, target_y), 20, (0, 255, 255), -1)
            cv2.circle(display, (target_x, target_y), 5, (0, 0, 255), -1)
            
            # Instructions
            elapsed = time.time() - start_time
            if elapsed < 2.0:
                text = "Look at the yellow dot..."
                color = (255, 255, 255)
            elif elapsed < 4.0:
                text = "Recording..."
                color = (0, 255, 0)
                
                # Collect data
                gaze = get_gaze_vector(frame, face_detector, gaze_detector, device, idx_tensor, args)
                if gaze:
                    samples.append(gaze)
            else:
                break

            cv2.putText(display, text, (screen_w//2 - 100, screen_h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow("Calibration", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        if samples:
            avg_pitch = np.mean([s[0] for s in samples])
            avg_yaw = np.mean([s[1] for s in samples])
            calibration_data.append({
                "screen_x": target_x,
                "screen_y": target_y,
                "pitch": avg_pitch,
                "yaw": avg_yaw
            })
            logging.info(f"Recorded: Pitch={avg_pitch:.2f}, Yaw={avg_yaw:.2f}")
        else:
            logging.warning("No face detected for this point.")

    cap.release()
    cv2.destroyAllWindows()

    # Compute Mapping (Linear Regression)
    # x = a*yaw + b*pitch + c
    # y = d*yaw + e*pitch + f
    
    if len(calibration_data) < 4:
        logging.error("Not enough data points for calibration.")
        return

    X_data = np.array([[d['yaw'], d['pitch'], 1] for d in calibration_data])
    Y_x = np.array([d['screen_x'] for d in calibration_data])
    Y_y = np.array([d['screen_y'] for d in calibration_data])

    # Solve for weights
    # w_x = (X^T * X)^-1 * X^T * Y_x
    w_x = np.linalg.lstsq(X_data, Y_x, rcond=None)[0]
    w_y = np.linalg.lstsq(X_data, Y_y, rcond=None)[0]

    calibration_result = {
        "screen_width": screen_w,
        "screen_height": screen_h,
        "weights_x": w_x.tolist(),
        "weights_y": w_y.tolist()
    }

    with open(args.output, 'w') as f:
        json.dump(calibration_result, f, indent=4)
    
    logging.info(f"Calibration saved to {args.output}")
    logging.info(f"Weights X: {w_x}")
    logging.info(f"Weights Y: {w_y}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
