import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
import argparse
import json
import time
import uniface
import pyautogui
from config import data_config
from utils.helpers import get_model, pre_process

logging.basicConfig(level=logging.INFO, format='%(message)s')
pyautogui.FAILSAFE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Mouse Control")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name")
    parser.add_argument("--weight", type=str, default="weights/resnet34.pt", help="Path to weights")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name")
    parser.add_argument("--calibration", type=str, default="calibration.json", help="Calibration file")
    parser.add_argument("--smoothing", type=float, default=0.5, help="Smoothing factor (0-1)")
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
        return None, None

    # Take the largest face
    bbox = max(bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    h, w, _ = frame.shape
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)

    image = frame[y_min:y_max, x_min:x_max]
    if image.size == 0:
        return None, None

    image = pre_process(image).to(device)
    pitch, yaw = gaze_detector(image)

    pitch_predicted = F.softmax(pitch, dim=1)
    yaw_predicted = F.softmax(yaw, dim=1)

    pitch_deg = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
    yaw_deg = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

    return pitch_deg.item(), yaw_deg.item()

def main(args):
    # Load Calibration
    try:
        with open(args.calibration, 'r') as f:
            calib = json.load(f)
        w_x = np.array(calib["weights_x"])
        w_y = np.array(calib["weights_y"])
        screen_w = calib["screen_width"]
        screen_h = calib["screen_height"]
        logging.info("Calibration loaded successfully.")
    except FileNotFoundError:
        logging.error("Calibration file not found! Run calibration.py first.")
        return

    # Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    face_detector = uniface.RetinaFace()
    gaze_detector = get_model(args.model, args.bins, inference_mode=True)
    gaze_detector.load_state_dict(torch.load(args.weight, map_location=device))
    gaze_detector.to(device)
    gaze_detector.eval()

    idx_tensor = torch.arange(args.bins, device=device, dtype=torch.float32)

    cap = cv2.VideoCapture(0)
    
    # Smoothing variables
    curr_x, curr_y = pyautogui.position()
    
    logging.info("Starting Control Loop. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        pitch, yaw = get_gaze_vector(frame, face_detector, gaze_detector, device, idx_tensor, args)
        
        if pitch is not None and yaw is not None:
            # Apply Calibration Mapping
            # x = w_x[0]*yaw + w_x[1]*pitch + w_x[2]
            # y = w_y[0]*yaw + w_y[1]*pitch + w_y[2]
            
            target_x = w_x[0]*yaw + w_x[1]*pitch + w_x[2]
            target_y = w_y[0]*yaw + w_y[1]*pitch + w_y[2]
            
            # Clamp to screen
            target_x = max(0, min(screen_w, target_x))
            target_y = max(0, min(screen_h, target_y))
            
            # Smoothing
            curr_x = curr_x * args.smoothing + target_x * (1 - args.smoothing)
            curr_y = curr_y * args.smoothing + target_y * (1 - args.smoothing)
            
            # Move Mouse
            pyautogui.moveTo(curr_x, curr_y)
            
            # Visualization (Optional)
            cv2.putText(frame, f"Gaze: ({int(curr_x)}, {int(curr_y)})", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gaze Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)
