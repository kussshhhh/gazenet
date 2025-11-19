import cv2
import logging
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze, pre_process

import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet34`")
    parser.add_argument(
        "--weight",
        type=str,
        default="weights/resnet34.pt",
        help="Path to gaze esimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="0",
                        help="Path to source video file or camera index")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args





def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    logging.info(f"Using device: {device}")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = uniface.RetinaFace()  # third-party face detection library

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")
        return

    gaze_detector.to(device)
    gaze_detector.eval()

    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            success, frame = cap.read()

            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            bboxes, keypoints = face_detector.detect(frame)
            
            # If faces are detected
            if len(bboxes) > 0:
                for bbox, keypoint in zip(bboxes, keypoints):
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])

                    # Ensure bbox is within frame
                    h, w, _ = frame.shape
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    image = frame[y_min:y_max, x_min:x_max]
                    
                    if image.size == 0:
                        continue
                        
                    image = pre_process(image)
                    image = image.to(device)

                    pitch, yaw = gaze_detector(image)

                    pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                    # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                    # Degrees to Radians
                    pitch_predicted = np.radians(pitch_predicted.cpu())
                    yaw_predicted = np.radians(yaw_predicted.cpu())
                    
                    # Log the gaze angles (Pitch, Yaw)
                    # logging.info(f"Pitch: {pitch_predicted.item():.4f}, Yaw: {yaw_predicted.item():.4f}")

                    # draw box and gaze direction
                    draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)
            
            if params.view:
                cv2.imshow('GazeNet Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
