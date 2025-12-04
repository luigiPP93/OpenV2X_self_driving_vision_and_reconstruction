import sys
import cv2
import torch

# This method load yolo_model with PyTorch Hub


yolo_repo_path = 'yolo/yolov5-master'
model_path = 'yolo/yolov5_vehicle_oriented.pt'

def load_yolo_model():
    
    sys.path.append(yolo_repo_path)
    yolo_model = torch.hub.load(yolo_repo_path, 'custom', path=model_path, source='local')
    return yolo_model

# This method detects objects in a frame using a YOLO model.
def detect_objects(frame, yolo_model):
    yolo_outputs = yolo_model(frame)
    output = yolo_outputs.xyxy[0]
    return output


def process_detections(output, names, distance_factor):
    rectangles = []
    texts = []
    detection_data = []
    
    for j in range(len(output)):
        output_j = output[j]
        label = names[int(output_j[5])]
        confidence = round(output_j[4].item(), 2)
        coordinates = output_j[:4].int().tolist()
        xmin, ymin, xmax, ymax = coordinates
        
        rectangles.append(((xmin, ymin), (xmax, ymax)))
        texts.append(((xmin, ymin - 10), f'{label} {confidence}'))
        
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        bbox_height_px = ymax - ymin
        distance_m = distance_factor / bbox_height_px
        distance_text = f'{distance_m:.2f}m'
        texts.append(((center_x, center_y), distance_text))
        
        detection_data.append({
            'coordinates': coordinates,
            'label': label,
            'confidence': confidence,
            'center': (center_x, center_y),
            'distance_m': distance_m
        })
    
    return rectangles, texts, detection_data

def draw_rectangles_and_text(img, rectangles, texts):
    for (pt1, pt2) in rectangles:
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    for ((x, y), text) in texts:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    