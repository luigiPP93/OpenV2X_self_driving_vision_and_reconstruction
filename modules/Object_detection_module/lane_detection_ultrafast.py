import cv2
from UltraFast import LaneDetector, ModelType

# Imposta i parametri del modello
model_path = "models/culane_18.pth"
model_type = ModelType.CULANE
#model_path = "models/tusimple_18.pth"
#model_type = ModelType.TUSIMPLE
use_gpu = True

# Inizializza il modello di rilevamento delle corsie
laneDetector = LaneDetector(model_path, model_type, use_gpu)

def detect_lanes_in_frame(frame):
    # Rileva le corsie
    output_img, lanes_status,text,mask = laneDetector.detect_lanes(frame)
    height = frame.shape[0]  # Altezza di gray_background
    width = frame.shape[1]   # Larghezza di gray_background

        
    img_resized = cv2.resize(output_img, (width, height))  # Mantiene la larghezza originale di img
    
    return img_resized, lanes_status,text,mask