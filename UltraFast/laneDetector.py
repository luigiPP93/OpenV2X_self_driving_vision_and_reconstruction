import cv2
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum

from UltraFast.model import parsingNet


lane_colors = [(0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 255, 0)]

tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                       168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                       220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                       272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class ModelType(Enum):
    TUSIMPLE = 0
    CULANE = 1


class ModelConfig():

    def __init__(self, model_type):

        if model_type == ModelType.TUSIMPLE:
            self.init_tusimple_config()
        else:
            self.init_culane_config()

    def init_tusimple_config(self):
        self.img_w = 1280
        self.img_h = 720
        self.row_anchor = tusimple_row_anchor
        self.griding_num = 100
        self.cls_num_per_lane = 56

    def init_culane_config(self):
        self.img_w = 1640
        self.img_h = 590
        self.row_anchor = culane_row_anchor
        self.griding_num = 200
        self.cls_num_per_lane = 18


class LaneDetector():

    def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

        self.use_gpu = use_gpu

        # Load model configuration based on the model type
        self.cfg = ModelConfig(model_type)

        # Initialize model
        self.model = self.initialize_model(model_path, self.cfg, use_gpu)

        # Initialize image transformation
        self.img_transform = self.initialize_image_transform()

    @staticmethod
    def initialize_model(model_path, cfg, use_gpu):

        # Load the model architecture
        net = parsingNet(pretrained=False, backbone='18', cls_dim=(cfg.griding_num + 1, cfg.cls_num_per_lane, 4),
                         use_aux=False)  # do not need auxiliary segmentation in testing

        # Load the weights from the downloaded model
        if use_gpu:
            if torch.backends.mps.is_built():
                net = net.to("mps")
                state_dict = torch.load(model_path, map_location='mps')['model']  # Apple GPU
                print("Using Apple GPU")
            else:
                net = net.cuda()
                state_dict = torch.load(model_path, map_location='cuda')['model']  # CUDA
                print("Using CUDA")
        else:
            state_dict = torch.load(model_path, map_location='cpu')['model']  # CPU
            print("Using CPU")

        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        # Load the weights into the model
        net.load_state_dict(compatible_state_dict, strict=False)
        
        #net.eval()

        return net

    @staticmethod
    def initialize_image_transform():
        # Create transform operation to resize and normalize the input images
        img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return img_transforms

    def detect_lanes(self, image, draw_points=True):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        output = self.inference(input_tensor)

        # Process output data
        self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

        # Draw depth image
        visualization_img,text,mask = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)
        
        lanes_status = {
            'left': self.lanes_detected[1],   # Corsia sinistra
            'right': self.lanes_detected[2]   # Corsia destra
        }

        return visualization_img, lanes_status, text, mask

    def prepare_input(self, img):
        # Transform the image for inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        input_img = self.img_transform(img_pil)
        input_tensor = input_img[None, ...]

        if self.use_gpu:
            if not torch.backends.mps.is_built():
                input_tensor = input_tensor.cuda()

        return input_tensor

    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)

        return output

    @staticmethod
    def process_output(output, cfg):
        # Parse the output of the model
        processed_output = output[0].data.cpu().numpy()
        processed_output = processed_output[:, ::-1, :]
        prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        processed_output = np.argmax(processed_output, axis=0)
        loc[processed_output == cfg.griding_num] = 0
        processed_output = loc

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        lanes_points = []
        lanes_detected = []

        max_lanes = processed_output.shape[1]
        for lane_num in range(max_lanes):
            lane_points = []
            # Check if there are any points detected in the lane
            if np.sum(processed_output[:, lane_num] != 0) > 2:

                lanes_detected.append(True)

                # Process each of the points for each lane
                for point_num in range(processed_output.shape[0]):
                    if processed_output[point_num, lane_num] > 0:
                        lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1,
                                      int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane - 1 - point_num] / 288)) - 1]
                        lane_points.append(lane_point)
            else:
                lanes_detected.append(False)

            lanes_points.append(lane_points)
        # return np.array(lanes_points), np.array(lanes_detected)
        return lanes_points, lanes_detected


    @staticmethod
    def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
        # Initialize status
        text = "UNKNOWN"
        mask_img = np.zeros_like(input_img)
        
        # Resize image for visualization
        visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)
        
        # Draw detected lane points if requested
        if draw_points:
            for lane_num, lane_points in enumerate(lanes_points):
                for lane_point in lane_points:
                    cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)

        # Process only if both main lanes are detected
        if lanes_detected[1] and lanes_detected[2]:
            lane_segment_img = visualization_img.copy()

            mask_img = np.zeros_like(lane_segment_img)
            
            # Draw lane areas
            main_lane_points = np.vstack((lanes_points[1], np.flipud(lanes_points[2])))
            cv2.fillPoly(lane_segment_img, pts=[main_lane_points], color=(0, 255, 0))
            
            # Draw adjacent lanes if detected
            if lanes_points[0]:
                left_adjacent = np.vstack((lanes_points[0], np.flipud(lanes_points[1])))
                cv2.fillPoly(lane_segment_img, pts=[left_adjacent], color=(0, 0, 255))
            if lanes_points[3]:
                right_adjacent = np.vstack((lanes_points[2], np.flipud(lanes_points[3])))
                cv2.fillPoly(lane_segment_img, pts=[right_adjacent], color=(0, 0, 255))
                
            # Blend the lane overlay
            visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

            # Calculate lane characteristics
            left_lane = np.array(lanes_points[1])
            right_lane = np.array(lanes_points[2])
            
            # Calculate the lane center line
            center_points = []
            min_points = min(len(left_lane), len(right_lane))
            for i in range(min_points):
                center_x = (left_lane[i][0] + right_lane[i][0]) // 2
                center_y = (left_lane[i][1] + right_lane[i][1]) // 2
                center_points.append([center_x, center_y])
            center_line = np.array(center_points)
            
            # Calculate lane direction using polynomial fit
            if len(center_line) > 2:
                # Fit a second degree polynomial to the center line
                poly_coeffs = np.polyfit(center_line[:, 1], center_line[:, 0], 2)
                
                # Calculate curvature
                deriv_coeffs = np.polyder(poly_coeffs)
                curvature = np.abs(deriv_coeffs[0])
                
                # Calculate deviation from center
                height = visualization_img.shape[0]
                width = visualization_img.shape[1]
                bottom_center_x = np.polyval(poly_coeffs, height)
                center_deviation = bottom_center_x - width/2
                
                # Determine direction based on curvature and deviation
                CURVE_THRESHOLD = 0.1
                DEVIATION_THRESHOLD = 30
                
                if curvature < CURVE_THRESHOLD:
                    if abs(center_deviation) < DEVIATION_THRESHOLD:
                        text = "STRAIGHT"
                    elif center_deviation < -DEVIATION_THRESHOLD:
                        text = "TURN LEFT"
                    else:
                        text = "TURN RIGHT"
                else:
                    if center_deviation < -DEVIATION_THRESHOLD:
                        text = "TURN LEFT"
                    elif center_deviation > DEVIATION_THRESHOLD:
                        text = "TURN RIGHT"
                    if abs(center_deviation) > DEVIATION_THRESHOLD * 2:
                        text = "CHANGING LANE"
                
                # Draw visualization elements
                # Center line
                for i in range(len(center_line)-1):
                    cv2.line(visualization_img, 
                            tuple(center_line[i]), 
                            tuple(center_line[i+1]), 
                            (0, 255, 255), 2)
                
                # Reference line
                cv2.line(visualization_img, 
                        (width//2, height), 
                        (width//2, height-150), 
                        (255, 255, 255), 2)
                
                # Status indicators
                status_bg = visualization_img.copy()
                cv2.rectangle(status_bg, 
                            (width//2-200, height-100),
                            (width//2+200, height-40),
                            (0, 0, 0), -1)
                visualization_img = cv2.addWeighted(visualization_img, 0.7, status_bg, 0.3, 0)
                
                # Draw text with outline
                cv2.putText(visualization_img, text,
                            (width//2-180, height-60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4)
                cv2.putText(visualization_img, text,
                            (width//2-180, height-60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 200, 50), 2)
                
                # Draw deviation indicator
                deviation_x = int(width/2 + center_deviation/2)
                cv2.circle(visualization_img, 
                        (deviation_x, height-70),
                        10, (0, 255, 255), -1)
                
        return visualization_img, text,mask_img
    
    @staticmethod
    def draw_lanes_metrics(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
        text = "UNKNOWN"
        
        # Resize image for visualization
        visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)

        if draw_points:
            for lane_num, lane_points in enumerate(lanes_points):
                for lane_point in lane_points:
                    cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)

        # Draw a mask for the current lane
        mask_img = np.zeros_like(input_img)
        if lanes_detected[1] and lanes_detected[2]:
            lane_segment_img = visualization_img.copy()

            # Disegna l'area tra le corsie principali
            cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[1], np.flipud(lanes_points[2])))],
                        color=(0, 255, 0))
            
            mask_img = np.zeros_like(lane_segment_img)

            # Disegna il poligono verde sulla maschera
            cv2.fillPoly(mask_img, pts=[np.vstack((lanes_points[1], np.flipud(lanes_points[2])))],
                        color=(0, 255, 0))  # Colore verde (0, 255, 0)

            # Salva la maschera verde
            cv2.imwrite('./output_images/pp3.png', mask_img)

            
            # Disegna le aree delle corsie adiacenti se rilevate
            if lanes_points[0]:
                cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[0], np.flipud(lanes_points[1])))],
                            color=(0, 0, 255))
            if lanes_points[3]:
                cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[2], np.flipud(lanes_points[3])))],
                            color=(0, 0, 255))
            
            visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

            # Dimensioni finestra
            windowWidth = visualization_img.shape[1]
            windowHeight = visualization_img.shape[0]

            

            # Linea centrale di riferimento
            center_reference_x = round(windowWidth / 2)
            center_reference_y_top = round(windowHeight * 2 / 4)
            center_reference_y_bottom = windowHeight
            
            # Disegna linea centrale di riferimento
            cv2.line(visualization_img, 
                    [center_reference_x, center_reference_y_top], 
                    [center_reference_x, center_reference_y_bottom], 
                    (255, 255, 255), 5)

            # Calcola i punti di riferimento sulle corsie
            left_lane_idx = round(3 / 5 * len(lanes_points[1]))
            right_lane_idx = round(3 / 5 * len(lanes_points[2]))

            left_point = lanes_points[1][left_lane_idx]
            right_point = lanes_points[2][right_lane_idx]

            # Calcola il punto centrale tra le corsie
            center_point_x = round((left_point[0] + right_point[0]) / 2)
            center_point_y = round((left_point[1] + right_point[1]) / 2)

            # Calcola la deviazione dal centro
            deviation = center_reference_x - center_point_x
            # Calcolo dell'angolo di sterzata
            deviation = center_reference_x - center_point_x
            distance_to_bottom = windowHeight - center_point_y
            steering_angle = np.arctan2(deviation, distance_to_bottom) * 180 / np.pi
            
            # Soglie per la classificazione della direzione
            STRAIGHT_THRESHOLD = 15
            TURN_THRESHOLD = 90

            # Buffer per stabilizzare il cambio di stato
            direction_buffer = []
            buffer_size = 10

            # Determina la direzione in base alla deviazione
            if abs(deviation) <= STRAIGHT_THRESHOLD or abs(steering_angle) < 18.0:
                current_direction = "STRAIGHT"
            elif deviation > TURN_THRESHOLD:
                current_direction = "CHANGING LANE"
            elif deviation > STRAIGHT_THRESHOLD:
                current_direction = "TURN LEFT"
            elif deviation < -TURN_THRESHOLD:
                current_direction = "CHANGING LANE"
            elif deviation < -STRAIGHT_THRESHOLD:
                current_direction = "TURN RIGHT"
            
            # Aggiorna il buffer e determina la direzione finale
            direction_buffer.append(current_direction)
            if len(direction_buffer) > buffer_size:
                direction_buffer.pop(0)
            
            # La direzione più frequente nel buffer diventa il testo finale
            text = max(set(direction_buffer), key=direction_buffer.count)

            # Visualizzazione
            # Linee verticali sulle corsie
            cv2.line(visualization_img, [left_point[0], left_point[1] - 15],
                    [left_point[0], left_point[1] + 15], (255, 0, 0), 3)
            cv2.line(visualization_img, [right_point[0], right_point[1] - 15],
                    [right_point[0], right_point[1] + 15], (255, 0, 0), 3)

            # Linea centrale effettiva
            cv2.line(visualization_img, [center_point_x, center_point_y - 15],
                    [center_point_x, center_point_y + 15], (0, 255, 0), 3)

            # Linea di direzione
            cv2.line(visualization_img, [center_point_x, center_point_y],
                    [center_reference_x, center_reference_y_bottom], (0, 0, 255), 2)

            # Visualizza il testo con outline
            text_position = (center_reference_x - 140, round(windowHeight - 70))
            cv2.putText(visualization_img, text+f"(Steer: {steering_angle:.1f})", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
            cv2.putText(visualization_img, text+f"(Steer: {steering_angle:.1f})", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        return visualization_img, text,mask_img
