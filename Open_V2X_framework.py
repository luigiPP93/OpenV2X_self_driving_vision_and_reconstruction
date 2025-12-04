import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from camera_calibration import calib
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from parameter import ProcessingParams
import pandas as pd
import time
from codecarbon import EmissionsTracker,OfflineEmissionsTracker
import csv
#from detect_line import Line

from modules.Object_detection_module.lane_detection_ultrafast import detect_lanes_in_frame
from modules.Object_detection_module.object_detection import load_yolo_model,detect_objects,draw_rectangles_and_text
from modules.Reconstruction_module.reconstruction import overlay_png,overlay_fixed_car_image
from modules.Object_detection_module.lane_detection_pipline import pipeline
from modules.MQTT_module.mqtt_V2X import MQTTClient
from modules.Interface_module.interface_user import add_to_frames_to_save, run_tkinter

'''sudo systemctl start mosquitto
sudo systemctl status mosquitto
sudo systemctl stop mosquitto'''

# Variabili globali
type_of_inconvenient = None
last_overlay_time = 0

# Aggiungi queste variabili globali
 
incident_active = False
traffic_active = False
road_close_active=False

road_close_time=0
incident_time = 0
traffic_time = 0

icon_width = None
spacing = 20
total_icons_width = None
background_color = (200, 200, 200)  # RGB
background_height = None
background_width = None
x_start = None
x_end = None
center_x=0
img_back=True
icon=True
gray_background=None
original_gray_background=None



pipeline_chek = True



frames_to_save = []  # Lista per memorizzare i frame da salvare
MAX_FRAMES_TO_SAVE = 50 #Immagini da salvare
#frames_to_save = deque(maxlen=MAX_FRAMES_TO_SAVE) # Coda per memorizzare i frame da salvare
detected_front_vehicles = []
detected_vehicles = {}
vehicle_counter = 0
current_offsets={}
target_offsets = {}
destra = False
sinistra = False
use_car_fix = False

performance_data = pd.DataFrame(columns=['Frame', 'Processing Time'])

def process_frame(params, mqtt_client=None):
    global vehicle_counter,detected_front_vehicles,detected_vehicles, destra,sinistra

    frame = params.frame_resized
    yolo_model = params.yolo_model
    window_scale_factor = params.window_scale_factor
    car_fix = params.car_fix
    car_fix2 = params.car_fix2
    car_back_img = params.car_back_img
    car_back_imgS = params.car_back_imgS
    car_front_imgS = params.car_front_imgS
    car_front_img = params.car_front_img
    stop_img = params.stop_img
    mtx = params.mtx
    dist = params.dist
    focal_length_px = params.focal_length_px
    vehicle_height_m = params.vehicle_height_m
    moto_back_img = params.moto_back_img
    moto_back_imgS = params.moto_back_imgS
    car_fix_curve_left = params.car_fix_curve_left
    car_fix_curve_right = params.car_fix_curve_right
    car_fix_move = params.car_fix_move
    car_back_imgM = params.car_back_imgM
    car_front_imgM = params.car_front_imgM
    moto_back_imgM = params.moto_back_imgM
    car_fix2_move = params.car_fix2_move
    car_fix_curve_left_move = params.car_fix_curve_left_move
    car_fix_curve_right_move = params.car_fix_curve_right_move
    truck_back_img = params.truck_back_img
    truck_back_imgS = params.truck_back_imgS
    truck_back_imgM = params.truck_back_imgM
    traffic=params.traffic
    accident=params.accident
    road_close=params.road_close
    

    detected_vehicles.clear()
    vehicle_counter = 0
    global icon_width, total_icons_width, background_height, background_width, x_start, x_end, gray_background,img_back,original_gray_background,background_color,center_x
    global incident_active, traffic_active,road_close_active, road_close_time, incident_time, traffic_time
    global icon_width, total_icons_width, background_width,icon,center_x,spacing,destra,sinistra,use_car_fix,use_car_fix
    
    # Sistema modulare per il rilevamento delle lane
    if (pipeline_chek):
        img, lane, parameters = pipeline(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mtx, dist,"video")#"Day","Night", "Rain"
    else:
        img, lane, parameters, mask = detect_lanes_in_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    final_image, background_height, x_start, x_end, original_gray_background, fixed_image, destra, sinistra, use_car_fix = overlay_fixed_car_image(
    car_fix, car_fix2, car_fix_curve_left, car_fix_curve_right,
    window_scale_factor, parameters, car_fix_move, car_fix2_move,
    car_fix_curve_left_move, car_fix_curve_right_move, traffic, accident, road_close,
    mqtt_client.get_incident_status()['incident_active'],
    mqtt_client.get_incident_status()['traffic_active'],
    mqtt_client.get_incident_status()['road_close_active'],
    mqtt_client.get_incident_status()['road_close_time'],
    mqtt_client.get_incident_status()['incident_time'],
    mqtt_client.get_incident_status()['traffic_time'],
    icon_width, total_icons_width, background_width, icon, center_x, spacing,
    destra, sinistra, use_car_fix, original_gray_background,
    background_height, x_start, x_end,frame
)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = detect_objects(img, yolo_model)
    
    names = yolo_model.names
    distance_factor = vehicle_height_m * focal_length_px

    rectangles = []
    texts = []
    detected_vehicles = {}
    
    # Lista per accumulare i dati da trasmettere via MQTT
    mqtt_data = {
        "frame_id": params.frame_id if hasattr(params, "frame_id") else None,
        "vehicles": []
    }
    
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
        
        vehicle_counter = len(detected_vehicles) + 1
        detected_vehicles[vehicle_counter] = (xmin, ymin, xmax, ymax)
        
        overlay_png(
            final_image, coordinates, label, window_scale_factor, car_back_img, car_back_imgS,
            car_front_imgS, car_front_img, stop_img, confidence, moto_back_img, moto_back_imgS,
            distance_m, car_back_imgM, car_front_imgM, moto_back_imgM, truck_back_img, truck_back_imgS, truck_back_imgM,
            destra,sinistra #global

        )
        
        # Aggiungi i dati del veicolo rilevato al messaggio MQTT
        mqtt_data["vehicles"].append({
            "id": vehicle_counter,
            "label": label,
            "confidence": confidence,
            "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "center": {"x": center_x, "y": center_y},
            "distance_m": round(distance_m, 2)
        })
    
    draw_rectangles_and_text(img, rectangles, texts)
    
    rows, cols = img.shape[:2]

    # Invia i dati rilevati tramite MQTT
    if mqtt_client:
        mqtt_client.transmit("intelligent-driving/vehicles", mqtt_data)


    return final_image, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



def add_performance_measure(frame_number, processing_time):
    global performance_data
    new_row = pd.DataFrame({'Frame': [frame_number], 'Processing Time': [processing_time]})
    performance_data = pd.concat([performance_data, new_row], ignore_index=True)


def main():
    
    yolo_model = load_yolo_model()

    video_paths = [
    'test_video/project_video.mp4', #[0]
    'Lane_detect/advanced-lane-detection-for-self-driving-cars-master/harder_challenge_video.mp4',
    'test_video/prova1.mp4',
    '/media/vrlab/video/video/output_compressed_video.mp4',
    '/media/vrlab/video/video/normal.mp4',  # Night [4]
    '/media/vrlab/video/video/pioggia_2224x1080.mp4',  # Rain [5]
    '/media/vrlab/video/video/output_2224x1080.mp4',  # Day [6]
    'advanced-lane-detection-for-self-driving-cars-master/challenge_video.mp4',
    'videoStrada2.mp4'
    ]
    



    # Seleziona il video desiderato, ad esempio l'indice 0 per il primo video
    video_path = video_paths[0]

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore nell'apertura del video.")
        return
    
    #video_path=None

    # Definizione delle dimensioni ridotte delle finestre di visualizzazione
    window_scale_factor = 1/2  # Riduzione del 50%
    original_width = 960
    original_height = 540

    # Nuova risoluzione
    new_width = 960
    new_height = 540
    #426 x 240
    #960, 540
    #640,360

    # Calcolo il fattore di scala per la nuova risoluzione
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    window_scale_factor = (scale_x + scale_y) / 2
    # Creiamo un ThreadPoolExecutor con un numero arbitrario di thread
    executor = ThreadPoolExecutor(max_workers=4)
    window_scale_factor = 1/2 #1/4 per 500x300
    #devo fare tutto più piccolo in base all dimenzione dello schemro
    
    
    car_back_img = cv2.imread('img_Project/car_back2.png', cv2.IMREAD_UNCHANGED)
    car_front_img = cv2.imread('img_Project/car_front2.png', cv2.IMREAD_UNCHANGED)
    stop_img = cv2.imread('img_Project/stop.png', cv2.IMREAD_UNCHANGED)
    moto_back = cv2.imread('img_Project/moto_back.png', cv2.IMREAD_UNCHANGED)
    truck_back = cv2.imread('img_Project/truck_back.png', cv2.IMREAD_UNCHANGED)
    
    car_back_img_original = car_back_img.copy()
    car_front_img_original = car_front_img.copy()
    truck_back_img_original = truck_back.copy()

    # Ridimensiona direttamente dall'immagine originale
    car_back_img = cv2.resize(car_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    car_back_imgM = cv2.resize(car_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    car_back_imgS = cv2.resize(car_back_img_original, (int(170 * window_scale_factor), int(170 * window_scale_factor)), interpolation=cv2.INTER_AREA)

    car_front_img = cv2.resize(car_front_img_original, (int(80 * window_scale_factor), int(80 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    car_front_imgM = cv2.resize(car_front_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    car_front_imgS = cv2.resize(car_front_img_original, (int(170 * window_scale_factor), int(170 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    moto_back_img = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    moto_back_imgM = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    moto_back_imgS = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    truck_back_img = cv2.resize(truck_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    truck_back_imgM = cv2.resize(truck_back_img_original, (int(160 * window_scale_factor), int(160 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    truck_back_imgS = cv2.resize(truck_back_img_original, (int(200 * window_scale_factor), int(200 * window_scale_factor)), interpolation=cv2.INTER_AREA)

   
    car_fix = cv2.imread('img_Project/carline.png', cv2.IMREAD_UNCHANGED)
    car_fix = cv2.resize(car_fix, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)

    car_fix_move = cv2.imread('img_Project/carline2.png', cv2.IMREAD_UNCHANGED)
    car_fix_move = cv2.resize(car_fix_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    
    car_fix2 = cv2.imread('img_Project/no_carline.png', cv2.IMREAD_UNCHANGED)
    car_fix2 = cv2.resize(car_fix2, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    car_fix2_move = cv2.imread('img_Project/no_carline2.png', cv2.IMREAD_UNCHANGED)
    car_fix2_move = cv2.resize(car_fix2_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    
    car_fix_curve_left = cv2.imread('img_Project/carline_left.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_left = cv2.resize(car_fix_curve_left, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)

    car_fix_curve_left_move = cv2.imread('img_Project/carline_left2.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_left_move = cv2.resize(car_fix_curve_left_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    
    car_fix_curve_right = cv2.imread('img_Project/carline_right.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_right = cv2.resize(car_fix_curve_right, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    car_fix_curve_right_move = cv2.imread('img_Project/carline_right2.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_right_move = cv2.resize(car_fix_curve_right_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    

    accident = cv2.imread('img_Project/accident.png', cv2.IMREAD_UNCHANGED)
    accident = cv2.resize(accident, (int(100 * window_scale_factor), int(100 * window_scale_factor)),interpolation=cv2.INTER_AREA)

    traffic = cv2.imread('img_Project/traffic-jam.png', cv2.IMREAD_UNCHANGED)
    traffic = cv2.resize(traffic, (int(100 * window_scale_factor), int(100 * window_scale_factor)),interpolation=cv2.INTER_AREA)

    road_close = cv2.imread('img_Project/road.png', cv2.IMREAD_UNCHANGED)
    road_close = cv2.resize(road_close, (int(100 * window_scale_factor), int(100 * window_scale_factor)),interpolation=cv2.INTER_AREA)



    #car_back_img=cv2.resize(car_back_img, (int(100 * window_scale_factor), int(100 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    mtx, dist = calib()
    # th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
    # th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)
    
    # Parametri della telecamera
    focal_length_px = mtx[1, 1]  # Distanza focale in pixel (f_y)
    vehicle_height_m = 1.5  # Altezza media del veicolo in metri
    # Assicurati che l'immagine sia stata caricata correttamente
    
    # Inizializzazione della finestra Tkinter in un thread separato
    tkinter_thread = Thread(target=run_tkinter)
    tkinter_thread.start()
    frame_number = 0
    #tracker = EmissionsTracker()
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    tracker.start()
    csv_filename = "manual_labels.csv"
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Label"])  # Intestazione CSV
    mqtt_client = MQTTClient(broker="127.0.0.1", port=1883)
    mqtt_client.start()

    while True:
        
        if video_path != None:
            ret, frame = cap.read()

            if not ret:
                break
        else:
            frame = cv2.imread("runs/detect/predict5/t.png")
        window_scale_factor= 1/2
        # Ridimensioniamo il frame letto per corrispondere alle dimensioni ridotte
        add_to_frames_to_save(frames_to_save)
        frame_resized = cv2.resize(frame, (960 ,540), interpolation=cv2.INTER_AREA) #640 ,360 va più veloce devo scalare tutto in proporzione a questo però. grazie dmani si prova, devo fare la proporzione anche con i poligoni 426 x 240
        #provare a lasciare così e fare il resize alle fine
        
        start_time = time.time()
        
        #add_to_frames_to_save(frame)
        
    
        
        params = ProcessingParams(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), yolo_model, window_scale_factor, car_fix, car_fix2, car_back_img,
                                  car_back_imgS, car_front_imgS, car_front_img, stop_img, mtx, dist,focal_length_px, vehicle_height_m, moto_back_img,
                                  moto_back_imgS, car_fix_curve_left, car_fix_curve_right, car_fix_move, car_back_imgM, car_front_imgM, moto_back_imgM,
                                  car_fix2_move, car_fix_curve_left_move, car_fix_curve_right_move,
                                  truck_back_img, truck_back_imgM, truck_back_imgS,traffic,accident,road_close)
        
        future = executor.submit(process_frame, params,mqtt_client)
        
        # Recuperiamo il risultato del processamento del frame
        gray_background, img = future.result()
        #rint("Shape img TTTT", img.shape)
        #print("Shape gray_background UUUUU", gray_background.shape)
        #print("img_width:",img.shape[1],"img_height:",img.shape[0])
        height = gray_background.shape[0]  # Altezza di gray_background
        width = gray_background.shape[1]   # Larghezza di gray_background
        #print("width:", width, "height:", height)
        img_resized = cv2.resize(img, (width, height))  # Mantiene la larghezza originale di img
        # Ora puoi concatenare
        concatenated_img = np.concatenate((gray_background, img_resized), axis=1)
        #print("concat_width:",concatenated_img.shape[1],"concat_height:",concatenated_img.shape[0])
        frames_to_save.append(concatenated_img.copy())
        end_time = time.time()
        processing_time = end_time - start_time
        cv2.namedWindow('Object Detection Overlay', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection Overlay', int(img.shape[1]), int(img.shape[0]))

        cv2.imshow('Object Detection Overlay', concatenated_img)
    
        
        # # Aggiungi la misura prestazionale
        add_performance_measure(frame_number, processing_time)
        
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    performance_data.to_csv('performance_data_video.csv', index=False)
    #print(performance_data)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Ferma il tracker di CodeCarbon
    tracker.stop()
    

if __name__ == "__main__":
    main()
