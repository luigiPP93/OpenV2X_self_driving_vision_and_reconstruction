from modules.Object_detection_module.image_processing import get_combined_gradients, get_combined_hls, combine_grad_hls, canny_edge_detection, remove_unwanted_objects
from detect_line import Line, get_perspective_transform, get_lane_lines_img, illustrate_driving_lane, illustrate_info_panel, illustrate_driving_lane_with_topdownview
from camera_calibration import calib, undistort
import concurrent.futures
import cv2
import numpy as np

c_rows, c_cols = None, None
s_LTop2, s_RTop2 = None, None
s_LBot2, s_RBot2 = None, None
punti,src,dst,punti2 = None, None, None, None
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)
left_line = Line()
right_line = Line()

def pipeline(frame, mtx, dist,condizione):
    global c_rows, c_cols, s_LTop2, s_RTop2, s_LBot2, s_RBot2, punti, src, dst, punti2
    identificate_lane_lines = True
    
    # Correcting for Distortion
    undist_img = undistort(frame, mtx, dist)
    rows, cols = undist_img.shape[:2]
    #cv2.imwrite('./output_images/pp100.png', undist_img)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        edge_future = executor.submit(canny_edge_detection, undist_img)
        hls_future = executor.submit(get_combined_hls, undist_img, th_h, th_l, th_s)

    combined_gradient = edge_future.result()
    combined_hls = hls_future.result()

    combined_result = combine_grad_hls(combined_gradient, combined_hls)
    
    # Dimensioni di riferimento
    reference_rows, reference_cols = 308, 960
    
    if s_LTop2 is None or c_rows != rows or c_cols != cols:
        # Aggiorna le dimensioni correnti
        c_rows, c_cols = combined_result.shape[:2]
        
        # Calcola i fattori di scala rispetto alle dimensioni di riferimento
        scale_x = c_cols / reference_cols
        scale_y = c_rows / reference_rows
        
        # Definisci la condizione attuale
        condizione = condizione  # Cambia il valore per testare altre condizioni

        # Dizionario di configurazione con valori normalizzati rispetto alle dimensioni di riferimento
        configurazioni_base = {
            "video": {
                "s_LTop2": [reference_cols / 2 - 55, 120], 
                "s_RTop2": [reference_cols / 2 + 50, 120], 
                "s_LBot2": [190, reference_rows - 40], 
                "s_RBot2": [reference_cols - 100, reference_rows - 40]
            },
            "Night": {
                "s_LTop2": [reference_cols / 2 - 130, 155], 
                "s_RTop2": [reference_cols / 2 + 60, 155], 
                "s_LBot2": [160, reference_rows], 
                "s_RBot2": [reference_cols - 160, reference_rows]
            },
            "Day": {
                "s_LTop2": [reference_cols / 2 - 100, 150], 
                "s_RTop2": [reference_cols / 2 + 100, 150], 
                "s_LBot2": [170, reference_rows], 
                "s_RBot2": [reference_cols - 100, reference_rows]
            },
            "Rain": {
                "s_LTop2": [reference_cols / 2 - 50, 80], 
                "s_RTop2": [reference_cols / 2 + 50, 80], 
                "s_LBot2": [240, reference_rows - 120], 
                "s_RBot2": [reference_cols - 300, reference_rows - 120]
            }
        }
        
        # Crea un nuovo dizionario con le coordinate scalate
        configurazioni = {}
        for key, config in configurazioni_base.items():
            configurazioni[key] = {
                "s_LTop2": [config["s_LTop2"][0] * scale_x, config["s_LTop2"][1] * scale_y],
                "s_RTop2": [config["s_RTop2"][0] * scale_x, config["s_RTop2"][1] * scale_y],
                "s_LBot2": [config["s_LBot2"][0] * scale_x, config["s_LBot2"][1] * scale_y],
                "s_RBot2": [config["s_RBot2"][0] * scale_x, config["s_RBot2"][1] * scale_y]
            }

        # Seleziona la configurazione corretta
        if condizione in configurazioni:
            s_LTop2, s_RTop2 = configurazioni[condizione]["s_LTop2"], configurazioni[condizione]["s_RTop2"]
            s_LBot2, s_RBot2 = configurazioni[condizione]["s_LBot2"], configurazioni[condizione]["s_RBot2"]
        else:
            raise ValueError(f"Condizione '{condizione}' non riconosciuta")
        
        # Anche dst deve scalare con l'immagine
        dst = np.float32([
            [170 * scale_x, 720 * scale_y], 
            [170 * scale_x, 0], 
            [550 * scale_x, 0], 
            [550 * scale_x, 720 * scale_y]
        ])
        
        # Aggiorna punti, punti2 e src
        punti = np.array([[s_LBot2, s_LTop2, s_RTop2, s_RBot2]], dtype=np.int32)
        punti2 = np.array([[dst[0], dst[1], dst[2], dst[3]]], dtype=np.int32)
        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        

    combined_result2=combined_result
    
    
    # Disegna il rettangolo sull'immagine
    cv2.polylines(combined_result2, [punti], isClosed=True, color=(255, 0, 0), thickness=2) 
    #cv2.imwrite('./output_images/pp1.png', combined_result2)
    combined_result = combine_grad_hls(combined_gradient, combined_hls)
    #cv2.imwrite('./output_images/pp2.png', combined_result)
    warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (int(720 * scale_y), int(720 * scale_x)))
    #cv2.imwrite('./output_images/pp3.png', warp_img)
    cv2.polylines(warp_img, [punti], isClosed=True, color=(255, 0, 0), thickness=2)

    mask = np.zeros_like(warp_img)
    # Disegna il rettangolo bianco (o altro colore) esterno
    cv2.fillPoly(mask, [punti2], color=(255, 255, 255))
    # Applica la maschera a warp_img per mantenere solo la regione interna
    masked_img = cv2.bitwise_and(warp_img, mask)
    searching_img = get_lane_lines_img(masked_img, left_line, right_line)

    if left_line.detected and right_line.detected:
        identificate_lane_lines
    else:
        identificate_lane_lines = False

    w_comb_result, w_color_result = illustrate_driving_lane(searching_img, left_line, right_line)
    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    lane_color = np.zeros_like(undist_img)
    # Calcola il fattore di scala per le righe
    scale_y = rows / 540  # Adattato all'altezza dell'immagine

    # Adatta dinamicamente i valori di ritaglio
    crop_top = int(220 * scale_y)
    crop_bottom = int(12 * scale_y)

    lane_color[crop_top:rows - crop_bottom, 0:cols] = cv2.resize(color_result, (cols, rows - crop_top - crop_bottom))

    #lane_color[220:rows - 12, 0:cols] = color_result
    result = cv2.addWeighted(undist_img, 1, lane_color, 0.5, 0)
    birdeye_view_panel, parameters = illustrate_info_panel(result, left_line, right_line)
    

    return result,identificate_lane_lines,parameters
