import cv2
import numpy as np
import time

global destra
global sinistra
gray_background=None
original_gray_background=None
background_color = (200, 200, 200)  # RGB
background_height = None
background_width = None


def overlay_png(image, coordinates, labels, window_scale_factor, car_back_img, car_back_imgS, car_front_imgS, car_front_img, stop_img, confidence, moto_back, moto_backS, distance_m, car_back_imgM, car_front_imgM, moto_back_imgM, truck_back_img, truck_back_imgS, truck_back_imgM,destra, sinistra):
    label_to_image = {
        'car_back': car_back_img,
        'car_front': car_front_img,
        'stop sign': stop_img,
        'motorcycle_back': moto_back,
        'truck_back': truck_back_img
    }

    if labels not in label_to_image:
        #(f"Errore: L'etichetta '{labels}' non è presente nel dizionario.")
        return image  # Esce dalla funzione senza fare nulla se l'etichetta non esiste nel dizionario


    

    png_image = label_to_image[labels]
    png_height, png_width, _ = image.shape
    x1, y1, x2, y2 = map(int, coordinates)
    x = max(0, x1)
    y = max(0, y1)
    distance = distance_m
    x_offset = (1 - (distance / 250)) * png_height
    y = int(x_offset)

    if confidence <= 0.30 or labels not in label_to_image or distance > 109:
        return image

    # Adjustments for night
     # Se aumenatto sposta in alto le auto
    # Se aumento sposta a destra le auto
    y-=200 #immagine più grande se aumento più in alto
    angle_factor = 0  # Nessuna correzione di default
    if destra:
        angle_factor = -1  # Sposta i veicoli più a destra se curvi a destra
    elif sinistra:
        angle_factor = 1   # Sposta i veicoli più a sinistra se curvi a sinistra

    
    x += angle_factor
    #y-=90 #immagine inetermedia
    #y-=60 # immagine più piccola

    

    def resize_and_rotate(image, distance, labels, destra, sinistra, x,y,img_width):
        c=False
        '''if x > img_width // 2 and distance < 20:
            x -=20
        elif x > img_width // 2 and distance<20:
            x +=20'''

        if labels == 'car_front':
            if distance <= 10:
                x -= 1
                return car_front_imgS, x,y
            elif distance <= 50:
                x -= 1  # Correzione posizione per la prospettiva
                return car_front_imgS, x,y
            elif distance <= 60:
                return car_front_imgM, x,y
            else:
                return car_front_img, x,y
        elif labels == 'car_back':
            x +=15
            if distance <= 18:
                x-=100
                if destra:
                    x -= 30
                elif sinistra:
                    x += 30
                if x > img_width // 2:
                    x -=1
                else:
                    x +=1
                resized_png = car_back_imgS
            elif distance <= 50:
                resized_png = car_back_imgS
            elif distance <= 60:
                if sinistra:
                    y +=1
                    resized_png = car_back_imgM
                else:
                    resized_png = car_back_imgM
            else:
                resized_png = car_back_img

            if x < img_width // 2:
        
                #print("Questa è la metà=",img_width // 2,"x=",x)
                x -=15 #Sposta il veicolo a sinistra
                c=True
                resized_png = cv2.flip(resized_png, 1)

            if destra:
                if not c:
                    resized_png = cv2.flip(resized_png, 1)
                    c=False
                angle = 5
            elif sinistra:
                if c:
                    resized_png = cv2.flip(resized_png, 1)
                angle = 2
            else:
                angle = 3
            rotated_image = cv2.warpAffine(
                resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), angle, 1.0),
                (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )
            return rotated_image, x,y
        elif labels == 'motorcycle_back':
            if distance < 30:
                resized_png = moto_back
            elif distance <= 50:
                resized_png = moto_back_imgM
            else:
                resized_png = moto_backS
            if x < img_width // 2:
                x -=1 #Sposta il veicolo a sinistra
                resized_png = cv2.flip(resized_png, 1)

            if destra:
                angle = -3
            elif sinistra:
                angle = 3
            else:
                angle = 0
            rotated_image = cv2.warpAffine(
                resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), angle, 1.0),
                (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )
            return rotated_image, x,y
        elif labels == 'truck_back':
            x +=30
            if distance <= 15:
                if destra:
                    x -= 30
                elif sinistra:
                    x += 30
                if x > img_width // 2:
                    x -=30
                else:
                    x +=1
                resized_png = truck_back_imgS
            if distance <= 30:
                resized_png = truck_back_imgS
            elif distance <= 50:
                if sinistra:
                    y +=1
                    resized_png = truck_back_imgM
                else:
                    resized_png = truck_back_imgM
            else:
                resized_png = truck_back_img
            if x < img_width // 2:
                c=True
                x -=15 #Sposta il veicolo a sinistra
                resized_png = cv2.flip(resized_png, 1)
            if destra:
                if not c:
                    resized_png = cv2.flip(resized_png, 1)
                    c=False
                angle = 5
            elif sinistra:
                angle = 2
            else:
                angle = 3
            rotated_image = cv2.warpAffine(
                resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), angle, 1.0),
                (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )
            return rotated_image, x,y
        else:
            resized_png = cv2.resize(image, (int(100 * window_scale_factor), int(100 * window_scale_factor)), interpolation=cv2.INTER_AREA)
            return resized_png, x,y


    resized_png,x,y = resize_and_rotate(png_image, distance, labels, destra, sinistra,x,y,image.shape[1])

    overlay_height, overlay_width, _ = resized_png.shape
    if y + overlay_height > image.shape[0]:
        overlay_height = image.shape[0] - y
    if x + overlay_width > image.shape[1]:
        overlay_width = image.shape[1] - x

    overlay_height = int(overlay_height)
    overlay_width = int(overlay_width)
    overlay = resized_png[:overlay_height, :overlay_width, :3]
    alpha_mask = resized_png[:overlay_height, :overlay_width, 3] / 255.0

    overlay_width = max(1, overlay_width)
    overlay_height = max(1, overlay_height)

    if x < 0: x = 0
    if y < 0: y = 0
    if x + overlay_width > image.shape[1]: overlay_width = image.shape[1] - x
    if y + overlay_height > image.shape[0]: overlay_height = image.shape[0] - y

    if overlay_height > 0 and overlay_width > 0:
        for c in range(3):
            image[y:y + overlay_height, x:x + overlay_width, c] = (
                alpha_mask * overlay[:, :, c] + (1 - alpha_mask) * image[y:y + overlay_height, x:x + overlay_width, c]
            ).astype(np.uint8)

    return image
            
        


def get_fixed_image(parameters, car_fix, car_fix2, car_fix_curve_left, car_fix_curve_right, car_fix_move, car_fix2_move, car_fix_curve_left_move, car_fix_curve_right_move,destra,sinistra,use_car_fix):
    """
    Seleziona l'immagine della macchina da sovrapporre in base alle condizioni stradali.
    """
    
    # Mappatura delle condizioni stradali con le immagini corrispondenti
    road_conditions = {
        "Straight": (car_fix, car_fix_move, False, False),
        "curving to Left": (car_fix_curve_left, car_fix_curve_left_move, False, True),
        "curving to Right": (car_fix_curve_right, car_fix_curve_right_move, True, False),
        "STRAIGHT": (car_fix, car_fix_move, False, False),
        "UNKNOWN": (car_fix, car_fix_move, False, False),
        "CHANGING LANE": (car_fix2, car_fix2_move, None, None),
        "TURN LEFT": (car_fix_curve_left, car_fix_curve_left_move, False, True),
        "TURN RIGHT": (car_fix_curve_right, car_fix_curve_right_move, True, False)
    }
    
    if isinstance(parameters, dict):
        road_info = parameters.get("road_info", "UNKNOWN")
    elif isinstance(parameters, str):
        road_info = parameters
    else:
        road_info = "UNKNOWN"
    
    fixed_images = road_conditions.get(road_info, (car_fix, car_fix_move, False, False))
    fixed_image = fixed_images[0] if use_car_fix else fixed_images[1]
    
    if fixed_images[2] is not None:
        destra, sinistra = fixed_images[2], fixed_images[3]
    
    use_car_fix = not use_car_fix
    
    return fixed_image,destra,sinistra,use_car_fix

def overlay_fixed_car_image( car_fix, car_fix2, car_fix_curve_left, car_fix_curve_right, 
                          window_scale_factor, parameters, car_fix_move, car_fix2_move, 
                          car_fix_curve_left_move, car_fix_curve_right_move, traffic, accident,road_close,
                          incident_active, traffic_active,road_close_active, road_close_time, incident_time, traffic_time,
                          icon_width, total_icons_width, background_width,icon,center_x,spacing,destra,sinistra,use_car_fix,original_gray_background,background_height,x_start, x_end,frame):
    """
    Overlays a fixed car image and traffic/incident images on the background image.
    """

    background_color = (200, 200, 200)  # RGB

    if original_gray_background is None:  # Lo creiamo solo una volta
        #print("🛠 Inizializzazione del background...")

        # ✅ Controlla che background_color sia corretto
        if background_color is None:
            background_color = (200, 200, 200)  # Default: Grigio chiaro

        # 🎨 Crea il background con colore scuro (100, 100, 100)
        original_gray_background = np.ones_like(frame) * 100  

        # 🎨 Disegna il rettangolo chiaro (200, 200, 200) sulla parte superiore
        original_gray_background[0:background_height, x_start:x_end] = background_color

        # ✅ Mantieni una copia di sicurezza e una copia modificabile
        image = original_gray_background.copy()
        
        #print("✅ Background inizializzato con due colori!")

    
    # 🚀 Resettiamo il background ogni volta
    gray_background = original_gray_background.copy()
    # Sistema modulare per il rilevamento delle lane

    if incident_active and time.time() - incident_time > 5:  # 5 secondi
        incident_active = False

    if traffic_active and time.time() - traffic_time > 5:  # 5 secondi
        traffic_active = False

    if road_close_active and time.time() - road_close_time > 5:  # 5 secondi
        road_close_active=False



    fixed_image,destra,sinistra,use_car_fix = get_fixed_image(parameters, car_fix, car_fix2, car_fix_curve_left, 
                                car_fix_curve_right, car_fix_move, car_fix2_move, 
                                car_fix_curve_left_move, car_fix_curve_right_move,destra,sinistra,use_car_fix)
    
    height, width, _ = image.shape
    fixed_height, fixed_width, _ = fixed_image.shape
    
    # Overlay the car image at the bottom-center
    x = (width - fixed_width) // 2
    y = height - fixed_height - 10
    
    fixed_height = min(fixed_height, height - y)
    fixed_width = min(fixed_width, width - x)
    
    overlay = fixed_image[:fixed_height, :fixed_width, :3]
    alpha_mask = fixed_image[:fixed_height, :fixed_width, 3] / 255.0
    
    for c in range(3):
        image[y:y + fixed_height, x:x + fixed_width, c] = (
            alpha_mask * overlay[:, :, c] + (1 - alpha_mask) * image[y:y + fixed_height, x:x + fixed_width, c]
        ).astype(np.uint8)
    
    # Lista di overlay da mostrare
    overlays_to_show = []
    if icon:
        icon=False
        icon_width = traffic.shape[1]  # Supponiamo che entrambe le icone abbiano la stessa larghezza
        

        # Calcola la posizione centrale
        center_x = width // 2
        # Larghezza totale delle icone + lo spazio tra di esse
        total_icons_width = (3 * icon_width) + (2*spacing)
        # Colore dello sfondo (grigio chiaro)

        # Dimensioni del riquadro di sfondo
        background_height = traffic.shape[0]
        #print("Beckround_height",background_height)  # Altezza del rettangolo
        background_width = total_icons_width  # Larghezza: giusta per contenere le icone
        #print("Beckround_width",background_width)

        # Calcola la posizione centrale del rettangolo
        center_x = width // 2
        x_start = center_x - (background_width // 2)
        x_end = center_x + (background_width // 2)
        #print("X_Start",x_start)
        #print("X_end",x_end)
        original_gray_background = None

        
        # Disegna il rettangolo grigio SOLO nella parte centrale in alto
        
        
    y_position = 0  # Altezza fissa per entrambe le icone
    # Calcola la posizione delle icone in modo che siano centrate
    x_position_accident = center_x - (total_icons_width // 2)
    x_position_traffic = x_position_accident + icon_width + spacing
    x_position_road_colose = x_position_traffic + icon_width +spacing

    # Aggiungi le icone alla lista
    if incident_active:
        overlays_to_show.append((accident, x_position_accident, y_position))

    if traffic_active:
        overlays_to_show.append((traffic, x_position_traffic, y_position))

    if road_close_active:
        overlays_to_show.append((road_close, x_position_road_colose, y_position))

    


    # Applica tutti gli overlay necessari
    for overlay_img, overlay_x, overlay_y in overlays_to_show:
        overlay_height, overlay_width, overlay_channels = overlay_img.shape
        
        # Assicura che non esca fuori dai bordi
        overlay_x = max(0, min(overlay_x, width - overlay_width))
        overlay_y = max(0, min(overlay_y, height - overlay_height))
        
        overlay_height = min(overlay_height, height - overlay_y)
        overlay_width = min(overlay_width, width - overlay_x)
        
        if overlay_channels == 4:  # Se ha canale alpha
            overlay_rgb = overlay_img[:overlay_height, :overlay_width, :3]
            overlay_alpha = overlay_img[:overlay_height, :overlay_width, 3] / 255.0
            
            for c in range(3):
                image[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width, c] = (
                    overlay_alpha * overlay_rgb[:, :, c] + 
                    (1 - overlay_alpha) * image[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width, c]
                ).astype(np.uint8)
        else:
            image[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width, :3] = (
                overlay_img[:overlay_height, :overlay_width, :3]
            ).astype(np.uint8)

    
    return image,background_height,x_start,x_end,original_gray_background,fixed_image,destra,sinistra,use_car_fix
