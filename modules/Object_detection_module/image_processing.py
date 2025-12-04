import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', thresh=(20, 100)):
    """
    #--------------------- 
    # This function applies Sobel x or y, and then 
    # takes an absolute value and applies a threshold.
    #
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
   
    # Create a binary mask where mag thresholds are met  
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    #---------------------
    # This function takes in an image and optional Sobel kernel size, 
    # as well as thresholds for gradient magnitude. And computes the gradient magnitude, 
    # applies a threshold, and creates a binary output image showing where thresholds were met.
    #
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Create a binary mask where mag thresholds are met    
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):
    """
    #---------------------
    # This function applies Sobel x and y, 
    # then computes the direction of the gradient,
    # and then applies a threshold.
    #
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the x and y gradients 
    # and calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
   
    # Create a binary mask where direction thresholds are met 
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
    
    # Return the binary image
    return binary_output.astype(np.uint8)



def get_combined_gradients(img, thresh_x, thresh_y, thresh_mag, thresh_dir):
    rows, cols = img.shape[:2]

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel thresholds
    sobelx = abs_sobel_thresh(gray, 'x', thresh_x)
    sobely = abs_sobel_thresh(gray, 'y', thresh_y)
    mag_binary = mag_thresh(gray, 3, thresh_mag)
    dir_binary = dir_thresh(gray, 15, thresh_dir)

    # Crop the image to match HLS cropping
    sobelx = sobelx[220:rows - 12, 0:cols]
    sobely = sobely[220:rows - 12, 0:cols]
    mag_binary = mag_binary[220:rows - 12, 0:cols]
    dir_binary = dir_binary[220:rows - 12, 0:cols]

    # Combine Sobel x, Sobel y, magnitude, and direction thresholds
    gradient_combined = np.zeros_like(dir_binary).astype(np.uint8)
    gradient_combined[((sobelx > 1) & (mag_binary > 1) & (dir_binary > 1)) | ((sobelx > 1) & (sobely > 1))] = 255

    return gradient_combined



def channel_thresh(channel, thresh=(80, 255)):
    """
    #---------------------
    # This function takes in a channel of an image and
    # returns thresholded binary image
    # 
    """
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 255
    return binary


def channel_thresh(channel, thresh):
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary


def canny_edge_detection(image):
    orig_rows=540
    #orig_cols=960
    rows, cols = image.shape[:2]
    #print("pp1",rows, cols)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 150)

    scale_y = rows / orig_rows  

    # Adatta dinamicamente i valori di ritaglio
    crop_top = int(220 * scale_y)
    crop_bottom = int(12 * scale_y)

    # Esegui il ritaglio scalato
    edges_cropped = edges[crop_top:rows - crop_bottom, 0:cols]
    return edges_cropped

def get_combined_hls(img, th_h, th_l, th_s):
    # Convert the image to HLS color space
    orig_rows=540
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    rows, cols = img.shape[:2]
    #print("pp2", rows, cols)

    # Calcola il fattore di scala per le righe
    scale_y = rows / orig_rows  

    # Adatta dinamicamente i valori di ritaglio
    crop_top = int(220 * scale_y)
    crop_bottom = int(12 * scale_y)

    # Extract regions of interest for H, L, and S channels
    H = hls[crop_top:rows - crop_bottom, 0:cols, 0]
    L = hls[crop_top:rows - crop_bottom, 0:cols, 1]
    S = hls[crop_top:rows - crop_bottom, 0:cols, 2]

    # Apply thresholds to H, L, and S channels
    h_channel = channel_thresh(H, th_h)
    l_channel = channel_thresh(L, th_l)
    s_channel = channel_thresh(S, th_s)

    average_saturation = np.mean(hls[:, :, 2])
    white_pixels = np.sum(np.all(hls >= [0, 200, 200], axis=-1))

    # Define color ranges for white and yellow in HLS space
    lower_white = np.array([0, 221, 30], dtype=np.uint8)  
    upper_white = np.array([180, 225, 255], dtype=np.uint8)  
    lower_yellow = np.array([10, 170, 50], dtype=np.uint8) 
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # Create masks for white and yellow colors
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    # Combine the masks
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # Extract the same portion as H, L, and S
    combined_mask = combined_mask[crop_top:rows - crop_bottom, 0:cols]

    # Combine the masks with the thresholded channels
    hls_comb = np.zeros_like(s_channel).astype(np.uint8)
    hls_comb[((s_channel > 1) & (l_channel == 0)) | 
             ((s_channel == 0) & (h_channel > 1) & (l_channel > 1)) | 
             (combined_mask > 0)] = 255

    return hls_comb

# Funzione per filtrare le linee della corsia
def filter_lane_lines(contours, image_shape):
    filtered_contours = []
    rows, cols = image_shape[:2]

    # Parametri di filtraggio: puoi modificarli per perfezionare il risultato
    min_length = 4  # Lunghezza minima per una linea della corsia
    max_width = 8  # Larghezza massima per le linee della corsia
    vertical_tolerance = 90  # Tolleranza angolare per la verticalità

    for contour in contours:
        # Calcola il rettangolo minimo che racchiude il contorno
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calcola la lunghezza e la larghezza del rettangolo
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])

        # Calcola l'angolo del rettangolo rispetto all'orientamento verticale
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle

        # Filtra solo le linee della corsia (orientamento verticale, lunghe e strette)
        if (min(width, height) < max_width and max(width, height) > min_length and
            abs(angle) < vertical_tolerance):  # Linee verticali, strette e lunghe
            filtered_contours.append(contour)

    return filtered_contours

def remove_unwanted_objects(image):
    # Trova i contorni nell'immagine
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra i contorni per ottenere solo le linee della corsia
    lane_lines = filter_lane_lines(contours, image.shape)

    # Crea una maschera vuota per l'immagine filtrata
    filtered_image = np.zeros_like(image)

    # Disegna solo i contorni che rappresentano le linee della corsia
    cv2.drawContours(filtered_image, lane_lines, -1, (255), thickness=cv2.FILLED)

    return filtered_image

def combine_grad_hls(gradient_combined, hls_comb):
    # Ensure the shapes are consistent
    if gradient_combined.shape != hls_comb.shape:
        #print("Shapes are not consistent, resizing hls_comb")
        hls_comb = cv2.resize(hls_comb, (gradient_combined.shape[1], gradient_combined.shape[0]))

    # Combine gradient and HLS results
    combined_result = np.zeros_like(gradient_combined)
    combined_result[(gradient_combined > 0) | (hls_comb > 0)] = 255

    return combined_result
