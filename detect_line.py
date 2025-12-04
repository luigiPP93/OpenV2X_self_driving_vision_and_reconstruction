import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image

import cupy as cp


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_info = None
        self.curvature = None
        self.deviation = None


def get_perspective_transform(img, src, dst, size):
    """ 
    #---------------------
    # This function takes in an image with source and destination image points,
    # generates the transform matrix and inverst transformation matrix, 
    # warps the image based on that matrix and returns the warped image with new perspective, 
    # along with both the regular and inverse transform matrices.
    #
    """

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv


def measure_curvature(left_lane, right_lane):
    """ 
    #---------------------
    # This function measures curvature of the left and right lane lines
    # in radians. 
    # This function is based on code provided in curvature measurement lecture.
    # 
    """

    ploty = left_lane.ally

    leftx, rightx = left_lane.allx, right_lane.allx

    leftx = leftx[::-1]     # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]   # Reverse to match top-to-bottom in y

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image    
    y_eval = np.max(ploty)

    # U.S. regulations that require a  minimum lane width of 12 feet or 3.7 meters, 
    # and the dashed lane lines are 10 feet or 3 meters long each.
    # >> http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC
    
    # Below is the calculation of radius of curvature after correcting for scale in x and y
    # Define conversions in x and y from pixels space to meters
    lane_width = abs(right_lane.startx - left_lane.startx)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7*(720/1280) / lane_width  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    # radius of curvature result
    left_lane.radius_of_curvature = left_curverad
    right_lane.radius_of_curvature = right_curverad


def smoothing(lines, prev_n_lines=3):
    # collect lines & print average line
    """
    #---------------------
    # This function takes in lines, averages last n lines
    # and returns an average line 
    # 
    """
    lines = np.squeeze(lines)       # remove single dimensional entries from the shape of an array
    avg_line = np.zeros((720))

    for i, line in enumerate(reversed(lines)):
        if i == prev_n_lines:
            break
        avg_line += line
    avg_line = avg_line / prev_n_lines

    return avg_line


def line_search_reset2(binary_img, left_lane, right_line):
    """
    #---------------------
    # After applying calibration, thresholding, and a perspective transform to a road image, 
    # I have a binary image where the lane lines stand out clearly. 
    # However, I still need to decide explicitly which pixels are part of the lines 
    # and which belong to the left line and which belong to the right line.
    # 
    # This lane line search is done using histogram and sliding window
    #
    # The sliding window implementation is based on lecture videos.
    # 
    # This function searches lines from scratch, i.e. without using info from previous lines.
    # However, the search is not entirely a blind search, since I am using histogram information. 
    #  
    # Use Cases:
    #    - Use this function on the first frame
    #    - Use when lines are lost or not detected in previous frames
    #
    """

    # I first take a histogram along all the columns in the lower half of the image
    histogram = np.sum(binary_img[int(binary_img.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] / 2)
    leftX_base = np.argmax(histogram[:midpoint])
    rightX_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    num_windows = 9
    
    # Set height of windows
    window_height = int(binary_img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    current_leftX = leftX_base
    current_rightX = rightX_base

    # Set minimum number of pixels found to recenter window
    min_num_pixel = 50

    # Create empty lists to receive left and right lane pixel indices
    win_left_lane = []
    win_right_lane = []

    window_margin = left_lane.window_margin

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]
        # Append these indices to the lists
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_inds) > min_num_pixel:
            current_leftX = int(np.mean(nonzerox[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_rightX = int(np.mean(nonzerox[right_window_inds]))

    # Concatenate the arrays of indices
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Extract left and right line pixel positions
    leftx= nonzerox[win_left_lane]
    lefty =  nonzeroy[win_left_lane]
    rightx = nonzerox[win_right_lane]
    righty = nonzeroy[win_right_lane]

    out_img[lefty, leftx] = [255, 0, 255]  # Colora i pixel della linea sinistra con viola
    out_img[righty, rightx] = [0, 255, 255]  # Colora i pixel della linea destra con giallo

    
    if len(leftx) == 0 or len(rightx) == 0:
        # Handle the case where no pixels are found within the sliding windows
        # You can add your logic here, such as returning a default image or raising an exception.
        # For simplicity, let's assume a basic handling:
        print("No lane pixels found within the windows")
        return out_img  # or handle differently based on your application

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_lane.current_fit = left_fit
    right_line.current_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_lane.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_lane.prevx) > 10:
        left_avg_line = smoothing(left_lane.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_lane.current_fit = left_avg_fit
        left_lane.allx, left_lane.ally = left_fit_plotx, ploty
    else:
        left_lane.current_fit = left_fit
        left_lane.allx, left_lane.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    left_lane.startx, right_line.startx = left_lane.allx[len(left_lane.allx)-1], right_line.allx[len(right_line.allx)-1]
    left_lane.endx, right_line.endx = left_lane.allx[0], right_line.allx[0]

    # Set detected=True for both lines
    left_lane.detected, right_line.detected = True, True
    
    measure_curvature(left_lane, right_line)
    
    return out_img


def line_search_reset_gpu(binary_img, left_lane, right_line):
    # Copia l'immagine binaria sulla GPU
    binary_img_gpu = cp.asarray(binary_img)

    # Calcola l'istogramma solo nella metà inferiore dell'immagine
    histogram = cp.sum(binary_img_gpu[binary_img_gpu.shape[0] // 2:, :], axis=0)

    # Prepara l'immagine di output per la visualizzazione
    out_img_gpu = cp.dstack((binary_img_gpu, binary_img_gpu, binary_img_gpu)) * 255

    # Trova i picchi dell'istogramma per sinistra e destra
    midpoint = histogram.shape[0] // 2
    leftX_base = cp.argmax(histogram[:midpoint])
    rightX_base = cp.argmax(histogram[midpoint:]) + midpoint

    # Parametri dei finestrini
    num_windows = 15
    window_height = binary_img_gpu.shape[0] // num_windows
    window_margin = left_lane.window_margin

    # Trova i pixel non zero e separa i componenti x e y
    nonzero = binary_img_gpu.nonzero()
    nonzeroy_gpu, nonzerox_gpu = nonzero[0], nonzero[1]

    # Posizioni di partenza per i finestrini
    current_leftX = leftX_base
    current_rightX = rightX_base

    # Liste di indici per i pixel dei binari sinistro e destro
    win_left_lane = []
    win_right_lane = []
    min_num_pixel = 50

    # Ciclo ottimizzato sui finestrini
    for window in range(num_windows):
        # Calcolo dei limiti dei finestrini per ogni finestra
        win_y_low = binary_img_gpu.shape[0] - (window + 1) * window_height
        win_y_high = binary_img_gpu.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        # Individua i pixel nei finestrini
        left_inds = ((nonzeroy_gpu >= win_y_low) & (nonzeroy_gpu < win_y_high) &
                     (nonzerox_gpu >= win_leftx_min) & (nonzerox_gpu < win_leftx_max)).nonzero()[0]
        right_inds = ((nonzeroy_gpu >= win_y_low) & (nonzeroy_gpu < win_y_high) &
                      (nonzerox_gpu >= win_rightx_min) & (nonzerox_gpu < win_rightx_max)).nonzero()[0]

        win_left_lane.append(left_inds)
        win_right_lane.append(right_inds)

        # Aggiorna la posizione del finestrino se sono presenti abbastanza pixel
        if len(left_inds) > min_num_pixel:
            current_leftX = cp.mean(nonzerox_gpu[left_inds])
        if len(right_inds) > min_num_pixel:
            current_rightX = cp.mean(nonzerox_gpu[right_inds])

    # Concatena solo una volta per aumentare la velocità
    win_left_lane = cp.concatenate(win_left_lane)
    win_right_lane = cp.concatenate(win_right_lane)

    # Estrai posizioni dei pixel della linea
    leftx = nonzerox_gpu[win_left_lane]
    lefty = nonzeroy_gpu[win_left_lane]
    rightx = nonzerox_gpu[win_right_lane]
    righty = nonzeroy_gpu[win_right_lane]

    # Colora i pixel della linea trovati (può essere rimosso per ulteriore velocità)
    out_img_gpu[lefty, leftx] = [255, 0, 255]  # Linea sinistra in viola
    out_img_gpu[righty, rightx] = [0, 255, 255]  # Linea destra in giallo

    if len(leftx) == 0 or len(rightx) == 0:
        #print("No lane pixels found within the windows")
        # Converte l'immagine per restituirla
        out_img = cp.asnumpy(out_img_gpu).astype(np.uint8)
        return out_img

    # Adattamento polinomiale di ordine 2
    left_fit = cp.polyfit(lefty, leftx, 2)
    right_fit = cp.polyfit(righty, rightx, 2)

    left_lane.current_fit = cp.asnumpy(left_fit)
    right_line.current_fit = cp.asnumpy(right_fit)

    ploty = cp.linspace(0, binary_img_gpu.shape[0] - 1, binary_img_gpu.shape[0])
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Smoothing
    if len(left_lane.prevx) > 10:
        left_avg_line = smoothing(left_lane.prevx, 10)
        left_avg_fit = np.polyfit(ploty.get(), left_avg_line, 2)
        left_lane.allx = np.polyval(left_avg_fit, ploty.get())
        left_lane.ally = ploty.get()
    else:
        left_lane.allx = cp.asnumpy(left_plotx)
        left_lane.ally = cp.asnumpy(ploty)

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty.get(), right_avg_line, 2)
        right_line.allx = np.polyval(right_avg_fit, ploty.get())
        right_line.ally = ploty.get()
    else:
        right_line.allx = cp.asnumpy(right_plotx)
        right_line.ally = cp.asnumpy(ploty)

    left_lane.startx = left_lane.allx[-1]
    right_line.startx = right_line.allx[-1]
    left_lane.endx = left_lane.allx[0]
    right_line.endx = right_line.allx[0]

    left_lane.detected = True
    right_line.detected = True
    measure_curvature(left_lane, right_line)

    # Converte l'immagine per restituirla
    out_img = cp.asnumpy(out_img_gpu).astype(np.uint8)
    return out_img




def line_search_reset2(binary_img, left_lane, right_line):
    
    #Riduzione delle Operazioni su Array: Evita duplicazioni nelle ricerche e utilizza variabili predefinite quando possibile.
    #Minimizzazione delle Creazioni di Oggetti Temporanei: Uso di np.concatenate() solo una volta per win_left_lane e win_right_lane.
    #Eliminazione di Operazioni di Visualizzazione Non Essenziali: Rimuovi le operazioni di disegno dei rettangoli se non necessarie.
    # Calcola l'istogramma solo nella metà inferiore dell'immagine
    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)

    # Prepara l'immagine di output per la visualizzazione
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # Trova i picchi dell'istogramma per sinistra e destra
    midpoint = histogram.shape[0] // 2
    leftX_base = np.argmax(histogram[:midpoint])
    rightX_base = np.argmax(histogram[midpoint:]) + midpoint

    # Parametri dei finestrini
    num_windows = 15
    window_height = binary_img.shape[0] // num_windows
    window_margin = left_lane.window_margin

    # Trova i pixel non zero e separa i componenti x e y
    nonzero = binary_img.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]

    # Posizioni di partenza per i finestrini
    current_leftX, current_rightX = leftX_base, rightX_base

    # Liste di indici per i pixel dei binari sinistro e destro
    win_left_lane, win_right_lane = [], []
    min_num_pixel = 50

    # Ciclo ottimizzato sui finestrini
    for window in range(num_windows):
        # Calcolo dei limiti dei finestrini per ogni finestra
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        # Rettangoli solo per la visualizzazione (puoi rimuovere se non necessario)
        cv2.rectangle(out_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Ottimizza l'individuazione dei pixel nei finestrini senza duplicare calcoli
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_leftx_min) & (nonzerox < win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_rightx_min) & (nonzerox < win_rightx_max)).nonzero()[0]

        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # Aggiorna la posizione del finestrino se sono presenti abbastanza pixel
        if len(left_window_inds) > min_num_pixel:
            current_leftX = int(np.mean(nonzerox[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_rightX = int(np.mean(nonzerox[right_window_inds]))

    # Concatena solo una volta per aumentare la velocità
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Estrai posizioni dei pixel della linea
    leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
    rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

    # Colora i pixel della linea trovati (può essere rimosso per ulteriore velocità)
    out_img[lefty, leftx] = [255, 0, 255]  # Linea sinistra in viola
    out_img[righty, rightx] = [0, 255, 255]  # Linea destra in giallo

    if len(leftx) == 0 or len(rightx) == 0:
        print("No lane pixels found within the windows")
        return out_img

    # Adattamento polinomiale di ordine 2
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_lane.current_fit = left_fit
    right_line.current_fit = right_fit

    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Smoothing
    if len(left_lane.prevx) > 10:
        left_avg_line = smoothing(left_lane.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_lane.allx, left_lane.ally = np.polyval(left_avg_fit, ploty), ploty
    else:
        left_lane.allx, left_lane.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_line.allx, right_line.ally = np.polyval(right_avg_fit, ploty), ploty
    else:
        right_line.allx, right_line.ally = right_plotx, ploty

    left_lane.startx, right_line.startx = left_lane.allx[-1], right_line.allx[-1]
    left_lane.endx, right_line.endx = left_lane.allx[0], right_line.allx[0]

    left_lane.detected, right_line.detected = True, True
    measure_curvature(left_lane, right_line)

    return out_img

def line_search_tracking_gpu(binary_img, left_line, right_line):
    # Trasferisce l'immagine binaria sulla GPU
    binary_img_gpu = cp.asarray(binary_img)

    # Crea l'immagine di output sulla GPU
    out_img_gpu = cp.dstack((binary_img_gpu, binary_img_gpu, binary_img_gpu)) * 255

    # Identifica i pixel non zero senza duplicazione
    nonzero = binary_img_gpu.nonzero()
    nonzeroy_gpu, nonzerox_gpu = nonzero[0], nonzero[1]

    # Margine della finestra e fit delle linee
    window_margin = left_line.window_margin
    left_fit = cp.asarray(left_line.current_fit)
    right_fit = cp.asarray(right_line.current_fit)

    # Calcola le posizioni x min e max per i pixel all'interno della finestra per sinistra e destra
    left_fit_y = left_fit[0] * nonzeroy_gpu ** 2 + left_fit[1] * nonzeroy_gpu + left_fit[2]
    right_fit_y = right_fit[0] * nonzeroy_gpu ** 2 + right_fit[1] * nonzeroy_gpu + right_fit[2]
    
    left_x_min = left_fit_y - window_margin
    left_x_max = left_fit_y + window_margin
    right_x_min = right_fit_y - window_margin
    right_x_max = right_fit_y + window_margin

    # Individua i pixel all'interno delle finestre con una maschera booleana diretta
    left_inds = (nonzerox_gpu >= left_x_min) & (nonzerox_gpu <= left_x_max)
    right_inds = (nonzerox_gpu >= right_x_min) & (nonzerox_gpu <= right_x_max)
    
    leftx_gpu = nonzerox_gpu[left_inds]
    lefty_gpu = nonzeroy_gpu[left_inds]
    rightx_gpu = nonzerox_gpu[right_inds]
    righty_gpu = nonzeroy_gpu[right_inds]

    # Colora i pixel delle linee se necessario per visualizzazione
    out_img_gpu[lefty_gpu, leftx_gpu] = [255, 0, 255]
    out_img_gpu[righty_gpu, rightx_gpu] = [0, 255, 255]

    # Converte i risultati sulla CPU per controllare le lunghezze
    leftx = cp.asnumpy(leftx_gpu)
    rightx = cp.asnumpy(rightx_gpu)

    # Verifica se entrambe le linee hanno pixel sufficienti
    if len(leftx) < 50 or len(rightx) < 50:
        left_line.detected, right_line.detected = False, False
        left_line.prevx.clear()
        right_line.prevx.clear()
        return line_search_reset_gpu(binary_img, left_line, right_line)

    # Adattamento polinomiale
    if len(leftx) > 0:
        left_fit = cp.polyfit(lefty_gpu, leftx_gpu, 2)
    else:
        left_fit = None

    if len(rightx) > 0:
        right_fit = cp.polyfit(righty_gpu, rightx_gpu, 2)
    else:
        right_fit = None

    if left_fit is not None and right_fit is not None:
        # Calcola i valori di x per il tracciamento delle linee
        ploty_gpu = cp.linspace(0, binary_img_gpu.shape[0] - 1, binary_img_gpu.shape[0])
        left_plotx_gpu = left_fit[0] * ploty_gpu ** 2 + left_fit[1] * ploty_gpu + left_fit[2]
        right_plotx_gpu = right_fit[0] * ploty_gpu ** 2 + right_fit[1] * ploty_gpu + right_fit[2]

        # Calcola la distanza media tra le due linee
        mean_distance = cp.mean(cp.abs(right_plotx_gpu - left_plotx_gpu))
        distance_threshold = 200  # Limite di distanza accettabile tra le linee

        if mean_distance < distance_threshold:
            left_line.detected, right_line.detected = False, False
            left_line.prevx.clear()
            right_line.prevx.clear()
            return line_search_reset_gpu(binary_img, left_line, right_line)

        # Aggiorna i fit delle linee e i punti se non sono sovrapposti
        left_line.current_fit = cp.asnumpy(left_fit)
        right_line.current_fit = cp.asnumpy(right_fit)

        left_plotx = cp.asnumpy(left_plotx_gpu)
        right_plotx = cp.asnumpy(right_plotx_gpu)
        ploty = cp.asnumpy(ploty_gpu)

        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        # Aggiorna le posizioni e le linee tracciate
        left_line.allx, left_line.ally = left_plotx, ploty
        right_line.allx, right_line.ally = right_plotx, ploty
        left_line.startx, right_line.startx = left_line.allx[-1], right_line.allx[-1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    # Misura della curvatura
    measure_curvature(left_line, right_line)

    # Converte l'immagine di output sulla CPU
    out_img = cp.asnumpy(out_img_gpu).astype(np.uint8)

    return out_img

import numpy as np
from concurrent.futures import ThreadPoolExecutor

def line_search_tracking(binary_img, left_line, right_line):
    # Crea l'immagine di output solo se necessario
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # Identifica i pixel non zero, usando già l'array senza duplicazione
    nonzero = binary_img.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]

    # Margine della finestra e fit delle linee
    window_margin = left_line.window_margin
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # Calcola le posizioni x per i pixel nelle due linee (in un'unica operazione)
    left_fit_y = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2]
    right_fit_y = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2]

    # Calcola i limiti per i pixel in base al margine della finestra
    left_x_min, left_x_max = left_fit_y - window_margin, left_fit_y + window_margin
    right_x_min, right_x_max = right_fit_y - window_margin, right_fit_y + window_margin

    # Funzione per ottenere gli indici dei pixel all'interno della finestra
    def get_indices(x_min, x_max):
        return (nonzerox >= x_min) & (nonzerox <= x_max)

    # Esegui in parallelo il calcolo per la sinistra e la destra
    with ThreadPoolExecutor() as executor:
        left_inds_future = executor.submit(get_indices, left_x_min, left_x_max)
        right_inds_future = executor.submit(get_indices, right_x_min, right_x_max)

        # Attendi il completamento dei calcoli
        left_inds = left_inds_future.result()
        right_inds = right_inds_future.result()

    # Raccogli i pixel
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    # Colora i pixel delle linee se necessario per visualizzazione
    out_img[lefty, leftx] = [255, 0, 255]
    out_img[righty, rightx] = [0, 255, 255]

    # Verifica se entrambe le linee hanno abbastanza pixel
    if len(leftx) < 50 or len(rightx) < 50:
        left_line.detected, right_line.detected = False, False
        left_line.prevx.clear()
        right_line.prevx.clear()
        return line_search_reset(binary_img, left_line, right_line)

    # Adattamento polinomiale con protezione per il caso di nessun pixel valido
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    if left_fit is not None and right_fit is not None:
        # Calcola i valori di x per il tracciamento delle linee
        ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        left_plotx = np.polyval(left_fit, ploty)
        right_plotx = np.polyval(right_fit, ploty)

        # Calcola la distanza media tra le due linee
        mean_distance = np.mean(np.abs(right_plotx - left_plotx))
        distance_threshold = 200  # Limite di distanza accettabile tra le linee

        # Reset in caso di sovrapposizione eccessiva
        if mean_distance < distance_threshold:
            left_line.detected, right_line.detected = False, False
            left_line.prevx.clear()
            right_line.prevx.clear()
            return line_search_reset(binary_img, left_line, right_line)

        # Aggiorna i fit delle linee
        left_line.current_fit, right_line.current_fit = left_fit, right_fit
        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        # Aggiorna le posizioni e le linee tracciate
        left_line.allx, left_line.ally = left_plotx, ploty
        right_line.allx, right_line.ally = right_plotx, ploty
        left_line.startx, right_line.startx = left_line.allx[-1], right_line.allx[-1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    # Misura della curvatura
    measure_curvature(left_line, right_line)

    return out_img

def line_search_reset(binary_img, left_lane, right_line):
    
    # Calcola l'istogramma solo nella metà inferiore dell'immagine
    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # Trova i picchi dell'istogramma per sinistra e destra
    midpoint = histogram.shape[0] // 2
    leftX_base = np.argmax(histogram[:midpoint])
    rightX_base = np.argmax(histogram[midpoint:]) + midpoint

    # Parametri dei finestrini
    num_windows = 15
    window_height = binary_img.shape[0] // num_windows
    window_margin = left_lane.window_margin

    # Trova i pixel non zero e separa i componenti x e y
    nonzero = binary_img.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]

    # Posizioni di partenza per i finestrini
    current_leftX, current_rightX = leftX_base, rightX_base

    # Liste di indici per i pixel dei binari sinistro e destro
    win_left_lane, win_right_lane = [], []
    min_num_pixel = 50

    # Funzione per calcolare i pixel nei finestrini
    def get_window_indices(window, current_leftX, current_rightX):
        # Calcolo dei limiti dei finestrini per ogni finestra
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        # Individuazione dei pixel per sinistra e destra nei finestrini
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_leftx_min) & (nonzerox < win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_rightx_min) & (nonzerox < win_rightx_max)).nonzero()[0]

        return left_window_inds, right_window_inds

    # Esegui il calcolo in parallelo per ogni finestra
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_window_indices, window, current_leftX, current_rightX) for window in range(num_windows)]
        for future in futures:
            left_window_inds, right_window_inds = future.result()
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)

            # Aggiorna la posizione del finestrino se sono presenti abbastanza pixel
            if len(left_window_inds) > min_num_pixel:
                current_leftX = int(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds) > min_num_pixel:
                current_rightX = int(np.mean(nonzerox[right_window_inds]))

    # Concatena solo una volta per aumentare la velocità
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Estrai posizioni dei pixel della linea
    leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
    rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

    # Colora i pixel della linea trovati (può essere rimosso per ulteriore velocità)
    out_img[lefty, leftx] = [255, 0, 255]  # Linea sinistra in viola
    out_img[righty, rightx] = [0, 255, 255]  # Linea destra in giallo

    if len(leftx) == 0 or len(rightx) == 0:
        print("No lane pixels found within the windows")
        return out_img

    # Adattamento polinomiale di ordine 2
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_lane.current_fit = left_fit
    right_line.current_fit = right_fit

    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Smoothing
    if len(left_lane.prevx) > 10:
        left_avg_line = smoothing(left_lane.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_lane.allx, left_lane.ally = np.polyval(left_avg_fit, ploty), ploty
    else:
        left_lane.allx, left_lane.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_line.allx, right_line.ally = np.polyval(right_avg_fit, ploty), ploty
    else:
        right_line.allx, right_line.ally = right_plotx, ploty

    left_lane.startx, right_line.startx = left_lane.allx[-1], right_line.allx[-1]
    left_lane.endx, right_line.endx = left_lane.allx[0], right_line.allx[0]

    left_lane.detected, right_line.detected = True, True
    measure_curvature(left_lane, right_line)

    return out_img


def line_search_tracking6(binary_img, left_line, right_line):
    # Crea l'immagine di output solo se necessario
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # Identifica i pixel non zero, usando già l'array senza duplicazione
    nonzero = binary_img.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]

    # Margine della finestra e fit delle linee
    window_margin = left_line.window_margin
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # Calcola le posizioni x min e max per i pixel all'interno della finestra per sinistra e destra
    left_fit_y = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2]
    right_fit_y = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2]
    
    left_x_min = left_fit_y - window_margin
    left_x_max = left_fit_y + window_margin
    right_x_min = right_fit_y - window_margin
    right_x_max = right_fit_y + window_margin

    # Individua i pixel all'interno delle finestre con una maschera booleana diretta
    left_inds = (nonzerox >= left_x_min) & (nonzerox <= left_x_max)
    right_inds = (nonzerox >= right_x_min) & (nonzerox <= right_x_max)
    
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    # Colora i pixel delle linee se necessario per visualizzazione
    out_img[lefty, leftx] = [255, 0, 255]
    out_img[righty, rightx] = [0, 255, 255]

    # Verifica se entrambe le linee hanno pixel sufficienti
    if len(leftx) < 50 or len(rightx) < 50:
        left_line.detected, right_line.detected = False, False
        left_line.prevx.clear()
        right_line.prevx.clear()
        #print("Resetting lines due to insufficient pixel count")
        return line_search_reset(binary_img, left_line, right_line)

    # Adattamento polinomiale
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

    if left_fit is not None and right_fit is not None:
        # Calcola i valori di x per il tracciamento delle linee
        ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        left_plotx = np.polyval(left_fit, ploty)
        right_plotx = np.polyval(right_fit, ploty)

        # Calcola la distanza media tra le due linee
        mean_distance = np.mean(np.abs(right_plotx - left_plotx))
        distance_threshold = 200  # Limite di distanza accettabile tra le linee

        if mean_distance < distance_threshold:
            left_line.detected, right_line.detected = False, False
            left_line.prevx.clear()
            right_line.prevx.clear()
            #print("Resetting lines due to overlapping detection")
            return line_search_reset(binary_img, left_line, right_line)

        # Aggiorna i fit delle linee e i punti se non sono sovrapposti
        left_line.current_fit, right_line.current_fit = left_fit, right_fit
        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        # Aggiorna le posizioni e le linee tracciate
        left_line.allx, left_line.ally = left_plotx, ploty
        right_line.allx, right_line.ally = right_plotx, ploty
        left_line.startx, right_line.startx = left_line.allx[-1], right_line.allx[-1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    # Misura della curvatura
    measure_curvature(left_line, right_line)

    return out_img

def line_search_tracking5(binary_img, left_line, right_line):
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # Usa i polinomi dei frame precedenti per predire la posizione delle linee
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    left_fit_pred = np.polyval(left_line.current_fit, ploty)
    right_fit_pred = np.polyval(right_line.current_fit, ploty)

    # Seleziona i pixel vicino alle linee predette
    nonzero = binary_img.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]
    
    left_inds = np.abs(nonzerox - left_fit_pred) < left_line.window_margin
    right_inds = np.abs(nonzerox - right_fit_pred) < right_line.window_margin
    
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    # Se c'è un numero sufficiente di pixel, esegui il fitting
    if len(leftx) > 50 and len(rightx) > 50:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        # Calcola la posizione finale dei pixel
        left_line.allx = np.polyval(left_fit, ploty)
        right_line.allx = np.polyval(right_fit, ploty)

        # Misura la curvatura
        measure_curvature(left_line, right_line)

    return out_img


def line_search_tracking2(b_img, left_line, right_line):
    # Immagine di output
    out_img = np.dstack((b_img, b_img, b_img)) * 255

    # Identificazione dei pixel non zero
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Margine della finestra
    window_margin = left_line.window_margin

    # Fit della linea corrente
    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit

    # Calcola le posizioni minime e massime delle linee
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # Identifica i pixel all'interno delle finestre
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]
    
    # Posizioni dei pixel delle linee
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    # Colora i pixel delle linee
    out_img[lefty, leftx] = [255, 0, 255]
    out_img[righty, rightx] = [0, 255, 255]

    # Verifica che entrambe le linee abbiano un numero sufficiente di pixel
    if len(leftx) < 50 or len(rightx) < 50:
        left_line.detected = False
        right_line.detected = False
        left_line.prevx.clear()
        right_line.prevx.clear()
        #("Resetting lines due to insufficient pixel count")
        out_img = line_search_reset(b_img, left_line, right_line)
        return out_img

    # Adattamento polinomiale
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

    if left_fit is not None and right_fit is not None:
        # Identificazione delle linee per il tracciamento
        ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Calcola la distanza media tra le due linee
        mean_distance = np.mean(np.abs(right_plotx - left_plotx))

        # Se la distanza è troppo piccola, resetta
        distance_threshold = 200  # Regola in base alla larghezza della corsia
        if mean_distance < distance_threshold:
            left_line.detected = False
            right_line.detected = False
            left_line.prevx.clear()
            right_line.prevx.clear()
            #print("Resetting lines due to overlapping detection")
            out_img = line_search_reset(b_img, left_line, right_line)
            return out_img

        # Se le linee sono a sinistra e a destra come atteso, aggiorna i fit e i punti
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)
        
        # Aggiornamento delle posizioni e delle linee
        left_line.allx, left_line.ally = left_plotx, ploty
        right_line.allx, right_line.ally = right_plotx, ploty
        left_line.startx, right_line.startx = left_line.allx[-1], right_line.allx[-1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]
        
    # Misura della curvatura
    measure_curvature(left_line, right_line)

    return out_img




def get_lane_lines_img(binary_img, left_line, right_line):
    """
    #---------------------
    # This function finds left and right lane lines and isolates them. 
    # If first frame or detected==False, it uses line_search_reset,
    # else it tracks/finds lines using history of previously detected lines, with line_search_tracking
    # 
    """
    
    if left_line.detected == True and right_line.detected == True:
        return line_search_tracking(binary_img, left_line, right_line)
    else:
        return line_search_reset(binary_img, left_line, right_line)


def illustrate_driving_lane(img, left_line, right_line, lane_color=(0, 255, 0), road_color=(120, 120, 120)):
    """ 
    #---------------------
    # This function draws lane lines and drivable area on the road
    # 
    """

    # Create an empty image to draw on
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), lane_color)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.5, 0)

    return result, window_img


def get_measurements(left_line, right_line):
    """
    #---------------------
    # This function calculates and returns follwing measurements:
    # - Radius of Curvature
    # - Distance from the Center
    # - Whether the lane is curving left or right
    # 
    """

    # take average of radius of left curvature and right curvature 
    curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

    # calculate direction using X coordinates of left and right lanes 
    direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2
     
    if curvature > 2000 and abs(direction) < 100:
        road_info = 'Straight'
        curvature = -1
    elif curvature <= 2000 and direction < - 50:
        road_info = 'curving to Left'
    elif curvature <= 2000 and direction > 50:
        road_info = 'curving to Right'
    else:
        if left_line.road_info != None:
            road_info = left_line.road_info
            curvature = left_line.curvature
        else:
            road_info = 'None'
            curvature = curvature

    center_lane = (right_line.startx + left_line.startx) / 2
    lane_width = right_line.startx - left_line.startx

   
    center_car = 720 / 2
    if center_lane > center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Left'
    elif center_lane < center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Right'
    else:
        deviation = 'by 0 (Centered)'

    """
    center_car = 720 / 2
    if center_lane > center_car:
        deviation = 'Left by ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%' + ' from center'
    elif center_lane < center_car:
        deviation = 'Right by ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%' + ' from center'
    else:
        deviation = 'by 0 (Centered)'
    """


    
    left_line.road_info = road_info
    left_line.curvature = curvature
    left_line.deviation = deviation

    return road_info, curvature, deviation


def illustrate_info_panel(img, left_line, right_line):
    """
    #---------------------
    # This function illustrates details below in a panel on top left corner.
    # - Lane is curving Left/Right
    # - Radius of Curvature:
    # - Deviating Left/Right by _% from center.
    #
    """

    road_info, curvature, deviation = get_measurements(left_line, right_line)
    cv2.putText(img, 'Measurements ', (75, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (80, 80, 80), 2)

    lane_info = 'Lane is ' + road_info
    if curvature == -1:
        lane_curve = 'Radius of Curvature : <Straight line>'
    else:
        lane_curve = 'Radius of Curvature : {0:0.3f}m'.format(curvature)
    #deviate = 'Deviating ' + deviation  # deviating how much from center, in %
    deviate = 'Distance from Center : ' + deviation  # deviating how much from center

    cv2.putText(img, lane_info, (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    cv2.putText(img, lane_curve, (10, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    cv2.putText(img, deviate, (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    
    parameters = {
        "road_info": road_info,
        "curvature": curvature,
        "deviation": deviation,
        "lane_info": lane_info,
        "lane_curve": lane_curve,
        "deviate": deviate
    }

    return img, parameters

def illustrate_driving_lane_with_topdownview(image, left_line, right_line):
    """
    #---------------------
    # This function illustrates top down view of the car on the road.
    #  
    """

    img = cv2.imread('img_gui/tesla.png', -1)
    img = cv2.resize(img, (120, 246))

    rows, cols = image.shape[:2]
    window_img = np.zeros_like(image)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally
    lane_width = right_line.startx - left_line.startx
    lane_center = (right_line.startx + left_line.startx) / 2
    lane_offset = cols / 2 - (2*left_line.startx + lane_width) / 2
    car_offset = int(lane_center - 360)
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - lane_width+ window_margin / 4, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (140, 0, 170))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (140, 0, 170))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

    #window_img[10:133,300:360] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    img = Image.fromarray(img)
    road_map.paste(window_img, (0, 0))
    road_map.paste(img, (300-car_offset, 590), mask=img)
    road_map = np.array(road_map)
    road_map = cv2.resize(road_map, (95, 95))
    road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)

    return road_map

