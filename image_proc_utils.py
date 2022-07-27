# Image Processing Utility Functions

from turtle import width
import numpy as np
import math

# Gaussian Filter Implememtation
def gauss_filter(window_size: int, sigma: int) -> np.array:

    gauss_fil = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            gauss_fil[i, j] = np.exp((-i**2) / (2*(sigma**2))) * np.exp((-j**2) / (2*(sigma**2)))
    
    return gauss_fil


# Calculate Convolution's output size for one dimension
def calculate_target_size(img_size: int, kernel_size: int) -> int:
    
    num_pixels = 0
    
    # From 0 up to img size (if img size = 224, then up to 223)
    for i in range(img_size):
        # Add the kernel size (let's say 3) to the current i
        added = i + kernel_size
        # It must be lower than the image size
        # Increment if so
        num_pixels += 1 if added <= img_size else num_pixels
            
    return num_pixels


# Perform Convolution of an image with a kernel
def convolve(img: np.array, kernel: np.array) -> np.array:
    
    # Getting Target's Image X and Y size
    height = calculate_target_size(
        img_size=img.shape[0],
        kernel_size=kernel.shape[0]
    )

    width = calculate_target_size(
        img_size=img.shape[1],
        kernel_size=kernel.shape[1]
    )
    # Get kernel size
    k = kernel.shape[0]
    
    # 2D array of zeros
    convolved_img = np.zeros((height, width))
    
    # Iterate over the rows
    for i in range(height):
        # Iterate over the columns
        for j in range(width):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i:i+k, j:j+k]
            
            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
            
    return convolved_img

# Calculate directions/orientations
def calc_dir(filtered_x: np.array, filtered_y: np.array) -> np.array:

    height, width = filtered_x.shape

    theta = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            theta[i, j] = math.atan2(filtered_y[i, j], filtered_x[i, j])
            theta[i, j] = theta[i, j] * 180 / math.pi

    return theta

# Adjustment for negative directions, making all directions positive
def pos_dir(theta: np.array) -> np.array:

    height, width = theta.shape[0], theta.shape[1]

    # Add 360 to negative angles
    for i in range(height):
        for j in range(width):
            theta[i, j] = 360 + theta[i, j] if theta[i, j] < 0 else theta[i, j]
            
    return theta

# Adjusting directions to nearest 0, 45, 90, or 135 degree
def adjust_dir_nearest(theta: np.array) -> np.array:

    height, width = theta.shape

    theta_adj = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if (theta[i, j] >= 0) and (theta[i, j] < 22.5) or (theta[i, j] >= 157.5) and (theta[i, j] < 202.5) or (theta[i, j] >= 337.5) and (theta[i, j] <= 360):
                theta_adj[i, j] = 0
            elif (theta[i, j] >= 22.5) and (theta[i, j] < 67.5) or (theta[i, j] >= 202.5) and (theta[i, j] < 247.5):
                theta_adj[i,j] = 45
            elif (theta[i, j] >= 67.5) and (theta[i, j] < 112.5) or (theta[i, j] >= 247.5) and (theta[i, j] < 292.5):
                theta_adj[i, j] = 90
            elif (theta[i, j] >= 112.5) and (theta[i, j] < 157.5) or (theta[i, j] >= 292.5) and (theta[i, j] < 337.5):
                theta_adj[i, j] = 135

    return theta_adj

# Calculate magnitude of each edge
def calc_mag(filtered_x: np.array, filtered_y: np.array) -> np.array:

    magnitude = np.sqrt((filtered_x ** 2) + (filtered_y ** 2))
    
    return magnitude

# Non-Maximum Supression
def non_max_supr(magnitude: np.array, theta: np.array) -> np.array:

    height, width = magnitude.shape

    BW = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
                if theta[i, j] == 0:
                    BW[i, j] = magnitude[i, j] == max([magnitude[i, j], magnitude[i, j+1], magnitude[i, j-1]])
                elif theta[i, j] == 45:
                    BW[i, j] = magnitude[i, j] == max([magnitude[i, j], magnitude[i+1, j+1], magnitude[i-1, j-1]])
                elif theta[i, j] == 90:
                    BW[i, j] = magnitude[i, j] == max([magnitude[i, j], magnitude[i+1, j], magnitude[i-1, j]])
                elif theta[i, j] == 135:
                    BW[i, j] = magnitude[i, j] == max([magnitude[i, j], magnitude[i+1, j-1], magnitude[i-1, j+1]])

    return BW

# Hysteresis Thresholding
def hysterisis_thresh(BW: np.array, t_low: int, t_high: int) -> np.array:

    t_low = (t_low/255) * np.max(BW)
    t_high = (t_high/255) * np.max(BW)

    height, width = BW.shape[0], BW.shape[1]

    t_res = np.zeros((height, width))

    for i in range(height - 1):
        for j in range(width - 1):
           t_res[i, j] = 1 if ((BW[i+1, j]   > t_high or BW[i-1, j]   > t_high or 
                                BW[i, j+1]   > t_high or BW[i, j-1]   > t_high or 
                                BW[i-1, j-1] > t_high or BW[i-1, j+1] > t_high or
                                BW[i+1, j+1] > t_high or BW[i+1, j-1] > t_high) or BW[i, j] > t_high) else 0

    return t_res