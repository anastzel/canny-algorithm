# Canny Edge Detection Algorithm Implementation

from imports import *
from  image_proc_utils import *
from  utils import *
 
def main():

    # Load an image
    img = mpimg.imread("sphere.jpg")

    # Convert to Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Plot Input Image and Grayscale
    plot_two_images_gray(img, img_gray, "Input Image", "Grayscale Image")

    # Gaussian Filter Parameters
    window_size = 5
    sigma = 5

    # Convolution of image by Gaussian Filter
    img_conv = convolve(img_gray, gauss_filter(window_size, sigma))
    # Plot Grayscale Image and Filtered Image
    plot_two_images_gray(img_gray, img_conv, "Grayscale Image", "Filtered Image")

    # Filter for horizontal and vertical direction
    Gx = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])

    Gy = np.array([[1, 2, 1],
                  [0, 0, 0], 
                  [-1, -2, -1]])

    #Convolution by image by horizontal and vertical filter
    filtered_x = convolve(img_conv, Gx)
    filtered_y = convolve(img_conv, Gy)
    # Plot Filtered Images in X and Y axis
    plot_two_images_gray(filtered_x, filtered_y, "Filtered X-axis", "Filtered Y-axis")

    # Calculate directions/orientations
    theta = calc_dir(filtered_x, filtered_y)
    # Adjustment for negative directions, making all directions positive
    theta = pos_dir(theta)
    # Adjusting directions to nearest 0, 45, 90, or 135 degree
    theta_adjusted = adjust_dir_nearest(theta)
    # Plot Directions and Positive Directions
    plot_two_images(theta, theta_adjusted, "Directions", "Positive Directions")

    # Calculate magnitude of each edge
    magn = calc_mag(filtered_x, filtered_y)
    # Non-Maximum Supression
    BW = non_max_supr(magn, theta_adjusted)
    # Keeping only the most Dominant Edges in a Neighborhood
    dom_BW = np.multiply(magn, BW)
    # Plot Edges' Magnitude and Dominant Edges
    plot_two_images_gray(BW, dom_BW, "Edges\' Magnitude", "Dominant Edges")

    # Thresholding Values
    t_low = 5
    t_high = 10

    # Hysteresis Thresholding
    img_edges = 255 - (hysterisis_thresh(dom_BW, t_low, t_high) *  255)
    # Plot Input Image and Image with Edges
    plot_two_images_gray(img, img_edges, "Input Image", "Image with Edges")

if __name__ == '__main__':
    main()


