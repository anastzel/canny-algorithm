# Utility Functions

import matplotlib.pyplot as plt
import numpy as np

# Plot two grayscale images
def plot_two_images_gray(img_1: np.array, img_2: np.array, title_1: str, title_2: str):

    plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    plt.title(title_1)
    ax1.imshow(img_1, cmap='gray')

    ax2 = plt.subplot(1, 2, 2)
    plt.title(title_2)
    ax2.imshow(img_2, cmap='gray')

    plt.show()

# Plot two images
def plot_two_images(img_1: np.array, img_2: np.array, title_1: str, title_2: str):

    plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    plt.title(title_1)
    ax1.imshow(img_1)

    ax2 = plt.subplot(1, 2, 2)
    plt.title(title_2)
    ax2.imshow(img_2)

    plt.show()


