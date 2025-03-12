
# Maya Atwan, 314813494
# Obaida Khateeb, 201278066

# Please replace the above comments with your names and ID numbers in the same format.
import cv2
import matplotlib.pyplot as plt
import numpy as np


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    # pre processing the image to calculate the bilateral filter
    im = im.astype(np.float64)
    rows, cols = im.shape
    cleanIm = np.zeros_like(im)
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing='ij')
    gs = np.exp(-(x**2 + y**2) / (2 * stdSpatial**2))

    for i in range(rows):
        for j in range(cols):
			# calculate the window of the image 
            x_min = max(0, i - radius)
            x_max = min(rows, i + radius + 1)
            y_min = max(0, j - radius)
            y_max = min(cols, j + radius + 1)
            window = im[x_min:x_max, y_min:y_max]

			# calculate the bilateral filter for the window
            gs_local = gs[:x_max - x_min, :y_max - y_min]
            gi = np.exp(-((window - im[i, j])**2) / (2 * stdIntensity**2))
            g = gs_local * gi
            cleanIm[i, j] = np.sum(g * window) / np.sum(g) #the new pixel value

    cleanIm = np.clip(cleanIm, 0, 255).astype(np.uint8)# clip the image to be between 0 and 255 and convert it to uint8        
    return cleanIm

# # Apply Bilateral Filter
image = cv2.imread("broken.jpg", cv2.IMREAD_GRAYSCALE)
image_gussianed = clean_Gaussian_noise_bilateral(image, 10, 22, 20)
# Apply Median Filter
image_medianed = cv2.medianBlur(image_gussianed, 5) 
output_image_path = "a.jpg"
cv2.imwrite(output_image_path, image_medianed)

# part B
# Load the noisy images
noised_images = np.load("noised_images.npy")
stacked_median = np.median(noised_images, axis=0).astype(np.uint8)
cv2.imwrite("b.jpg", stacked_median)