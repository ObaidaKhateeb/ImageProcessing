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

# change this to the name of the image you'll try to clean up
original_image_path = 'NoisyGrayImage.png'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
clear_image_b = clean_Gaussian_noise_bilateral(image, 10, 60 , 100)
output_image_path = 'NoisyGrayImage_edited.jpg'
cv2.imwrite(output_image_path, clear_image_b)

original_image_path = 'taj.jpg'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
clear_image_b = clean_Gaussian_noise_bilateral(image, 9, 20, 30)
output_image_path = 'taj_edited.jpg'
cv2.imwrite(output_image_path, clear_image_b)

original_image_path = 'balls.jpg'
image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
# 2% = 6.5
clear_image_b = clean_Gaussian_noise_bilateral(image, 3, 6, 5)
output_image_path = 'balls_edited.jpg'
cv2.imwrite(output_image_path, clear_image_b)


plt.subplot(121)
plt.imshow(image, cmap='gray')

plt.subplot(122)
plt.imshow(clear_image_b, cmap='gray')

plt.show()
