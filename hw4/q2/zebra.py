# Maya Atwan, 314813494
# Obaida Khateeb, 201278066

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Your code goes here

#Fourier transform of the image - top right image  
fourier_transform = fft2(image)
fourier_transform = fftshift(fourier_transform)
fourier_spectrum_log = np.log(1 + np.abs(fourier_transform))

#Fourier transform with zero padding - middle left image
M, N = image.shape
fourier_transform_zero_padding = np.zeros((2*M, 2*N), dtype=complex) 
fourier_transform_zero_padding[M//2:3*M//2, N//2:3*N//2] = fourier_transform #placing the fourier transform in the center
fourier_transform_zero_padding_log = np.log(1 + np.abs(fourier_transform_zero_padding))

#Two times larger grayscale image - middle right image
upscaled_image = np.abs(ifft2(ifftshift(fourier_transform_zero_padding)))
upscaled_image *= 4 #multiplying each pixel by 4 
upscaled_image = np.clip(upscaled_image, 0, 255)

#Fourier Spectrum Four Copies - bottom left image
fourier_transform_four_copies = np.zeros((2*M, 2*N), dtype=complex)
fourier_transform_four_copies[::2, ::2] = fourier_transform #placing each (i,j) un (2*i, 2*j)
fourier_transform_four_copies *= 4
fourier_transform_four_copies_log = np.log(1 + np.abs(fourier_transform_four_copies))

#Four Copies Grayscale Image - bottom right image
four_copies_image = np.abs(ifft2(ifftshift(fourier_transform_four_copies)))

plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(fourier_spectrum_log, cmap='gray')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(fourier_transform_zero_padding_log, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(upscaled_image, cmap='gray')

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(fourier_transform_four_copies_log, cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(four_copies_image, cmap='gray')
plt.savefig('zebra_scaled.png')
plt.show()