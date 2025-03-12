# Maya Atwan, 314813494
# Obaida Khateeb, 201278066

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_fix(image, id):
	# Your code goes here
	if id ==1:
		return cv2.equalizeHist(image)
	elif id ==2:
		return 	gamma_correction(image, 0.5)
	else:
		return 	gamma_correction(image, 1.15)


def gamma_correction(image, gamma):
    return np.array(255 * (image / 255) ** gamma, dtype='uint8')

def brightness_and_contrast(image, a, b):
	m = np.mean(image)
	brightness_contrast= a * (image - m) + m + b 
	#making sure the values are in the range [0,255]
	brightness_contrast[brightness_contrast < 0] = 0
	brightness_contrast[brightness_contrast > 255] = 255
	#making sure the values are integers 
	brightness_contrast = np.round(brightness_contrast).astype(np.uint8)
	return brightness_contrast
	
for i in range(1,4):
	if i==1 :
		path=f'{i}.png'
	else :
		path = f'{i}.jpg'
	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	fixed_image = apply_fix(image, i)
	plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray',vmin=0,vmax=255)

