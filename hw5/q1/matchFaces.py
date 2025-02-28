# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import warnings
warnings.filterwarnings("ignore")

def scale_down(image, resize_ratio):
	# Your code goes here
	length, width = image.shape

	#new dimensions of the image
	new_length = int(length * resize_ratio)
	new_width = int(width * resize_ratio)

	#converting the image to the frequency domain
	frequency_image = fft2(image)
	frequency_image = fftshift(frequency_image)

	#cropping the higher frequencies
	frequency_image = frequency_image[length//2 - new_length//2 : length//2 + new_length//2, width//2 - new_width//2 : width//2 + new_width//2]	

	#returning the image to the spatial domain
	frequency_image = ifftshift(frequency_image)
	image = np.abs(ifft2(frequency_image))

	return image

def scale_up(image, resize_ratio):
	# Your code goes here
	length, width = image.shape

	#new dimensions of the image
	new_length = int(length * resize_ratio)
	new_width = int(width * resize_ratio)

	#converting the image to the frequency domain
	frequency_image = fft2(image)
	frequency_image = fftshift(frequency_image)

	#zero padding the image
	new_image = np.zeros((new_length, new_width))
	new_image[new_length//2 - length//2 : new_length//2 + length//2, new_width//2 - width//2 : new_width//2 + width//2] = frequency_image

	#returning the image to the spatial domain
	new_image = ifftshift(new_image)
	new_image = np.abs(ifft2(new_image))

	return new_image


def ncc_2d(image, pattern):
	# Your code goes here
	pattern_length, pattern_width = pattern.shape
	#normalizing the image and the pattern
	normalized_image = image - np.mean(image)
	normalized_pattern = pattern - np.mean(pattern)
	#go over all the windows 
	for i in range(image.shape[0] - pattern_length):
		for j in range(image.shape[1] - pattern_width):
			window = normalized_image[i:i+pattern_length, j:j+pattern_width]
			#calculating the NCC value
			ncc = np.sum(window * normalized_pattern) / (np.sqrt(np.sum(normalized_window ** 2)) * np.sqrt(np.sum(normalized_pattern ** 2))
			#storing the NCC value in the image
			normalized_image[i + pattern_length//2, j + pattern_width//2] = ncc



def display(image, pattern):
	
	plt.subplot(2, 3, 1)
	plt.title('Image')
	plt.imshow(image, cmap='gray')
		
	plt.subplot(2, 3, 3)
	plt.title('Pattern')
	plt.imshow(pattern, cmap='gray', aspect='equal')
	
	ncc = ncc_2d(image, pattern)
	
	plt.subplot(2, 3, 5)
	plt.title('Normalized Cross-Correlation Heatmap')
	plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto') 
	
	cbar = plt.colorbar()
	cbar.set_label('NCC Values')
		
	plt.show()

def draw_matches(image, matches, pattern_size):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	for point in matches:
		y, x = point
		top_left = (int(x - pattern_size[1]/2), int(y - pattern_size[0]/2))
		bottom_right = (int(x + pattern_size[1]/2), int(y + pattern_size[0]/2))
		cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)
	
	plt.imshow(image, cmap='gray')
	plt.show()
	
	cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)



CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
display(image, pattern)





############# Students #############

image_scaled = # Your code goes here. If you choose not to scale the image, just remove it.
pattern_scaled =  # Your code goes here. If you choose not to scale the pattern, just remove it.

display(image_scaled, pattern_scaled)

ncc = # Your code goes here
real_matches = # Your code goes here

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"





############# Crew #############

image_scaled = # Your code goes here. If you choose not to scale the image, just remove it.
pattern_scaled =  # Your code goes here. If you choose not to scale the pattern, just remove it.

display(image_scaled, pattern_scaled)

ncc = # Your code goes here
real_matches = # Your code goes here

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"
