# Maya Atwan , ID:314813494
# Obaida Khateeb , ID: 201278066

# Please replace the above comments with your names and ID numbers in the same format.

import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import maximum_filter

warnings.filterwarnings("ignore")

def scale_down(image, resize_ratio):
	# Your code goes here
	image=cv2.GaussianBlur(image,(5,5),0) #blurring the image before scaling it down

	#if the image is colored, convert it to grayscale
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#apply the fourier transform to the image	
	f_transform = fft2(image)
	#shift the fourier transform
	f_shift =fftshift(f_transform)

	# get the dimensions of the image
	rows, cols = image.shape
	#calculate the new dimensions
	new_rows, new_cols = int(rows * resize_ratio), int(cols * resize_ratio)
	start_row, start_col = (rows - new_rows) // 2, (cols - new_cols) // 2

	#crop the frequency domain image to the new dimensions
	cropped_f_image = f_shift[start_row:start_row + new_rows, start_col:start_col + new_cols]
	#apply the inverse shift to the cropped frequency domain image
	scaled_down_image = np.abs(ifft2(ifftshift(cropped_f_image)))

	# Normalize the image to uint8
	scaled_down_image = cv2.normalize(scaled_down_image, None, 0, 255, cv2.NORM_MINMAX)
	scaled_down_image = scaled_down_image.astype(np.uint8)

	return scaled_down_image




def scale_up(image, resize_ratio):
	# Your code goes here
	#if the image is colored, convert it to grayscale
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#apply the fourier transform to the image
	f_transform = fft2(image)
	#shift the fourier transform
	f_shift = fftshift(f_transform)

	# get the dimensions of the image
	rows, cols = f_shift.shape
	#calculate the new dimensions
	new_rows, new_cols = int(rows * resize_ratio), int(cols * resize_ratio)
	start_row, start_col = (new_rows - rows) // 2, (new_cols - cols) // 2

	#initialize a new zero matrix with the new dimensions
	padded_transform = np.zeros((new_rows, new_cols), dtype=complex)

	#center the frequency domain in the new zero padded matrix
	padded_transform[start_row:start_row + rows, start_col:start_col + cols] = f_shift

	#apply the inverse shift to the padded frequency domain image
	scaled_up_image = np.abs(ifft2(ifftshift(padded_transform)))

	# Normalize the image to uint8
	scaled_up_image = cv2.normalize(scaled_up_image, None, 0, 255, cv2.NORM_MINMAX)
	scaled_up_image = scaled_up_image.astype(np.uint8)

	return scaled_up_image



def ncc_2d(image, pattern):
	# Your code goes here
	windows=np.lib.stride_tricks.sliding_window_view(image, pattern.shape)
	pattern_mean = np.mean(pattern)

	ncc = np.zeros((windows.shape[0], windows.shape[1]))
	for i in range(windows.shape[0]):
		for j in range(windows.shape[1]):
			window = windows[i, j]
			window_mean = np.mean(window)

			numerator = np.sum((window - window_mean) * (pattern - pattern_mean))
			denominator = np.sqrt(np.sum((window - window_mean) ** 2) * np.sum((pattern - pattern_mean) ** 2))
            
            # handle the case where the denominator=0
			if denominator == 0:
				ncc[i, j] = 0
			else:
				ncc[i, j] = numerator / denominator
				
	return ncc


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



CURR_IMAGE = "students" # Change this to "students" or "crew" to switch between the two images

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
display(image, pattern)





############# Students #############

image_scaled = scale_up(image,1.32)# Your code goes here. If you choose not to scale the image, just remove it.
pattern_scaled = scale_down(pattern,0.7) # Your code goes here. If you choose not to scale the pattern, just remove it.

ncc = ncc_2d(image_scaled,pattern_scaled)# Your code goes here
ncc[ncc!= maximum_filter(ncc, size=(40,20))]=0
threshold = 0.52
real_matches = np.argwhere(ncc >= threshold)

display(image_scaled, pattern_scaled)
######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
real_matches[:,0]=(real_matches[:,0]*(1/1.32)).astype(int)
real_matches[:,1]=(real_matches[:,1]*(1/1.32)).astype(int)
draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"



############# Crew #############

image_scaled =scale_up(image,1.66) # Your code goes here. If you choose not to scale the image, just remove it.
pattern_scaled =scale_down(pattern,0.38)  # Your code goes here. If you choose not to scale the pattern, just remove it.

ncc = ncc_2d(image_scaled, pattern_scaled)
ncc[ncc!= maximum_filter(ncc, size=(40,20))]=0
threshold = 0.452
real_matches =  np.argwhere(ncc >= threshold)
display(image_scaled, pattern_scaled)
######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:,0] += pattern_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:,1] += pattern_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
real_matches[:,0]=(real_matches[:,0]*(1/1.66)).astype(int)
real_matches[:,1]=(real_matches[:,1]*(1/1.66)).astype(int)

draw_matches(image, real_matches, pattern_scaled.shape)	# if pattern was not scaled, replace this with "pattern"
