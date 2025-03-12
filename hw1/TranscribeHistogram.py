# Obaida Khateeb, 201278066
# Maya Atwan, 314813494

# Please replace the above comments with your names and ID numbers in the same format.

import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
	img_size = imgs_arr[0].shape
	res = []
	
	for img in imgs_arr:
		X = img.reshape(img_size[0] * img_size[1], 1)
		km = KMeans(n_clusters=n_colors)
		km.fit(X)
		
		img_compressed = km.cluster_centers_[km.labels_]
		img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

		res.append(img_compressed.reshape(img_size[0], img_size[1]))
	
	return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
	image_arrays = []
	lst = sorted([file for file in os.listdir(folder) if file.endswith(formats)])
	for filename in lst:
		file_path = os.path.join(folder, filename)
		image = cv2.imread(file_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_arrays.append(gray_image)
	return np.array(image_arrays), lst

# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
	# Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
	x_pos = 70 + 40 * idx
	y_pos = 274
	while image[y_pos, x_pos] == 1:
		y_pos-=1
	return 274 - y_pos

# Sections c, d
# Remember to uncomment compare_hist before using it!
#Input: two images, source and target
#Output: True if the target detected into the source, otherwise False 
def compare_hist(src_image, target):
	# Your code goes here
	# return True
	# or
	# return False
	window_height = target.shape[0]
	window_width = target.shape[1]
	relevant_image = src_image[100:135, 20:55] #slicing the region of the image where the topmost number appears 
	windows = np.lib.stride_tricks.sliding_window_view(relevant_image, (window_height, window_width))
	target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
	#computing the cumulative target histogram 
	for i in range(1, len(target_hist)):
		target_hist[i] += target_hist[i-1]
	#finding EMD for each of the windows 
	for y in range(len(windows)):
		for x in range(len(windows[0])):
			windows_hist = cv2.calcHist([windows[y,x]], [0], None, [256], [0, 256]).flatten()
			#computing the cumulative windows histogram
			for i in range(1, len(windows_hist)):
				windows_hist[i] += windows_hist[i-1]
			emd = np.sum(np.abs(windows_hist - target_hist)) #computing EMD
			if emd < 260:
				return True
	return False 


# Sections a, b

images, names = read_dir('data')
numbers, _ = read_dir('numbers')

#cv2.imshow(names[0], images[0]) 
#cv2.waitKey(0)
#cv2.destroyAllWindows() 
#exit()

# The following intends to insure that the digits are loaded by displaying them (section b)
#for i, number in enumerate(numbers): 
#	cv2.imshow(str(i), number) 
#cv2.waitKey(0)	
#cv2.destroyAllWindows() 
#exit()

#Testing the functionality of compare_hist (section d)
#Input: image object and its name. 
#Output: the topmost digit recognized along the horizontal axis
def topmost_detect(image, image_name):
	for i, number in enumerate(reversed(numbers)): #iterate over the digits images in reverse order 
		if compare_hist(image, number):
			#print(f"No. {len(numbers) - i - 1} detected as the topmost in {image_name}") #digit detected 
			return len(numbers) - i - 1
	#print("No number was recognized in the image") #no digit detected
	return None

topmost_digits = []
for i in range(len(images)):
	topmost = topmost_detect(images[i], names[i])
	topmost_digits.append(topmost)

	
#Quantize the first image with different gray (section e - first part)
#quantized_images = quantization([images[0]])
#quantized_image = quantized_images[-1]
#cv2.imshow('a.jpg, n_colors = 3', quantization([images[0]], 3)[-1]) 
#cv2.imshow('a.jpg, n_colors = 2', quantization([images[0]], 2)[-1]) 
#cv2.waitKey(0)
#cv2.destroyAllWindows() 
#exit()

#Quantize all the images with 3 levels of gray (section e - second part)
quantized_images = quantization(images, 3)
gray_degrees = np.unique(quantized_images[0]) #checking the different gray levels 
#print(gray_degrees)

#Quantize the images to black and white (section e - third part)
#Input: Image and integer number represent threshold
#Output: Modified binary image in which the colors above the threshold are black, and the rest are white
def bw_convert(img, threshold):
	bw_img = np.zeros(img.shape)
	bw_img[img < threshold] = 1 #gray levels below the threshold become white, the other remain black 
	return bw_img

bw_images = []
for img in quantized_images:
	bw_img = bw_convert(img, 225)
	bw_images.append(bw_img)

#extract the bar heights (section f)
#Input: image and number of the bars 
#Output: A list contains the heights of the bars in pixels
def img_bar_heights(img, bars_no):
	bar_heights = []
	for i in range(bars_no):
		bar_height = get_bar_height(img, i)
		bar_heights.append(bar_height)
	return bar_heights

images_bar_heights = []
for img in bw_images:
	img_bar_height = img_bar_heights(img, 10)
	images_bar_heights.append(img_bar_height)

#computing the no. of students in each range/bar (section g)
for id in range(len(images)):
	topmost_digit = topmost_digits[id] #max-student-num
	max_bar_height = max(images_bar_heights[id]) #max-bin-height
	heights = []
	for j in range(len(images_bar_heights[id])):
		bar_height = round(topmost_digit * images_bar_heights[id][j]/max_bar_height) #according to the given formula
		heights.append(bar_height)
	print(f'Histogram {names[id]} gave {",".join(map(str, heights))}')


# The following print line is what you should use when printing out the final result - the text version of each histogram, basically.

#print(f'Histogram {names[id]} gave {heights}')
