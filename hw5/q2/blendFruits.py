# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
	laplacian_pyr = []
	image = image = image.astype(np.float32) 
	for i in range(levels - 1):
		blurred_img = cv2.GaussianBlur(image, (13,13), 0)
		layer = cv2.subtract(image, blurred_img)
		laplacian_pyr.append(layer)
		new_width = max(1, int(image.shape[1] * resize_ratio)) 
		new_height = max(1, int(image.shape[0] * resize_ratio))
		image = cv2.resize(blurred_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	laplacian_pyr.append(image)
	return laplacian_pyr


def restore_from_pyramid(pyramidList, resize_ratio=2):
	image = pyramidList[-1]
	for i in range(len(pyramidList) - 2, -1, -1):
		image = cv2.resize(image, (pyramidList[i].shape[1], pyramidList[i].shape[0]), interpolation=cv2.INTER_CUBIC)
		image = cv2.add(image, pyramidList[i])
		plt.imshow(image, cmap='gray')
		plt.show()
	return image


def validate_operation(img):
	pyr = get_laplacian_pyramid(img, 5)
	img_restored = restore_from_pyramid(pyr)

	plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
	plt.imshow(img_restored, cmap='gray')

	plt.show()
	

def blend_pyramids(levels):
	# Your code goes here
	pass


apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

validate_operation(apple)
validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple)
pyr_orange = get_laplacian_pyramid(orange)



pyr_result = []

# Your code goes here

final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)

