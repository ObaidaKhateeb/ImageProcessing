# Obaida Khateeb, 201278066
# Maya Atwan, 314813494

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
	laplacian_pyr = []
	image = image = image.astype(np.float32) 
	for i in range(levels- 1):
		blurred_img = cv2.GaussianBlur(image,(27,27),0)
		layer = cv2.subtract(image, blurred_img)
		laplacian_pyr.append(layer)
		new_width = max(1,int(image.shape[1]*resize_ratio)) 
		new_height = max(1,int(image.shape[0]*resize_ratio))
		image = cv2.resize(blurred_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	laplacian_pyr.append(image)
	return laplacian_pyr


def restore_from_pyramid(pyramidList, resize_ratio=2):
	image = pyramidList[-1]
	for i in range(len(pyramidList) - 2, -1, -1):
		image = cv2.resize(image, (pyramidList[i].shape[1], pyramidList[i].shape[0]), interpolation=cv2.INTER_CUBIC)
		image = cv2.add(image, pyramidList[i])
	return image


def validate_operation(img):
	pyr = get_laplacian_pyramid(img, 6)
	img_restored = restore_from_pyramid(pyr)

	plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
	plt.imshow(img_restored, cmap='gray')

	plt.show()
	

def blend_pyramids(levels):
	global pyr_apple, pyr_orange
	blended_pyr = [] #initializing blended pyramid
	for apple_layer, orange_layer in zip(pyr_apple, pyr_orange):
		rows, cols = apple_layer.shape
		
		#Initiallizing the mask
		mask = np.zeros((rows, cols), dtype=np.float32)
		mask[:, :cols// 2] = 1 #Setting the mask to 1 on one side 

		blend_width = cols// 7 #Defining the blending window width
		for j in range(cols// 2 - blend_width, cols // 2 + blend_width):
			if 0 <= j < cols:
				mask[:, j] = 0.5 + 0.5 *np.cos(np.pi * (j -(cols //2 -blend_width)) / (2*blend_width))
		
		#Cross-Dissolve
		blended_layer = mask *orange_layer + (1 -mask) *apple_layer

		blended_pyr.append(blended_layer) #adding the blended layer to the blended pyramid
	return blended_pyr


apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

validate_operation(apple)
validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple, 6)
pyr_orange = get_laplacian_pyramid(orange, 6)


pyr_result = blend_pyramids(6)

final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite("result.jpg", final)