# Maya Atwan, 314813494
# Obaida Khateeb, 201278066

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import median_filter


def clean_baby(im):
	# applying affine / projective transformation as we did in assignment 2 and the median as we did in assignment 3
	clean_imge = cv2.medianBlur(im, ksize=3)
	# define the points of the image corners
	up_left_img=np.float32([[6, 20],[6, 130], [111, 20],[111, 130]])
	up_right_img=np.float32([ [181, 5],[121, 51], [249, 70],[176,119]])
	down_img=np.float32([[78, 163],[132, 244], [145, 117],[244, 161]])
	result_img = np.float32([[0, 0], [0, 255], [255, 0],[255, 255]])
	# get perspective transformation matrix
	img1 = cv2.getPerspectiveTransform(up_left_img, result_img)
	img2 = cv2.getPerspectiveTransform(up_right_img, result_img)
	img3 = cv2.getPerspectiveTransform(down_img, result_img)
	# get the transformed images 
	clean_M1=cv2.warpPerspective(clean_imge, img1, (256, 256), flags=cv2.INTER_CUBIC)
	clean_M2=cv2.warpPerspective(clean_imge, img2, (256, 256), flags=cv2.INTER_CUBIC)
	clean_M3=cv2.warpPerspective(clean_imge, img3, (256, 256), flags=cv2.INTER_CUBIC)
	# avg the 3 images to get the final image
	clean_image=np.average([clean_M1, clean_M2, clean_M3], axis=0)
	return clean_image

def clean_windmill(im):
	# we will use fft 
	im_fft = fft2(im)
	im_fft_shifted = fftshift(im_fft)
	val=(im_fft_shifted[123][100] + fftshift(im_fft)[124][99] + fftshift(im_fft)[125][100] )/3
	im_fft_shifted[124][100]=val
	im_fft_shifted[132][156]=val
	# shift the image back to the original position
	return np.abs(ifft2(ifftshift(im_fft_shifted)))


def clean_watermelon(im):
	# applying the sharpening filter with the kernel we used in tutorial 7
	return cv2.filter2D(src=im, ddepth=-1, kernel=np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]]))


def clean_umbrella(im):
	# we will use fft 
	im_fft = fft2(im)
	# shift the CD to the center as we always do
	im_fft_shifted = fftshift(im_fft)
	# create the mask and ftt of the mask and shift it to the center
	mask=np.zeros(im.shape)
	mask[0][0] = 0.5
	mask[4][79]=0.5
	mask_fft = fft2(mask)
	mask_fft_shifted = fftshift(mask_fft)
	mask_fft_shifted[abs(mask_fft_shifted)<0.0001] = 1 # to avoid division by zero
	# the mask is ready now we can divide the image fft by the mask fft
	clean_im_fft =np.divide(im_fft_shifted, mask_fft_shifted)
	# # shift the image back to the original position
	return np.abs(ifft2(clean_im_fft))


def clean_USAflag(im):
	# after checking the up left corner its 90 * 140 so we will crop the image to this size but it makes the line of the corener not good so we tried 145 and its pretty good
	corner= im[0:90, 0:145].copy()
	# we will use median filter to remove the noise hirozontally smoothing
	filtered = median_filter(im, [1,20]) 
	filtered[0:90, 0:145] = corner
	return filtered
	

def clean_house(im):
	# its obvious we need to use fft
	im_fft = fft2(im)
	# shift the CD to the center as we always do
	im_fft_shifted = fftshift(im_fft)
	# create the mask and ftt of the mask and shift it to the center  
	mask=np.zeros(im.shape)
	for i in range(10):
		mask[0][i] = 0.1
	mask_fft = fft2(mask)
	mask_fft_shifted = fftshift(mask_fft)
	mask_fft_shifted[mask_fft_shifted == 0] = 1 # to avoid division by zero
	# the mask is ready now we can divide the image fft by the mask fft
	clean_im_fft = np.divide(im_fft_shifted, mask_fft_shifted)
	# shift the image back to the original position
	clean_im = np.abs(ifft2(ifftshift(clean_im_fft)))
	return clean_im


def clean_bears(im):
	# normalize the image using min max values of the original image
	return cv2.normalize(im, None, alpha = 0, beta = 255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)