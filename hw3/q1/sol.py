# Maya Atwan, 314813494
# Obaida Khateeb, 201278066

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)

# function to plot original, target, and edited images with MSE
def plot_images_with_mse(original, target, recreated, mse, title="Image Comparison"):
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Target Image
    plt.subplot(1, 3, 2)
    plt.imshow(target, cmap='gray')
    plt.title("Target Image")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(recreated, cmap='gray')
    plt.title(f"Edited Image\nMSE: {mse:.2f}")
    plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
# **************************** 
# ******* Image 1  ***********
# image1, avg filter, kernel 1x1183
#tried gaussian mean filter and didn't work, iterated over all the std mean kernels (1-30)x(1-1250), the one below is the best 
img1 = cv2.imread('image_1.jpg', cv2.IMREAD_GRAYSCALE)
# Define the average filter kernel (1x1183)
kernel = np.ones((1, 1183), np.float32) / 1183
new_img = cv2.filter2D(img, -1, kernel)
new_img = np.clip(new_img, 0, 255).astype(np.uint8)
mse = np.mean((new_img - img1) ** 2)
cv2.imwrite('edited_image_1.jpg', new_img)
plot_images_with_mse(img, img1, new_img, mse, title="Image 1 - Average Filter")

# # ****************************
# # ******* Image 2  ***********
# #image2 , gaussian blur , kernel size = 11x11 , sigma = 15 , border type = wrap
img2 = cv2.imread('image_2.jpg',cv2.IMREAD_GRAYSCALE)
new_img = cv2.GaussianBlur(img,(11,11), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_WRAP)
mse = np.mean((new_img - img2)**2)
cv2.imwrite('edited_image_2.jpg', new_img)
plot_images_with_mse(img, img2, new_img, mse, title="Image 2 - Gaussian Blur")

# **************************** 
# ******* Image 3  ***********

#image3, median filter, kernel size = 11x11
img3 = cv2.imread('image_3.jpg',cv2.IMREAD_GRAYSCALE)
new_img = cv2.medianBlur(img,11)
mse = np.mean((new_img - img3)**2)
cv2.imwrite('edited_image_3.jpg', new_img)
plot_images_with_mse(img, img3, new_img, mse, title="Image 3 - Median Filter")

# ****************************
# ******* Image 4  ***********
img4 = cv2.imread('image_4.jpg',cv2.IMREAD_GRAYSCALE)
kernel_mat = np.ones((15,1),np.float32)/15
# choosing 7 beacuse its the half of the kernel size
new_img = cv2.filter2D(np.pad(img, ((7, 7), (0, 0)), mode='wrap'),-1,kernel_mat)
new_img =new_img[7:-7, :]
mse = np.mean((new_img - img4)**2)
cv2.imwrite('edited_image_4.jpg', new_img)
plot_images_with_mse(img, img4, new_img, mse, title="Image 4 - Average Filter")

# ****************************
# ******* Image 5  ***********
# Sharpening Filter
img5 = cv2.imread('image_5.jpg', cv2.IMREAD_GRAYSCALE)
kernel_size = (11, 11)
sigma = 15 
blurred_image = cv2.GaussianBlur(img, kernel_size, sigma)
new_img = img- blurred_image +128
mse = np.mean((new_img - img5) ** 2)
cv2.imwrite('edited_image_5.jpg', new_img)
plot_images_with_mse(img, img5, new_img, mse, title="Image 5 - sharpening Filter")

#****************************
#******* Image 6 ***********
#image6, laplacian filter, kernel size 3x3 - failed 
img6 = cv2.imread('image_6.jpg',cv2.IMREAD_GRAYSCALE)
laplacian_kernel = np.array([[-0.2, -0.65239, -0.2],
                              [0, 0, 0],
                              [0.2, 0.65239, 0.2]], dtype=np.float32)
laplacian = cv2.filter2D(img, cv2.CV_64F,laplacian_kernel)
laplacian = np.clip(laplacian,0,255).astype(np.uint8)
mse = np.mean((laplacian - img6)**2)
cv2.imwrite('edited_image_6.jpg', laplacian)
plot_images_with_mse(img, img6, laplacian, mse, title="Image 6 - Laplacian Filter")

#****************************
#******* Image 7 ***********
#image7, translation filter, shift in y direction by 399 pixels 
img7 = cv2.imread('image_7.jpg',cv2.IMREAD_GRAYSCALE)
rows,cols = img7.shape
translation_matrix = np.float32([[1, 0, 0], [0, 1, 399]])
img7_second_part = img7[0:399,:]
new_img = cv2.warpAffine(img,translation_matrix,(cols,rows))
new_img[0:399,:] = img7_second_part
mse = np.mean((new_img - img7)**2)
cv2.imwrite('edited_image_7.jpg', new_img)
plot_images_with_mse(img, img7, new_img, mse, title="Image 7 - Translation Filter")

#****************************
#******* Image 8 ***********
# no operation
img8 = cv2.imread('image_8.jpg',cv2.IMREAD_GRAYSCALE)
mse = np.mean((img - img8)**2)
cv2.imwrite('edited_image_8.jpg', img)
plot_images_with_mse(img, img8, img, mse, title="Image 8 - Original Image")

#****************************
#******* Image 9 ***********
# sharpness filter using matrix convolution
img9 = cv2.imread('image_9.jpg',cv2.IMREAD_GRAYSCALE)
sharpening_kernel = np.array([
    [-0.25, -0.25, -0.25],
    [-0.25,  3.0,  -0.25],
    [-0.25, -0.25, -0.25]
])
new_img = cv2.filter2D(img, -1, sharpening_kernel)
mse = np.mean((new_img - img9)**2)
cv2.imwrite('edited_image_9.jpg', new_img)
plot_images_with_mse(img, img9, new_img, mse, title="Image 9 - Sharpening Filter")

#****************************
