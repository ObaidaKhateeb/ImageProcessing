# Maya Atwan, 314813494
# Obaida Khateeb, 201278066

# Please replace the above comments with your names and ID numbers in the same format.
import os
import shutil
import sys

import cv2
import numpy as np


#matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
	src_points, dst_points = matches[:,0], matches[:,1]
	if is_affine:
		T, _ = cv2.estimateAffine2D(src_points, dst_points)
	else:
		T, _ = cv2.findHomography(src_points, dst_points)
	return T

def stitch(img1, img2):
    return np.maximum(img1, img2)

# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
	# Add your code here
	#  cv2.estimateAffine2D returns 2*3 matrix, so we need to convert it to 3*3
	if original_transform.shape == (2, 3):
		original_transform = np.vstack([original_transform, [0, 0, 1]])
	inverse_transform = np.linalg.inv(original_transform)
	warped_img = cv2.warpPerspective(target_img, inverse_transform, output_size, borderMode=cv2.BORDER_TRANSPARENT)
	return warped_img


def prepare_puzzle(puzzle_dir):
	edited = os.path.join(puzzle_dir, 'abs_pieces')
	if os.path.exists(edited):
		shutil.rmtree(edited)
	os.mkdir(edited)
	
	affine = 4 - int("affine" in puzzle_dir)
	
	matches_data = os.path.join(puzzle_dir, 'matches.txt')
	n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

	matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images-1,affine,2,2)
	
	return matches, affine == 3, n_images
	
if __name__ == '__main__':
	lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
	#lst = ['puzzle_affine_1']
	for puzzle_dir in lst:
		print(f'Starting {puzzle_dir}')
		puzzle = os.path.join('puzzles', puzzle_dir)
		pieces_pth = os.path.join(puzzle, 'pieces')
		edited = os.path.join(puzzle, 'abs_pieces')
		matches, is_affine, n_images = prepare_puzzle(puzzle)
		#prints to ensure the correctness of the data returned by prepare_puzzle
		# print(f'is_affine: {is_affine}') 
		# print(f'n_images: {n_images}') 
		# for i in range(n_images-1): 
		# 	print(f'matches {i}:') 
		# 	print(matches[i])
		# Add your code here
		firstImage=cv2.imread(f'{pieces_pth}/piece_1.jpg')
		first_image_path = os.path.join(edited, 'piece_1_absolute.jpg')
		cv2.imwrite(first_image_path, firstImage)
		final_puzzle=firstImage
		for i in range(1, n_images):
			transform=get_transform(matches[i-1], is_affine)
			if transform is not None:
				secondImage=cv2.imread(f'{pieces_pth}/piece_{i+1}.jpg')
				warped=inverse_transform_target_image(secondImage, transform, (firstImage.shape[1], firstImage.shape[0]))
				path_inversed =os.path.join(f'{edited}/piece_{i+1}_absolute.jpg')
				cv2.imwrite(path_inversed, warped)
				final_puzzle = stitch(final_puzzle, warped)
		sol_file = f'solution.jpg'
		cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle) 