# Advanced Finding Lane Lines

# Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math 
from moviepy.editor import VideoFileClip
import glob

# Calibrate the camera using the chessboard calibration images
import calibrate_camera
images = glob.glob('camera_cal/calibration*.jpg')
ret, mtx, dist, rvecs, tvecs = calibrate_camera.get_calibration(images)

import threshold
import find_lane_lines_utils

# Return color processed image with lane lines overlaid
def process_image(image, mtx, dist):

	# Undistort image
	dist = cv2.undistort(image, mtx, dist, None, mtx)
	# # Plot the result
	# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	# f.tight_layout()
	# ax1.imshow(image)
	# ax1.set_title('Original Image', fontsize=50)
	# ax2.imshow(dist)
	# ax2.set_title('Undistorted Image', fontsize=50)
	# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	# plt.savefig('output_images/undistorted_lane_lines.png')

	# Apply thresholding to the image
	combined_binary = threshold.combined_threshold(dist, thresh=(20,200), r_thresh=(225,255), s_thresh=(170,255))

	# create masked edges image using cv2.fillPoly()
	mask = np.zeros_like(combined_binary)
	ignore_mask_color = 255

	# define four sided polygon to mask
	imshape = image.shape
	left_bottom = (np.int(imshape[1]*0.05),imshape[0])
	left_top = (np.int(0.4*imshape[1]), np.int(0.65*imshape[0]))
	right_top = (np.int(0.6*imshape[1]), np.int(0.65*imshape[0]))
	right_bottom = (np.int(imshape[1]-imshape[1]*0.05),imshape[0])
	vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(combined_binary, mask)

	# define contours to determine the 4 corners of the polygon surrounding lane lines
	contour_image = dist.copy()
	_, cnts, _ = cv2.findContours(masked_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]
	points_left = []
	points_right = []
	max_area_left = 0
	max_area_right = 0
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)
		if (approx[0][0][0] < np.int(imshape[1] / 2)):
			max_area_left = cv2.contourArea(c)
			points_left.append([approx[0][0], approx[1][0]])
		if (approx[0][0][0] >= np.int(imshape[1] / 2)):
			max_area_right = cv2.contourArea(c)
			points_right.append([approx[0][0], approx[1][0]])
	points_left = np.array(points_left, np.int32)
	points_right = np.array(points_right, np.int32)

	m_left, b_left = find_lane_lines_utils.slope_and_intcpt_from_points(points_left)
	m_right, b_right = find_lane_lines_utils.slope_and_intcpt_from_points(points_right)

	upper_bound = np.int(0.65*imshape[0])
	lower_bound = np.int(imshape[0])
	lower_left, upper_left = find_lane_lines_utils.new_bounded_points(m_left, b_left, upper_bound, lower_bound)
	lower_right, upper_right = find_lane_lines_utils.new_bounded_points(m_right, b_right, upper_bound, lower_bound)

	lower_left = [190, 720]
	lower_right = [1100, 720]
	upper_left = [575, 468]
	upper_right = [715, 468]
	cv2.line(contour_image,(lower_left[0],lower_left[1]), (upper_left[0],upper_left[1]),(0,255,0),3)
	cv2.line(contour_image,(lower_right[0],lower_right[1]), (upper_right[0],upper_right[1]),(0,255,0),3)
	# contour_points = np.concatenate((points_left, points_right), axis=0,)
	# cv2.drawContours(contour_image, contour_points, -1, (0,255,0), 3)

	# Use perspective transform to define a top down view of the image
	src = [lower_left, lower_right, upper_left, upper_right]
	dest = [[320, 720], [960, 720], [320, 0], [960, 0]]
	src = np.array(src, np.float32)
	dest = np.array(dest, np.float32)
	M = cv2.getPerspectiveTransform(src, dest)
	img_size = (image.shape[1], image.shape[0])
	warped = cv2.warpPerspective(masked_image, M, img_size, flags=cv2.INTER_LINEAR)

	# Detect lane lines

	# Determin lane curvature

	# Define and return the processed image
	processed = contour_image

	return warped

# Process test images
for filename in os.listdir("test_images/"):
    if filename.endswith(".jpg"): 
        # Identify the image
        image = mpimg.imread(os.path.join("test_images/", filename))
        output = process_image(image, mtx, dist)

        # Save the file as overlay
        mpimg.imsave((os.path.join("output_images/test_images_with_lane_line_overlay/", filename)),output)


# output = 'output.mp4'
# clip = VideoFileClip("project_video.mp4")
# output_clip = clip.fl_image(process_image) 
# %time output_clip.write_videofile(output, audio=False)

