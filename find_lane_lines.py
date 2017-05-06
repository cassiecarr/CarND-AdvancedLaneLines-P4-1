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
	combined_binary = threshold.combined_threshold(dist, thresh=(30,200), r_thresh=(225,255), s_thresh=(170,255))

	# create masked edges image using cv2.fillPoly()
	mask = np.zeros_like(combined_binary)
	ignore_mask_color = 255

	# define four sided polygon to mask
	imshape = image.shape
	left_bottom = (np.int(imshape[1]*0.15),imshape[0])
	left_top = (np.int(0.48*imshape[1]), np.int(0.56*imshape[0]))
	right_top = (np.int(0.52*imshape[1]), np.int(0.56*imshape[0]))
	right_bottom = (np.int(imshape[1]-imshape[1]*0.05),imshape[0])
	vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(combined_binary, mask)

	# # Use perspective transform to define a top down view of the image
    # src = [[70,70], [230, 70], [70, 230], [230, 230]]
    # dest = [[70,70], [230, 70], [70, 230], [230, 230]]
    # src = np.array(src, np.float32)
    # dest = np.array(dest, np.float32)
    # M = cv2.getPerspectiveTransform(src, dest)
    # warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

	# Detect lane lines

	# Determin lane curvature

	# Define and return the processed image
	processed = masked_image

	return processed

# Process test images
for filename in os.listdir("test_images/"):
    if filename.endswith(".jpg"): 
        # Identify the image
        image = mpimg.imread(os.path.join("test_images/", filename))
        output = process_image(image, mtx, dist)

        # Save the file as overlay
        mpimg.imsave((os.path.join("output_images/test_images_with_lane_line_overlay/", filename)),output, cmap='gray')


# output = 'output.mp4'
# clip = VideoFileClip("project_video.mp4")
# output_clip = clip.fl_image(process_image) 
# %time output_clip.write_videofile(output, audio=False)

