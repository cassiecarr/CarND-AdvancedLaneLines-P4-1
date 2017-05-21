# Advanced Finding Lane Lines

# Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math 
import moviepy
from moviepy.editor import VideoFileClip
import glob
from functools import partial

# Calibrate the camera using the chessboard calibration images
import calibrate_camera
images = glob.glob('camera_cal/calibration*.jpg')
ret, mtx, dist, rvecs, tvecs = calibrate_camera.get_calibration(images)

import threshold
import find_lane_lines_utils

# Return color processed image with lane lines overlaid
def process_image(mtx, dist, image):

	# Undistort the image
	dist = cv2.undistort(image, mtx, dist, None, mtx)

	# Apply thresholding to the image
	combined_binary = threshold.combined_threshold(dist, thresh=(30,200), r_thresh=(225,255), s_thresh=(170,255))

	# Create masked edge image using cv2.fillPoly()
	mask = np.zeros_like(combined_binary)
	ignore_mask_color = 255
	# Define four sided polygon to mask
	imshape = image.shape
	left_bottom = (np.int(imshape[1]*0.05),imshape[0])
	left_top = (np.int(0.4*imshape[1]), np.int(0.65*imshape[0]))
	right_top = (np.int(0.6*imshape[1]), np.int(0.65*imshape[0]))
	right_bottom = (np.int(imshape[1]-imshape[1]*0.05),imshape[0])
	vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
	# Mask the image with defined polygon
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(combined_binary, mask)

	# Define the 4 corners of a polygon surrounding the lane lines
	contour_image = dist.copy()
	lower_left = [190, 720]
	lower_right = [1100, 720]
	upper_left = [575, 468]
	upper_right = [715, 468]
	cv2.line(contour_image,(lower_left[0],lower_left[1]), (upper_left[0],upper_left[1]),(0,255,0),3)
	cv2.line(contour_image,(lower_right[0],lower_right[1]), (upper_right[0],upper_right[1]),(0,255,0),3)

	# Use perspective transform to define a top down view of the image
	src = [lower_left, lower_right, upper_left, upper_right]
	dest = [[320, 720], [960, 720], [320, 0], [960, 0]]
	src = np.array(src, np.float32)
	dest = np.array(dest, np.float32)
	M = cv2.getPerspectiveTransform(src, dest)
	img_size = (image.shape[1], image.shape[0])
	warped = cv2.warpPerspective(masked_image, M, img_size, flags=cv2.INTER_LINEAR)

	# Find lane lines using window search
	# Set window settings and find centroids
	window_width = 50 
	window_height = 120 # Break image into 9 vertical layers since image height is 720
	margin = 80 # Define how much to slide left and right for searching
	window_centroids = find_lane_lines_utils.find_window_centroids(warped, window_width, window_height, margin)
	# If any window centers found
	if len(window_centroids) > 0:

	    # Points used to draw all the left and right windows
	    l_points = np.zeros_like(warped)
	    r_points = np.zeros_like(warped)

	    # Go through each level and draw the windows 	
	    for level in range(0,len(window_centroids)):
	        # Use window_mask to draw window areas
		    l_mask = find_lane_lines_utils.window_mask(window_width,window_height,warped,window_centroids[level][0],level)
		    r_mask = find_lane_lines_utils.window_mask(window_width,window_height,warped,window_centroids[level][1],level)
		    # Add graphic points from window mask here to total pixels found 
		    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
		    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

	    # Draw the results 
	    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
	    zero_channel = np.zeros_like(template) # create a zero color channel
	    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
	    warped[warped == 1] = 255
	    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
	    output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0) # overlay the orignal road image with window results
	 
	# If no window centers found, just display orginal road image
	else:
	    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

	# Identify only the pixels that are contained within the windows found
	warped_binary = np.zeros_like(warped)
	warped_binary[(warped > 0)] = 255
	template_binary = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
	template_binary[(template_binary > 0)] = 255
	combined = np.zeros_like(template_binary)
	combined[(warped_binary == 255) & (template_binary == 255)] = 255
	
	# Identify the indices of the left and right pixels
	half = np.int(img_size[0]/2)
	nonzero = np.argwhere(combined > 1)
	nonzero_left = nonzero[(nonzero[:, 1] < half)]
	nonzero_right = nonzero[(nonzero[:, 1] > half)]
	x_left = nonzero_left[:, 1]
	y_left = nonzero_left[:, 0]
	x_right = nonzero_right[:, 1]
	y_right = nonzero_right[:, 0]

	# Use polyfit to determine the line curve the identified left and right pixels
	left_fit = np.polyfit(y_left, x_left, 2)
	right_fit = np.polyfit(y_right, x_right, 2)
	ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Plot the polyfit results
	overlay = np.array(cv2.merge((zero_channel,combined,zero_channel)),np.uint8)
	plt.imshow(overlay)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.axis("off")
	plt.savefig("output_images/dwg.jpg")
	plt.close()
	overlay_image = mpimg.imread("output_images/dwg.jpg")

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 3/55 # meters per pixel in y dimension
	xm_per_pix = 3.7/250 # meters per pixel in x dimension

	# Define the max y value
	y_eval = np.max(ploty)

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(y_left*ym_per_pix, x_left*xm_per_pix, 2)
	right_fit_cr = np.polyfit(y_right*ym_per_pix, x_right*xm_per_pix, 2)
	y_max = np.max(y_right*ym_per_pix)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	curve = np.mean(np.array(left_curverad, right_curverad))
	# Calculate the distance from the center of the lane
	left_pos = left_fit_cr[0]*y_max**2 + left_fit_cr[1]*y_max + left_fit_cr[2]
	right_pos = right_fit_cr[0]*y_max**2 + right_fit_cr[1]*y_max + right_fit_cr[2]
	car_center = (image.shape[1]/2)*xm_per_pix
	lane_center = left_pos + np.int((right_pos - left_pos)/2)
	diff = car_center - lane_center
	# Identify direction and define the text to display over the image
	direction = "Left"
	if diff > 0:
		direction = "Right"
	diff_direction = str(str("{0:.2f}".format(abs(diff))) + 'm ' + direction + ' of Center')
	font = cv2.FONT_HERSHEY_SIMPLEX
	text = str(int(curve)) + 'm Curve'

	# Define an image to draw the lane line polynomials on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Draw the lane line polynomials onto blank image
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank image with lane lines back to original image space using inverse perspective matrix (Minv)
	Minv = cv2.getPerspectiveTransform(dest, src)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	
	# Combine the result with the original image
	result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
	cv2.putText(result,text,(10,50), font, 1,(255,255,0),2,cv2.LINE_AA)
	cv2.putText(result,diff_direction,(10,100), font, 1,(255,255,0),2,cv2.LINE_AA)

	return result

# Bind the process image and calibration data
bound_process_image = partial(process_image, mtx, dist)

# Process test images with process image function
for filename in os.listdir("test_images/"):
    if filename.endswith(".jpg"): 
        # Identify the image
        image = mpimg.imread(os.path.join("test_images/", filename))
        output = bound_process_image(image)

        # Save the file as overlay
        mpimg.imsave((os.path.join("output_images/test_images_with_lane_line_overlay/", filename)),output)

# Process video with process image function
output = 'output.mp4'
clip = VideoFileClip("project_video.mp4")
output_clip = clip.fl_image(bound_process_image) 
output_clip.write_videofile(output, audio=False)

