# Calibrate the camera

# Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

# Define function to return camera calibration
def get_calibration(images):
	# Define object points and image points array
	objpoints = []
	imgpoints = []

	# Prepare object points array
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	# Loop through example images and add to imgpoints and objpoints arrays 
	for fname in images:
		img = mpimg.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)
			# img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
			# plt.imshow(img)
			# plt.title('Chessboard Corners', fontsize=30)
			# plt.savefig('camera_cal/draw_corners.png')

	# Use open CV calibrate camera function to get camera calibration
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	return ret, mtx, dist, rvecs, tvecs


# Read image
images = glob.glob('camera_cal/calibration*.jpg')

# Return calibration
ret, mtx, dist, rvecs, tvecs = get_calibration(images)

# Undistort and example image
img = mpimg.imread('camera_cal/calibration3.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)

# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(dst, cmap='gray')
# ax2.set_title('Chessboard Undistorted', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.savefig('output_images/undistorted_chessboard.png')
