import numpy as np
import cv2

# Define functions for sobel, magnitude and directional thresholding

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
    	abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
    	abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    grad_binary = sbinary
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelxy_mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*sobelxy_mag/np.max(sobelxy_mag))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    mag_binary = sxbinary  
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)
    sobel_arctan = np.arctan2(sobely_abs, sobelx_abs)
    sxbinary = np.zeros_like(sobel_arctan)
    sxbinary[(sobel_arctan >= thresh[0]) & (sobel_arctan <= thresh[1])] = 1
    dir_binary = sxbinary
    return dir_binary

def combined_threshold(img, thresh=(0, 255), r_thresh=(0, 255), s_thresh=(0, 255)):
	# Define the r channel
	r_channel = img[:,:,0]

	# Convert to HLS color space and separate the S channel
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]

	# Grayscale image
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# Sobel x
	sxbinary = abs_sobel_thresh(gray, orient='x', sobel_kernel=9, thresh=thresh)

	# Sobel y
	sybinary = abs_sobel_thresh(gray, orient='y', sobel_kernel=9, thresh=thresh)

	# Threshold r color channel
	r_binary = np.zeros_like(r_channel)
	r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

	# Threshold s color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# Combine the r color channel and x & y thresholds 
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(r_binary == 1) | ((sxbinary == 1) & (sybinary == 1))] = 1

	return combined_binary



