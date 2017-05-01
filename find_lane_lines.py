# Advanced Finding Lane Lines

# import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math 
from moviepy.editor import VideoFileClip

import calibrate_camera

# return color processed image with lane lines overlaid
def process_image(image):
	processed = image
return processed

# process test images
for filename in os.listdir("test_images/"):
    if filename.endswith(".jpg"): 
        # identify the image
        image = mpimg.imread(os.path.join("test_images/", filename))
        output = process_image(image)
        plt.figure()
        plt.imshow(output)
        
        # save the file as overlay
        mpimg.imsave((os.path.join("output/test_images_with_lane_line_overlay/", filename)),output)


output = 'output.mp4'
clip = VideoFileClip("project_video.mp4")
output_clip = clip.fl_image(process_image) 
%time output_clip.write_videofile(output, audio=False)

