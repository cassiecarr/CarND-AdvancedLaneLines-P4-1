## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive) 

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Project Files
---

My project includes the following files:
* [find_lane_lines.py](find_lane_lines.py) containing the main pipeline for processing the lane line video and overlays the found lane lines, curvature, and distance from lane center
* [calibrate_camera.py](calibrate_camera.py) containing the preprocessing calibrating the camera
* [threshold.py](threshold.py) contains the preprocessing for applying thresholding to the images or video frames
* [find_lane_lines_utils.py](find_lane_lines_utils.py) containing the functions for finding windows surrounding lane lines
* [writeup.md](writeup.md) explains the results

Results
---
[//]: # (Image References)

[image7]: ./output_images/test_images_with_lane_line_overlay/test3.jpg "Result Output"
[video1]: https://www.youtube.com/watch?v=tTAImRSo81s "Video"

Here is an example of the output you will get for each lane line image:

![alt text][image7]

Here's a [link to my video result](https://www.youtube.com/watch?v=tTAImRSo81s)
