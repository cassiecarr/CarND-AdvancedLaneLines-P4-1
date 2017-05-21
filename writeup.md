## Project Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chessboard.png "Undistorted Chessboard"
[image2]: ./output_images/undistorted_lane_lines.png "Undistorted Lane Lines"
[image3]: ./output_images/test_images_overlay_step1_threshold/test3.jpg "Binary Example"
[image4]: ./output_images/test_images_overlay_step2_perspective/test3.jpg "Warped Example"
[image5]: ./output_images/test_images_overlay_step3_windows/test3.jpg "Window Overlay Example"
[image6]: ./output_images/test_images_overlay_step4_polyfit/test3.jpg "Polyfit Line Example"
[image7]: ./output_images/test_images_with_lane_line_overlay/test3.jpg "Result Output"
[image8]: ./test_images/test3.jpg "Undistorted Lane Line Image"
[video1]: https://www.youtube.com/watch?v=tTAImRSo81s "Video"

---
### Submission Files

My project includes the following files:
* [find_lane_lines.py](find_lane_lines.py) containing the main pipeline for processing the lane line video and overlays the found lane lines, curvature, and distance from lane center
* [calibrate_camera.py](calibrate_camera.py) containing the preprocessing calibrating the camera
* [threshold.py](threshold.py) contains the preprocessing for applying thresholding to the images or video frames
* [find_lane_lines_utils.py](find_lane_lines_utils.py) containing the functions for finding windows surrounding lane lines
* writeup.md (this file) explains the results

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in [calibrate_camera.py](calibrate_camera.py) and is referenced in line 16 of [find_lane_lines.py](find_lane_lines.py).

The chessboard images in the [camera_cal](camera_cal) folder were read and input into the `get_calibration` function in order to compute the camera calibration and distortion coefficients. 

The `get_calibration()` function defines the `objpoints` and `imgpoints` for the `cv2.calibrateCamera()` function. The `imgpoints` are found using `cv2.findChessboardCorners()`, which idetifies the chessboard corners in each image. The `objpoints` are defined by creating a grid of the (x, y, z) coordinates for the corners of a square flat chessboard. 

The camera calibration and distortion coefficients output from `get_calibration()` are then applied to the test images using the `cv2.undistort()` in [find_lane_lines.py](find_lane_lines.py) line 27. The result below is a chessboard image before and after it is undistorted: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

`cv2.undistort()` is used in [find_lane_lines.py](find_lane_lines.py) line 27. Below is an example of one of the test images before and after it is undistorted:
![alt text][image2]

#### Original Image

I will use the image below to walk though rest of the steps for my pipeline:
![alt text][image8]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The `combined_threshold()` function in [threshold.py](threshold.py) was used to output a thresholded image in [find_lane_lines.py](find_lane_lines.py) line 30. `combined_threshold()` utilized Sobel thesholding in the x and y direction for the grayscale image. In addition, utilized thesholding in the red color channel. Each of these were combined to develop the final thesholded image. Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in [find_lane_lines.py](find_lane_lines.py) line 55. `src` and `dst` points were defined and `cv2.getPerspectiveTransform(src, dest)` was used to determine the perspective transform coefficients. The output was then input into the function `cv2.warpPerspective()` in order to create a top-down view of the image.  

I chose to hardcode the source and destination points and verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Here is an example of a warped image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I found the lane line pixels by first applying the `find_window_centroids()` function, found in [find_lane_lines_utils.py](find_lane_lines_utils.py), in line 69 of [find_lane_lines.py](find_lane_lines.py). The function defines the window centroids so window search regions can be used to identify the lane line pixels. It does this by identifying where the majority of the pixels lie in the warped images, assumed to be where the lane line pixels are located. 

Here is an example of the windows found:

![alt text][image5]

In line 71 of [find_lane_lines.py](find_lane_lines.py), I looped though each window centroid and added the pixels found within each window to a new image, defined as `combined`. 

After the `combined` image was defined, these pixels were used to find a best fit polynomial for the left and right lane lines. This can be found starting in line 106 of [find_lane_lines.py](find_lane_lines.py). The left and right pixel indicies were first defined. Then, `np.polyfit()` was used to determine the best fit 2nd order polynomial. Finally, the results were plotted. 

Here is an example of the lane line pixels with the overlay of the 2nd order polynomial:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Starting in line 133 of [find_lane_lines.py](find_lane_lines.py), the radius of curvature of the lane and position of the vehicle were determined. Using example images, conversion factors were defined to convert pixels to meters in real world space (assumptions were 3m between dashed lane lines and 3.7m between left and right lane lines). 

Using this conversion, new 2nd order best fit polynomials were defined for the lane lines. These new 2nd order polynomials were then used to determin the radius of curvature of the lane by utilizing the radius of curature formula defined [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).

The distance from the center of the lane was calculated by finding the center of the lane at the bottom of the image and subtracting it from the center of the image (or car center, assuming the car center is the center of the image). 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Starting in line 162 of [find_lane_lines.py](find_lane_lines.py), I drew the lane line polynomials, radius of curvature, and distance from lane center onto the original image. An example of this can be seen here:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=tTAImRSo81s)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
