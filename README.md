#Advanced Lane Finding Project

---

[//]: # (Image References)

[imagechessboard]: ./doc/chessboard.jpg "Chessboard Image"
[calimagechessboard]: ./doc/cal_chessboard.jpg "Undistorted Chessboard Image"
[distorted]: ./doc/distorted.jpg "Distorted Image"
[undistorted]: ./doc/undistorted.jpg "Undistorted Image"
[binary]: ./doc/binary.jpg "Binary Image"
[birdeyeview]: ./doc/birdeyeview.jpg "Bird eye view Image"
[final]: ./doc/final.jpg "Final Result"

##Camera Calibration

The code for this step is contained in the `calibrate()` method of `Calibration` class in `calibration.py`.  

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I saved the `objpoints` and `imgpoints` in `calibration.p` file for future use.

I have then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

###Distorted chessboard Image
![Chessboard Image][imagechessboard]

###Undistorted chessboard Image
![Undistorted chessboard Image][calimagechessboard]

##Pipeline (test images)
Each image goes through the following steps implemented in `process_image()` method of `find_lane_lines.py`

1. Undistort using `objpoints` and `imgpoints` determined from camera calibration
2. Create binary image using several thresholding methods to make lane lines prominent 
3. Transform the binary image to bird eye view to make the lane lines significant
4. Find lane line points from the transformed image
5. Draw the lanes on undistorted image

###1. Undistort: 
This step is implemented in `undistort()` method of `image_procesing.py` 

```python
def undistort(img, objpoints, imgpoints):
    """ Undistort image
    Args:
        img: image in BGR
        objpoints: correct image points
        imgpoints: corresponding distorted image points
    Returns:
        Undistorted image
    """
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst
    
undistort_image = undistort(img, objpoints, imgpoints)
```

####Distorted Road Image
![Distorted Image][distorted]

####Undistorted Road Image
![Undistorted Image][undistorted]

###2. Processing undistorted image to binary: 
To create a binary image from undistorted image I have implemented `process_binary()` method of `image_processing.py`. Here, two separate processes are combined to create the thresholded binary. 

1. image is converted to gray scale => applied sobel operator on x axis => get the absolute sobel values => scale the values between 0 and 255 => apply binary thesholding between 30 and 150 pixel values
2. convert the image to HLS color space => extract S channel => get pixels having S values between 175 and 250
3. combine two binary images


```python
def process_binary(img):
    """ Process image to generate a sanitized binary image
    Args:
        img: undistorted image in BGR
    Returns:
        Binary image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    retval, sxthresh = cv2.threshold(scaled_sobel, 30, 150, cv2.THRESH_BINARY)
    sxbinary[(sxthresh >= 30) & (sxthresh <= 150)] = 1


    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_thresh = cv2.inRange(s_channel.astype('uint8'), 175, 250)

    combined_binary = np.zeros_like(gray)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary
    
processed_image = process_binary(undistort_image)
```

####Undistorted Road Image
![Undistorted Image][undistorted]

####Binary Image
![Binary Image][binary]

###3. Perspective transformation: 
To create a bird eye view of the binary image, I have implemented `transform()` method of `PerspectiveTransformer` class in `perspective_transformer.py`. I chose hardcoded source and destination points in the following manner:

Source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 240, 719      | 300, 719      |
| 579, 450      | 300, 0        |
| 712, 450      | 900, 0        |
| 1165, 719     | 900, 719      |

```python
def transform(self, img):
    """ Transform perspective of image
    Args:
        img: input image
    """
    return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

src = np.float32([[240,719],[579,450],[712,450],[1165,719]])
dst =  np.float32([[300,719],[300,0],[900,0],[900,719]])
transformer = PerspectiveTransformer(src, dst)
transformed_image = transformer.transform(binary_image)
```

####Binary Image
![Binary Image][binary]

####Bird Eye View Image
![Bird Eye View Image][birdeyeview]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

###6. Draw lanes on image.

I implemented this in `fit_lane()` method of `image_processing.py`.
```python
def fit_lane(warped_img, undist, yvals, left_fitx, right_fitx, transformer):
    """ Draw lane in image
    Args:
        warped_img: binary third eye view image
        undist: undistorted image
        yvals: y points of the lane
        left_fitx: x points of left lane
        right_fitx: x points of right lane
        transformer: perspective transformer
    Returns:
        undistored image with lanes drawn 
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warped_img, warped_img, warped_img))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = transformer.inverse_transform(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result
    
processed_image = fit_lane(bird_eye_view_image, undistort_image, yvals, left_fit, right_fit, transformer)
```
![Final result][final]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
