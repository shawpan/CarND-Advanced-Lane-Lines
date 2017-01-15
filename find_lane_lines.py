import numpy as np
import cv2
import glob
import pickle
import os
import ntpath
import matplotlib.pyplot as plt
from calibration import Calibration

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

def process_binary(img, thresh_min, thresh_max, s_thresh_min, s_thresh_max):
    """ Apply color and gradient threshold to undistorted image
    Args:
        img: undistorted image in BGR
        thresh_min: minimum gradient threshold
        thresh_max: maximum gradient threshold
        s_thresh_min: minimum color threshold on s channel
        s_thresh_max: maximum color threshold on s channel
    Returns:
        Binary image created by applying color and gradient thresholds
    """
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Threshold s channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def find_lane_lines():
    # Calibrate
    calibration = Calibration('camera_cal', 9, 5)
    objpoints, imgpoints = calibration.calibrate()
    images = glob.glob('test_images/*.jpg')
    for idx, fname in enumerate(images):
        print('Processing image ', idx)
        image = cv2.imread(fname)
        # Undistort
        undistorted_image = undistort(image, objpoints, imgpoints)
        # Apply thresholds
        binary_image = process_binary(undistorted_image, 20, 100, 170, 255)

        output_filename = 'output_images/' + ntpath.basename(fname)
        cv2.imwrite(output_filename, binary_image * 255)


find_lane_lines()
