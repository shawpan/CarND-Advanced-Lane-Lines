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
    """
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

def find_lane_lines():
    calibration = Calibration('camera_cal', 9, 5)
    objpoints, imgpoints = calibration.calibrate()
    images = glob.glob('camera_cal/*.jpg')
    for idx, fname in enumerate(images):
        print('Processing image ', idx)
        image = cv2.imread(fname)
        undistorted_image = undistort(image, objpoints, imgpoints)
        output_filename = 'output_images/' + ntpath.basename(fname)
        cv2.imwrite(output_filename, undistorted_image)


find_lane_lines()
