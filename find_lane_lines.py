import numpy as np
import cv2
import glob
import pickle
import os
import ntpath
import matplotlib.pyplot as plt
from image_processing import *

def process_image(img):
    src = np.float32([[240,719],[579,450],[712,450],[1165,719]])
    dst =  np.float32([[300,719],[300,0],[900,0],[900,719]])
    transformer = PerspectiveTransformer(src, dst)

    undistort_image = undistort(img)
    processed_image = process_binary(undistort_image)
    processed_image = transformer.transform(processed_image);
    left_fit, right_fit, yvals, out_img = find_lanes(processed_image)
    processed_image = fit_lane(processed_image, undistort_image, yvals, left_fit, right_fit, transformer)

    return processed_image

def find_lane_lines():
    images = glob.glob('test_images/test*.jpg')
    for idx, fname in enumerate(images):
        print('Processing image ', idx)
        image = cv2.imread(fname)
        processed_image = process_image(image)
        print('Processing done!!! ', idx)
        output_filename = 'output_images/' + ntpath.basename(fname)
        cv2.imwrite(output_filename, processed_image)


find_lane_lines()
