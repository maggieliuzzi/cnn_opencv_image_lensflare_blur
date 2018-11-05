#!/usr/bin/python

# Standard imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
# /Users/maggieliuzzi/predict_images/13.JPG
im = cv2.imread("/Users/maggieliuzzi/predict_images/12.jpg",1)
# cv2.imshow("Original", im)
# cv2.waitKey(0)

size = 30 # 15

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# applying the kernel to the input image
output = cv2.filter2D(im, -1, kernel_motion_blur)

cv2.imshow('Motion Blur', output)
cv2.waitKey(0)

exit(0)
im_gaussian_blur = cv2.GaussianBlur(im,(49,49), 0) # width and height values must be odd
cv2.imshow("Gaussian filter", im_gaussian_blur)
cv2.waitKey(0)

exit(0)

for x in range(1, 50, 2):
    im_gaussian_blur = cv2.GaussianBlur(im,(x,x), 0) # ,0
    cv2.imshow("Gaussian filter", im_gaussian_blur)
    cv2.waitKey(5000)


exit(0)
# Blurrying
kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(im, -1, kernel)
plt.subplot(121), plt.imshow(im), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()




exit(0)
# Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()

'''
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
'''

# Create a detector with the parameters
# detector = cv2.SimpleBlobDetector_create(params)
detector = cv2.SimpleBlobDetector_create()


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)