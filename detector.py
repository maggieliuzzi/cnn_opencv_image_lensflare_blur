from utils import warning, error, timer
from predict_functions import prepare_model, predict_from_file
import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt # Optional

print("hola")
# Load an color image in grayscale
# /Users/maggieliuzzi/predict_images/2.png
img = cv2.imread('/Users/maggieliuzzi/predict_images/13.JPG', 1)
height, width, channels = img.shape
print(img.shape)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(imgray.shape)
plt.imshow(imgray, cmap = 'gray', interpolation = 'bicubic')
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
imgray, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imgcont = cv2.drawContours(imgray, contours, -1, (0,255,0), 3)
print(imgcont.shape)
# cnt = contours[4]
# cv2.drawContours(imgray, [cnt], 0, (0,255,0), 3)

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
# Detect blobs.
keypoints = detector.detect(imgray)
'''
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(imgray, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
'''
# Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

plt.imshow(imgcont, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

print("Hey")
exit(0)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

cv2.imshow('image',img)
cv2.imsave('/Users/maggieliuzzi/predict_images/saved/2.png', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


parser = argparse.ArgumentParser(
    description="Classifies images in good, lens_flare or blurry",
    epilog="Created by Maggie Liuzzi")
parser.add_argument('--image_name', default=None, required=True,
                    help="the name of the image to be classified; required.")
parser.add_argument('--model', default=None, required=True,
                    help="the model to be used; required.")

args = parser.parse_args()

if not os.path.isfile(args.model):
    print("ERROR: Could not find the gender-recognition model file.")
    exit(1)
if not os.path.isfile(args.image_name):
    print("ERROR: Could not find the image to be classified.")
    exit(1)

image_name = args.image_name
model = args.model



def cv2_func():
    pass

def detector(image_name):
    ''' Takes in as input an image name and outputs a 1 if the image is 'faulty' and 0 otherwise. '''

    quality = 0
    return quality

def main():
    cv2_func()
    print("Hola")
    model = prepare_model(args.model)
    raw_prediction = predict_from_file(model, args.image_name)
    print("\nProbabilities: " + str(raw_prediction))
# timer(main)