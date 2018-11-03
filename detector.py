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
img = cv2.imread('/Users/maggieliuzzi/predict_images/2.png', 0)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

cv2.imshow('image',img)
cv2.imsave('/Users/maggieliuzzi/predict_images/saved/2.png', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("Hey")

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