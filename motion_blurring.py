import argparse
import os
import cv2
import numpy as np


parser = argparse.ArgumentParser(
    description="Blurs images in folder passed as argument and saves in new folder.",
    epilog="Created by Maggie Liuzzi")
parser.add_argument('--folder_to_blur', default=None, required=True,
                    help="the fault to train for (i.e. flare or blurry; required.")
args = parser.parse_args()
folder_to_blur = args.folder_to_blur


home_path = os.path.dirname(os.path.abspath(__file__))
blurred_images_path = os.path.join(home_path, "blurred_images")
if not os.path.isdir(blurred_images_path):
    os.makedirs(blurred_images_path)
    print("Created directory: " + blurred_images_path)


def blur_move_files(directory):
    print(directory)
    for root, dirs, files in os.walk(directory):
        for image in files:
            image_path = os.path.join(root, image)
            print(image)
            img = cv2.imread(image_path, 1)
            print(img)
            # blur
            size = 50
            # generating the kernel
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            # applying the kernel to the input image
            output = cv2.filter2D(img, -1, kernel_motion_blur)
            cv2.imwrite(blurred_images_path+"/"+image, output)


blur_move_files(folder_to_blur)
