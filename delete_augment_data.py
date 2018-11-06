# Based on https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt # Optional
import matplotlib.image as mpimg
import cv2
import os
import glob
import argparse


parser = argparse.ArgumentParser(
    description="Retrains top layers of MobileNetV2 for classifying images into good and faulty.",
    epilog="Created by Maggie Liuzzi")
parser.add_argument('--fault', default=None, required=True,
                    help="the fault to train for (i.e. flare or blurry; required.")
args = parser.parse_args()
fault = args.fault


home_path = os.path.dirname(os.path.abspath(__file__)) # abspath?
source_path = os.path.join(home_path, "training-data/")
good_data_list = glob.glob(os.path.join(source_path,'good-data/*.JPG')) + glob.glob(os.path.join(source_path,'good-data/*.jpg')) + glob.glob(os.path.join(source_path,'good-data/*.jpeg')) + glob.glob(os.path.join(source_path,'good-data/*.png'))
faulty_data_list = glob.glob(os.path.join(source_path,fault+'-data/*.JPG')) + glob.glob(os.path.join(source_path,fault+'-data/*.jpg')) + glob.glob(os.path.join(source_path,fault+'-data/*.jpeg')) + glob.glob(os.path.join(source_path,fault+'-data/*.png'))



home_path = os.path.dirname(os.path.abspath(__file__))
augmented_data_path = os.path.join(home_path, "augmented_data/")
if not os.path.isdir(augmented_data_path):
    os.makedirs(augmented_data_path)
    print("Created directory: " + augmented_data_path)


# Image eg: /Users/maggieliuzzi/predict_images/13.JPG

# First, load the image again
# dir_path = os.path.dirname(os.path.realpath(__file__))
# filename = dir_path + "/MarshOrchid.jpg"
filename = "/Users/maggieliuzzi/predict_images/L.png"
raw_image_data = mpimg.imread(filename)

image = raw_image_data.reshape(1, 64, 64, 3)
image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()






''' Flip '''
def flip(img_path, img_name, height, width, channels):
    path = img_path + img_name
    img = cv2.imread(path, 1)
    # NumPy.'img' = A single image.
    print(img.shape)
    flip_1 = np.fliplr(img)
    # TensorFlow. 'x' = A placeholder for an image.
    shape = [height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    flip_2 = tf.image.flip_up_down(x)
    flip_2.save(os.path.join(augmented_data_path, img_name, "_flip_2.jpeg"))
    cv2.imshow("Image", flip_2)
    cv2.waitKey(0)
    flip_3 = tf.image.flip_left_right(x)
    flip_4 = tf.image.random_flip_up_down(x)
    flip_5 = tf.image.random_flip_left_right(x)

# flip('/Users/maggieliuzzi/predict_images/', '2.png', 50, 50, 3)

''' Rotation '''
''' 1 '''
def rotate_images(X_imgs, img_width, img_height):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(img_width, img_height, 3)) # correct order: width, height?
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                print(X_imgs)
                print(tf_img)
                print(img)
                print(i)
                rotated_img = sess.run(tf_img, feed_dict={X: img, k: i + 1}) # tf_img
                X_rotate.append(rotated_img)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate

# rotated_imgs = rotate_images(good_data_list,50,50)

''' 2 '''
def rotate():
    # Placeholders: 'x' = A single image, 'y' = A batch of images
    # 'k' denotes the number of 90 degree anticlockwise rotations
    shape = [height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    rot_90 = tf.image.rot90(img, k=1)
    rot_180 = tf.image.rot90(img, k=2)
    # To rotate in any angle. In the example below, 'angles' is in radians
    shape = [batch, height, width, 3]
    y = tf.placeholder(dtype = tf.float32, shape = shape)
    rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)
    # Scikit-Image. 'angle' = Degrees. 'img' = Input Image
    # For details about 'mode', checkout the interpolation section below.
    rot = skimage.transform.rotate(img, angle=45, mode='reflect')

''' Scale '''
def scale():
    # Scikit Image. 'img' = Input Image, 'scale' = Scale factor
    # For details about 'mode', checkout the interpolation section below.
    scale_out = skimage.transform.rescale(img, scale=2.0, mode='constant')
    scale_in = skimage.transform.rescale(img, scale=0.5, mode='constant')
    # Don't forget to crop the images back to the original size (for
    # scale_out)

''' Crop '''
def crop():
    # TensorFlow. 'x' = A placeholder for an image.
    original_size = [height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = original_size)
    # Use the following commands to perform random crops
    crop_size = [new_height, new_width, channels]
    seed = np.random.randint(1234)
    x = tf.random_crop(x, size = crop_size, seed = seed)
    output = tf.images.resize_images(x, size = original_size)

''' Translation '''
def translate():
    # pad_left, pad_right, pad_top, pad_bottom denote the pixel
    # displacement. Set one of them to the desired value and rest to 0
    shape = [batch, height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    # We use two functions to get our desired augmentation
    x = tf.image.pad_to_bounding_box(x, pad_top, pad_left, height + pad_bottom + pad_top, width + pad_right + pad_left)
    output = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)

''' Gaussian Noise '''
def gaussian_noise():
    #TensorFlow. 'x' = A placeholder for an image.
    shape = [height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    # Adding Gaussian noise
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0,
    dtype=tf.float32)
    output = tf.add(x, noise)

''' Interpolation '''
''' Constant, edge, reflect, symmetric and wrap modes '''