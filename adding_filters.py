from utils import warning, error, timer
import cv2

# eg. image_path: /Users/maggieliuzzi/predict_images/2.png
def load_image(image_path):
    img = cv2.imread(image_path, 0)
    return img

def adding_lensflare(image_path):
    img = load_image(image_path)
    # Add lens flare
    return img

def adding_blur(image_path):
    img = load_image(image_path)
    # Add blur
    return img

