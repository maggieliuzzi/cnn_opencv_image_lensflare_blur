from utils import warning, error, timer
import math
import os
import argparse
import shutil
import csv
import glob # Try
from random import shuffle


parser = argparse.ArgumentParser(
    description="Retrains top layers of MobileNetV2 for classifying images into good and faulty.",
    epilog="Created by Maggie Liuzzi")
parser.add_argument('--fault', default=None, required=True,
                    help="the fault to train for (i.e. flare or blurry; required.")
args = parser.parse_args()
fault = args.fault


# I first manually duplicated the images to reach 200 in total in each category + added images sourced by web scrapping

home_path = os.path.dirname(os.path.abspath(__file__)) # abspath?
source_path = os.path.join(home_path, "training-data/")
good_data_list = glob.glob(os.path.join(source_path,'good-data/*.JPG')) # Try
flare_data_list = glob.glob(os.path.join(source_path,'flare-data/*.JPG'))
blurry_data_list = glob.glob(os.path.join(source_path,'blurry-data/*.JPG'))
csv_path = os.path.join(source_path,'data.csv')

with open(csv_path, 'w') as wf:
    writer = csv.writer(wf, delimiter=',', quoting=csv.QUOTE_ALL)
    for item in good_data_list:
        writer.writerow([item] + ["good"])
    for item in flare_data_list:
        writer.writerow([item] + ["flare"])
    for item in blurry_data_list:
        writer.writerow([item] + ["blurry"])


if fault == 'flare':
    dataset_path = os.path.join(home_path, "training-data-formatted-flare/")
elif fault == 'blurry':
    dataset_path = os.path.join(home_path, "training-data-formatted-blurry/")

train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")
train_path_0 = os.path.join(train_path, "good")
if fault == 'flare':
    train_path_1 = os.path.join(train_path, "flare")
elif fault == 'blurry':
    train_path_1 = os.path.join(train_path, "blurry")
validate_path_0 = os.path.join(validate_path, "good")
if fault == 'flare':
    validate_path_1 = os.path.join(validate_path, "flare")
elif fault == 'blurry':
    validate_path_1 = os.path.join(validate_path, "blurry")
test_path = os.path.join(dataset_path, "test")

processed_paths = [train_path_0, train_path_1, validate_path_0, validate_path_1, test_path]

for path in processed_paths:
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Created directory: " + path)
    else:
        print("Directory already exists: " + path)


with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)

    total_images = 0
    usable = []
    # Assuming all training images are usable - Add filtering
    for row in reader:
        usable.append(row)
        total_images += 1
    print("Loaded data from " + csv_path)

    # Randomising data to ensure even distribution between train, validate and test folders
    usable = [i for i in usable]
    shuffle(usable)

    usable_images = len(usable)
    train_cutoff = 0.64
    validate_cutoff = 0.8
    train_images = int(math.floor(usable_images * train_cutoff))
    validate_images = int(math.floor(usable_images * (validate_cutoff - train_cutoff)))
    test_images = usable_images - (train_images + validate_images)

    train_images_data = []
    validate_images_data = []
    test_images_data = []
    current_point = 0

    with open(dataset_path + "form_data.csv", "w") as form_data_file, open(dataset_path + "test.csv", "w") as test_data_file:
        # For all training images
        for i in range(0, train_images):
            train_images_data.append(usable[i])
            role = "train"
            quality = usable[i][1]
            source = os.path.join(source_path, quality+"-data", usable[i][0])
            destination = os.path.join(train_path, quality)
            shutil.copy(source, destination)
            writer = csv.writer(form_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [quality] + [role])
            current_point += 1
        # For all validation images
        for i in range(train_images, train_images + validate_images):
            validate_images_data.append(usable[i])
            role = "validate"
            quality = usable[i][1]
            source = os.path.join(source_path, quality+"-data", usable[i][0])
            destination = os.path.join(validate_path, quality)
            shutil.copy(source, destination)
            writer = csv.writer(form_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [quality] + [role])
            current_point += 1
        # For all testing images
        for i in range(train_images + validate_images, usable_images):
            test_images_data.append(usable[i])
            role = "test"
            quality = usable[i][1]
            source = os.path.join(source_path, quality+"-data", usable[i][0])
            destination = os.path.join(test_path)
            shutil.copy(source, destination)
            writer = csv.writer(form_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [quality] + [role])
            writer = csv.writer(test_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [quality] + [role])
            current_point += 1

    print("Total images: " + str(total_images))
    print("Usable images: " + str(usable_images))
    print("Training images: " + str(len(train_images_data)))
    print("Validation images: " + str(len(validate_images_data)))
    print("Testing images: " + str(len(test_images_data)))

print("Pre-processing for 'good' and '"+fault+"' done.")
