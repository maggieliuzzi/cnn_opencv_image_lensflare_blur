from utils import warning, error, timer

import math
import os
import shutil
import csv
import glob # Try

home_path = os.path.dirname(os.path.abspath(__file__)) # abspath?
source_path = os.path.join(home_path, "training-data/")
good_data_list = glob.glob(os.path.join(source_path,'good-data/*.JPG')) # Try
faulty_data_list = glob.glob(os.path.join(source_path,'faulty-data/*.JPG'))
data_list = good_data_list + faulty_data_list

with open(os.path.join(source_path,'data.csv'), 'w') as wf:
    for item in data_list:
        writer = csv.writer(wf, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow([item])
dataset_path = os.path.join(home_path, "training-data-formatted/")
csv_path = os.path.join(source_path, "data.csv")

train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")
train_path_0 = os.path.join(train_path, "g")
train_path_1 = os.path.join(train_path, "f")
validate_path_0 = os.path.join(validate_path, "g")
validate_path_1 = os.path.join(validate_path, "f")
test_path = os.path.join(dataset_path, "test/")
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
    # Assuming all training images are usable
    usable = []
    for row in reader:
        image_folder = row[0]
        original_image = row[1]
        face_id = row[2]
        image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
        actual_age = row[3]
        binned_age = row[12]
        if int(actual_age) > 0 and binned_age is not None:
            usable.append(row)
        gender = row[4]
        if gender is not None:
            usable.append(row)
        role = None
        total_images += 1
    print("Loaded data from " + csv_path)

    good_images = len(usable)
    train_cutoff = 0.64
    validate_cutoff = 0.8
    train_images = int(math.floor(good_images * train_cutoff))
    validate_images = int(math.floor(good_images * (validate_cutoff - train_cutoff)))
    test_images = good_images - (train_images + validate_images)

    train_images_data = []
    validate_images_data = []
    test_images_data = []
    current_point = 0

    with open(dataset_path + "labels.csv", "w") as file, open(test_path + "test_labels.csv", "w") as test_labels_file:
        # For all training images
        for i in range(0, train_images):
            train_images_data.append(usable[i])
            role = "train"
            image_folder = usable[i][0]
            original_image = usable[i][1]
            face_id = usable[i][2]
            image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
            age = usable[i][12]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(dataset_path, filepath)
            destination = os.path.join(train_path, age)
            shutil.copy(source, destination)
            file.write(str(usable[i]) + '\n')
            current_point += 1
        # For all validation images
        for i in range(train_images, train_images + validate_images):
            validate_images_data.append(usable[i])
            role = "validate"
            image_folder = usable[i][0]
            original_image = usable[i][1]
            face_id = usable[i][2]
            image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
            age = usable[i][12]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(dataset_path, filepath)
            destination = os.path.join(validate_path, age)
            shutil.copy(source, destination)
            file.write(str(usable[i]) + '\n')
            current_point += 1
        # For all testing images
        for i in range(train_images + validate_images, good_images):
            test_images_data.append(usable[i])
            role = "test"
            image_folder = usable[i][0]
            original_image = usable[i][1]
            face_id = usable[i][2]
            image_name = "coarse_tilt_aligned_face." + face_id + "." + original_image
            age = usable[i][12]
            filepath = 'faces/' + image_folder + '/' + image_name
            source = os.path.join(dataset_path, filepath)
            destination = os.path.join(test_path)
            shutil.copy(source, destination)
            file.write(str(usable[i]) + '\n')
            test_labels_file.write(str(usable[i]) + '\n')
            current_point += 1

    print("Total images: " + str(total_images))
    print("Usable images: " + str(good_images))
    print("Training images: " + str(len(train_images_data)))
    print("Validation images: " + str(len(validate_images_data)))
    print("Testing images: " + str(len(test_images_data)))

print("Processing done.")
