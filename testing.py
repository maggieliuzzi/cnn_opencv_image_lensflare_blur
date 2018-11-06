''' Tests a model against the images in a test folder '''

from predict_functions import predict_from_file, prepare_model
import argparse
import csv
import os

parser = argparse.ArgumentParser(
        description="Tests a model against the images in a test folder.")
parser.add_argument('--model', default=None, required=True,
                        help="required; path to the model file.")
parser.add_argument('--fault', default=None, required=True,
                    help="the fault to train for (i.e. flare or blurry); required.")
args = parser.parse_args()
model = prepare_model(args.model)
fault = args.fault

home_path = os.path.dirname(os.path.abspath(__file__))


with open(home_path+"/training-data-formatted-"+fault+"/test.csv",'r') as f, open(home_path+"/training-data-formatted-"+fault+"/test_pred.csv",'w') as newf:
    reader = csv.reader(f)
    writer = csv.writer(newf)

    line_count = 0
    faulty = 0
    good = 0
    correct_guesses = 0
    corr_faulty = 0
    corr_good = 0

    for line in reader:
        newline = line
        actual_quality = newline[1]

        path_to_test = home_path+"/training-data-formatted-"+fault+"/test/"
        original_image = newline[0]
        to_replace_1 = home_path+'/training-data/'+fault+'-data/'
        to_replace_2 = home_path+'/training-data/good-data/'
        original_image = original_image.replace(to_replace_1, '')
        original_image = original_image.replace(to_replace_2, '')
        image = path_to_test + original_image
        print("image: "+image)

        probability_vector = predict_from_file(model, image)

        if probability_vector[0] >= probability_vector[1]:
            predicted_quality = "f"
            faulty += 1
        else:
            predicted_quality = "g"
            good += 1

        newline.append(predicted_quality)

        if predicted_quality == actual_quality:
            accuracy = "Y"
            correct_guesses += 1
            if actual_quality == "f":
                corr_faulty += 1
            elif actual_quality == "g":
                corr_good += 1
        else:
            accuracy = "N"

        newline.append(accuracy)

        writer.writerow(newline)
        line_count += 1
        print("line_count: " + str(line_count))

    print("Number of test images: " + str(line_count))
    print("correct_guesses: " + str(correct_guesses))
    perc_corr_guesses = correct_guesses / line_count
    print("% correct guesses: " + perc_corr_guesses)
    perc_corr_faulty = corr_faulty / faulty
    print("% correct faulty/ total faulty: " + perc_corr_faulty)
    perc_corr_good = corr_good / good
    print("% correct good/ total good: " + perc_corr_good)

print("End of file.")
