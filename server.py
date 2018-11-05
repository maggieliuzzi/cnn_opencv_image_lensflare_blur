# Starts a server which can receive images via HTTP POST and return predictions
# Based on: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

from predict_functions import prepare_model, predict_from_file, predict_from_pil
import flask
import os
import argparse
import numpy as np

app = flask.Flask(__name__)
model = None

# Receive images via POST at /predict and respond with JSON containing the prediction vector
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    print("\nReceived a POST request.")

    # Ensure there is an 'image' attribute in POST request
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()

        raw_prediction = predict_from_pil(model, image)
        prediction = "Good" if raw_prediction[0] >= raw_prediction[1] else "Faulty"

        print("Probability of [Faulty, Good]: " + str(raw_prediction))
        data["prediction"] = {"Faulty": float(raw_prediction[0]), "Good": float(raw_prediction[1]), "Guess": prediction}

        data["success"] = True
    else:
        print("Image attribute not found in POST request.")

    return flask.jsonify(data)

# When this script is ran directly, prepare the model + server to take requests
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts server to predict a person's age and gender using a CNN model.",
        epilog="Created by Maggie Liuzzi")
    parser.add_argument('--model', default=None, required=True,
                        help="required; path to the neural network model file.")
    parser.add_argument('--port', default=4000, help="the port the server occupies; defaults to 4000.")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print("ERROR: Could not find model file.")
        exit(1)
    try:
        n = int(args.port)
        if n < 1:
            print("ERROR: Port number must be greater than or equal to 1.")
            exit(1)
    except ValueError:
        print("ERROR: Port number must be an integer.")
        exit(1)

    print("\nLoading Keras model...")
    model = prepare_model(args.model)
    print("\nLoading Flask server...")
    app.run(host="0.0.0.0", port='4000')
