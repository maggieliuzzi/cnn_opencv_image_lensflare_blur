# Image Classification Project

This project is for the development of a convolutional neural network (CNN) for classifying images into good and faulty (i.e. blurry and/or with lens flare).

It was developed by Maggie Liuzzi.

The dataset used for training consists of:
```shell
75 images provided (25 good, 25 flare, 25 blurry)
Web-scrapping (https://github.com/hardikvasa/google-images-download, uses ChromeDriver) (keywords: )
    Commands:
    pip install google_images_download
    brew tap homebrew/cask
    brew cask install chromedriver
    (which chromedriver)
    eg. googleimagesdownload -k 'lens flare daytime' -l 212 -cd /usr/local/bin/chromedriver
```


## Installation

The project may be moved to a docker file at a later date, but for the time being, here is how it should be installed:

1. Install **Python 2.7** and **virtualenv** if you haven't - (should also work with Python 3.6, but needs further testing):

```shell
pip install virtualenv
```

2. Clone this repo to a directory of your choice:

```shell
git clone https://gitlab.com/maggieliuzzi/cnn_lensflare_blur.git
```

3. Enter the folder and create a new virtual environment:

```
cd cnn_lensflare_blur
virtualenv -p {path to your python 2.7 interpreter} venv
```


4. Install and add cv2 to your virtual environment:

```shell
For macOS (based on https://www.youtube.com/watch?v=iluST-V757A):
brew update (or install)
brew tap brewsci/bio
brew install opencv3 --with-contrib
cd usr/local/Cellar/opencv (use your path)
cd {version you want to use}
cd lib/python2.7/site-packages (or the version of python you want to use)
pwd (and copy the location)
{path/to/cv2.{}.so} (should find it, even if permission is denied)
cd venv (find and cd into the virtual environment you created)
source venv/bin/activate (to activate your virtual environment; omit "source" if on Windows)
cd lib/python2.7/site-packages
ln -s {path/to/cv2.{}.so} {cv2.so} (saving as cv2.so is optional)
cd ../../../ (to root of virtual environment)
pip install numpy
python
>>> import cv2 (should give no errors)
```


5. Install the project's dependencies:

```shell
pip install -r requirements.txt
```

With that, you should be good to go!



Predicting:

* **detector.py** takes a model and an image as arguments and classifies the image as 'Good' or 'Faulty'.
* **server.py** scripts start a server that receives HTTP POST request with a test image and outputs the estimated probabilities. 
Eg: http://0.0.0.0:4000/predict
"prediction": {
    "Good": 0.01101667433977127,
    "Faulty": 0.9543766379356384
}
* **predict_functions.py** defines functions used to make predictions.


Preprocessing and training:

* **train_good_faulty.py** trains the network over a certain number of epochs and outputs an .hdf5 model every epoch.
* **preprocessing_good_faulty.py** takes target fault to train for and separate data into training, validation and testing sets.
* **motion_blurrying.py** motion blurs images in a folder and saves them to a new folder.


Testing:

* **test.py** tests the quality of a model with the images in the test/ folder generated running preprocessing_good_faulty.py.


Extras:

* **utils.py** defines extra error, warning and timer functions.