# import the necessary packages
from localbinarypatterns import LocalBinaryPatterns
from imutils import paths
from sklearn.model_selection import train_test_split
import argparse
import cv2
import os
import pickle
from PIL import Image
import numpy as np
from pre_process.pre_process import pre_process_frame

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data_real = []
labels_real = []
data_fake = []
labels_fake = []

detector = cv2.CascadeClassifier("face_detector/haarcascade_frontalface_default.xml")
real_id = 0

# loop over the real images
for imagePath in paths.list_images("frames_reais/"):
    real_id += 1

	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)

	face, x1, x2, y1, y2 = pre_process_frame(image,detector)

	if face is not None:
		face = image_resize(face, height = 250)
		gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # save image
        cv2.imwrite("faces_reais/face"+str(real_id), face)

		hist = desc.describe(gray)

		# extract the label from the image path, then update the
		# label and data lists
		im = imagePath.split(os.path.sep)[-1]
		im = im[0:4]
		labels_real.append(im)
		data_real.append(hist)

fake_id  = 0

# loop over fake images
for imagePath in paths.list_images("frames_fakes/"):
    fake_id += 1
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)

	face, x1, x2, y1, y2 = pre_process_frame(image,detector)

	if face is not None:
		
		face = image_resize(face, height = 250)
		gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # save image
        cv2.imwrite("faces_fakes/face"+str(fake_id), face)

		hist = desc.describe(gray)

		# extract the label from the image path, then update the
		# label and data lists
		im = imagePath.split(os.path.sep)[-1]
		im = im[0:4]
		labels_fake.append(im)
		data_fake.append(hist)


# Split and join data
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, labels_real,
                                                                        test_size=0.25, random_state=42)
X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, labels_fake,
                                                                        test_size=0.25, random_state=42)
X_train = X_train_real + X_train_fake
X_test = X_test_real + X_test_fake
y_train = y_train_real + y_train_fake
y_test = y_test_real + y_test_fake

# Save data
pickle_out = open("data/data_train.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("data/labels_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("data/data_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("data/labels_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()
