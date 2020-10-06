# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
from imutils.video import VideoStream
import argparse
import cv2
import os
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
from imutils.video import FPS
from PIL import Image

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str,default = "face_detector",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-m", "--model", type=str,
	help="Path to model")
ap.add_argument("-p", "--pca", type=str, default=None,
	help="Path to pca model")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
# histogram equal
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# model
model = joblib.load(args["model"]) 

# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier("face_detector/haarcascade_frontalface_default.xml")


# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
print("[INFO] starting video stream...")
#vs = cv2.VideoCapture('rtsp://admin:admin@192.168.1.168:554/ch01/0')
vs = FileVideoStream('rtsp://admin:admin@192.168.1.168:554/ch01/0').start()
#vs = VideoStream(src=0).start()
time.sleep(1.0)


HISTS = []

fps = FPS().start()

# loop over the frames from the video stream
while vs.more():
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()

	# load the image, convert it to grayscale, and describe it
	#image = cv2.imread(imagePath)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect face
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
									flags=cv2.CASCADE_SCALE_IMAGE)


	if len(rects) > 0:
		# get only the image of the face
		face_gray = gray[rects[0][1]:rects[0][1]+rects[0][3], rects[0][0]:rects[0][0]+rects[0][2]]
		pil_gray = Image.fromarray(face_gray)
		open_cv_image = np.array(pil_gray)
	
		open_cv_image = image_resize(open_cv_image, height = 150)

		hist = desc.describe(open_cv_image)
		HISTS.append(hist)
		prediction = model.predict(hist.reshape(1, -1))
		print(prediction)
		cv2.rectangle(frame, (rects[0][0], rects[0][1]), (rects[0][0]+rects[0][2], rects[0][1]+rects[0][3]),
		(0, 0, 255), 2)
		cv2.putText(frame, prediction[0], (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

means = []
stds = []

X = np.array(HISTS)

var_imp = [16, 15, 17, 14, 13, 18, 12, 11, 10, 19, 20,  9,  8,  7,  6,  5,  4,
        3,  2, 21, 22, 23, 24,  1,  0, 25]

for i in var_imp:
    #print(i)
    #print("Mean: ",np.mean(X[:,i])," std:", np.std(X[:,i]))
    means.append(np.mean(X[:,i]))
    stds.append(np.std(X[:,i]))

fig,ax=plt.subplots(figsize=(5, 5))

ax.set_ylim([0,0.25])

ax.plot(means)
ax.plot(stds)
plt.show()
