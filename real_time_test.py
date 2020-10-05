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



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str,default = "face_detector",
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-m", "--model", type=str,
	help="Path to model")
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
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


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

	#time.sleep(0.1)

	#cv2.imshow("Frame", frame)

	#key = cv2.waitKey(1) & 0xFF
	#fps.update()
 
	# if the `q` key was pressed, break from the loop
	#if key == ord("q"):
	#	break

	#continue


	# grab the frame dimensions and construct a blob from the frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))


	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()


	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			hist = desc.describe(gray)
			HISTS.append(hist)
			prediction = model.predict(hist.reshape(1, -1))
			print(prediction)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
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
