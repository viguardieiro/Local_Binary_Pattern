#!/usr/bin/env python
# coding: utf-8

# In[6]:


# import the necessary packages
import numpy as np
import argparse
import cv2
import os


# In[7]:


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector","res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# In[8]:


# get the index from folder

lista = os.listdir("frames_reais")
if len(lista) > 1:
	# get the index of frames
	index = [int(i[5:-4]) for i in lista]
	# sort the index
	index.sort()
	# get the last index
	saved = len(index) - 1
else:
	saved = 0


# In[9]:


# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture("videos_reais/videos_real.mp4")
read = 0


# In[10]:


# loop over frames from the video file stream
while True:
    # grab the frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # increment the total number of frames read thus far
    read += 1

    # check to see if we should process this frame
    if read % 1 != 0:
        continue

    # grab the frame dimensions and construct a blob from the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
        
            # write the frame to disk
            nome = "frames_reais" + "/" + "real" + "_" + str(saved) + ".png"
            cv2.imwrite(nome, face)
            saved += 1
            print("[INFO] saved {} to disk".format(nome))
# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()


# In[ ]:


lista = os.listdir("frames_fakes")
if len(lista) > 1:
    # get the index of frames
    index = [int(i[5:-4]) for i in lista]
    # sort the index
    index.sort()
    # get the last index
    saved = len(index) - 1
else:
    saved = 0


# In[ ]:


# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture("videos_fakes/videos_fakes.mp4")
read = 0


# In[ ]:


# loop over frames from the video file stream
while True:
    # grab the frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # increment the total number of frames read thus far
    read += 1

    # check to see if we should process this frame
    if read % 1 != 0:
        continue

    # grab the frame dimensions and construct a blob from the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
        
            # write the frame to disk
            nome = "frames_fakes" + "/" + "fake" + "_" + str(saved) + ".png"
            cv2.imwrite(nome, face)
            saved += 1
            print("[INFO] saved {} to disk".format(nome))
# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()

