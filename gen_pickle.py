#!/usr/bin/env python
# coding: utf-8

# # Library

# In[18]:


# import the necessary packages
from localbinarypatterns import LocalBinaryPatterns
from imutils import paths
from sklearn.model_selection import train_test_split
import argparse
import cv2
import os
import pickle


# # Initialize variables

# In[12]:


# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data_real = []
labels_real = []
data_fake = []
labels_fake = []


# # Loop over real and fake images

# In[13]:


# loop over the real images
for imagePath in paths.list_images("frames_reais/"):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# extract the label from the image path, then update the
	# label and data lists
	im = imagePath.split(os.path.sep)[-1]
	im = im[0:4]
	labels_real.append(im)
	data_real.append(hist)


# In[16]:


# loop over fake images
for imagePath in paths.list_images("frames_fakes/"):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# extract the label from the image path, then update the
	# label and data lists
	im = imagePath.split(os.path.sep)[-1]
	im = im[0:4]
	labels_fake.append(im)
	data_fake.append(hist)


# # Split Data

# In[19]:


# Split and join data
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, labels_real,
                                                                        test_size=0.25, random_state=42)
X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, labels_fake,
                                                                        test_size=0.25, random_state=42)
X_train = X_train_real + X_train_fake
X_test = X_test_real + X_test_fake
y_train = y_train_real + y_train_fake
y_test = y_test_real + y_test_fake


# # Save data pickle

# In[29]:


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

