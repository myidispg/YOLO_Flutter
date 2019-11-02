from __future__ import division
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import os 
from darknet import Darknet
import pandas as pd
import random

start = 0
CUDA = torch.cuda.is_available()

confidence = 0.5
nms_thresh = 0.4
cfg_file = "cfg/yolov3.cfg"
weightsfile = "yolov3.weights"

num_classes = 80    #For COCO
classes = load_classes("data/coco.names")
print(classes)


#Set up the neural network
print("Loading network.....")
model = Darknet(cfg_file)
model.load_weights(weightsfile)
print("Network successfully loaded")

# model.net_info["height"] = args.reso
inp_dim = 416

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()

# Read the images
# img = cv2.imread('dog-cycle-car.png')
img = cv2.imread('Screenshot (2).png')
orig_shape = img.shape
scaling_factor = [orig_shape[0]/inp_dim, orig_shape[1]/inp_dim]
# scaling_factor = [inp_dim/orig_shape[0], inp_dim/orig_shape[1]]
# scaling_factor = np.array([inp_dim/orig_shape[0], inp_dim/orig_shape[1]]).min()
print(orig_shape)
print(scaling_factor)
# Preprocess the image
img_tensor= prep_image(img, inp_dim)
# Convert to PyTorch tensor
if CUDA:
    img_tensor = img_tensor.cuda()

# Predict
prediction = model(img_tensor, CUDA)
prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
# The output is of the shape No_detected_objects x 8.
# The first is ignored. The last is the index in the classes list.

prediction = prediction.detach().cpu().numpy()

# Go over all the predictions and draw on the image
for single_prediction in prediction:
    # Find the class label
    class_label = classes[int(single_prediction[-1])]
    print(class_label)
    # Resize the coordinates according to the original image
    single_prediction[[1, 3]] *= scaling_factor[1]
    single_prediction[[2, 4]] *= scaling_factor[0]
    # single_prediction[[1, 3]] -= (inp_dim - scaling_factor[0]*orig_shape[0])/2
    # single_prediction[[2, 4]] -= (inp_dim - scaling_factor[1]*orig_shape[1])/2
    # single_prediction[[1, 3]] /= scaling_factor[0]
    # single_prediction[[2, 4]] /= scaling_factor[1]
    # Clip any bounding boxes with dimensions outside our image.
    single_prediction[[1, 3]] = np.clip(single_prediction[[1, 3]], 0.0, orig_shape[0])
    single_prediction[[2, 4]] = np.clip(single_prediction[[2, 4]], 0.0, orig_shape[1])

    # Draw the bounding boxes
    c1 = (int(single_prediction[1]), int(single_prediction[2]))
    c2 = (int(single_prediction[3]), int(single_prediction[4]))
    cv2.rectangle(img, c1, c2, (255, 0, 0), 1)
cv2.imwrite('image.png', img)
