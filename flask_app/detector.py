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

class Detector:
    def __init__(self):
        self.start = 0
        self.CUDA = torch.cuda.is_available()

        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.cfg_file = "cfg/yolov3.cfg"
        self.weightsfile = "yolov3.weights"

        self.num_classes = 80    #For COCO

        self.inp_dim = 416
        self.classes = load_classes("data/coco.names")
        #Set up the neural network
        print("Loading network.....")
        self.model = Darknet(self.cfg_file)
        self.model.load_weights(self.weightsfile)
        print("Network successfully loaded")

        if self.CUDA:
            self.model.cuda()

        self.model.eval()

    def detect_and_draw(self, img):
        orig_shape = img.shape
        scaling_factor = [orig_shape[0]/self.inp_dim, orig_shape[1]/self.inp_dim]

        # Preprocess the image
        img_tensor= prep_image(img, self.inp_dim)
        # Convert to PyTorch tensor
        if self.CUDA:
            img_tensor = img_tensor.cuda()

        # Predict
        prediction = self.model(img_tensor, self.CUDA)
        prediction = write_results(prediction, self.confidence, self.num_classes, nms_conf = self.nms_thresh)
        # The output is of the shape No_detected_objects x 8.
        # The first is ignored. The last is the index in the classes list.
        
        # If no object detected
        if type(prediction) == int:
            return 0

        prediction = prediction.detach().cpu().numpy()

        # Go over all the predictions and draw on the image
        for single_prediction in prediction:
            # Find the class label
            class_label = self.classes[int(single_prediction[-1])]
            # Resize the coordinates according to the original image
            single_prediction[[1, 3]] *= scaling_factor[1]
            single_prediction[[2, 4]] *= scaling_factor[0]

            # Clip any bounding boxes with dimensions outside our image.
            single_prediction[[1, 3]] = np.clip(single_prediction[[1, 3]], 0.0, orig_shape[0])
            single_prediction[[2, 4]] = np.clip(single_prediction[[2, 4]], 0.0, orig_shape[1])

            # Draw the bounding boxes
            c1 = (int(single_prediction[1]), int(single_prediction[2]))
            c2 = (int(single_prediction[3]), int(single_prediction[4]))
            cv2.rectangle(img, c1, c2, (255, 0, 0), 1)

            # Write the class label
            text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.putText(img, class_label, (c1[0], c1[1] + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)
        return img

if __name__ == '__main__':
    detector = Detector()
    # Read the image, predict, draw and save.
    img = cv2.imread('dog-cycle-car.png')
    img = detector.detect_and_draw(img)
    cv2.imwrite('image.png', img)