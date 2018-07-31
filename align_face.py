import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt
import glob
import os
import random

e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
# estimate human poses from a single image !
with open('/Users/rui/Downloads/FDDB-folds/FDDB.txt') as f:
    files_grabbed = [line.rstrip('\n') for line in f]

i = random.randint(0,len(files_grabbed))
path = 'originalPics/'+files_grabbed[i]+'.jpg'


path = '/Users/rui/Desktop/rui.jpg'
image = common.read_imgfile(path, None, None)
humans = e.inference(image, resize_to_default=(432 > 0 and 368 > 0),
                         upsample_size=4)

img_h, img_w = image.shape[:2]
faces = [human.get_face_box(img_w,img_h,mode=1) for human in humans]
faces = [face for face in faces if face is not None and face['h'] > 25]
    
x_1 = faces[0]['x']
x_2 = faces[0]['x'] + faces[0]['w']
y_1 = faces[0]['y']
y_2 = faces[0]['y'] + faces[0]['h']
cropped = image[y_1:y_2,x_1:x_2, :]
cropped = cv2.resize(cropped,(160,160))
cv2.imwrite('test.jpg',cropped)