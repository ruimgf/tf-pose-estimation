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

base_path = '/home/ruimgf/datasets/WIDER_test/images'
results_path = 'resultsWider'

#e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
# estimate human poses from a single image !
with open('paths_wider.txt') as f:
    files_grabbed = [line.rstrip('\n') for line in f]

if not os.path.isdir(results_path):
	os.mkdir(results_path)
sys.exit()

for folder in files_grabbed:
	if not os.path.isdir(os.path.join(results_path,folder)):
    	os.mkdir(os.path.join(results_path,folder))

	for image_name in os.listdir(os.path.join(base_path,folder))
	    path = os.path.join(base_path,folder,image_name)

	    image = common.read_imgfile(path, None, None)
	    humans = e.inference(image, resize_to_default=(432 > 0 and 368 > 0),
	                         upsample_size=4)
	    
	    
	    with open(os.path.join(results_path,folder,image_name.split('.')[0] + '.txt'))
	    	f.write(image_name.split('.')[0] + '\n')
	    	img_h, img_w = image.shape[:2]
	   		faces = [human.get_face_box(img_w, img_h, mode=1) for human in humans]
	    	faces = [face for face in faces if face is not None and face['h'] > 25]
	    	f.write(str(len(faces)) + '\n')
	    	for face in faces:
	            f.write('{} {} {} {} {}\n'.format(face['x'], face['y'], face['w'], face['h'], face['confidence']))
