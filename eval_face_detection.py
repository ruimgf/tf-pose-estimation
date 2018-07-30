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

with open('tf_pose_estimation.txt', "w") as f:
    for image_name in files_grabbed[:10]:
        path = 'originalPics/' + image_name + '.jpg'
        image = common.read_imgfile(path, None, None)
        humans = e.inference(image, resize_to_default=(432 > 0 and 368 > 0),
                             upsample_size=4)
        f.write(image_name + '\n')
        img_h, img_w = image.shape[:2]
        faces = [face for face in faces if face is not None and face['h'] > 25]
        f.write(str(len(faces)) + '\n')
        for face in faces:
            f.write('{} {} {} {} {}\n'.format(face['x'], face['y'], face['w'], face['h'], face['confidence']))
