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
from IPython.display import Image
import PIL


e = TfPoseEstimator(get_graph_path('cmu'), target_size=(368, 368))

set_ = 'set00'
path = os.path.join('eth',set_,'V00')

images_path = os.listdir(path)
images_path.sort()

with open(os.path.join('eth_res',set_,'V000.txt'),'w') as f:
    for i in range(0,len(images_path)):
        image = common.read_imgfile(os.path.join(path,images_path[i]), None, None)
        humans = e.inference(image, resize_to_default=(432 > 0 and 368 > 0),
                             upsample_size=15)
        for human in humans:
            bb = human.get_full_body_box(image.shape[1],image.shape[0])
            
            if bb is not None:
                f.write('{},{},{},{},{ {}\n'.format(i+1,bb['x'],bb['y'],bb['w'],bb['h'],bb['confidence'])

