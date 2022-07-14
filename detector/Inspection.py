import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

import utils
import visualize
from visualize import display_images
import model as modellib
from config import Config
from model import log

import seaborn as sns
import pandas as pd

import imgaug as ia
from imgaug import augmenters as iaa
import imageio


import csv
import dataset

# Load class map - these tables map the original TACO classes to your desired class system
# and allow you to discard classes that you don't want to include.
class_map = {}
with open("./taco_config/map_1.csv") as csvfile:
    reader = csv.reader(csvfile)
    class_map = {row[0]:row[1] for row in reader}

# Load full dataset or a subset
TACO_DIR = "../data"
round = None # Split number: If None, loads full dataset else if int > 0 selects split no 
subset = "train" # Used only when round !=None, Options: ('train','val','test') to select respective subset
dataset = dataset.Taco()
taco = dataset.load_taco(TACO_DIR, round, subset, class_map=class_map, return_taco=True)

# Must call before using the dataset
dataset.prepare()

print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

#Mask RCNN Configuration
class TacoTestConfig(Config):
    NAME = "taco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3
    NUM_CLASSES = dataset.num_classes
config = TacoTestConfig()

#I feel like I need to start in training mode before attempting to use my own things 
model=modellib.MaskRCNN(mode="inference", model_dir=TACO_DIR, config=config)

#inpath="./models/logs/mask_rcnn_taco_0100.h5"
#outpath = "C:/Users/blais/Internship/TACO-master/TACO-master/data"
#outpath="./models/logs/mask_rcnn_taco_0100.h5"
inpath="C:/Users/blais/models/logs/taco20220705T1712"

outpath="C:/Users/blais/models/logs/taco20220705T1712"

#I guess I need to figure out whihc weights to start with
model.load_weights(weights_in_path=inpath, weights_out_path=outpath, by_name=False, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])

image_id = np.random.choice(dataset.image_ids, 1)[0]
orimage, meta, class_ids, bbox, mask=modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

print(image_id)
log("orimage", orimage)
log("meta", meta)
log("class_ids", class_ids)
log("bbox", bbox)
log("mask", mask)

visualize.display_instances(orimage, bbox, mask, class_ids, dataset.class_names)

print(dataset.class_names)

results=model.detect([orimage], verbose=1)
r=results[0]

visualize.display_instances(orimage, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])

for x in range(5):
	model=modellib.MaskRCNN(mode="inference", model_dir=TACO_DIR, config=config)
	model.load_weights(weights_in_path=inpath, weights_out_path=outpath, by_name=False, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
	results=model.detect([orimage], verbose=1)
	r=results[0]
	visualize.display_instances(orimage, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])