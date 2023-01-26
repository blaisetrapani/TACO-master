import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import track_test_notebook

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
import utils
import visualize
from visualize import display_images
import model as modellib
from config import Config
from model import log
import detector
import seaborn as sns
import pandas as pd
import imageio
import norfair
from norfair import Detection, Paths, Tracker, Video
from typing import List


def setup():
    # Root directory of the project
    ROOT_DIR = os.path.abspath(".")
    print(ROOT_DIR)
    MODEL_DIR=os.path.join(ROOT_DIR, "models\logs")
    print(MODEL_DIR)
    COCO_MODEL_PATH=os.path.join(ROOT_DIR, "models\mask_rcnn_taco0100.h5")
    print(COCO_MODEL_PATH)

    import csv
    import dataset
    # Load class map - these tables map the original TACO classes to your desired class system
    # and allow you to discard classes that you don't want to include.
    class_map = {}
    with open("./taco_config/map_10.csv") as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]:row[1] for row in reader}

    # Load full dataset or a subset
    TACO_DIR = "../data"
    round = None # Split number: If None, loads full dataset else if int > 0 selects split no 
    subset = "test" # Used only when round !=None, Options: ('train','val','test') to select respective subset
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
        DETECTION_MIN_CONFIDENCE = 3
        NUM_CLASSES = dataset.num_classes
    config = TacoTestConfig()

    model=modellib.MaskRCNN(mode="inference", model_dir=TACO_DIR, config=config)
    print(MODEL_DIR)

    model_path="./models/logs/mask_rcnn_taco_0100.h5"

    model.load_weights(weights_in_path=model_path, weights_out_path=model_path, by_name=True)

def norftrack(res,class_list):
#res: the list of dicts returned by the model.detect function
#class_list: the list of class ids for each detection

    norftection: List[Detection]=[]
    for x in range(len(res['rois'])):
        yc=(res['rois'][x][0]+res['rois'][x][2])/2
        xc=(res['rois'][x][1]+res['rois'][x][3])/2
        
        #the class of the particular detection
        det_class = res['class_ids'][x] #int class ID
        class_list.append(det_class)
        
        centroid=np.array([xc, yc])
        scores=np.array([res['scores'][x]])
        norftection.append(Detection(points=centroid, scores=scores))
        
    return norftection

def euc_distance(detection, tracked_obj):
    #print(tracked_obj.estimate)
    return np.linalg.norm(detection.points-tracked_obj.estimate)

def key_distance(detection, tracked_obj):
    box1=np.concatenate([detection.points[0], detection.points[1]])
    box2=np.concatenate([tracked_obj.estimate[0], tracked_obj.estimate[1]])
    
    #print(box1)
    #print(box2)
    ya=max(box1[0], box2[0])
    xa=max(box1[1], box2[1])
    yb=max(box1[2], box2[2])
    xb=max(box1[3], box2[3])
    
    area=max(0, xb-xa+1)*max(0, yb-ya+1)
    
    aarea=(box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
    barea=(box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
    
    iou=area/float(aarea+barea-area)
    
    return 1/iou if iou else (10000)

def center(pnt):
    return[np.mean(np.array(pnt), axis=0)]

def process(inputVid):
    setup()
    import cv2 as cv
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing.image import array_to_img
    # make a list to store all the frames
    vid_array = []
    # getting the video
    vid2 = cv.VideoCapture(inputVid)

    #norfair tracker
    tracker=norfair.Tracker(distance_function=euc_distance, distance_threshold=30)
    path=Paths(center, attenuation=0.01)

    # checks whether frames were extracted
    success = 1
    while success:
        # read the frame and add it to the frame list
        success, image = vid2.read()
        vid_array.append(image)
    #color correction
    count=0
    for img in vid_array:
        if count<(len(vid_array)-1):
            img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
            vid_array[count]=img
        count+=1
        
    #each class is represented by a set that will contain detection ids without duplication
    totalitems=set()
    bottles=set()
    botcap=set()
    cans=set()
    cigarettes=set()
    cups=set()
    lids=set()
    others=set()
    wrapperbag=set()
    poptabs=set()
    straws=set()
    #list of sets can be indexed by class_id
    setlist = [totalitems,bottles,botcap,cans,cigarettes,cups,lids,others,wrapperbag,poptabs,straws]

    # go through and detect all frames in the list  
    frame_counter=1
    #line=400
    obj_class = []

    # set up for video export
    frameSize = (1280,720)
    out = cv.VideoWriter('../data/vid2/test_vid.mp4',cv.VideoWriter_fourcc(*"mp4v"), 10, frameSize)

    for vid_img in vid_array:
        if frame_counter<(len(vid_array)-1): #ignore the last image, which is blank
            orimage=img_to_array(vid_img)
            results=model.detect([orimage], verbose=1)
            r=results[0]
            print("Frame " + str(frame_counter))
            
            #convert to norfair detection and add to tracked items
            det=norftrack(r,obj_class)
            tracked=tracker.update(detections=det) #list of tracked items
            
            norfair.draw_points(orimage, det)
            norfair.draw_tracked_objects(orimage, tracked, color=(0, 225, 0), id_size=3, id_thickness=2)
            
            if (frame_counter>7):
                x = 0
                while x < len(tracked):
                    item = tracked[x] #specific tracked item
                    item_class = obj_class[x] #item's corresponding class id
                    #if ((line+15)>item.estimate[0][1]>(line-15)): #if an object is within range of the line
                    setlist[0].add(item.id) #add the item to the list of counted items
                    print("item", item.id, "detected",dataset.class_names[item_class])
                    setlist[item_class].add(item.id) #add the item id to its class's corresponding set
                    x += 1
                print("bottles", bottles)
                print("bottle caps", botcap)
                print("cans", cans)
                print("cigarettes", cigarettes)
                print("cups", cups)
                print("lids", lids)
                print("other", others)
                print("wrapper + bag", wrapperbag)
                print("pop tabs", poptabs)
                print("straws", straws)
                print("total", totalitems)

            frame=path.draw(vid_img, tracked)
            #orimage=cv.line(orimage, (0,line), (2000, line), (255, 0, 0), 6)
            frame=cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            out.write(frame)
            visualize.display_instances(orimage, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
            plt.close()  # close the figure after displaying it to free up memory
            frame_counter+=1
    out.release()        
    print("bottles", len(bottles))
    print("bottle caps", len(botcap))
    print("cans", len(cans))
    print("cigarettes", len(cigarettes))
    print("cups", len(cups))
    print("lids", len(lids))
    print("other", len(others))
    print("wrapper + bag", len(wrapperbag))
    print("pop tabs", len(poptabs))
    print("straws", len(straws))
    print("total", len(totalitems))

#-----------------------------GUI-----------------------
def processVid(inputvid):
    process(inputvid)
    

class CostiaUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.vidFile = "" #video to be processed
        self.geometry('500x500')
        self.fileButton = tk.Button(self,text="Browse Files",command=self.getFile)
        self.fileButton.pack()
        self.fileLabel = tk.Label (self,text="File name:")
        self.fileLabel.pack()
        self.procButton = tk.Button(self,text="Process Video",command=processVid(self.vidFile),state='disabled')
        self.procButton.pack()

    #file explorer, gets the video to be processed
    def getFile(self):
        self.vidFile = filedialog.askopenfilename(initialdir = "/",title = "Select a File")
        self.fileLabel.configure(text="File name" + self.vidFile)
        self.procButton['state'] = 'normal'


exGUI = CostiaUI()
exGUI.mainloop()