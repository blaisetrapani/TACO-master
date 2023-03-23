#Ellen, Blaise
#2/23/2023
#--------------------------
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
#import track_test_notebook
import webbrowser
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


def norftrack(res,class_list):
#res: the list of dicts returned by the model.detect function
#class_list: the list of class ids for each detection

    norftection: List[Detection]=[]
    for x in range(len(res['rois'])):
        yc=(res['rois'][x][0]+res['rois'][x][2])/2
        xc=(res['rois'][x][1]+res['rois'][x][3])/2
        
        #the class of the particular detection
        det_class = res['class_ids'][x] #int class ID
        print(det_class)
        class_list.append(det_class)
        
        centroid=np.array([xc, yc])
        scores=np.array([res['scores'][x]])
        #label=np.array(class_list[res['class_ids'][x]])
        norftection.append(Detection(points=centroid, scores=scores))
        #norftection.append(Detection(points=centroid, scores=scores, label=label))
        
    return norftection

def norftrack2(res, labels):
    norftection: List[Detection]=[]
    for x in range(len(res['rois'])):
        yc=(res['rois'][x][0]+res['rois'][x][2])/2
        xc=(res['rois'][x][1]+res['rois'][x][3])/2
        
        centroid=np.array([xc, yc])
        scores=np.array([res['scores'][x]])
        label=np.array(labels[res['class_ids'][x]])
        norftection.append(Detection(points=centroid, scores=scores, label=label))
        
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
    spectracker=norfair.Tracker(distance_function=euc_distance, distance_threshold=30)
    
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
    #setting video dimensions
    height,width = vid_array[0].shape[:2]
    
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
    specscores=[]
    
    # set up for video export
    frameSize = (height,width)
    #frameSize = (1280,720)
    #make a saves folder in the detector folder
    savepath = './saves'
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath)
    out = cv.VideoWriter('./saves/test.mp4',cv.VideoWriter_fourcc(*"mp4v"), 10, frameSize)

    for vid_img in vid_array:
        if frame_counter<(len(vid_array)-1): #ignore the last image, which is blank
            orimage=img_to_array(vid_img)
            results=model.detect([orimage], verbose=1)
            r=results[0]
            print("Frame " + str(frame_counter))
            
            #convert to norfair detection and add to tracked items
            det=norftrack(r,obj_class)
            specdet=norftrack2(r, dataset.class_names)
            
            print(det)
            
            tracked=tracker.update(detections=det) #list of tracked items
            spectracked=spectracker.update(detections=specdet)
            print(tracked)
            print(spectracked)
            

            norfair.draw_points(orimage, det)
            norfair.draw_tracked_objects(orimage, tracked, color=(0, 225, 0), id_size=3, id_thickness=2, draw_labels=True)
            objdet=False
            if (frame_counter>7):
                x = 0
                while x < len(tracked):
                    print(tracked[0].age)
                    item = tracked[x] #specific tracked item
                    print(item)
                    item_class = obj_class[x] #item's corresponding class id
                    print(obj_class)
                    #if ((line+15)>item.estimate[0][1]>(line-15)): #if an object is within range of the line
                    setlist[0].add(item.id) #add the item to the list of counted items
                    #print("item", item.id, "detected",dataset.class_names[item_class])
                    print("item", item.id, "detected")
                    #setlist[item_class].add(item.id) #add the item id to its class's corresponding set 
                    x += 1
                    print(item.last_detection.points[0][0])

                    if (item.id==2):
                        objdet=True

                x=0
                for item in spectracked:
                    #for x in tracked:
                                            
                    if (len(specscores)<item.id):
                        specscores.append([])
                    if item.label=="Bottle":
                        bottles.add(item.id)
                    elif item.label=="Bottle cap":
                        botcap.add(item.id)
                    elif item.label=="Can":
                        cans.add(item.id)
                    elif item.label=="Cigarette":
                        cigarettes.add(item.id)
                    elif item.label=="Cup":
                        cups.add(item.id)
                    elif item.label=="Lid":
                        lids.add(item.id)
                    elif item.label=="Other":
                        others.add(item.id)
                    elif item.label=="Plastic bag + wrapper":
                        wrapperbag.add(item.id)
                    elif item.label=="Pop tab":
                        poptabs.add(item.id)
                    elif item.label=="Straw":
                        straws.add(item.id)
                    print("item", item.id, "detected",item.label)
                    print("item", item.id, "detected",item.last_detection.points)
                    specscores[(item.id)-1].append(item.last_detection.scores[0])
                    

                    x+=1

                    
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
            #print(r['scores'])
            #visualize.display_instances(orimage, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
            #debugging tool
            if objdet==True:
                visualize.display_instances(orimage, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])

            save_dir = ("./saves/"+str(frame_counter).rjust(5,'0')+".png")
            #change visualize.py so that display_instances takes a save directory in the function call and saves the figure as a png at that directory
            visualize.display_instances(orimage, save_dir,r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
            frame_counter+=1

    for file in os.listdir("./saves"):
        frame = cv.imread("./saves/" + file)
        out.write(frame)
        
    out.release()
    outfile=open("results.txt", "w")
    outfile.write("Plastics Count Costia \n")
    outfile.write("\nbottles ")
    outfile.write(str(len(bottles)))
    outfile.write("\nbottle caps ")
    outfile.write(str(len(botcap)))
    outfile.write("\ncans ")
    outfile.write(str(len(cans)))
    outfile.write("\ncigarettes ")
    outfile.write(str(len(cigarettes)))
    outfile.write("\ncups ")
    outfile.write(str(len(cups)))
    outfile.write("\nlids ")
    outfile.write(str(len(lids)))
    outfile.write("\nother ")
    outfile.write(str( len(others)))
    outfile.write("\nwrapper + bag ")
    outfile.write(str(len(wrapperbag)))
    outfile.write("\npop tabs ")
    outfile.write(str(len(poptabs)))
    outfile.write("\nstraws ")
    outfile.write(str(len(straws)))
    outfile.write("\ntotal ")
    outfile.write(str(len(totalitems)))

    for x in range(len(specscores)):
        outfile.write("\nScore of item ")
        outfile.write(str(x))
        outfile.write(str(specscores[x]))
        average=sum(specscores[x])/len(specscores[x])
        outfile.write("\nAverage:")
        outfile.write(str(average))

def train():
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

    TACO_DIR = "../data"
    round = None # Split number: If None, loads full dataset else if int > 0 selects split no 
    subset = "train" # Used only when round !=None, Options: ('train','val','test') to select respective subset
    dataset = dataset.Taco()
    taco = dataset.load_taco(TACO_DIR, round, subset, class_map=class_map, return_taco=True)
    dataset.prepare()
    nr_classes = dataset.num_classes
    
    dataset_val=dataset.load_taco(dataset, round, "val", class_map=class_map, auto_download=None)
    dataset_val.prepare()

    class TacoTrainConfig(Config):
        NAME = "taco"
        #originally Images_per_gpu equalled 2
        IMAGES_PER_GPU = 1
        GPU_COUNT = 1
        STEPS_PER_EPOCH = min(1000,int(dataset.num_images/(IMAGES_PER_GPU*GPU_COUNT)))
        #STEPS_PER_EPOCH=100000
        USE_MINI_MASK = True
        MINI_MASK_SHAPE = (512, 512)
        NUM_CLASSES = nr_classes
    config = TacoTrainConfig()

    model=modellib.MaskRCNN(mode="training", model_dir=TACO_DIR, config=config)
    print(MODEL_DIR)

    model_path="./models/logs/mask_rcnn_taco_0100.h5"

    model.load_weights(model_path, model_path, by_name=True)

    training_meta = {
        'number of classes': nr_classes,
        'round': round,
        #'use_augmentation': args.aug,
        #'use_transplants': args.use_transplants != None,
        'learning_rate': config.LEARNING_RATE,
        'layers_trained': 'all'}

    subdir = os.path.dirname(model.log_dir)
    if not os.path.isdir(subdir):
        os.mkdir(subdir)

    if not os.path.isdir(model.log_dir):
        os.mkdir(model.log_dir)

    train_meta_file = model.log_dir + '_meta.json'
    with open(train_meta_file, 'w+') as f:
        f.write(json.dumps(training_meta))

    # Training all layers
    model.train(taco, dataset_val,learning_rate=config.LEARNING_RATE, epochs=100,
                layers='all', augmentation=None)




#-----------------------------GUI-----------------------
    

class CostiaUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('500x500')
        self.detections = []
        self.vidFile = "" #video to be processed
        lButton = tk.Button(self,text="Manual image labeling",command=self.label_mode)
        dButton = tk.Button(self,text="Automated trash detection",command=self.detect_mode)
        tButton = tk.Button(self,text="Train the model",command=self.train_mode)

        lButton.pack()
        dButton.pack()
        tButton.pack()
        #display video w detections for playback
        #display list of detections 
        #download processed video (mp4)
        #download list of detections (csv)

    def enableMode(self):
        self.selectButton['state'] = 'normal'
        print(self.mode.get() + 1)  

    def l_or_d(self):
        if self.mode.get() == 0:
            self.label_mode()
        else:
            self.detect_mode()

    def label_mode(self):
        os.system('cmd /c "docker pull heartexlabs/label-studio:latest"')
        os.system('cmd /c "docker run -d -p 8081:8080 heartexlabs/label-studio:latest"')
        #implement some way to wait for the container to load properly so you won't get an error page
        #is there a way to "ask" docker if the container is ready?
        webbrowser.open("http://localhost:8081/")
        print("label")
        

    def detect_mode(self):
        self.fileButton = tk.Button(self,text="Browse Files",command=self.getFile)
        self.fileButton.pack()
        self.fileLabel = tk.Label (self,text="File name:")
        self.fileLabel.pack()
        self.procButton = tk.Button(self,text="Process Video",state='disabled')
        self.procButton.pack()
        print("detect")

    def train_mode(self):
        train()

    #file explorer, gets the video to be processed
    def getFile(self):
        self.vidFile = filedialog.askopenfilename(initialdir = "/",title = "Select a File")
        self.fileLabel.configure(text="File name" + self.vidFile)
        self.procButton['state'] = 'normal'
        self.procButton['command'] = self.processVid

    def processVid(self):
        print("processing")
        process(self.vidFile)  


#get image
os.system('cmd /c "docker pull garnes96/tf-gpu-v1.1.0:latest"')
#launch image
os.system('cmd /c "docker run -d -p 8080:8080 garnes96/tf-gpu-v1.0.13:latest"')
exGUI = CostiaUI()
exGUI.mainloop()
