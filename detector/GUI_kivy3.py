# Ellen O'Brien
# 4.12.2023
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import csv
import numpy as np
import cv2 as cv
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
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
import kivy
from kivy.app import App
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.screenmanager import ScreenManager, Screen
import kivy.uix.progressbar
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.lang import Builder
import threading


def norftrack(res, class_list):
    # res: the list of dicts returned by the model.detect function
    # class_list: the list of class ids for each detection

    norftection: List[Detection] = []
    for x in range(len(res['rois'])):
        yc = (res['rois'][x][0] + res['rois'][x][2]) / 2
        xc = (res['rois'][x][1] + res['rois'][x][3]) / 2

        # the class of the particular detection
        det_class = res['class_ids'][x]  # int class ID
        print(det_class)
        class_list.append(det_class)

        centroid = np.array([xc, yc])
        scores = np.array([res['scores'][x]])
        # label=np.array(class_list[res['class_ids'][x]])
        norftection.append(Detection(points=centroid, scores=scores))
        # norftection.append(Detection(points=centroid, scores=scores, label=label))

    return norftection


def norftrack2(res, labels):
    norftection: List[Detection] = []
    for x in range(len(res['rois'])):
        yc = (res['rois'][x][0] + res['rois'][x][2]) / 2
        xc = (res['rois'][x][1] + res['rois'][x][3]) / 2

        centroid = np.array([xc, yc])
        scores = np.array([res['scores'][x]])
        label = np.array(labels[res['class_ids'][x]])
        norftection.append(Detection(points=centroid, scores=scores, label=label))

    return norftection


def euc_distance(detection, tracked_obj):
    # print(tracked_obj.estimate)
    return np.linalg.norm(detection.points - tracked_obj.estimate)


def key_distance(detection, tracked_obj):
    box1 = np.concatenate([detection.points[0], detection.points[1]])
    box2 = np.concatenate([tracked_obj.estimate[0], tracked_obj.estimate[1]])

    # print(box1)
    # print(box2)
    ya = max(box1[0], box2[0])
    xa = max(box1[1], box2[1])
    yb = max(box1[2], box2[2])
    xb = max(box1[3], box2[3])

    area = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    aarea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    barea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = area / float(aarea + barea - area)

    return 1 / iou if iou else (10000)


def center(pnt):
    return [np.mean(np.array(pnt), axis=0)]


# kv
kivy.require('2.0.0')
Builder.load_string("""  
<HomeScreen>:
    BoxLayout:
        orientation: 'vertical'
        Image:
            source: 'mylogo.png'
            size: self.texture_size
        Label:
            text: 'Costia'
            font_size: 70
        Label:
            text: 'Environmentally Focused Trash Detection'
        Button:
            text: 'Automated trash detection'
            on_press: root.manager.current = 'detect'

<DetectScreen>:
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            Button:
                text: 'TACO'
                on_release: root.model_choose("./models/logs/mask_rcnn_taco_0100.h5")
            Button:
                text: 'TACO+'
                on_release: root.model_choose("./models/logs/taco_10_0/mask_rcnn_taco_0074.h5")
        Button:
            text: 'Choose file'
            on_release: root.choose_file()

<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: 'vertical'
        FileChooserListView:
            id: filechooser
        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: 'Cancel'
                on_release: root.cancel()
            Button:
                text: 'Load'
                on_press: root.load(filechooser.path, filechooser.selection)

<ProgressScreen>:
    on_enter: root.process()
    BoxLayout:
        orientation: 'horizontal'
        BoxLayout:
            orientation: 'vertical'
            Label:
                id: file_label
            ProgressBar:
                id: pb
            Label:
                id: prog_label
                text: ''
        Label:
            id: count_label
            text: 'Trash Count'

<ResultScreen>:
    on_enter: root.get_scores()
    BoxLayout:
        VideoPlayer:
            source: './saves/test.mp4'
        Label:
            id: scores
""")


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class HomeScreen(Screen):
    # license information and credit
    pass


class DetectScreen(Screen):
    # dropdown menu: choose model
    # file explorer button
    # "Start processing" button
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ""
    model_choice = "./models/logs/mask_rcnn_taco_0100.h5"

    def dismiss_popup(self):
        self._popup.dismiss()

    def load(self, path, filename):
        self.text_input = os.path.join(path, filename[0])
        self.dismiss_popup()
        if self.text_input is not None:
            self.manager.add_widget(ProgressScreen(self.text_input, self.model_choice, name='progress'))
            self.manager.current = 'progress'

    def choose_file(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def model_choose(self,mpath):
        self.model_choice = mpath


class ProgressScreen(Screen):
    # while processing:
    #  track progress, "x of n frames processed"
    #  loading bar?
    #  costia logo spinny :)
    # change to ResultScreen
    frameTotal = 0
    frame_count = 0
    vid_array = []

    def __init__(self, txt_in, mpath, **kwargs):
        super().__init__(**kwargs)
        self.in_path = txt_in
        self.model_path = mpath
        self.ids.file_label.text = self.in_path


    def process(self):
        t1 = threading.Thread(target=self.vid_process)
        t1.start()

    def loadVid(self, inputVid):
        # getting the video
        vid2 = cv.VideoCapture(inputVid)
        # checks whether frames were extracted
        success = 1
        count = 0
        while success:
            # read the frame and add it to the frame list
            success, image = vid2.read()
            self.vid_array.append(image)
            count += 1
            print(count)
        # color correction
        count = 0
        for img in self.vid_array:
            if count < (len(self.vid_array) - 1):
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                self.vid_array[count] = img
            count += 1
        self.frameTotal = len(self.vid_array) - 1  # number of frames in video

    def vid_process(self):
        self.loadVid(self.in_path)
        # Root directory of the project
        ROOT_DIR = os.path.abspath(".")
        print(ROOT_DIR)
        MODEL_DIR = os.path.join(ROOT_DIR, "models\logs")
        print(MODEL_DIR)
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, self.model_path)  #"models\mask_rcnn_taco0100.h5")
        print(COCO_MODEL_PATH)

        import csv
        import dataset
        # Load class map - these tables map the original TACO classes to your desired class system
        # and allow you to discard classes that you don't want to include.
        class_map = {}
        with open("./taco_config/map_10.csv") as csvfile:
            reader = csv.reader(csvfile)
            class_map = {row[0]: row[1] for row in reader}

        # Load full dataset or a subset
        TACO_DIR = "../data"
        round = None  # Split number: If None, loads full dataset else if int > 0 selects split no
        subset = "test"  # Used only when round !=None, Options: ('train','val','test') to select respective subset
        dataset = dataset.Taco()
        taco = dataset.load_taco(TACO_DIR, round, subset, class_map=class_map, return_taco=True)

        # Must call before using the dataset
        dataset.prepare()

        print("Class Count: {}".format(dataset.num_classes))
        for i, info in enumerate(dataset.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        # Mask RCNN Configuration
        class TacoTestConfig(Config):
            NAME = "taco"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 3
            NUM_CLASSES = dataset.num_classes

        config = TacoTestConfig()

        self.ids.file_label.text += "\nLoading model: " + self.model_path
        model = modellib.MaskRCNN(mode="inference", model_dir=TACO_DIR, config=config)
        print(MODEL_DIR)

        model.load_weights(weights_in_path=self.model_path, weights_out_path=self.model_path, by_name=True)

        # norfair tracker
        tracker = norfair.Tracker(distance_function=euc_distance, distance_threshold=30)
        spectracker = norfair.Tracker(distance_function=euc_distance, distance_threshold=30)

        path = Paths(center, attenuation=0.01)

        # each class is represented by a set that will contain detection ids without duplication
        totalitems = set()
        bottles = set()
        botcap = set()
        cans = set()
        cigarettes = set()
        cups = set()
        lids = set()
        others = set()
        wrapperbag = set()
        poptabs = set()
        straws = set()
        # list of sets can be indexed by class_id
        setlist = [totalitems, bottles, botcap, cans, cigarettes, cups, lids, others, wrapperbag, poptabs, straws]

        # go through and detect all frames in the list
        frame_counter = 1
        # line=400
        obj_class = []
        specscores = []

        # set up for video export
        frameSize = (1600, 1600)
        # frameSize = (1280,720)
        # make a saves folder in the detector folder
        out = cv.VideoWriter('./saves/test.mp4', cv.VideoWriter_fourcc(*"mp4v"), 10, frameSize)

        n = self.frameTotal - 1
        self.ids.pb.max = n
        import time
        start = time.time()
        for vid_img in self.vid_array:
            if frame_counter < self.frameTotal:  # ignore the last image, which is blank
                p = frame_counter - 1
                self.ids.pb.value = p
                self.ids.prog_label.text = 'Processing frame {} of {}'.format(p, n)
                orimage = img_to_array(vid_img)
                results = model.detect([orimage], verbose=1)
                r = results[0]
                print("Frame " + str(frame_counter))

                # convert to norfair detection and add to tracked items
                det = norftrack(r, obj_class)
                specdet = norftrack2(r, dataset.class_names)

                print(det)

                tracked = tracker.update(detections=det)  # list of tracked items
                spectracked = spectracker.update(detections=specdet)
                print(tracked)
                print(spectracked)

                norfair.draw_points(orimage, det)
                norfair.draw_tracked_objects(orimage, tracked, color=(0, 225, 0), id_size=3, id_thickness=2,
                                             draw_labels=True)
                objdet = False
                if (frame_counter > 7):
                    x = 0
                    while x < len(tracked):
                        print(tracked[0].age)
                        item = tracked[x]  # specific tracked item
                        print(item)
                        item_class = obj_class[x]  # item's corresponding class id
                        print(obj_class)
                        # if ((line+15)>item.estimate[0][1]>(line-15)): #if an object is within range of the line
                        setlist[0].add(item.id)  # add the item to the list of counted items
                        # print("item", item.id, "detected",dataset.class_names[item_class])
                        print("item", item.id, "detected")
                        # setlist[item_class].add(item.id) #add the item id to its class's corresponding set
                        x += 1
                        # print(item.last_detection.points[0][0])

                        if (item.id == 2):
                            objdet = True

                    x = 0
                    for item in spectracked:
                        # for x in tracked:

                        if (len(specscores) < item.id):
                            specscores.append([])
                        if item.label == "Bottle":
                            bottles.add(item.id)
                        elif item.label == "Bottle cap":
                            botcap.add(item.id)
                        elif item.label == "Can":
                            cans.add(item.id)
                        elif item.label == "Cigarette":
                            cigarettes.add(item.id)
                        elif item.label == "Cup":
                            cups.add(item.id)
                        elif item.label == "Lid":
                            lids.add(item.id)
                        elif item.label == "Other":
                            others.add(item.id)
                        elif item.label == "Plastic bag + wrapper":
                            wrapperbag.add(item.id)
                        elif item.label == "Pop tab":
                            poptabs.add(item.id)
                        elif item.label == "Straw":
                            straws.add(item.id)
                        print("item", item.id, "detected", item.label)
                        print("item", item.id, "detected", item.last_detection.points)
                        print(item.id)
                        print(item.last_detection.scores[0])
                        print(specscores)
                        # specscores[(item.id)-1].append(item.last_detection.scores[0])
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
                frame = path.draw(vid_img, tracked)
                # orimage=cv.line(orimage, (0,line), (2000, line), (255, 0, 0), 6)
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                out.write(frame)
                # print(r['scores'])
                # visualize.display_instances(orimage, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
                # debugging tool
                save_dir = ("./saves/" + str(frame_counter).rjust(5, '0') + ".png")
                visualize.display_instances(orimage, save_dir, r['rois'], r['masks'], r['class_ids'],
                                            dataset.class_names, r['scores'])
                #time metrics
                stop = time.time()
                totalTime = stop - start
                avgTime = totalTime / frame_counter
                #update the onscreen tracker
                count_str = "Trash Count \n" + "\nbottles " + str(len(bottles)) \
                            + "\nbottle caps " + str(len(botcap)) \
                            + "\ncans " + str(len(cans)) \
                            + "\ncigarettes " + str(len(cigarettes)) \
                            + "\ncups " + str(len(cups)) \
                            + "\nlids " + str(len(lids)) \
                            + "\nother " + str(len(others)) \
                            + "\nwrapper + bag " + str(len(wrapperbag)) \
                            + "\npop tabs " + str(len(poptabs)) \
                            + "\nstraws " + str(len(straws)) \
                            + "\ntotal " + str(len(totalitems)) \
                            + "\nTime elapsed: " + str(totalTime) + " sec" \
                            + "\nAverage time per frame: " + str(avgTime) + " sec"
                self.ids.count_label.text = count_str
                frame_counter += 1
        for file in os.listdir("./saves"):
            frame = cv.imread("./saves/" + file)
            out.write(frame)

        out.release()
        outfile = open("results.txt", "w")
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
        outfile.write(str(len(others)))
        outfile.write("\nwrapper + bag ")
        outfile.write(str(len(wrapperbag)))
        outfile.write("\npop tabs ")
        outfile.write(str(len(poptabs)))
        outfile.write("\nstraws ")
        outfile.write(str(len(straws)))
        outfile.write("\ntotal ")
        outfile.write(str(len(totalitems)))
        outfile.write("\nTime elapsed: " + str(totalTime) +" sec")
        outfile.write("\nAverage time per frame: " + str(avgTime) +" sec")
        for x in range(len(specscores)):
            outfile.write("\nScore of item ")
            outfile.write(str(x))
            outfile.write(str(specscores[x]))
        self.manager.current = 'result'




class ResultScreen(Screen):
    # playable results video
    # video download
    # text summary
    # text summary download
    def get_scores(self):
        with open("results.txt", "r") as rf:
            result_txt = rf.read()
        self.ids.scores.text = result_txt


class GUI_App(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(DetectScreen(name='detect'))
        sm.add_widget(ResultScreen(name='result'))
        return sm


Factory.register('DetectScreen', cls=DetectScreen)
Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    GUI_App().run()
