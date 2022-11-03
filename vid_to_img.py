#Oct 27 2022

#This program will convert a video to a collection of jpgs
#More specifically, it will save every [Xth] frame as a jpg
# cropped to [dimension] in a specified folder

import cv2 as cv #you'll need opencv-python installed to run this
import os

ROOT_DIR = os.path.abspath(".")
#^ the path, wherever you're running this file from
#for me, this is a subfolder of my ENGR441 folder

# make a list to store all the frames
vid_array = []
# getting the video
vid = cv.VideoCapture('./sample.mp4') #add directory of video file
# checks whether frames were extracted
success = 1
while success:
    # read the frame and add it to the frame list
    success, image = vid.read()
    vid_array.append(image)

#saving as jpg
count=0
for img in vid_array:
    if count<(len(vid_array)-1):
        #cropping **ADD CROPPING LATER**
        #img = image cropped to [dimension]
        #**ADD OPTION TO CHANGE FRAME RATE LATER**
        if (count % 5) == 0: #to test, we're doing every 5 frames
            fname = str(count) + '.jpg'
            #save frame to folder
            cv.imwrite(fname,img)
    count+=1

