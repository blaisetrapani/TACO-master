import os, random, shutil

ROOT_DIR = os.path.abspath(".")
#^ the path, wherever you're running this file from
#for me, this is my ENGR441 folder
origin = './vid/' #ENGR441 subfolder with the video images
newdir = './selection/' #ENGR441 subfolder i'm saving the selected images in
file_list = os.listdir(origin) #get a list of all the video image file names
new_list = []
counter = 0
while counter < 20:
	new_file = random.choice(file_list) #pick a random file name
	file_list.remove(new_file) #take it off the original list
	new_list.append(new_file) #add the file name to the new list
	counter += 1
for img_file in new_list:
        #copy the files in new_list over to newdir
	shutil.copy(origin + img_file, newdir + img_file)
	os.remove(origin + img_file)
