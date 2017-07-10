import os
import shutil

kitti_dir = "/ext/Data/Kitti"

subdirs = ['City', 'Person', 'Residential', 'Road']

for subdir in subdirs:
    dir = os.path.join(kitti_dir, subdir)
    print("--------------")
    for parent,dirnames,filenames in os.walk(dir):
        print("+++")
        for dirname in dirnames:
            print("parent is: %s"%dirname)
            print("dirname is: %s" % dirname)
