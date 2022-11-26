#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""hist matching."""

import cv2
import os
import rospy
import roslib
# import imgsim

TARGET_FILE = roslib.packages.get_pkg_dir('image_matching') + '/data/real_image/center1760.png'
# IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/'
# REAL_IMG_DIR = roslib.packages.get_pkg_dir('image_matching') + '/data/real_image'
# SIM_IMG_DIR = roslib.packages.get_pkg_dir('image_matching') + '/data/sim_image'
IMG_DIR = roslib.packages.get_pkg_dir('image_matching') + '/data/sim_image/'
IMG_SIZE = (64, 48)

# target_img_path = IMG_DIR + TARGET_FILE
target_img_path = TARGET_FILE
target_img = cv2.imread(target_img_path)
# vtr = imgsim.Vectorizer()
# vec0 = vtr.vectorize(target_img)

# target_img = cv2.resize(target_img, IMG_SIZE)
target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])

print('TARGET_FILE: %s' % (TARGET_FILE))

files = os.listdir(IMG_DIR)
for file in files:
    if file == '.DS_Store' or file == TARGET_FILE:
        continue

    comparing_img_path = IMG_DIR + file
    comparing_img = cv2.imread(comparing_img_path)
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    # vec1 = vtr.vectorize(comparing_img)

    # dist = imgsim.distance(vec0, vec1)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

    ret = cv2.compareHist(target_hist, comparing_hist, 0)
    print(file, ret)