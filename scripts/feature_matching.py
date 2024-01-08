#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import os
import roslib

IMG_DIR1 = roslib.packages.get_pkg_dir('image_matching') + '/data/00_02/'
IMG_DIR2 = roslib.packages.get_pkg_dir('image_matching') + '/data/694_520_rename/'
IMG_SIZE = (64, 48)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING)
flann = cv2.FlannBasedMatcher()
# detector = cv2.ORB_create()
# detector = cv2.AKAZE_create(threshold=0.001)
detector = cv2.AKAZE_create()

files1 = os.listdir(IMG_DIR1)
files2 = os.listdir(IMG_DIR2)

common_files = set(files1) & set(files2)

for file in common_files:
    if file == '.DS_Store':
        continue

    img1_path = os.path.join(IMG_DIR1, file)
    img2_path = os.path.join(IMG_DIR2, file)

    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, IMG_SIZE)
        (kp1, des1) = detector.detectAndCompute(img1, None)

        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, IMG_SIZE)
        (kp2, des2) = detector.detectAndCompute(img2, None)

        # matches = bf.match(des1, des2)
        matches = flann.knnMatch(des1, des2, k=2)

        if len(matches) > 0:
            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)
        else:
            ret = 100000  # マッチング失敗時のエラー値
    except cv2.error:
        ret = 100000

    print(file, ret)
