#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import os
import roslib
import time
import csv

IMG_DIR1 = roslib.packages.get_pkg_dir('image_matching') + '/data/00_02/'
IMG_DIR2 = roslib.packages.get_pkg_dir('image_matching') + '/data/694_520_rename/'
IMG_SIZE = (64, 48)

start_time = time.strftime("%Y%m%d_%H:%M:%S")
os.makedirs("/home/y-takahashi/catkin_ws/src/image_matching/result/"+start_time+"/img", exist_ok=True)
os.makedirs("/home/y-takahashi/catkin_ws/src/image_matching/result/"+start_time+"/similarity", exist_ok=True)

bf = cv2.BFMatcher()
detector = cv2.SIFT_create()

files1 = os.listdir(IMG_DIR1)
files2 = os.listdir(IMG_DIR2)

common_files = set(files1) & set(files2)
ret_dict = {}  # ファイルごとのマッチング数を保存する辞書

# 画像上に全てのマッチング結果を表示するためのリスト
all_matches_images = []

# CSVファイルのヘッダ
csv_header = ["File1", "File2", "Distance", "Similarity"]

# CSVファイルへの書き込み
with open("/home/y-takahashi/catkin_ws/src/image_matching/result/"+start_time+"/similarity/similarity.csv", 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

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

            if des1 is not None and des2 is not None:
                matches = bf.knnMatch(des1, des2, k=2)

                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                ret = len(good_matches)
            else:
                ret = 0

            ret_dict[file] = ret  # ファイルごとのマッチング数を保存

            # マッチングした特徴を描画して表示
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            all_matches_images.append(img_matches)

            # マッチングした箇所を新しい画像として保存
            output_path = os.path.join("/home/y-takahashi/catkin_ws/src/image_matching/result/"+start_time+"/img", f"matched_{file}")
            cv2.imwrite(output_path, img_matches)

            # 特徴点間の距離や類似度をCSVファイルに書き込む
            for match in good_matches:
                distance = match.distance
                similarity = 1 / (1 + distance)
                csv_row = [file, file, distance, similarity]
                csv_writer.writerow(csv_row)

        except cv2.error:
            ret_dict[file] = 0  # エラー時はマッチング失敗として0を保存

# マッチング数が大きい順にファイル名とマッチング数を出力
for file in sorted(common_files, key=lambda x: ret_dict[x], reverse=True):
    if file == '.DS_Store':
        continue

    ret = ret_dict[file]
    print(file, ret)
