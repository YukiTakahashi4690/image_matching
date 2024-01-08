import cv2
import os
import roslib

IMG_DIR1 = roslib.packages.get_pkg_dir('image_matching') + '/data/00_02/'
IMG_DIR2 = roslib.packages.get_pkg_dir('image_matching') + '/data/694_520_rename/'
IMG_SIZE = (64, 48)

for file in os.listdir(IMG_DIR1):
    if file == '.DS_Store':
        continue

    img1_path = os.path.join(IMG_DIR1, file)
    img2_path = os.path.join(IMG_DIR2, file)

    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, IMG_SIZE)

        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, IMG_SIZE)

        # ヒストグラム比較
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

        # 類似度の計算
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        print(file, similarity)
    except cv2.error:
        print(file, "Error")
