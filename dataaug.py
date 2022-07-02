# 数据增强

import cv2
import os
import numpy as np
import mediapipe as mp
path = './data/Pajinsen'
savepath = './data/videos'
if not os.path.exists(savepath):
    os.mkdir(savepath)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编码格式
for classify_name in os.listdir(path):
    path_sub = os.path.join(path,classify_name)
    for video_name in os.listdir(path_sub):
        save_path = os.path.join(savepath,classify_name,video_name.split('.')[0]+'_2.avi')
        print(save_path)
        video_path = os.path.join(path_sub,video_name)
        out = cv2.VideoWriter(save_path, fourcc, 20, (960, 640), True)
        cap = cv2.VideoCapture(video_path)
        print('读取视频数据')
        num = 0
        while True:
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.resize(image, (960, 640))
            dist = cv2.flip(image, -1)  # 12420
            out.write(dist)
            print('写入视频')


        cap.release()
        out.release()

0
0