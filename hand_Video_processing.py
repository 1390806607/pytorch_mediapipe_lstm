#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import numpy as np
import mediapipe as mp



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=640)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.6)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args




def main():
    # 引数解析 #################################################################
    # 引数解析 #################################################################
    args = get_args()
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    print(use_static_image_mode,min_tracking_confidence,min_detection_confidence)

    # mediapipe  调用解决手的包
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,)
    path = 'D:\\my_Ai_project\\hand-gesture-recognition-using-mediapipe-main\\data\\Pajinsen'
    sava_path = './data_process/Pajinsen'
    classify = {}
    # all_landmarks = []   # 所有视频的关键点
    for index,name in  enumerate(os.listdir(path)):
        classify[index] = name
        classifies_dir = os.path.join(path,name)
        train,val = train_test_split(os.listdir(classifies_dir),train_size=0.9,shuffle=True)
        print('训练集视频数据开始预处理')
        for train_i in train:
            video_path = os.path.join(classifies_dir,train_i)
            savepath = os.path.join(sava_path,f"train/{train_i.split('.')[0]}.txt")
            video_preprocessing(path=video_path, cap_width=cap_width, cap_height=cap_height, hands=hands, index=index, savepath=savepath)
        print('验证集视频数据开始预处理')
        for val_i in val:
            video_path = os.path.join(classifies_dir, val_i)
            savepath = os.path.join(sava_path, f"val/{val_i.split('.')[0]}.txt")
            video_preprocessing(path=video_path, cap_width=cap_width, cap_height=cap_height, hands=hands, index=index,
                                savepath=savepath)
    with open('./dataloader/pajinsen_labels.txt','a') as f:
        length = len(classify)
        for i in range(length):
            text = str(i) +' '+ classify[i] +'\n'
            f.write(text)
        f.close()
    # shuff = []
    # for i in range(len(all_landmarks)):
    #     shuff.append(i)
    # np.random.shuffle(shuff)
    # print(shuff)
    # save_path_train = 'D:\\my_Ai_project\\hand-gesture-recognition-using-mediapipe-main\\data_process\\Pajinsen\\train'
    # for i in range(int(len(all_landmarks)*0.8)):
    #     filename_path = os.path.join(save_path_train, 'video' +str(i)+ '.txt')
    #     for j in range(len(all_landmarks[shuff[i]])):
    #         logging_csv(all_landmarks[shuff[i]][j], filename_path)
    #
    # save_path_train = 'D:\\my_Ai_project\\hand-gesture-recognition-using-mediapipe-main\\data_process\\Pajinsen\\val'
    # for i in range(int(len(all_landmarks) * 0.8+1),len(all_landmarks)):
    #     filename_path = os.path.join(save_path_train, 'video' +str(i)+  '.txt')
    #     for j in range(len(all_landmarks[shuff[i]])):
    #         logging_csv(all_landmarks[shuff[i]][j], filename_path)


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return w,h


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def logging_csv(landmark_list, filename):
    with open(filename, 'a') as f:
        f.write(str(landmark_list).replace('[','').replace(']','')+'\n')
        f.close()

def video_preprocessing(path, cap_width, cap_height,hands,index,savepath):
    every_n_frames = 1
    cap = cv.VideoCapture(path)
    num = 0
    print('一个视频的手关键点坐标开始记录')
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if (num % every_n_frames == 0):
            image = cv.resize(image, (cap_width, cap_height))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)  # 检测图片中是否有手
            image.flags.writeable = True
            #  results.multi_hand_landmarks 每只手的关键点的信息，坐标，关键点的id
            if results.multi_hand_landmarks is not None:
                # 针对一只手进行操作
                for hand_landmarks in results.multi_hand_landmarks:
                    # 学习数据存储
                    base_x, base_y = 0, 0
                    landmark_point = []   # 记录每个图片的关键点坐标
                    landmark_point.append(index)
                    w,h = calc_bounding_rect(image,hand_landmarks)
                    for point_index, landmark in enumerate(hand_landmarks.landmark):
                        # landmark_z = landmark.z
                        print(point_index)
                        if point_index == 0:
                            base_x, base_y = (landmark.x*cap_width)/w, (landmark.y*cap_height)/h
                        landmark_point.append((landmark.x*cap_width)/w - base_x)
                        landmark_point.append((landmark.y*cap_height)/h - base_y)
                    print('一张图片的手坐标记录结束')
                    logging_csv(landmark_point, filename=savepath)



        num += 1
    print('一个视频的手关键点坐标记录结束')
    cap.release()


if __name__ == '__main__':
    main()
