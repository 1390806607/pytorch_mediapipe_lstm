import mediapipe as mp
import cv2
import argparse
import cv2
import os
import numpy as np


################################# 视频增强操作test
path = './1.jpg'
image = cv2.imread(path)
print(image.shape)
image = cv2.resize(image,(500,500))
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)   # 图像的中心点
M = cv2.getRotationMatrix2D((cX, cY), -45, 1.0)   # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75

cos = np.abs(M[0,0])
sin = np.abs(M[0,1])

newW = int((h*sin)+(w*sin))
newH = int((h*cos)+(w*sin))

M[0,2] += (newW/2) - cX
M[1,2] += (newH/2) - cY
rotation_image = cv2.warpAffine(image, M, (newW, newW))
# print(top_bottom_image.shape)
# print(left_rigth_image.shape)
# print(diagonal_image.shape)
print(rotation_image.shape)
# cv2.imshow('1',top_bottom_image)
# cv2.imshow('2',left_rigth_image)
# cv2.imshow('3',diagonal_image)
cv2.imshow('4',rotation_image)
cv2.waitKey(0)
cv2.destroyAllWindows()










# ############################################# 把视频图片增强然后保存
# path = './data/normal30.mov'
# save_path = './1.avi'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')   # 视频编码格式
# out = cv2.VideoWriter(save_path, fourcc, 20, (960, 640),True)
#
# cap = cv2.VideoCapture(path)
# print('读取视频数据')
# while True:
#     ret, image = cap.read()
#     if not ret:
#         break
#     image = cv2.resize(image, (960, 640))
#     # dist = cv2.flip(image,1)
#     dist = cv2.rotate(image,)
#     out.write(dist)
#     print('写入视频')
# cap.release()
# out.release()



# # #################### 把图片数据合成视频
# path = './data/image/normal1'
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# # for name in os.listdir(path):
# #     savepath = os.path.join(path,name + '_1.mp4')
# #     print(savepath)
# #     path_sub = os.path.join(path,name)
#
# videoWrite = cv2.VideoWriter('2.mp4', -1, 100, (960,640))
# for i in os.listdir(path):
#     pic_path = os.path.join(path,i)
#     image = cv2.imread(pic_path)
#     videoWrite.write(image)
# print('end')



# path = '2.mp4'
# cap = cv2.VideoCapture(path)
# while True:
#     ret, image = cap.read()
#     if not ret:
#         break
#
#     cv2.imshow('1',image)
# cap.release()














####################################################  整体关键点
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic

# holistic = mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)
#
# cap = cv2.VideoCapture(path)
# retaining = True
# while retaining:
#     retaining, frame = cap.read()
#     if not retaining and frame is None:
#         continue
#     image = cv2.resize(frame, (960, 540))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = holistic.process(image)  # 检测图片中是否有手
#     image.flags.writeable = True
#     if results.pose_landmarks:
#         print(len(results.pose_landmarks.landmark))
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles
#         .get_default_pose_landmarks_style())
#     cv2.imshow('1',image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()


