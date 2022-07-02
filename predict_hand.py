import os

import torch
import numpy as np
from models import RNN
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
torch.backends.cudnn.benchmark = True

input_size = 42
num_layers = 2
hidden_size = 256
num_classes = 2


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return w,h

def hands_coordinate_recording(image,multi_hand_landmarks):
    landmark_point = []
    image_w,image_y = image.shape[1],image.shape[0]
    for hand_landmarks in multi_hand_landmarks:
        w,h = calc_bounding_rect(image,hand_landmarks)
        for point_index, landmark in enumerate(hand_landmarks.landmark):
            if point_index == 0:
                base_x, base_y = (landmark.x*image_w)/w, (landmark.y*image_y)/h
            landmark_point.append((landmark.x*image_w)/w - base_x)
            landmark_point.append((landmark.y*image_y)/h - base_y)
    return landmark_point

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloader/pajinsen_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = RNN(input_size,hidden_size,num_layers,num_classes,device).to(device)
    # model = R3D_model.R3DClassifier(num_classes=2, layer_sizes=(2, 2, 2, 2))
    checkpoint = torch.load('./train_models/train4_models/epoch-770.pth.tar',
                            map_location=lambda storage, loc: storage)
    # checkpoint = torch.load('./train_models/hand_models/epoch-1999.pth.tar',
    #                         map_location=lambda storage, loc: storage)


    """
    state_dict = model.state_dict()
    for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
        state_dict[k1] = checkpoint[k2]
    model.load_state_dict(state_dict)
    """
    model.load_state_dict(checkpoint['state_dict'])  # 模型参数
    # optimizer.load_state_dict(checkpoint['opt_dict'])#优化参数

    model.to(device)
    model.eval()

    # read video
    # video = "./data/Pajinsen/normal/WIN_20220617_14_23_17_Pro.mp4"
    # video = "./data/Pajinsen/tremor/WIN_20220617_13_53_23_Pro.mp4"
    # videos = './test/TremorCodeTest/test1' #normal,normal
    # videos = './test/TremorCodeTest/test2' #tremor,tremor
    # videos = './test/TremorCodeTest/test3' #tremor,tremor
    videos = './test/TremorCodeTest/test4' #tremor,normal
    # video = './test/TremorCodeTest/Tremor/Tremor1.mov'
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,)
    result = []
    for video in os.listdir(videos):
        cap = cv2.VideoCapture(os.path.join(videos,video))
        retaining = True
        clip = []     # 记录视频的时序长度
        video_classify = {}   # 记录每个视频的每个类别出现的此时
        for i in range(len(class_names)):
            video_classify[class_names[i].split(' ')[-1].strip()] = 0
        while retaining:
            retaining, frame = cap.read()
            if not retaining and frame is None:
                continue
            image = cv2.resize(frame, (960, 540))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)  # 检测图片中是否有手
            image.flags.writeable = True
            cap_height,cap_width = image.shape[0],image.shape[1]
            #  results.multi_hand_landmarks 是否检测得到手，里面包括了关键的坐标和id
            if results.multi_hand_landmarks is not None:
                # 针对一只手进行操作
                print('hands detected，Please continue to')
                landmark_point = hands_coordinate_recording(image,results.multi_hand_landmarks)
                clip.append(landmark_point)
                if len(clip) == 32:
                    inputs = np.array(clip).astype(np.float32)
                    inputs = np.expand_dims(inputs, axis=0)
                    inputs = torch.from_numpy(inputs)
                    inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
                    with torch.no_grad():
                        outputs = model.forward(inputs)

                    # probs = torch.nn.Softmax(dim=1)(outputs)
                    probs = torch.nn.Sigmoid()(outputs)
                    label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                    print(probs[0][label])
                    video_classify[class_names[label].split(' ')[-1].strip()] += 1
                    cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    clip.pop(0)


            cv2.imshow('result', frame)
            cv2.waitKey(30)
        result.append(max(video_classify, key=lambda k: video_classify[k]))
    print(result)
    if len(set(result)) == 1 and list(set(result))[0] == 'tremor':
        print('2期：身体双边手部颤抖')
    elif len(set(result)) == 1 and list(set(result))[0] == 'normal':
        print('0期：无症状')
    elif len(set(result)) == 2:
        print('1期：身体单边手部颤抖')
    cap.release()
    cv2.destroyAllWindows()



def model_predict(input):
    pass

if __name__ == '__main__':
    main()









