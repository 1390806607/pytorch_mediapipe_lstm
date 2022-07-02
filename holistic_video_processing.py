import mediapipe as mp
import cv2
import argparse
import os
import copy
import numpy as np
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    cap_width = args.width
    cap_height = args.height
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    print(min_tracking_confidence,min_detection_confidence)
    use_brect = True

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
    path = './data/Pajinsen1'
    classify = {}
    every_n_frames = 2
    all_landmarks = []
    for index,name in enumerate(os.listdir(path)):
        classify[index] = name
        classifies_dir = os.path.join(path,name)
        for i in os.listdir(classifies_dir):
            video_path = os.path.join(classifies_dir,i)
            cap = cv2.VideoCapture(video_path)
            num = 0
            all_video_landmarks = []    # 每个视频的所有关键点
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                if (num%every_n_frames== 0):
                    image = cv2.resize(image, (cap_width, cap_height))
                    debug_image = copy.deepcopy(image)

                    # 検出実施 #############################################################
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    image.flags.writeable = False

                    results = holistic.process(image)  # 检测
                    image.flags.writeable = True

                    if results.pose_landmarks:
                        base_x, base_y = 0, 0
                        landmark_point = []
                        landmark_point.append(index)
                        print(len(results.pose_landmarks.landmark))
                        if len(results.pose_landmarks.landmark) ==33:
                            for point_index,pose_landmark in enumerate(results.pose_landmarks.landmark):
                                if point_index == 0:
                                    base_x, base_y = pose_landmark.x, pose_landmark.y
                                landmark_point.append(pose_landmark.x - base_x)
                                landmark_point.append(pose_landmark.y - base_y)
                            all_video_landmarks.append(landmark_point)
                num +=1
            if len(all_video_landmarks)>32:
                all_landmarks.append(all_video_landmarks)
            cap.release()
            cv2.destroyAllWindows()
    with open('./dataloader/pajinsen_label1s.txt','a') as f:
        length = len(classify)
        for i in range(length):
            text = str(i) +' '+ classify[i] +'\n'
            f.write(text)
        f.close()
    shuff = []
    for i in range(len(all_landmarks)):
        shuff.append(i)
    np.random.shuffle(shuff)
    save_path_train = 'D:\\my_Ai_project\\hand-gesture-recognition-using-mediapipe-main\\data_process\\Pajinsen1\\train'
    for i in range(int(len(all_landmarks)*0.8)):
        filename_path = os.path.join(save_path_train, 'video' +str(i)+ '.txt')
        for j in range(len(all_landmarks[shuff[i]])):
            logging_csv(all_landmarks[shuff[i]][j], filename_path)

    save_path_train = 'D:\\my_Ai_project\\hand-gesture-recognition-using-mediapipe-main\\data_process\\Pajinsen1\\val'
    for i in range(int(len(all_landmarks) * 0.8+1),len(all_landmarks)):
        filename_path = os.path.join(save_path_train, 'video' +str(i)+  '.txt')
        for j in range(len(all_landmarks[shuff[i]])):
            logging_csv(all_landmarks[shuff[i]][j], filename_path)



def logging_csv(landmark_list,filename):
    with open(filename, 'a') as f:
        f.write(str(landmark_list).replace('[','').replace(']','')+'\n')
        f.close()


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

if __name__ == '__main__':
    main()

