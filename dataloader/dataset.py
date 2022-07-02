from importlib.resources import path
import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import Dataset
import copy
import os
class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """
    # 注意第一次要预处理数据的  ，preprocess=True
    def __init__(self, path, clip_len=16,input_feature=66):
        self.root_dir = path  # 获取数据集的输入和输出路径
        self.clip_len = clip_len           # 时间数据的长度

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames = [os.path.join(self.root_dir,name) for name in os.listdir(self.root_dir)]
        self.input_feature = input_feature
    def __len__(self):
        return len(self.fnames)

    #需要重写__getitem__方法
    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer,label = self.load_data(self.fnames[index])
        buffer,label = self.crop(buffer,label, self.clip_len)
        return torch.from_numpy(buffer), torch.from_numpy(label)



    def load_data(self, file_dir):
        point_key_data = open(file_dir,'r').read().split('\n')
        length = len(point_key_data)
        buffer = np.empty((length, self.input_feature), np.dtype('float32'))
        label = np.empty((length,), np.dtype('float32'))
        for i, point_key in enumerate(point_key_data[:-1]):
            buffer[i] = point_key.strip().split(',')[1:]
            label[i] = point_key.strip().split(',')[0]
        return buffer,label

    def crop(self, buffer, label,clip_len):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(5,buffer.shape[0] - clip_len)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len]
        label = label[time_index:time_index + clip_len]

        return buffer,np.array(label[-1])





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    path = 'D:\my_Ai_project\hand-gesture-recognition-using-mediapipe-main\data_process\Pajinsen'
    train_data = VideoDataset(path=path, clip_len=8)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)

    for i, sample in enumerate(train_loader) :
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        # if i == 1:
        #     break