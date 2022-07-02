# 时序模型分类
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.dataset import VideoDataset
from models import RNN
import os
from focal_loss import FocalLoss
import torch.nn.functional as F

# input_size = 66
input_size = 42
num_layers = 2
hidden_size = 256
num_classes = 2
batch_size = 64
num_epochs = 2000
learning_rate = 0.001   # 0.0001
num_workers = 0
clip_len  = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = 'D:\my_Ai_project\hand-gesture-recognition-using-mediapipe-main\data_process\Pajinsen\\train'
train_data = VideoDataset(path=train_path, clip_len=clip_len,input_feature=input_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)


val_path = 'D:\my_Ai_project\hand-gesture-recognition-using-mediapipe-main\data_process\Pajinsen\\val'
val_data = VideoDataset(path=val_path, clip_len=clip_len,input_feature=input_size)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=num_workers)


model = RNN(input_size,hidden_size,num_layers,num_classes,device).to(device)
optimizer = optim.Adam(model.parameters(),lr= learning_rate)
# optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
loss = nn.CrossEntropyLoss()
BCEloss = nn.BCEWithLogitsLoss()
focalloss = FocalLoss()
ep = 1
model.train()
acc_ = 0.5
torch.backends.cudnn.enabled=False
highest_acc = 0
for epoch in tqdm(range(num_epochs), total=num_epochs):
   desc = f'Training model for epoch {ep}/ {num_epochs}'
   print(desc)
   acc =0
   for batch_idx, (data, target) in enumerate(train_loader):
      targets = target.long().to(device)
      scores = model(data.to(device))

      prob = torch.nn.Sigmoid()(scores)
      prob = torch.argmax(prob,dim=1)
      length = scores.size()[0]
      acc += (prob==targets).sum()/length
      mloss = loss(scores, targets) #+ focalloss(scores, F.one_hot(targets.long(),num_classes))
      # mloss = BCEloss(scores,F.one_hot(targets,2).float())
      optimizer.zero_grad()
      mloss.backward()
      optimizer.step()
      # scheduler.step()
   model.eval()
   val_acc = 0
   for val_batch_idx, (val_data, val_target) in enumerate(val_loader):
      val_targets = val_target.long().to(device)
      with torch.no_grad():
         val_scores = model(val_data.to(device))

      val_prob = torch.nn.Sigmoid()(val_scores)
      val_prob = torch.argmax(val_prob, dim=1)
      val_length = val_scores.size()[0]
      val_acc += (val_prob == val_targets).sum() / val_length
      # val_mloss = loss(val_scores, val_target) + focalloss(val_scores, F.one_hot(val_targets.long(),num_classes))
      # val_mloss = BCEloss(val_scores,F.one_hot(val_targets,2).float())
       # val_loss:{val_mloss}
   print(f'epoch: {epoch + 1} step: {batch_idx + 1}/{len(train_loader)} train_loss: {mloss},train_acc:{acc.item() / (batch_idx + 1)},val_acc:{val_acc.item() / (val_batch_idx + 1)}')

   ep += 1
   save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
   save_dir = os.path.join(save_dir_root, 'train_models/hand_models')
   if epoch>1400:
      torch.save({
         'epoch': epoch + 1,
         'state_dict': model.state_dict(),
         'opt_dict': optimizer.state_dict(),
      }, os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth.tar'))

   elif val_acc.item() / (val_batch_idx + 1) > acc_:
      acc_ = val_acc.item() / (val_batch_idx + 1)
      torch.save({
         'epoch': epoch + 1,

         'state_dict': model.state_dict(),
         'opt_dict': optimizer.state_dict(),
      }, os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth.tar'))
      print("Save model at {}\n".format(os.path.join(save_dir, 'epoch-' + str(epoch) + '.pth.tar')))

   highest_acc = max(highest_acc,val_acc.item() / (val_batch_idx + 1))
print ("Highest Accuracy when using these configs is:"+ str(highest_acc))







