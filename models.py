import torch.nn as nn
import torch
class RNN(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers, num_classes,device):
      super(RNN, self).__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)  #, dropout=0.3, bidirectional=True
      self.fc1 = nn.Linear(hidden_size, 128)
      self.fc2 = nn.Linear(128, num_classes)
      self.device = device

   def forward(self, data):
      h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(self.device)
      c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(self.device)
      # print(data.size())
      out, _ = self.lstm(data, (h0, c0))
      out = torch.nn.Dropout(0.3)(out[:,-1,:])    # 0.3
      out = self.fc1(out)
      out = nn.ReLU(inplace=True)(out)
      out = torch.nn.Dropout(0.6)(out)           # 0.6
      out = self.fc2(out)
      return out