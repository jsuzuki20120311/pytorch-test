import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = torch.nn.Linear(2, 8)
    self.fc2 = torch.nn.Linear(8, 8)
    self.fc3 = torch.nn.Linear(8, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    x = self.fc3(x)
    x = self.sigmoid(x)
    return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Net(nn.Module):

#   def __init__(self):
#     super(Net, self).__init__()
#     self.fc1 = nn.Linear(2, 2)
#     self.fc2 = nn.Linear(2, 1)

#   # nn.Moduleから継承された関数
#   # ここに順伝播の処理を書く
#   def forward(self, x):
#     x = self.fc1(x)
#     x = torch.sigmoid(x)
#     x = self.fc2(x)
#     return x
