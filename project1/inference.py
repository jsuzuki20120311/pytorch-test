import pathlib
import torch
import numpy as np
from models.net import Net


if __name__ == '__main__':
  net = Net()
  # 学習済みモデルの保存・ロード
  path_saved_model = pathlib.Path("./project1/dist/model.pth").resolve()
  net.load_state_dict(torch.load(path_saved_model))

  # 訓練用データ
  x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  # convert numpy array to tensor
  x_tensor = torch.from_numpy(x).float()
  print(x_tensor[0])

  y_out = net(x_tensor[0])
  print(y_out)
