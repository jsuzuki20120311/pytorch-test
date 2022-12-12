import pathlib
import torch
import numpy as np

from models.net import Net
# from utils.fix_seed import fix_seed


if __name__ == '__main__':
  # seed = 42
  # fix_seed(seed)

  print(torch.__version__)
  print(torch.cuda.is_available())
  print(torch.cuda.device_count())

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # 訓練用データ
  x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([[0], [1], [1], [0]])

  # convert numpy array to tensor
  x_tensor = torch.from_numpy(x).float()
  y_tensor = torch.from_numpy(y).float()

  # エポック数
  # 一つの訓練データを何回繰り返して学習させるか
  num_epochs = 10000

  # 学習率
  learning_rate = 0.01

  net = Net()

  optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
  criterion = torch.nn.MSELoss()

  # トレーニング
  # start to train
  print("----- トレーニング -----")
  net.train()
  epoch_loss = []
  for epoch in range(num_epochs):
    print(epoch)
    # forward
    outputs = net(x_tensor)

    # calculate loss
    loss = criterion(outputs, y_tensor)

    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save loss of this epoch
    epoch_loss.append(loss.data.numpy().tolist())

  print(net(torch.from_numpy(np.array([[0, 0]])).float()))
  print(net(torch.from_numpy(np.array([[1, 0]])).float()))
  print(net(torch.from_numpy(np.array([[0, 1]])).float()))
  print(net(torch.from_numpy(np.array([[1, 1]])).float()))


  # 評価
  print("----- 評価 -----")
  net.eval()
  for x_data in x_tensor:
    x_in = torch.Tensor(x_data)
    y_out = net(x_in)
    print("%d %d = %f" % (x_data[0], x_data[1], y_out))


  # 学習済みモデルの保存
  path_saved_model = pathlib.Path("./project1/dist/model.pth").resolve()
  torch.save(net.state_dict(), path_saved_model)
