from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from models.net import Net


if __name__ == '__main__':
  # アヤメのデータセットを読み込み
  iris = datasets.load_iris()

  # 正解ラベルをone-hot-vector表現のベクトルに変換
  y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
  y[np.arange(len(iris.target)), iris.target] = 1

  # 教師データの25%をテストデータとし、予測精度の評価のためにとっておく
  X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.25)

  x = torch.arange.Variable(torch.from_numpy(X_train).float(), requires_grad=True)
  y = torch.arange.Variable(torch.from_numpy(y_train).float())

  net = Net()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
  # 平均二乗誤差
  criterion = torch.nn.MSELoss()

  for i in range(3000):
    optimizer.zero_grad()
    output = net(x)

    loss = criterion(output, y)
    loss.backward()

    optimizer.step()



