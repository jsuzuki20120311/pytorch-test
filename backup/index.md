

```python
  # absolute_path = pathlib.Path().resolve()
  # script_path = pathlib.Path(__file__).resolve()

  # print(absolute_path)
  # print(script_path)
  # print(__file__ + '/dist/')


  # myModel.hello()

  # a = torch.zeros([2, 4], dtype=torch.int32)
  # b = torch.ones([2, 3, 3])
  # print(a)
  # print(b)

  # tensor = torch.tensor([1, 2, 3, 4, 5])
  # print(tensor)


  # dict_size = 10
  # depth = 3
  # hidden_size = 6

  # # モデル定義
  # embedding = torch.nn.Embedding(dict_size, depth)
  # lstm = torch.nn.LSTM(input_size=depth, hidden_size=hidden_size, batch_first=True)

  # linear = torch.nn.Linear(hidden_size, dict_size)
  # criterion = torch.nn.CrossEntropyLoss()

  # params = chain.from_iterable([
  #     embedding.parameters(),
  #     lstm.parameters(),
  #     linear.parameters(),
  #     criterion.parameters()
  # ])
  # optimizer = torch.optim.SGD(params, lr=0.01)

  # # 訓練用データ
  # x = [[1,2, 3, 4]]
  # y = [5]

  # # 学習
  # for i in range(100):
  #     tensor_y = torch.tensor(y)
  #     input_ = torch.tensor(x)
  #     tensor = embedding(input_)
  #     output, (tensor, c_n) = lstm(tensor)
  #     tensor = tensor[0]
  #     tensor = linear(tensor)
  #     loss = criterion(tensor, tensor_y)
  #     optimizer.zero_grad()
  #     loss.backward()
  #     optimizer.step()

  #     if (i + 1) % 10 == 0:
  #         print("{}: {}".format(i + 1, loss.data.item()))

  # tensor_y = torch.tensor(y)
  # input_ = torch.tensor(x)
  # tensor = embedding(input_)
  # output, (tensor, c_n) = lstm(tensor)
  # tensor = tensor[0]
  # tensor = linear(tensor)
  # print(tensor)
  # print(torch.argmax(tensor))
```
