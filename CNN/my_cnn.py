import my_layer
import my_func
import numpy as np

#卷积器
class Conv:

  #卷积器 初始边长 深度
  def __init__(self, first_len, deeps):
    self.deep_num = len(deeps)
    self.deeps = deeps
    self.layers = []
    self.errors = []
    self.rolls = []

    self.layers.append(my_layer.TwoDimLayer(first_len))
    self.errors.append(my_layer.TwoDimLayer(first_len))

    for i in range(self.deep_num):
      self.rolls.append(my_layer.Roll(deeps[i][0], deeps[i][1]))
      self.rolls[i].setWeightSame()
      length = 1 + int((self.layers[i].length  - deeps[i][0]) / deeps[i][1])
      print("cnn:", length)
      self.layers.append(my_layer.TwoDimLayer(length))
      self.errors.append(my_layer.TwoDimLayer(length))

  def calValue(self, input):
    #正向传播
    self.layers[0].value = input
    for i in range(self.deep_num):
      self.rolls[i].calValue(self.layers[i].value, self.layers[i + 1].value)
    return self.layers[self.deep_num]

  #导数回传学习
  def back_learn(self, output_error, learn_rate):
    #反向传播
    self.errors[self.deep_num] = output_error
    for i in range(self.deep_num):
      if (i == 0):
        self.rolls[self.deep_num - i - 1].calError(output_error.value, self.errors[self.deep_num - i - 1].value)
      else:
        self.rolls[self.deep_num - i - 1].calError(self.errors[self.deep_num - i].value, self.layers[self.deep_num - i - 1].value)

    #梯度优化
    for i in range(self.deep_num):
      self.rolls[i].learn(self.layers[i].value, self.errors[i + 1].value, learn_rate)
     
  def printWeight(self, deep_num):
    print(self.rolls[deep_num].weight)
