import numpy as np
import my_func

#各层定义
#二维
class TwoDimLayer:
  def __init__(self, length):
    self.value = [[0 for i in range(length)] for j in range(length)]
    self.length = length

  def changeOneValue(self):
    result = []
    for i in range(self.length):
      for j in range(self.length):
        result.append(self.value[i][j])
    return result

  def changeFromValue(self, value):
    result = []
    for i in range(self.length):
      for j in range(self.length):
        self.value[i][j] = value[i * self.length + j]

class OneDimLayer:
  def __init__(self, a):
    self.value = [0 for i in range(a)]

  def setValue(self, value):
    self.value = value

  def setValueByDim(self, oneDimLayer):
    h = len(oneDimLayer)
    w = len(oneDimLayer[0])
    for i in range(h):
      for j in range(j):
        self.value[i * w + j] = oneDimLayer[i][j]
  
  #代价函数为差平方的误差计算（求导）
  def calError(self, output, func, error):
    for i in range(len(self.value)):
      error[i] = (output[i] - self.value[i]) * func.calDerByValue(self.value[i])

#卷积计算
class Roll:
  #初始化
  def __init__(self, length, step):
    self.weight = [[0 for i in range(length)] for j in range(length)]
    self.step = step
    self.len = length

  def setWeight(self, weight):
    self.weight = weight

  def setWeightSame(self):
    for i in range(self.len):
      for j in range(self.len):
        self.weight[i][j] = 1.0 / self.len / self.len

  def setWeightRandom(self):
    self.weight = np.random.rand(self.len, self.len)

  #计算
  def calValue(self, input_layer, output_layer):
    h = len(input_layer)
    w = len(input_layer)
    i = 0 
    j = 0
    #print (h, w, self.len, self.step)
    while (i * self.step < h - self.len + 1):
      j = 0
      while  (j * self.step < w - self.len + 1):
        total_value = 0
        for len_h in range(self.len):
          for len_w in range(self.len):
            total_value += input_layer[i + len_h][j + len_w] * self.weight[len_h][len_w]
        #print(i , j)
        output_layer[i][j] = total_value
        j += 1
      i += 1
    #print(output_layer)

  #误差回归（求导）
  def calError(self, output_error, input_error):
    h = len(input_error)
    w = len(input_error)
    for i in range(h):
      for j in range(w):
        input_error[i][j] = 0
    i = 0 
    j = 0
    #print (h, w, self.len, self.step)
    while (i * self.step < h - self.len  + 1):
      j = 0
      while  (j * self.step < w - self.len  + 1):
        for len_h in range(self.len):
          for len_w in range(self.len):
            input_error[i + len_h][j + len_w] += output_error[i][j] * self.weight[len_h][len_w]
        j += 1
      i += 1 

  #学习
  def learn(self, input_layer, output_error, learn_rate):
    h = len(input_layer)
    w = len(input_layer)
    i = 0 
    j = 0
    #print (h, w, self.len, self.step)
    while (i * self.step < h - self.len + 1):
      j = 0
      while  (j * self.step < w - self.len + 1):
        total_value = 0
        for len_h in range(self.len):
          for len_w in range(self.len):
            #print(i, j, len_h, len_w)
            self.weight[len_h][len_w] +=  learn_rate * output_error[i][j] * input_layer[i + len_h][j + len_w]
            #print(self.weight[len_h][len_w])
        j += 1
      i += 1

#池化计算
#待完成

#全连接计算
class Link:
  #初始化
  def __init__(self, input_num, output_num):
    self.weight = [[0 for i in range(input_num)] for j in range(output_num)]
    self.b =  [0 for j in range(output_num)]
    self.out_num = output_num
    self.in_num = input_num

  def setWeight(self, weight, b):
    self.weight = weight
    self.b = b

  def setWeightRandom(self):
    self.weight = np.random.rand(self.out_num, self.in_num)
    for i in range(self.out_num):
      for j in range(self.in_num):
        self.weight[i][j] = self.weight[i][j] / self.out_num
    self.b = np.random.rand(self.out_num)

  #计算
  def calValue(self, input_layer, output_layer, func):
    for i in range(self.out_num):
      output_layer[i] = 0
      for j in range(self.in_num):
        output_layer[i] += input_layer[j] * self.weight[i][j]
      output_layer[i] += self.b[i]
      output_layer[i] = func.calValue(output_layer[i])

  #误差回归（求导）
  def calError(self, output_error, input_layer, input_error, func):
    for i in range(self.in_num):
      input_error[i] = 0
      for j in range(self.out_num):
        input_error[i] += output_error[j] * self.weight[j][i]
      input_error[i] = input_error[i] * func.calDerByValue(input_layer[i])

  #学习
  def learn(self, input_layer, output_error, learn_rate):
    for i in range(self.out_num):
      for j in range(self.in_num):
        self.weight[i][j] += learn_rate * input_layer[j] * output_error[i]
      self.b[i] += learn_rate * output_error[i]
    

#test case
# one_roll = Roll(3, 1)
# one_roll.setWeight([[1,2,1],[1,1,3],[0,0,0]])
# input = [[1,2,3,4,5],
#          [1,1,1,1,1],
#          [1,1,1,1,1],
#          [1,1,1,1,1],
#          [1,1,1,1,1]]
# output = np.random.rand(4,4)
# print(input)
# one_roll.calValue(input, output)
# print(output)

# weightsvalue1 = [[0.2, 0.8],[-0.7,-0.5]]
# weightsvalue2 = [[0.3, 0.5]]
# bvalue1 = [0,0]
# bvalue2 = [0]
# weight1 = Link(2,2)
# weight1.setWeight(weightsvalue1, bvalue1)

# weight2 = Link(2, 1)
# weight2.setWeight(weightsvalue2, bvalue2)

# input_items = [0.3, -0.7]
# center_items = [0 for i in range(2)]
# output_items = [0.1]
# result = [0]

# func = my_func.Sigmoid()
# weight1.calValue(input_items, center_items, func)
# print(center_items)
# weight2.calValue(center_items, result, func)
# print(result)