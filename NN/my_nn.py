import math

class Sigmoid:
  def calValue(x):
    y =  1.0/(1 + math.exp(-x))
    if (y < 0):
      print(y)
    return y

  #倒数计算(输入参数为函数计算值)
  def calDerByValue(value):
  	return value * (1.0 - value)

  #倒数计算
  def calDer(x):
  	return calDerByValue(calValue(x))

#层级
class Layer:
  def __init__(self, value):
    self.value = value
    self.size = len(value)

  #归一化
  def sigmoidLayer(self):
  	for i in range(self.size):
  	  self.value[i] = Sigmoid.calValue(self.value[i])

  def printValue(self):
  	print(self.value)

  def getNum(self):
    return len(self.value)

#权重和偏置
class Weight:
  def __init__(self, value, b):
    self.weight = value
    self.b = b
    self.m = len(value)
    self.n = len(value[0])

  def getM(self):
    return self.m

  def getN(self):
    return self.n

  def calValue(self, before_layer, next_layer):
    for i in range(self.m):
      value = self.b[i]
      for j in range(self.n):
        value += before_layer.value[j] * self.weight[i][j]
      #print(value, Sigmoid.calValue(value))
      next_layer.value[i] = Sigmoid.calValue(value)

#神经网络
class Neuralnetwork:
  def __init__(self, layers, weights):
    self.layers = layers
    #self.errors = layers
    self.errors = []
    for i in range(len(layers)):
      self.errors.append(Layer([0] * layers[i].getNum()))

    #参数缩小，防止节点计算结果太大，造成sigmoid函数去趋近于1
    self.weights = weights
    #这段不加貌似训练效果特别差
    for i in range(len(weights)):
      num = len(weights[i].weight)
      for j in range(len(weights[i].weight)):
        for l in range(len(weights[i].weight[j])):
          self.weights[i].weight[j][l] = weights[i].weight[j][l] #/ num
      print(num)

  def checkValue(self, layers, weights):
  	#check layers and weights
    layers_num = len(layers)
    weights_num = len(weights)
    if (layers_num > weights + 1):
      return false

  def calValue(self, input):
    layers_num = len(self.layers)
    for i in range(len(input)):
      self.layers[0].value[i] = input[i]

    for i in range(layers_num - 1):
      self.weights[i].calValue(self.layers[i], self.layers[i + 1])
      self.layers[i].printValue()
    self.layers[layers_num - 1].printValue()

  def printLayer(self, layers_num):
    self.layers[layers_num].printValue()

  def printLayerMax(self, layers_num):
    out = 0
    nums = len(self.layers[layers_num].value)
    for i in range(nums):
      if (self.layers[layers_num].value[i] > self.layers[layers_num].value[out]):
        out = i
    return out

  #训练(输入源、输出源、学习率)
  def train(self, input, output, learn_r):
    self.calValue(input)
    layers_num = len(self.layers)

    #计算误差
    for i in range(layers_num):
      for j in range(len(self.layers[layers_num - i - 1].value)):
        if (i == 0):
          self.errors[layers_num - i - 1].value[j] = Sigmoid.calDerByValue(self.layers[layers_num - i - 1].value[j]) * (output[j] - self.layers[layers_num - i - 1].value[j])
        else:
          total_error = 0;
          for l in range(len(self.layers[layers_num - i].value)):
            total_error += self.errors[layers_num - i].value[l] * self.weights[layers_num - i - 1].weight[l][j]
          self.errors[layers_num - i - 1].value[j] = Sigmoid.calDerByValue(self.layers[layers_num - i - 1].value[j]) * total_error
      
    #修正偏差和偏置
    for i in range(layers_num - 1):
      for j in range(len(self.weights[i].weight)):
        for l in range(len(self.weights[i].weight[j])):
          if (i < layers_num - 1):
            #print(i,j,l,self.weights[i].weight[j][l],self.errors[i + 1].value[j], self.layers[i].value[l])
            self.weights[i].weight[j][l] += learn_r * self.errors[i + 1].value[j] * self.layers[i].value[l];
        self.weights[i].b[j] += learn_r * self.errors[i + 1].value[j]
