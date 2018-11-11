import my_layer
import my_func
import numpy as np

#一般神经网络定义
class Normal_NN:

  #网络结构（层数），激活函数
  def __init__(self, deeps, func):
    self.deep_num = len(deeps)
    self.func = func
    self.layers = []
    self.links = []
    self.errors = []
    for i in range(self.deep_num  - 1):
      self.layers.append(my_layer.OneDimLayer(deeps[i]))
      self.errors.append(my_layer.OneDimLayer(deeps[i]))

      self.links.append(my_layer.Link(deeps[i], deeps[i + 1]))
      self.links[i].setWeightRandom()
      # print(self.links[i].weight)
      # print(self.links[i].b)
    self.layers.append(my_layer.OneDimLayer(deeps[self.deep_num - 1]))
    self.errors.append(my_layer.OneDimLayer(deeps[self.deep_num - 1]))

  def calValue(self, input):
    #正向传播
    self.layers[0].value = input
    for i in range(self.deep_num - 1):
      self.links[i].calValue(self.layers[i].value, self.layers[i + 1].value, self.func)
    return self.layers[self.deep_num - 1].value
    

  def learn(self, input, output, learn_rate):

  	#正向传播
    self.layers[0].value = input
    for i in range(self.deep_num - 1):
      self.links[i].calValue(self.layers[i].value, self.layers[i + 1].value, self.func)
      #print(self.layers[i + 1].value) 

    #反向传播
    for i in range(self.deep_num - 1):
      if (i == 0):
        self.layers[self.deep_num - i - 1].calError(output, self.func, self.errors[self.deep_num - i - 1].value)
      else:
        self.links[self.deep_num - i - 1].calError(self.errors[self.deep_num - i].value, self.layers[self.deep_num - i - 1].value, \
        self.errors[self.deep_num - i - 1].value, self.func)
    self.links[0].calError(self.errors[1].value, self.layers[0].value, self.errors[0].value, my_func.Self())

    #梯度优化
    for i in range(self.deep_num - 1):
      self.links[i].learn(self.layers[i].value, self.errors[i + 1].value, learn_rate)

  def getErrors(self, layer_num):
    return self.errors[layer_num].value
# test case
# func =  my_func.Sigmoid()
# nn = Normal_NN([5,3,2], func)

# t = 10000
# for i in range(t):
#   input_items = np.random.rand(5)
#   output_items = []
#   a = 0
#   b = 1
#   if (input_items[0] + input_items[1] + input_items[2] + input_items[3] + input_items[4] > 2.5):
#   	a = 1;
#   	b = 0;

#   output_items.append(a)
#   output_items.append(b)
#   #print(input_items)
#   #print(input_items[0] + input_items[1] + input_items[2] + input_items[3] + input_items[4])
#   nn.learn(input_items, output_items, 1)

# #验证学习结果
# input_test = [1.11, 0.2, 0.3, 0.4, 0.5]
# output = nn.calValue(input_test)
# print(input_test[0] + input_test[1] + input_test[2] + input_test[3] + input_test[4])
# print(output)

# input_test = [0.6, 0.2, 0.9, 0.4, 0.5]
# output = nn.calValue(input_test)
# print(input_test[0] + input_test[1] + input_test[2] + input_test[3] + input_test[4])
# print(output)

# input_test = [0.7, 0.2, 0.3, 0.4, 0.5]
# output = nn.calValue(input_test)
# print(input_test[0] + input_test[1] + input_test[2] + input_test[3] + input_test[4])
# print(output)

# input_test = [0.8, 0.7, 0.3, 0.4, 0.5]
# output = nn.calValue(input_test)
# print(input_test[0] + input_test[1] + input_test[2] + input_test[3] + input_test[4])
# print(output)