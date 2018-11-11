import math

#常用的激活函数库

class Self:
  def calValue(self, x):
    return x

  #导数计算(输入参数为函数计算值)
  def calDerByValue(self, value):
    return 1

  #导数计算
  def calDer(self, x):
    return 1

#用于输出层神经元
class Sigmoid:
  def calValue(self, x):
    #if (x > 20):
    #  print(x)
    y =  1.0/(1 + math.exp(-x))
    return y

  #导数计算(输入参数为函数计算值)
  def calDerByValue(self, value):
  	return value * (1.0 - value)

  #导数计算
  def calDer(self, x):
  	return calDerByValue(calValue(x))

#用于输出层神经元
class Tanh:
  def calValue(self, x):   
    exp_double_x = math.exp(-2 * x)
    y =  (1.0 - exp_double_x)/(1.0 + exp_double_x)
    return y

  #导数计算(输入参数为函数计算值)
  def calDerByValue(self, value):
    return 1.0 - value * value

  #导数计算
  def calDer(self, x):
    return calDerByValue(calValue(x))

#用于隐藏层神经元
class ReLU:
  def calValue(self, x):   
    if (x > 0):
      return x
    return 0

  #导数计算(输入参数为函数计算值)
  def calDerByValue(self, value):
    if (x > 0):
      return 1
    return 0

  #导数计算
  def calDer(self, x):
    if (x > 0):
      return 1
    return 0


