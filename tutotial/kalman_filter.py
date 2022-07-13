# -*- coding: utf-8 -*-
"""
@司南牧 
"""
import numpy as np


t = np.linspace(1,100,100) # 在1~100s内采样100次
a = 0.5 # 加速度值
position = (a * t**2)/2

position_noise = position+np.random.normal(0,120,size=(t.shape[0])) # 模拟生成GPS位置测量数据（带噪声）
import matplotlib.pyplot as plt
plt.plot(t,position,label='truth position')
plt.plot(t,position_noise,label='only use measured position')


#---------------卡尔曼滤波----------------
# 初始的估计导弹的位置就直接用GPS测量的位置
predicts = [position_noise[0]]
position_predict = predicts[0]

predict_var = 0
odo_var = 120**2 #这是我们自己设定的位置测量仪器的方差，越大则测量值占比越低，对应的是？
v_std = 50 # 测量仪器的方差（这个方差在现实生活中是需要我们进行传感器标定才能算出来的，可搜Allan方差标定）
for i in range(1,t.shape[0]):
  
    dv =  (position[i]-position[i-1]) + np.random.normal(0,50) # 模拟从IMU读取出的速度
    position_predict = position_predict + dv # 利用上个时刻的位置和速度预测当前位置
    predict_var += v_std**2 # 更新预测数据的方差
    # 下面是Kalman滤波
    position_predict = position_predict*odo_var/(predict_var + odo_var)+position_noise[i]*predict_var/(predict_var + odo_var)
    predict_var = (predict_var * odo_var)/(predict_var + odo_var)   # 高斯分布的乘法公式
    predicts.append(position_predict)

    
plt.plot(t,predicts,label='kalman filtered position')

plt.legend()
# plt.show()
plt.save("./kalman.png")

"""[summary]
卡尔曼滤波结果 = 估计值（上一个状态信息）*A + 观测值 （当前信息）*B
估计值：position_predict,上一个状态信息，通过上一个时刻推导
A：odo_var/(predict_var + odo_var)，估计值置信度信息，通过方差进行计算

观测值：position_noise[i]，当前时刻信息，通过传感器获取
B：predict_var/(predict_var + odo_var)，观测值置信度信息，通过方差进行计算

odo_var：位置测量仪器的方差，不变
predict_var：测量仪器的方差 + 累计误差

卡尔曼增益：B，是一个概率值，又称为置信度信息，通过传感器信息估计出来

有点类似于加权滑动平均值

"""