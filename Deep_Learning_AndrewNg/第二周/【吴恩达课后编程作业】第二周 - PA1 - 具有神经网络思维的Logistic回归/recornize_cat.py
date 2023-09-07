import numpy as np
import matplotlib.pyplot as plt
# 是在 Python 中导入 matplotlib 库的 pyplot 模块并为其指定一个简称 plt 的语句。matplotlib 是一个非常流行的 Python 数据可视化库，它提供了一套全面的绘图工具来制作各种静态、动态或交互式的图形。
# pyplot 是 matplotlib 的一个子模块，通常被认为是该库的核心。它提供了一个类似于 MATLAB 的绘图界面，使得创建图形变得非常简单。

import h5py 
# import h5py 是 Python 中引入 h5py 库的命令。
# h5py 是一个 Python 库，它提供了一种高效的方式来读取和写入 HDF5 格式的文件。
# HDF5（Hierarchical Data Format version 5）是一个流行的数据存储格式，常用于大型数据集，如科学计算或深度学习中的训练数据。
# HDF5 文件可以包含大量的数据集并支持高效的并行IO操作，它提供了一种结构化的方式来存储数据，其中数据可以被组织为不同的组和数据集。

from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes=load_dataset()

# print(train_set_x_orig.shape) #(209, 64, 64, 3) 训练数据集是由209个 64*64的3色域图片组成
# print(train_set_y_orig.shape) #(1, 209) 标签,1维度的209个标签

#打印测试单个
# index=25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# # print("label:"+str(train_set_y_orig[:,index]))

#降维度
#将 A(a,b,c,d)的矩阵变成 A(b*c*d,a)矩阵
#X_flatten =X.reshape(X.shape[0],-1).T 
#将训练集降维并且转制
#为什么要转制?
#这个数据集有209行,就是代表有209个数据,现在需要将非209的数据组合到一起,并且将矩阵变为209列,这样符合每一列代表一个图像.

# a=np.random.rand(5,2,3)
# print(a)
# print("-----------------------------------------")
# a_f=a.reshape(a.shape[0],-1).T
# print(a_f)

#Dimensionally reduce and Transpose the Training Data_set

# print(train_set_x_orig.shape) #(209, 64, 64, 3)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
# print(train_set_x_flatten.shape) #(12288, 209)

#Dimensionally reduce and Transpose the Testing Data_Set

# print(test_set_x_orig.shape) #(50, 64, 64, 3)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
# print(test_set_x_flatten.shape) #(12288, 50)

#将每一行的值控制在0-1之间，因为所有的都是图片的RGB，所以可以除以255

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255


# 建立神经网络的主要步骤是：

# 定义模型结构（例如输入特征的数量）

# 初始化模型的参数

# 循环：

# 3.1 计算当前损失（正向传播）

# 3.2 计算当前梯度（反向传播）

# 3.3 更新参数（梯度下降）


#1.构建sigmoid（）,计算sigmoid（w ^ T x + b）
#sigmoid()=1/1+e^-z
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

#2.初始化w和b
def initialize_zeros(dim):

    #为线性方程Z=W^T * X +b 做准备
    #为w创建一个维度为（dim，1）的0向量，并将b初始化0
    w=np.zeros(shape=(dim,1))
    b=0
    #使用断言来查看是否正确
    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return(w,b)
    #返回w,b

#初始化参数的函数已经构建好了，现在就可以执行“前向”和“后向”传播步骤来学习参数。也就是学习到最低点的w和b

#渐变函数
def propagate(w,b,X,Y):
    #实现前向传播和后向传播成本函数和梯度
    
    #传入参数：
    #   w -权重，大小不等的数组（图像高*图像的宽*3,1）
    #   b -偏差
    #   X -输入的矩阵 ，类型为（图像高*图像的宽*3,训练数）#(12288, 209)
    #   Y -标签矩阵 

    #返回：
    #  cost- 单个的点相较于那条线的成本的总和
    #  dw - w需要改变多少
    #  db - b需要改变多少

    m=X.shape[1] # 访问X的第二个元素，在这个例子里面代表样本的个数

    #正向传播
    # Z=w.T*X+b
    A=sigmoid(np.dot(w.T,X)+b) #计算激活值
    cost = (-1/m) * np.sum(Y * np.log(A)+(1-Y)*(np.log(1-A)))

    #反向传播（计算w和b需要改变多少）
    dz=A-Y
    dw = (1/m) * np.dot(X,dz.T)
    db = (1/m) * np.sum(dz)

    #将损失函数标量化，因为基于Y矩阵，所以有一个为1的下标
    cost = np.squeeze(cost)


    #确保数据是否正确
    assert(dw.shape == w.shape)
    assert(db.dtype == float)#断言会失败，导致程序抛出一个 AssertionError 异常。
    assert(cost.shape == ())

    #创建一个字典，把dw和db保存起来。
    grads={
        "dw":dw,
        "db":db
    }
    
    return (grads,cost)


# print("====================测试propagate====================")
# #初始化一些参数
# w, b, X, Y = np.array([[2], [2]]),1, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))


#通过最最小化成本函数J来学习w和b
def optimize (w,b,X,Y,num_itertions,learning_rate,print_cost = False):

    # 此函数通过运行梯度下降算法来优化w和b
    
    # 参数：
    #     w  - 权重，大小不等的数组（num_px * num_px * 3，1）
    #     b  - 偏差，一个标量
    #     X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
    #     Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
    #     num_iterations  - 优化循环的迭代次数
    #     learning_rate  - 梯度下降更新规则的学习率,就是那个阿尔法
    #     print_cost  - 每100步打印一次损失值
    # 返回：
    #     params  - 包含权重w和偏差b的字典
    #     grads  - 包含权重和偏差相对于成本函数的梯度的字典
    #     成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。
    
    # 提示：
    # 我们需要写下两个步骤并遍历它们：
    #     1）计算当前参数的成本和梯度，使用propagate（）。
    #     2）使用w和b的梯度下降法则更新参数。

    costs=[]

    for i in range (num_itertions):
        grads, cost =propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        #公式
        w = w-learning_rate *dw
        b = b-learning_rate *db

        #记录成本
        if i %100 ==0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i % 100 ==0):
            print("迭代的次数: %i ， 误差值： %f" % (i,cost))

    #创建字典保存w和b
    params={"w":w,"b":b}

    grads={"dw":dw,"db":db}

    return (params,grads,costs)

# print("====================测试optimize====================")
# #([[1],[2]])一维的数组，有两个元素[1]和[2]
# w,b,X,Y=np.array([[1],[2]]), 2 , np.array([[1,2],[3,4]]), np.array([[1,0]])
# params,grads,costs=optimize(w,b,X,Y,num_itertions=100,learning_rate = 0.009,print_cost = False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))

#optimize函数会输出已学习的w和b的值，我们可以使用w和b来预测数据集X的标签。

#现在我们要实现预测函数predict（）。计算预测有两个步骤：

#1.计算 Y ^ = A = σ ( w^T*X + b ) 
#2.将a变为0（如果激活值<=0.5）或者1（如果激活值>0.5）,a 或 A 通常表示神经元的激活值,A 是sigmoid激活函数σ 的输出。

#然后将预测值存储在向量Y_prediction中。

def predict(w,b,X):

    #使用学习逻辑回归参数logistic（w,b）预测标签是0还是1,
    
    # 参数：
    #     w  - 权重，大小不等的数组（num_px * num_px * 3，1）
    #     b  - 偏差，一个标量
    #     X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
    # 返回：
    #   Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）

    m=X.shape[1] #图片的数量
    Y_prediction =np.zeros((1,m)) #创建都是0的矩阵（1行m列）
    w=w.reshape(X.shape[0],1) #将w转为一个图片参数的累乘，维度为1

    #预测猫在图片中出现的概率
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        #将概率a[0,1]转化为实际预测的p[0.i]
        Y_prediction[0,i] = 1 if A[0,i] >0.5 else 0

    #使用断言
    assert(Y_prediction.shape==(1,m))

    return Y_prediction    

# print("====================测试predict====================")
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# print("predictions = " + str(predict(w, b, X)))

#集合
def model(X_train,Y_train,X_test,Y_test, num_iterations=2000,learning_rate =0.005, print_cost=False):
  
    #参数：
       # X_train   -numpy的数组，维度为（num_px*num_px*3,m_train）的训练集 
       # Y_train   -numpy的数组，维度为（1,m_train）(标签)（矢量）的训练集合

       # X_test   -numpy的数组，维度为（num_px*num_px*3,m_test）的测试集 
       # Y_test   -numpy的数组，维度为（1,m_test）(标签)（矢量）的测试集合

       # num_iterations - 用于优化参数的迭代次数
       # learning_rate --学习率
       # print_cost  - 设置为true以每100次迭代打印成本

    #返回
       #d -包含模型信息的字典

    w,b=initialize_zeros(X_train.shape[0])   #初始化w和b,根据训练集合的第一个参数（那一堆东西）

    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    #从字典“参数”中检索参数w和b
    w,b = parameters["w"],parameters["b"]

    #预测测试/训练集的例子

    Y_prediction_test =predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)

    #打印
    #用于计算数据的平均值：np.mean()
    print("训练集的准确度：",format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100),"%")
    print("测试集的准确度：",format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100),"%")

    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return d
print("====================测试model====================")    

d1=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations =2000,learning_rate =0.01,print_cost=True)
d2=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations =2000,learning_rate =0.001,print_cost=True)
d3=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations =2000,learning_rate =0.0001,print_cost=True)

#绘图
costs1=np.squeeze(d1['costs'])
plt.plot(costs1)
costs2=np.squeeze(d2['costs'])
plt.plot(costs2)
costs3=np.squeeze(d3['costs'])
plt.plot(costs3)

plt.ylabel('cost')
plt.xlabel(('iterations (per hundreds)'))
plt.title("Learning rate =" + str(d1["learning_rate"]))
plt.show()

