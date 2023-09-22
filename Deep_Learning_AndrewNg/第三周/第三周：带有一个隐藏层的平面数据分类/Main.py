import numpy as np #Python科学计算的一些包
import matplotlib.pyplot as plt #画图
from Week_3_DataSet.testCases import * #测试示例
from Week_3_DataSet.planar_utils import * #一些功能

#scikit-learn提供了许多常见的机器学习算法的简单和高效的工具，如分类、回归、聚类和降维等。它还包括了用于模型选择和评估的工具，如交叉验证和各种性能指标。
#导数数据挖掘和数据分析的一些包
import sklearn
import sklearn.datasets #它提供了一系列用于测试和学习的数据集。这个模块包含了多种类型的数据集
import sklearn.linear_model #它包含了许多线性模型，这些模型可以用于各种任务，如回归、分类和异常检测。

#1.定义神经网络的结构
def layer_sizes(X,Y):
    # X:样本集合 维度（数据的特征数量，数据量）  每一个数据有两个特征，即：每一个点有X和Y两个坐标值
    # Y：标签的集合（标签数量，数据量） 每一个数据有一个标签，即：0/1 == 红/蓝色
 
    Input_layer=X.shape[0]#输入层的神经元数量，它等于数据的特征数量，每个数据点有两个特征（x ，y ）坐标
    Hidden_layer=4 #隐藏层单元的数量，设置4
    Output_layer=Y.shape[0]#，表示输出层只有一个神经元来预测一个标签（0或1）。

    return (Input_layer,Hidden_layer,Output_layer)


#2.初始化函数
def init_parameters(Input_layer,Hidden_layer,Output_layer):
    np.random.seed(2) 

    #由于只有一层隐藏层，所以可以显式的写出初始化W1和W2

    W1=np.random.randn(Hidden_layer,Input_layer)*0.01 #W的维度（当前层的单元数，上一层的单元数）
    b1=np.zeros(shape=(Hidden_layer,1))  #b的维度（当前层的单元数，1）
    W2=np.random.randn(Output_layer,Hidden_layer)*0.01
    b2=np.zeros(shape=(Output_layer,1)) 

    #断言测试个格式
    assert(W1.shape==(Hidden_layer,Input_layer))
    assert(b1.shape==( Hidden_layer , 1))
    assert(W2.shape==(Output_layer,Hidden_layer))
    assert(b2.shape==(Output_layer,1))

    parameters ={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters

#3.激活函数
#Numpy里面有thah函数，但是为了练习还是自己写一个吧
def tanh(z): #传入一个线性的函数
    A=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return A #A是非线性输出

#因为是二分类任务，所以对于最后的结果输出一定要用sigmoid函数
def sigmoid(z):
     A=1/(1+np.exp(-z))
     return A
     
#4.构建前向传播(不用Y)
def propagate(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    Z1=np.dot(W1,X)+b1
    A1=tanh(Z1)

    #输出层计算
    Z2=np.dot(W2,A1)+b2

    #因为是二分类任务，所以对于最后的结果输出一定要用sigmoid函数
    A2=sigmoid(Z2)  #A2就是最后的输出

    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    } #为什么要存呢？

    return (A2,cache)

# #测试forward_propagation
# print("=========================测试forward_propagation=========================") 
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = propagate(X_assess, parameters)
# print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))


#5.损失函数构建
# 首先，A2是输出的结果，也就是预测，也就是Yhat，
# 其次，我们需要真实的标签Y
def Cost_Func(A2,Y):
      m=Y.shape[1] #总数据数

      sum_cost=-(
                 np.multiply(np.log(A2),Y)
                     +
                 np.multiply(np.log(1-A2),(1-Y))
                 )  
      cost=np.sum(sum_cost)/m

      #当前通过矩阵计算的得到的应该还是一个矩阵，这个不行，我们需要一个数
      cost=float(np.squeeze(cost))

      return cost

# #测试compute_cost
# print("=========================测试compute_cost=========================") 
# A2 , Y_assess , parameters = compute_cost_test_case()
# print("cost = " + str(Cost_Func(A2,Y_assess)))


#后向传播
#首先，我们需要损失函数想相较于最后一层输出的导数
#我们还需要使用链式法则计算每一层的W和b的梯度
def back_propagtion(parameters,cache,X,Y):
     m=Y.shape[1] #总数据数

     W1=parameters["W1"]
     W2=parameters["W2"]

     A1=cache["A1"]
     A2=cache["A2"]

     #就算损失函数较于最后一层输出的导数dz
     #带入公式就好
     dZ2=A2-Y
     dW2=(1/m) *(np.dot(dZ2,A1.T))
     db2=(1/m) * np.sum(dZ2,axis=1,keepdims=True)

     #使用公式 g`(z)=1-a^2
     dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
     dW1=(1/m)*np.dot(dZ1,X.T)
     db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

     back_con={
          "dW1":dW1,
          "dW2":dW2,
          "db2":db2,
          "db1":db1
     }
     return back_con
# #测试backward_propagation
# print("=========================测试backward_propagation=========================")
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

# grads = back_propagtion(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

#梯度下降，也就是更新参数
def update_parameter(parameters,back_con,learning_rate=1.2):
     #原数据
     W1,W2=parameters["W1"],parameters["W2"]
     b1,b2=parameters["b1"],parameters["b2"]
     #它们的导数
     dW1,dW2,db1,db2=back_con["dW1"],back_con["dW2"],back_con["db1"],back_con["db2"]

     W1=W1-learning_rate*dW1
     b1=b1-learning_rate*db1

     W2=W2-learning_rate*dW2
     b2=b2-learning_rate*db2

     update ={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
     }

     return update

# #测试update_parameters
# print("=========================测试update_parameters=========================")
# parameters, grads = update_parameters_test_case()
# parameters = update_parameter(parameters, grads)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def model(X,Y,num_iterations,print_cost=False):
    np.random.seed(3)

    #开始初始化
    Input_layer=layer_sizes(X,Y)[0] #导入输入层、隐藏层和输出层
    Hidden_layer=layer_sizes(X,Y)[1]
    Output_layer=layer_sizes(X,Y)[2]

    parameters=init_parameters(Input_layer,Hidden_layer,Output_layer)

    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]

    #初始化完成，开始神经网络

    for i in range(num_iterations):
         #前向传播，得到最后一个非线性输出结果和其它结果
         A2,cache=propagate(X,parameters)
         #计算损失
         cost=Cost_Func(A2,Y)
         #后向传播
         back_con=back_propagtion(parameters,cache,X,Y)
         #梯度下降/更新parameters
         parameters=update_parameter(parameters,back_con)

         #1000次打印
         if print_cost:
              if i%1000 == 0:
                   print("第 ",i," 次循环，成本为："+str(cost))

    #循环结束，返回最后的值
    return parameters

#构建预测
def predict(parameters,X):
		#parameters - 包含参数的字典类型的变量。
	    #X - 输入数据（n_x，m）
        A2,cache=propagate(X,parameters)

        predic=np.round(A2)

        return predic
# #测试predict
# print("=========================测试predict=========================")

# parameters, X_assess = predict_test_case()
# predictions = predict(parameters, X_assess)
# print("预测的平均值 = " + str(np.mean(predictions)))


def main():
    X,Y=load_planar_dataset()
    parameters=model(X,Y,num_iterations=10000,print_cost=True)
    #draw graphe
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()
    predictions = predict(parameters, X)

    # Y=Y*0  改变颜色玩一下
    # plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图
    # plt.show()
    # print(X.shape)#(2,400) 点的横坐标和纵坐标
    # print(Y.shape)#(1,400) 标签，红0蓝1

    # #原样测试：
    # plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图
    # plt.show()

    # #测试：使用自带的函数逻辑回归的线性分割
    # clf=sklearn.linear_model.LogisticRegressionCV()
    # clf.fit(X.T,Y.T)
    # plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
    # plt.title("Logistic Regression") #图标题、
    # plt.show()
    # LR_predictions  = clf.predict(X.T) #预测结果
    # print(Y.shape)
    # print(LR_predictions.shape)
    # print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
    #         np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
    #     "% " + "(正确标记的数据点所占的百分比)")


if __name__ =="__main__":
     main()   