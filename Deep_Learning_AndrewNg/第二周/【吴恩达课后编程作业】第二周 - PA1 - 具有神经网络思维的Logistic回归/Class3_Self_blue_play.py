import numpy as np
import matplotlib.pyplot as plt
import h5py 
from lr_utils import load_dataset
import seaborn as sns
from sklearn.metrics import confusion_matrix
train_set_x_orig,train_set_y_orig,test_set_x_roig,test_set_y_orig,classes=load_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_roig.reshape(test_set_x_roig.shape[0],-1).T
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255
def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a
def initialize_zeros(dim):
    
    w=np.zeros(shape=(dim,1))
    b=0
    #使用断言来查看是否正确
    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return(w,b)
    
def propagate(w,b,X,Y):


    m=X.shape[1] 


    A=sigmoid(np.dot(w.T,X)+b) #计算激活值

    cost = (-1/m) * np.sum(Y * np.log(A)+(1-Y)*(np.log(1-A)))

    cost = np.squeeze(cost)
    #反向传播（计算w和b需要改变多少）
    dz=A-Y
    dw = (1/m) * np.dot(X,dz.T)
    db = (1/m) * np.sum(dz)
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
def optimize (w,b,X,Y,num_itertions,learning_rate,print_cost = False):


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


def predict(w,b,X):
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

#这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 2000, learning_rate = 0.005, print_cost = True)



LR_predictions  = np.vstack((d['Y_prediction_test'].T,d['Y_prediciton_train'].T))

Y=np.vstack((test_set_y_orig.T,train_set_y_orig.T)).T

print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
       "% " + "(正确标记的数据点所占的百分比)")

# 假设 Y 是真实标签，LR_predictions 是模型的预测

print(Y.shape)
print(LR_predictions.shape)

cm = confusion_matrix(Y.T, LR_predictions)
# 使用 seaborn 绘制混淆矩阵
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

