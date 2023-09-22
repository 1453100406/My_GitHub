import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset
import sklearn
import sklearn.datasets #它提供了一系列用于测试和学习的数据集。这个模块包含了多种类型的数据集
import sklearn.linear_model #它包含了许多线性模型，这些模型可以用于各种任务，如回归、分类和异常检测。
import seaborn as sns
from sklearn.metrics import confusion_matrix

X_1,Y_1,X_2,Y_2,classes=load_dataset()
# print(X_1.shape)
# print(X_2.shape)
X_1=X_1.reshape(X_1.shape[0],-1)/255
X_2=X_2.reshape(X_2.shape[0],-1)/255
# print(X_1.shape)
# print(X_2.shape)
X=np.vstack((X_1,X_2)) #(259, 12288)
Y=np.vstack((Y_1.T,Y_2.T))#(259, 1)
# print(X.shape)
# print(Y.shape)
X=X/255
X=X.T
Y=Y.T


clf=sklearn.linear_model.LogisticRegressionCV(max_iter=10000)

clf.fit(X.T,Y.T)
LR_predictions  = clf.predict(X.T) #预测结果

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
